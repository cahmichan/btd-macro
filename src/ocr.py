"""Persistent Windows.Media.Ocr background worker using built-in WinRT OCR."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_PS_OCR_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Media.Ocr, ContentType=WindowsRuntime]

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]

function Await($WinRtTask, $ResultType) {
    $netTask = $asTaskGeneric.MakeGenericMethod($ResultType).Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}

$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {
    $lang = [Windows.Globalization.Language, Windows.Globalization, ContentType=WindowsRuntime]::new('en-US')
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
}
if ($null -eq $engine) {
    throw "Unable to initialize Windows.Media.Ocr.OcrEngine"
}

[Console]::Out.WriteLine("READY")
[Console]::Out.Flush()

while ($null -ne ($line = [Console]::In.ReadLine())) {
    if ($line -eq "__EXIT__") { break }
    $path = $line.Trim()
    if ([string]::IsNullOrEmpty($path) -or -not (Test-Path -LiteralPath $path)) {
        [Console]::Out.WriteLine("[]")
        [Console]::Out.Flush()
        continue
    }
    try {
        $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
        $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
        $bitmap = $null
        try {
            $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
            $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
            $ocrResult = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
            $lines = @()
            foreach ($l in $ocrResult.Lines) {
                if (-not [string]::IsNullOrWhiteSpace($l.Text)) {
                    $lines += [string]$l.Text
                }
            }
            if ($lines.Count -eq 0) {
                $json = "[]"
            } elseif ($lines.Count -eq 1) {
                $single = ConvertTo-Json -InputObject $lines[0] -Compress
                $json = "[$single]"
            } else {
                $json = ConvertTo-Json -InputObject $lines -Compress
                if (-not $json.StartsWith("[")) {
                    $json = "[$json]"
                }
            }
            [Console]::Out.WriteLine($json)
        } finally {
            if ($null -ne $bitmap) { $bitmap.Dispose() }
            if ($null -ne $stream) { $stream.Dispose() }
        }
    } catch {
        [Console]::Out.WriteLine("[]")
    }
    [Console]::Out.Flush()
}
"""


class WindowsOcrEngine:
    """Thread-safe persistent PowerShell worker wrapping Windows.Media.Ocr."""

    def __init__(
        self,
        timeout_sec: float = 4.0,
        startup_timeout_sec: float = 15.0,
        script_override: Optional[str] = None,
    ) -> None:
        self.timeout_sec = max(0.1, float(timeout_sec))
        self.startup_timeout_sec = max(0.5, float(startup_timeout_sec))
        self._script = script_override if script_override is not None else _PS_OCR_SCRIPT

        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen[str]] = None
        self._out_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._temp_dir: Optional[Path] = None
        self._closed = False

    def _ensure_temp_dir(self) -> Path:
        if self._temp_dir is None or not self._temp_dir.is_dir():
            self._temp_dir = Path(tempfile.mkdtemp(prefix="btd_ocr_"))
        return self._temp_dir

    def _stop_worker_locked(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write("__EXIT__\n")
                    proc.stdin.flush()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass

    def _start_worker_locked(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True

        self._stop_worker_locked()
        self._out_queue = queue.Queue()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", self._script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError:
            return False

        self._proc = proc
        out_q = self._out_queue

        def _reader() -> None:
            stdout = proc.stdout
            if stdout is None:
                out_q.put(None)
                return
            try:
                for raw_line in stdout:
                    out_q.put(raw_line.rstrip("\r\n"))
            except Exception:
                pass
            finally:
                out_q.put(None)

        self._reader_thread = threading.Thread(target=_reader, name="WindowsOcrReader", daemon=True)
        self._reader_thread.start()

        try:
            ready = out_q.get(timeout=self.startup_timeout_sec)
        except queue.Empty:
            self._stop_worker_locked()
            return False

        if ready is None or ready.strip() != "READY":
            self._stop_worker_locked()
            return False
        return True

    @staticmethod
    def _prepare_ocr_image(
        bgr_or_gray_crop: np.ndarray,
        invert: bool,
        scale: float,
    ) -> Optional[np.ndarray]:
        if bgr_or_gray_crop is None or bgr_or_gray_crop.size == 0:
            return None
        if bgr_or_gray_crop.ndim == 3:
            gray = cv2.cvtColor(bgr_or_gray_crop, cv2.COLOR_BGR2GRAY)
        elif bgr_or_gray_crop.ndim == 2:
            gray = np.ascontiguousarray(bgr_or_gray_crop, dtype=np.uint8)
        else:
            return None

        if invert:
            gray = cv2.bitwise_not(gray)
        if scale and abs(float(scale) - 1.0) > 1e-3:
            gray = cv2.resize(
                gray,
                None,
                fx=float(scale),
                fy=float(scale),
                interpolation=cv2.INTER_CUBIC,
            )
        border_val = 255 if invert else 0
        padded = cv2.copyMakeBorder(gray, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=border_val)
        return np.ascontiguousarray(padded, dtype=np.uint8)

    def _query_worker_locked(self, image_path: Path) -> Optional[list[str]]:
        if not self._start_worker_locked():
            return None
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            return None

        try:
            proc.stdin.write(f"{image_path}\n")
            proc.stdin.flush()
        except OSError:
            self._stop_worker_locked()
            return None

        try:
            line = self._out_queue.get(timeout=self.timeout_sec)
        except queue.Empty:
            self._stop_worker_locked()
            return None

        if line is None:
            self._stop_worker_locked()
            return None

        line = line.strip()
        if not line:
            return []
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed, str):
            text = parsed.strip()
            return [text] if text else []
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return []

    def read_lines(
        self,
        bgr_or_gray_crop: Optional[np.ndarray],
        invert: bool = True,
        scale: float = 3.0,
    ) -> list[str]:
        """Run Windows OCR on an image crop and return non-empty recognized text lines."""
        if bgr_or_gray_crop is None:
            return []
        prepared = self._prepare_ocr_image(bgr_or_gray_crop, invert=invert, scale=scale)
        if prepared is None:
            return []

        with self._lock:
            if self._closed:
                return []
            temp_dir = self._ensure_temp_dir()
            tmp_path = temp_dir / f"{uuid.uuid4().hex}.png"
            try:
                ok, buf = cv2.imencode(".png", prepared)
                if not ok:
                    return []
                buf.tofile(str(tmp_path))

                result = self._query_worker_locked(tmp_path)
                if result is None and not self._closed:
                    # Auto-restart once after unexpected pipe failure or timeout
                    result = self._query_worker_locked(tmp_path)
                return result if result is not None else []
            except Exception:
                return []
            finally:
                try:
                    if tmp_path.exists():
                        os.unlink(tmp_path)
                except OSError:
                    pass

    def close(self) -> None:
        """Cleanly shut down the persistent PowerShell OCR worker and remove temp files."""
        with self._lock:
            self._closed = True
            self._stop_worker_locked()
            if self._temp_dir is not None:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                self._temp_dir = None
