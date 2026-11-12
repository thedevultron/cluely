"""System-audio capture: computer sound only (WASAPI loopback), mic optional."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("cluely.audio")

TARGET_RATE = 16000
CHUNK_SAMPLES = 1024  # ~64ms @16kHz
# Latency / VAD tuning exposed for tests and diagnostics.
VAD_DEFAULT = 300  # linear abs-mean int16 threshold (legacy *_db name)
VAD_MIN = 50
VAD_MAX = 2000
HANGOVER_FRAMES = 2  # ~128ms trailing audio after speech ends
PREROLL_SAMPLES = 4800  # ~300ms @16kHz (>=3200 = 200ms onset protection)
AGC_TARGET_RMS = 1000.0
AGC_MAX_GAIN = 8.0

try:
    import sounddevice as sd
    _HAS_SD = True
except Exception as e:  # pragma: no cover
    sd = None  # type: ignore
    _HAS_SD = False
    log.warning("sounddevice unavailable: %s", e)

try:
    import soundcard as sc
    _HAS_SC = True
except Exception as e:  # pragma: no cover
    sc = None  # type: ignore
    _HAS_SC = False
    log.warning("soundcard unavailable: %s", e)


def list_microphones() -> List[Tuple[int, str]]:
    """Return [(index, name)] for input-capable devices."""
    if not _HAS_SD:
        return []
    try:
        devices = sd.query_devices()
        out: List[Tuple[int, str]] = []
        for i, d in enumerate(devices):
            try:
                if int(d.get("max_input_channels", 0)) > 0:
                    out.append((i, str(d.get("name", f"Device {i}"))))
            except Exception:
                continue
        return out
    except Exception as e:
        log.warning("list_microphones failed: %s", e)
        return []


def _ensure_com() -> bool:
    """CoInitialize the calling thread for WASAPI/COM calls. Returns True if
    this call owns a COM ref (caller must CoUninitialize). Never raises.

    S_OK (0) and S_FALSE (1) both obtain a reference that must be balanced
    with CoUninitialize (even when already initialized). Other HRESULTs
    (e.g. RPC_E_CHANGED_MODE) mean no ref was taken.
    """
    try:
        import ctypes
        ole = ctypes.windll.ole32
        try:
            ole.CoInitialize.argtypes = [ctypes.c_void_p]
            ole.CoInitialize.restype = ctypes.c_long
        except Exception:
            pass
        rc = ole.CoInitialize(None)
        try:
            return int(rc) in (0, 1)
        except Exception:
            return False
    except Exception:
        return False


def _release_com() -> None:
    try:
        import ctypes
        ctypes.windll.ole32.CoUninitialize()
    except Exception:
        pass


def list_loopbacks() -> List[str]:
    """Return speaker/loopback names via soundcard (canonical enumeration)."""
    if not _HAS_SC:
        return []
    mine = _ensure_com()  # soundcard needs COM on THIS thread (else 0x800401f0)
    try:
        names: List[str] = []
        seen = set()
        for m in sc.all_microphones(include_loopback=True):
            try:
                if getattr(m, "isloopback", False):
                    n = str(m.name)
                    if n not in seen:
                        seen.add(n)
                        names.append(n)
            except Exception:
                continue
        return names
    except Exception as e:
        log.warning("list_loopbacks failed: %s", e)
        return []
    finally:
        try:
            if mine:
                _release_com()
        except Exception:
            pass


def detect_system_audio(preferred: str = "") -> tuple[str, List[str]]:
    """Auto-detect the computer's internal sound device.

    Returns (selected_name, all_names). selected_name is "" when nothing found.
    preferred (substring, case-insensitive) wins when it matches; otherwise
    the first loopback device (system default output) is used.
    """
    try:
        names = list_loopbacks()
    except Exception:
        names = []
    if not names:
        return "", []
    pref = (preferred or "").strip().lower()
    if pref:
        for n in names:
            try:
                if pref in n.lower() or n.lower() in pref:
                    return n, names
            except Exception:
                continue
    return names[0], names


def _resample_mono(data: np.ndarray, src_rate: int, dst_rate: int = TARGET_RATE) -> np.ndarray:
    """Resample float32/int16 mono to dst_rate. Fallback linear interp."""
    try:
        if data is None:
            return np.zeros(0, dtype=np.float32)
    except Exception:
        return np.zeros(0, dtype=np.float32)
    try:
        if data.size == 0:
            return np.zeros(0, dtype=np.float32)
    except Exception:
        pass
    try:
        arr = np.asarray(data, dtype=np.float32).ravel()
    except Exception:
        return np.zeros(0, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    try:
        src_rate = int(src_rate)
        dst_rate = int(dst_rate)
    except Exception:
        return arr
    if src_rate <= 0 or dst_rate <= 0:
        return arr
    if src_rate == dst_rate:
        return arr
    try:
        ratio = dst_rate / float(src_rate)
        new_len = max(1, int(len(arr) * ratio))
        old_idx = np.linspace(0, len(arr) - 1, num=len(arr))
        new_idx = np.linspace(0, len(arr) - 1, num=new_len)
        return np.interp(new_idx, old_idx, arr).astype(np.float32)
    except Exception:
        return arr[: min(len(arr), CHUNK_SAMPLES)].astype(np.float32)


def _to_mono(frame) -> np.ndarray:
    """Average multi-channel frame to mono float32."""
    try:
        if frame is None:
            return np.zeros(0, dtype=np.float32)
    except Exception:
        return np.zeros(0, dtype=np.float32)
    try:
        arr = np.asarray(frame, dtype=np.float32)
        # np.asarray(None, float32) -> 0-d nan; treat scalar/None as empty.
        if arr.ndim == 0:
            return np.zeros(0, dtype=np.float32)
        if arr.ndim == 2:
            return arr.mean(axis=1).astype(np.float32)
        return arr.ravel().astype(np.float32)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _truncate_to_int16_len(buf: bytes) -> bytes:
    """Drop a trailing odd byte so frombuffer(..., int16) never raises."""
    try:
        if not buf:
            return b""
        n = len(buf) - (len(buf) % 2)
        if n == 0:
            return b""
        if n == len(buf):
            return bytes(buf)
        return bytes(buf[:n])
    except Exception:
        try:
            return bytes(buf) if buf else b""
        except Exception:
            return b""


def mix_audio_frames(mic_chunk: bytes, loopback_chunk: bytes) -> bytes:
    """Mix two int16 mono byte buffers with clipping."""
    try:
        mic_chunk = _truncate_to_int16_len(mic_chunk) if mic_chunk else b""
        loopback_chunk = _truncate_to_int16_len(loopback_chunk) if loopback_chunk else b""
        a = np.frombuffer(mic_chunk, dtype=np.int16).astype(np.float32) if mic_chunk else np.zeros(0, dtype=np.float32)
        b = np.frombuffer(loopback_chunk, dtype=np.int16).astype(np.float32) if loopback_chunk else np.zeros(0, dtype=np.float32)
        n = max(len(a), len(b))
        if n == 0:
            return b""
        if len(a) < n:
            a = np.pad(a, (0, n - len(a)))
        if len(b) < n:
            b = np.pad(b, (0, n - len(b)))
        mixed = np.clip(a + b, -32768, 32767).astype(np.int16)
        return mixed.tobytes()
    except Exception as e:
        log.warning("mix failed: %s", e)
        return mic_chunk or loopback_chunk or b""


def _safe_energy(v) -> float:
    try:
        f = float(v)
    except Exception:
        return 0.0
    # NaN / +-inf are not speech energy; clamp negatives (abs-mean is >=0).
    try:
        import math as _m
        if _m.isnan(f) or _m.isinf(f):
            return 0.0
    except Exception:
        pass
    return f if f > 0.0 else 0.0


def _safe_threshold(v) -> float:
    try:
        import math as _m
        f = float(v)
        if _m.isnan(f) or _m.isinf(f):
            return 300.0
        return f if f >= 0.0 else 0.0
    except Exception:
        return 300.0


def classify_speaker(mic_energy: float, lb_energy: float, threshold: float) -> str:
    """Lightweight dual-energy diarization. Returns interviewer/candidate/silence/both."""
    mic_energy = _safe_energy(mic_energy)
    lb_energy = _safe_energy(lb_energy)
    threshold = _safe_threshold(threshold)
    mic_active = mic_energy >= threshold
    lb_active = lb_energy >= threshold
    if not mic_active and not lb_active:
        return "silence"
    if lb_active and not mic_active:
        return "interviewer"
    if mic_active and not lb_active:
        return "candidate"
    # both active: whoever dominates by 1.4x wins, else both
    try:
        if lb_energy > mic_energy * 1.4:
            return "interviewer"
        if mic_energy > lb_energy * 1.4:
            return "candidate"
    except Exception:
        pass
    return "both"


class AudioEngine:
    """Capture computer (system) sound via WASAPI loopback, mic optional.

    Default mode is system-only: only the computer's internal audio is
    captured and sent to the live model. Set mic_enabled=True / audio_mode="dual"
    to also mix the microphone.
    """

    SPEAKER_MARKERS = {
        "interviewer": "[SPEAKER: SYSTEM AUDIO speaking — answer concisely with bullets]",
        "candidate": "[SPEAKER: CANDIDATE speaking — stay SILENT unless explicit help request]",
        "both": "[SPEAKER: BOTH talking / overlap — prioritize system audio question]",
        "system": "[SPEAKER: SYSTEM AUDIO speaking — answer concisely with bullets]",
    }

    def __init__(
        self,
        mic_index: Optional[int] = None,
        loopback_name: str = "",
        vad_threshold_db: int = VAD_DEFAULT,
        chunk_samples: int = CHUNK_SAMPLES,
        mic_enabled: bool = False,
        audio_mode: str = "system",
        enable_agc: bool = True,
    ) -> None:
        mode = str(audio_mode or "system").strip().lower()
        if mode in ("loopback", "computer"):
            mode = "system"
        if mode not in ("system", "dual"):
            mode = "system"
        # system mode forces mic off; dual forces mic on
        if mode == "system":
            mic_enabled = False
        elif mode == "dual":
            mic_enabled = True
        self.audio_mode = mode
        self.mic_enabled = bool(mic_enabled)
        try:
            if mic_index is None:
                self.mic_index = None
            else:
                self.mic_index = int(mic_index)
        except (TypeError, ValueError, OverflowError):
            self.mic_index = None
        try:
            self.loopback_name = str(loopback_name or "").strip()
        except Exception:
            self.loopback_name = ""
        # Legacy param name is *_db but value is a linear abs-mean threshold;
        # clamp identically to set_vad_threshold() so init never crashes or
        # accepts absurd values (negative / huge / NaN).
        try:
            _vad = int(vad_threshold_db)
        except (TypeError, ValueError, OverflowError):
            _vad = 300
        try:
            self.vad_threshold = max(VAD_MIN, min(VAD_MAX, _vad))
        except Exception:
            self.vad_threshold = VAD_DEFAULT
        try:
            _cs = int(chunk_samples)
        except (TypeError, ValueError, OverflowError):
            _cs = CHUNK_SAMPLES
        if _cs <= 0:
            _cs = CHUNK_SAMPLES
        self.chunk_samples = max(64, min(16384, _cs))
        self.output_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self.speaker_queue: "queue.Queue[str]" = queue.Queue(maxsize=50)
        self._mic_buffer: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)
        self._lifecycle_lock = threading.Lock()
        self._current_speaker = "silence"
        self._pending_speaker = "silence"
        self._pending_count = 0
        self._speaker_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._lb_ok = False
        self._lb_error = ""
        self._active_loopback = ""

        self._stop = threading.Event()
        self._paused = threading.Event()  # set() == paused
        self._threads: List[threading.Thread] = []
        self._mic_stream = None
        # Rolling loopback buffer keeps >= PREROLL_SAMPLES (~300ms) so
        # endpointing / speaker logic has onset context, not just the last
        # 1024-sample chunk. Mixer consumes the last chunk_samples of it.
        _buf_len = max(int(PREROLL_SAMPLES), int(self.chunk_samples))
        # also keep at least 4 chunks of history when chunk is large
        try:
            _buf_len = max(_buf_len, int(self.chunk_samples) * 4)
        except Exception:
            pass
        self._latest_loopback = np.zeros(_buf_len, dtype=np.float32)
        self._lb_lock = threading.Lock()
        self._running = False
        # -- adaptive VAD / AGC state ------------------------------------
        try:
            self.enable_agc = bool(enable_agc)
        except Exception:
            self.enable_agc = True
        self._agc_gain = 1.0
        self._rms_window: "deque[float]" = deque(maxlen=8)  # ~500ms @64ms
        self._rms_lock = threading.Lock()
        self._peak = 0.0
        self._last_speech_time = 0.0
        self._in_speech = False
        self._preroll_buf = np.zeros(0, dtype=np.float32)
        self._preroll_lock = threading.Lock()
        self._lb_silent_since: Optional[float] = None
        self._lb_failover_count = 0
        self._loopback_override = ""
        self._need_failover = False
        # -- diagnostics (console meter + HUD flow line) ------------------
        self._stats_lock = threading.Lock()
        self.frames_sent = 0
        self.last_lb_energy = 0.0
        self.last_mixed_energy = 0.0
        self.last_speaker = "silence"
        self.last_frame_time = 0.0
        self._hangover = 0  # trailing-silence frames left to emit after speech

    def get_audio_stats(self) -> dict:
        """Snapshot for console diagnostics: frames, energies, queue depth."""
        try:
            with self._stats_lock:
                d = {
                    "frames_sent": int(self.frames_sent),
                    "lb_energy": round(float(self.last_lb_energy), 1),
                    "mixed_energy": round(float(self.last_mixed_energy), 1),
                    "speaker": str(self.last_speaker),
                    "vad": int(self.vad_threshold),
                    "last_frame_age_s": round(time.time() - float(self.last_frame_time), 1) if self.last_frame_time else -1.0,
                }
            try:
                d["queue"] = int(self.output_queue.qsize())
            except Exception:
                d["queue"] = -1
            return d
        except Exception:
            return {}

    def _note_stats(self, lb_energy: float, mixed_energy: float, speaker: str, sent: bool) -> None:
        try:
            with self._stats_lock:
                self.last_lb_energy = float(lb_energy)
                self.last_mixed_energy = float(mixed_energy)
                self.last_speaker = str(speaker)
                if sent:
                    self.frames_sent += 1
                    self.last_frame_time = time.time()
        except Exception:
            pass

    # -- adaptive VAD / AGC ------------------------------------------------
    def _note_rms(self, rms: float, peak: float = 0.0) -> None:
        try:
            r = float(rms)
        except Exception:
            return
        try:
            import math as _m
            if _m.isnan(r) or _m.isinf(r) or r < 0:
                return
        except Exception:
            pass
        try:
            with self._rms_lock:
                if not hasattr(self, "_rms_window"):
                    self._rms_window = deque(maxlen=8)  # type: ignore[assignment]
                self._rms_window.append(float(r))
                self._peak = max(float(peak), float(self._peak) * 0.9)
        except Exception:
            pass

    def effective_threshold(self) -> float:
        """Adaptive VAD threshold in [VAD_MIN, VAD_MAX].

        Returns the configured base threshold, halved when recent (~500ms)
        RMS history shows speech (peak window > base) or we are inside
        speech. Lowering after speech onset keeps quiet meeting speech
        above the gate instead of requiring it to clear a fixed 300 gap.
        """
        try:
            base = float(getattr(self, "vad_threshold", VAD_DEFAULT))
        except Exception:
            base = float(VAD_DEFAULT)
        base = max(float(VAD_MIN), min(float(VAD_MAX), base))
        try:
            now = time.monotonic()
            last = float(getattr(self, "_last_speech_time", 0.0) or 0.0)
            if last and (now - last) < 0.5 and bool(getattr(self, "_in_speech", False)):
                return max(float(VAD_MIN), min(float(VAD_MAX), base * 0.5))
            win = getattr(self, "_rms_window", None)
            if win is not None and len(win) > 0:
                try:
                    if max(list(win)) > base:
                        return max(float(VAD_MIN), min(float(VAD_MAX), base * 0.5))
                except Exception:
                    pass
        except Exception:
            pass
        return base

    def _apply_agc(self, frame) -> np.ndarray:
        """Normalize loopback gain (+AGC) so low-volume meetings pass VAD.

        Boosts only quiet but non-silent signals toward TARGET RMS
        (1000 int16 scale), capped at 8x and smoothed to avoid pumping.
        True digital silence (peak ~0) is left untouched.
        """
        try:
            arr = np.asarray(frame, dtype=np.float32).ravel()
        except Exception:
            try:
                return np.asarray(frame, dtype=np.float32)
            except Exception:
                return np.zeros(0, dtype=np.float32)
        try:
            if not bool(getattr(self, "enable_agc", True)):
                return arr
        except Exception:
            pass
        try:
            if arr.size == 0:
                return arr
            peak = float(np.max(np.abs(arr)))
            if peak < 1.0:
                return arr
            rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
            if rms < 1e-6 or peak < 1.0:
                return arr
            gain_raw = float(AGC_TARGET_RMS) / rms if rms > 1e-6 else 1.0
            gain = max(1.0, min(float(AGC_MAX_GAIN), float(gain_raw)))
            old = float(getattr(self, "_agc_gain", 1.0) or 1.0)
            smoothed = 0.5 * old + 0.5 * gain
            self._agc_gain = float(smoothed)
            if smoothed <= 1.01:
                return arr
            return np.clip(arr * float(smoothed), -32767.0, 32767.0).astype(np.float32)
        except Exception:
            return arr

    # -- pre-roll ----------------------------------------------------------
    def _accumulate_preroll(self, frame) -> None:
        try:
            arr = np.asarray(frame, dtype=np.float32).ravel()
            if arr.size == 0:
                return
            with self._preroll_lock:
                if not hasattr(self, "_preroll_buf"):
                    self._preroll_buf = np.zeros(0, dtype=np.float32)
                buf = np.concatenate([self._preroll_buf, arr])
                keep = int(max(int(PREROLL_SAMPLES), int(getattr(self, "chunk_samples", CHUNK_SAMPLES) or CHUNK_SAMPLES)))
                if len(buf) > keep:
                    buf = buf[-keep:]
                self._preroll_buf = buf.astype(np.float32)
        except Exception:
            pass

    def _take_preroll_pcm(self) -> List[bytes]:
        """Drain buffered pre-roll as int16 PCM chunks (chunk_samples each)."""
        out: List[bytes] = []
        try:
            cs = int(getattr(self, "chunk_samples", CHUNK_SAMPLES) or CHUNK_SAMPLES)
            with self._preroll_lock:
                if not hasattr(self, "_preroll_buf") or self._preroll_buf is None or len(self._preroll_buf) == 0:
                    return out
                buf = np.asarray(self._preroll_buf, dtype=np.float32).ravel()
                self._preroll_buf = np.zeros(0, dtype=np.float32)
            for i in range(0, len(buf), cs):
                part = buf[i : i + cs]
                if len(part) < cs:
                    part = np.pad(part, (0, cs - len(part)))
                pcm = np.clip(part, -32768, 32767).astype(np.int16).tobytes()
                out.append(pcm)
        except Exception:
            pass
        return out

    def _enqueue_pcm(self, pcm: bytes) -> bool:
        try:
            if self.output_queue.full():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
            self.output_queue.put_nowait(pcm)
            return True
        except queue.Full:
            return False
        except Exception:
            return False

    # -- loopback failover -------------------------------------------------
    def _check_loopback_failover(self, energy: float, now: Optional[float] = None) -> bool:
        """True when loopback energy has been ~0 for >5s (healthy recorder).

        Called from the loopback record loop per chunk. Resets on any
        non-silent chunk. Never raises; COM stays balanced by caller.
        """
        try:
            if now is None:
                now = time.monotonic()
            try:
                e = float(energy)
            except Exception:
                e = 0.0
            try:
                import math as _m
                if _m.isnan(e) or _m.isinf(e):
                    e = 0.0
            except Exception:
                pass
            if e >= 1.0:
                self._lb_silent_since = None
                return False
            if getattr(self, "_lb_silent_since", None) is None:
                self._lb_silent_since = float(now)
                return False
            if (float(now) - float(self._lb_silent_since)) > 5.0:
                return True
            return False
        except Exception:
            return False

    # -- public control -------------------------------------------------
    def toggle_pause(self) -> bool:
        """Toggle pause. Returns True if now paused."""
        if self._paused.is_set():
            self._paused.clear()
        else:
            self._paused.set()
        return self.is_paused()

    def set_paused(self, paused: bool) -> bool:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()
        return self.is_paused()

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def get_speaker(self) -> str:
        with self._speaker_lock:
            return self._current_speaker

    def get_status(self) -> dict:
        """Return loopback health for UI: {ok, error, device, mode}."""
        with self._status_lock:
            return {
                "ok": bool(self._lb_ok),
                "error": str(self._lb_error or ""),
                "device": str(self._active_loopback or ""),
                "mode": str(getattr(self, "audio_mode", "system")),
                "mic": bool(getattr(self, "mic_enabled", False)),
            }

    def _set_lb_status(self, ok: bool, error: str = "", device: str = "") -> None:
        try:
            with self._status_lock:
                self._lb_ok = bool(ok)
                self._lb_error = str(error or "")
                if device:
                    self._active_loopback = str(device)
        except Exception:
            pass

    def _update_speaker(self, label: str) -> Optional[str]:
        """Debounce speaker changes (2 consistent frames). Returns marker or None."""
        if label == "silence":
            with self._speaker_lock:
                self._pending_speaker = "silence"
                self._pending_count = 0
            return None
        with self._speaker_lock:
            if label == self._current_speaker:
                self._pending_speaker = label
                self._pending_count = 0
                return None
            if label == self._pending_speaker:
                self._pending_count += 1
            else:
                self._pending_speaker = label
                self._pending_count = 1
            if self._pending_count >= 2:
                self._current_speaker = label
                self._pending_count = 0
                marker = self.SPEAKER_MARKERS.get(label)
                if marker:
                    try:
                        if self.speaker_queue.full():
                            try:
                                self.speaker_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.speaker_queue.put_nowait(marker)
                    except queue.Full:
                        pass
                    return marker
        return None

    def set_vad_threshold(self, value: int) -> None:
        try:
            self.vad_threshold = max(VAD_MIN, min(VAD_MAX, int(value)))
        except (TypeError, ValueError, OverflowError):
            pass

    def start(self) -> None:
        # Guard against double-start races: without a lock two threads could
        # both pass the _running check and spawn unbounded mixer/loopback
        # threads. Hold a lifecycle lock for the whole spawn sequence.
        try:
            lock = getattr(self, "_lifecycle_lock", None)
        except Exception:
            lock = None
        if lock is None:
            self._lifecycle_lock = threading.Lock()
            lock = self._lifecycle_lock
        with lock:
            if getattr(self, "_running", False):
                return
            self._stop.clear()
            self._running = True
            # Reset debounce state so a restarted engine never inherits a
            # stale speaker/marker from the previous session (marker leak).
            try:
                with self._speaker_lock:
                    self._pending_speaker = "silence"
                    self._pending_count = 0
                    self._current_speaker = "silence"
            except Exception:
                pass
            # Reset adaptive VAD / pre-roll / failover state on restart
            try:
                self._agc_gain = 1.0
                if hasattr(self, "_rms_window"):
                    self._rms_window.clear()  # type: ignore[attr-defined]
                self._last_speech_time = 0.0
                self._in_speech = False
                self._hangover = 0
                try:
                    with self._preroll_lock:
                        self._preroll_buf = np.zeros(0, dtype=np.float32)
                except Exception:
                    self._preroll_buf = np.zeros(0, dtype=np.float32)  # type: ignore[attr-defined]
                self._lb_silent_since = None
            except Exception:
                pass
            # Drop any stale mic frames queued from the previous session.
            try:
                while True:
                    self._mic_buffer.get_nowait()
            except Exception:
                pass
            t_mix = threading.Thread(target=self._mixer_loop, name="audio-mixer", daemon=True)
            t_lb = threading.Thread(target=self._loopback_loop, name="audio-loopback", daemon=True)
            self._threads = [t_mix, t_lb]
            for t in self._threads:
                try:
                    t.start()
                except Exception as e:
                    log.debug("thread start failed: %s", e)
        if self.mic_enabled:
            self._start_mic_stream()
        else:
            log.info("AudioEngine system-only mode: mic disabled, capturing computer sound.")
        log.info("AudioEngine started mode=%s mic_enabled=%s mic=%s loopback=%s vad=%s",
                 self.audio_mode, self.mic_enabled, self.mic_index, self.loopback_name or "<auto>", self.vad_threshold)

    def stop(self) -> None:
        try:
            self._stop.set()
        except Exception:
            pass
        try:
            self._running = False
        except Exception:
            pass
        try:
            mic = self._mic_stream
        except Exception:
            mic = None
        try:
            if mic is not None:
                try:
                    mic.stop()
                except Exception:
                    pass
                try:
                    mic.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self._mic_stream = None
        except Exception:
            pass
        try:
            threads = list(self._threads or [])
        except Exception:
            threads = []
        for t in threads:
            try:
                if t is threading.current_thread():
                    continue
                t.join(timeout=1.5)
            except Exception:
                pass
        try:
            self._threads = []
        except Exception:
            pass

    # -- mic ------------------------------------------------------------
    def _start_mic_stream(self) -> None:
        if not self.mic_enabled:
            return
        if not _HAS_SD:
            log.warning("No sounddevice; mic disabled (silence).")
            return
        try:
            def _cb(indata, frames, time_info, status):
                try:
                    if self._stop.is_set():
                        return
                    mono = _to_mono(np.array(indata))
                    res = _resample_mono(mono, 16000, TARGET_RATE)
                    # stash for mixer via queue attribute
                    self._on_mic_frame(res)
                except Exception as e:
                    log.debug("mic cb error: %s", e)

            # Reuse the init-time bounded queue (mixer may already be blocked
            # in Queue.get on it); recreating it here would orphan that wait.
            if getattr(self, "_mic_buffer", None) is None:
                self._mic_buffer = queue.Queue(maxsize=50)  # type: ignore[assignment]
            stream_kwargs = dict(samplerate=TARGET_RATE, channels=1, dtype="int16",
                                 blocksize=self.chunk_samples, callback=_cb)
            if self.mic_index is not None:
                stream_kwargs["device"] = int(self.mic_index)
            import sounddevice as _sd
            self._mic_stream = _sd.InputStream(**stream_kwargs)
            self._mic_stream.start()
        except Exception as e:
            log.warning("Mic stream failed (continuing with silence): %s", e)
            self._mic_stream = None

    def _on_mic_frame(self, frame: np.ndarray) -> None:
        try:
            if hasattr(self, "_mic_buffer"):
                if self._mic_buffer.full():
                    try:
                        self._mic_buffer.get_nowait()
                    except queue.Empty:
                        pass
                self._mic_buffer.put_nowait(np.asarray(frame, dtype=np.float32).ravel())
        except Exception:
            pass

    # -- loopback -------------------------------------------------------
    def _resolve_speaker(self, loopbacks):
        """Pick loopback device: override (failover) > preferred substring > first."""
        try:
            override = (getattr(self, "_loopback_override", "") or "").strip()
            if override:
                for m in loopbacks:
                    try:
                        if str(m.name) == override:
                            return m
                    except Exception:
                        continue
                # override not in fresh enumeration; clear and fall through
                try:
                    self._loopback_override = ""
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if self.loopback_name:
                pref = self.loopback_name.strip().lower()
                for m in loopbacks:
                    try:
                        if pref in str(m.name).lower() or str(m.name).lower() in pref:
                            return m
                    except Exception:
                        continue
                # Only try direct lookup when we have no enumeration to
                # fail over to; if a fresh list exists but the preferred
                # name is absent we fall through to first device below.
                if not loopbacks:
                    try:
                        return sc.get_microphone(self.loopback_name, include_loopback=True)
                    except Exception as e:
                        log.warning("get_microphone(%s, loopback) failed: %s", self.loopback_name, e)
                else:
                    log.debug("preferred loopback %s not in %s; using list", self.loopback_name, [getattr(m, "name", "") for m in loopbacks])
        except Exception:
            pass
        if loopbacks:
            return loopbacks[0]
        return None

    def _pick_failover_device(self, loopbacks, current_name: str = ""):
        """Return the next loopback device after ``current_name`` (cyclic)."""
        try:
            if not loopbacks:
                return None
            if len(loopbacks) == 1:
                return loopbacks[0]
            idx = -1
            for i, m in enumerate(loopbacks):
                try:
                    if str(getattr(m, "name", "")) == (current_name or ""):
                        idx = i
                        break
                except Exception:
                    continue
            nxt = loopbacks[(idx + 1) % len(loopbacks)]
            try:
                self._loopback_override = str(getattr(nxt, "name", "") or "")
            except Exception:
                pass
            return nxt
        except Exception:
            try:
                return loopbacks[0] if loopbacks else None
            except Exception:
                return None

    def _loopback_loop(self) -> None:
        if not _HAS_SC:
            self._set_lb_status(False, "soundcard not installed — pip install soundcard")
            log.warning("No soundcard; system-audio capture disabled.")
            return
        # COM must be initialized on this thread (WASAPI/soundcard), else
        # get_microphone/all_microphones fail with CO_E_NOTINITIALIZED (0x800401f0).
        # Use the shared helper so init/release stay balanced even when the
        # thread was already initialized (S_FALSE) or when CoInitialize
        # fails with a mode mismatch — never unconditionally CoUninitialize.
        _com_owned = _ensure_com()
        try:
            idle_cycles = 0
            while not self._stop.is_set():
                speaker = None
                active_name = ""
                try:
                    try:
                        loopbacks = [m for m in sc.all_microphones(include_loopback=True)
                                     if getattr(m, "isloopback", False)]
                    except Exception:
                        loopbacks = []
                    if getattr(self, "_need_failover", False):
                        prev = getattr(self, "_active_loopback", "") or self.loopback_name or ""
                        nxt = self._pick_failover_device(loopbacks, prev)
                        if nxt is not None:
                            speaker = nxt
                            try:
                                self._need_failover = False
                            except Exception:
                                pass
                        else:
                            speaker = self._resolve_speaker(loopbacks)
                    else:
                        speaker = self._resolve_speaker(loopbacks)
                    if speaker is None:
                        self._set_lb_status(False, "No system-audio device found. Play audio / check output device.")
                        with self._lb_lock:
                            self._latest_loopback = np.zeros(max(int(PREROLL_SAMPLES), int(self.chunk_samples) * 4), dtype=np.float32)
                        time.sleep(2.0)
                        continue
                    try:
                        active_name = str(getattr(speaker, "name", ""))
                    except Exception:
                        active_name = self.loopback_name or "<auto>"
                    sample_rate = 48000
                    self._set_lb_status(True, "", active_name)
                    log.info("loopback recording from %s @%dHz", active_name or "<auto>", sample_rate)
                    first_sound = False
                    with speaker.recorder(samplerate=sample_rate, channels=2) as rec:
                        while not self._stop.is_set():
                            try:
                                data = rec.record(numframes=2048)
                            except Exception as e:
                                log.debug("loopback record error: %s", e)
                                # device changed / unplugged -> break out and re-enumerate
                                self._set_lb_status(False, f"Device lost ({e}); retrying…", active_name)
                                with self._lb_lock:
                                    self._latest_loopback = np.zeros(max(int(PREROLL_SAMPLES), int(self.chunk_samples) * 4), dtype=np.float32)
                                time.sleep(0.5)
                                break
                            try:
                                # Resample in float32 (-1..1) then scale to int16
                                # range. Energy is computed on the pre-scaled
                                # float32 signal (pre-normalized) here so health
                                # checks are not distorted by AGC/mix gain.
                                mono = _to_mono(data)
                                res_f32 = _resample_mono(mono, sample_rate, TARGET_RATE)
                                try:
                                    energy_pre = float(np.abs(res_f32).mean() * 32767.0) if res_f32.size else 0.0
                                except Exception:
                                    energy_pre = 0.0
                                res = (res_f32 * 32767.0).astype(np.float32)
                                if not first_sound:
                                    try:
                                        if energy_pre > 500.0:
                                            first_sound = True
                                            log.info("loopback first sound detected on %s", active_name or "<auto>")
                                    except Exception:
                                        pass
                                # Auto-failover: sustained digital silence while
                                # the recorder itself is healthy means we are
                                # pinned to the wrong output device (e.g. a
                                # silent endpoint). Break out to re-enumerate
                                # and switch to the next loopback device.
                                try:
                                    if self._check_loopback_failover(energy_pre):
                                        log.warning("loopback silent >5s on %s; re-enumerating / failover", active_name or "<auto>")
                                        self._set_lb_status(False, f"Silent device {active_name}; switching…", active_name)
                                        self._lb_failover_count = int(getattr(self, "_lb_failover_count", 0) or 0) + 1
                                        self._need_failover = True
                                        self._lb_silent_since = None
                                        with self._lb_lock:
                                            keep = max(int(PREROLL_SAMPLES), int(self.chunk_samples) * 4)
                                            self._latest_loopback = np.zeros(keep, dtype=np.float32)
                                        break
                                except Exception:
                                    pass
                                with self._lb_lock:
                                    # Keep >= PREROLL_SAMPLES (~300ms) of
                                    # history for onset context. Mixer reads
                                    # the last chunk_samples of this buffer.
                                    keep = max(int(PREROLL_SAMPLES), int(self.chunk_samples) * 4)
                                    buf = np.concatenate([self._latest_loopback, res])[-keep:]
                                    if len(buf) >= self.chunk_samples:
                                        self._latest_loopback = buf.astype(np.float32)
                                    else:
                                        self._latest_loopback = np.pad(buf, (self.chunk_samples - len(buf), 0))
                                idle_cycles = 0
                            except Exception:
                                continue
                except Exception as e:
                    log.debug("loopback cycle error: %s", e)
                    self._set_lb_status(False, f"{e}; retrying…", active_name)
                    with self._lb_lock:
                        try:
                            self._latest_loopback = np.zeros(max(int(PREROLL_SAMPLES), int(self.chunk_samples) * 4), dtype=np.float32)
                        except Exception:
                            pass
                    time.sleep(2.0)
                    continue
        finally:
            if _com_owned:
                _release_com()

    # -- mixer ----------------------------------------------------------
    def _mixer_loop(self) -> None:
        try:
            zeros = np.zeros(int(getattr(self, "chunk_samples", CHUNK_SAMPLES) or CHUNK_SAMPLES), dtype=np.float32)
        except Exception:
            zeros = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        try:
            system_only = not bool(getattr(self, "mic_enabled", False))
        except Exception:
            system_only = True
        try:
            next_t = time.monotonic()
        except Exception:
            next_t = 0.0
        while not self._stop.is_set():
            try:
                if system_only:
                    # No mic clock in system-only mode: pace to real-time
                    # (~64ms per 1024-sample chunk). Without this the loop
                    # spins at full CPU re-queueing the same buffer.
                    now = time.monotonic()
                    if now < next_t:
                        time.sleep(next_t - now)
                    next_t = max(time.monotonic() + 0.064, next_t + 0.064)
                if system_only:
                    mic_frame = zeros.copy()
                    mic_energy = 0.0
                else:
                    mic_frame: Optional[np.ndarray] = None
                    if hasattr(self, "_mic_buffer"):
                        try:
                            mic_frame = self._mic_buffer.get(timeout=0.064)
                        except queue.Empty:
                            mic_frame = None
                    else:
                        time.sleep(0.064)
                    if mic_frame is None:
                        mic_frame = zeros.copy()
                    else:
                        a = np.asarray(mic_frame, dtype=np.float32).ravel()
                        if len(a) < self.chunk_samples:
                            a = np.pad(a, (0, self.chunk_samples - len(a)))
                        mic_frame = a[: self.chunk_samples]
                    mic_energy = float(np.abs(mic_frame).mean()) if mic_frame.size else 0.0

                with self._lb_lock:
                    lb_raw_full = np.asarray(self._latest_loopback, dtype=np.float32).ravel()
                # Mixer consumes the most-recent chunk_samples of the rolling
                # >=PREROLL_SAMPLES buffer (not the oldest samples).
                if len(lb_raw_full) >= self.chunk_samples:
                    lb_raw = lb_raw_full[-self.chunk_samples:].copy()
                elif len(lb_raw_full) > 0:
                    lb_raw = np.pad(lb_raw_full, (0, self.chunk_samples - len(lb_raw_full)))
                else:
                    lb_raw = zeros.copy()

                if self._paused.is_set():
                    continue  # context saver: emit nothing while paused

                # AGC-normalize loopback before energy / VAD checks so
                # quiet meeting speech clears the gate.
                try:
                    lb = self._apply_agc(lb_raw) if hasattr(self, "_apply_agc") else np.asarray(lb_raw, dtype=np.float32)
                except Exception:
                    lb = np.asarray(lb_raw, dtype=np.float32)
                try:
                    if lb is None or getattr(lb, "size", 0) == 0:
                        lb = zeros.copy()
                    else:
                        lb = np.asarray(lb, dtype=np.float32).ravel()
                        if len(lb) < self.chunk_samples:
                            lb = np.pad(lb, (0, self.chunk_samples - len(lb)))
                        lb = lb[: self.chunk_samples]
                except Exception:
                    lb = zeros.copy()

                try:
                    rms_agc = float(np.sqrt(np.mean(lb.astype(np.float32) ** 2))) if lb.size else 0.0
                    peak_agc = float(np.max(np.abs(lb))) if lb.size else 0.0
                    self._note_rms(rms_agc, peak_agc)
                except Exception:
                    pass
                try:
                    eff = float(self.effective_threshold())
                except Exception:
                    eff = float(getattr(self, "vad_threshold", VAD_DEFAULT))

                lb_energy = float(np.abs(lb).mean()) if lb.size else 0.0
                if system_only:
                    speaker = "interviewer" if lb_energy >= eff else "silence"
                    if speaker == "interviewer":
                        speaker = "system"
                else:
                    speaker = classify_speaker(mic_energy, lb_energy, eff)
                self._update_speaker(speaker)

                mixed = np.clip(mic_frame + lb, -32768, 32767)
                energy = float(np.abs(mixed).mean()) if mixed.size else 0.0
                is_speech = energy >= eff
                if is_speech:
                    now_mono = time.monotonic()
                    try:
                        self._last_speech_time = float(now_mono)
                    except Exception:
                        pass
                    # Flush pre-roll first: delivers the first ~200-300ms of
                    # the utterance that arrived below threshold.
                    if not bool(getattr(self, "_in_speech", False)):
                        try:
                            for pre_pcm in self._take_preroll_pcm():
                                self._enqueue_pcm(pre_pcm)
                                self._note_stats(lb_energy, energy, speaker, True)
                        except Exception:
                            pass
                    self._in_speech = True
                    self._hangover = int(HANGOVER_FRAMES)  # 128ms trailing
                    pcm = mixed.astype(np.int16).tobytes()
                    sent = self._enqueue_pcm(pcm)
                    self._note_stats(lb_energy, energy, speaker, bool(sent))
                else:
                    # Sub-threshold: buffer for onset pre-roll (never drop
                    # the first 200ms of the next utterance).
                    try:
                        self._accumulate_preroll(mixed)
                    except Exception:
                        pass
                    if bool(getattr(self, "_in_speech", False)) and int(getattr(self, "_hangover", 0) or 0) > 0:
                        # Trailing hangover: keep sending real trailing
                        # frames (2 frames) so endpointing fires fast.
                        try:
                            self._hangover -= 1
                            pcm = mixed.astype(np.int16).tobytes()
                            sent = self._enqueue_pcm(pcm)
                            if int(getattr(self, "_hangover", 0) or 0) <= 0:
                                self._in_speech = False
                            self._note_stats(lb_energy, energy, speaker, bool(sent))
                        except Exception:
                            self._in_speech = False
                    else:
                        self._in_speech = False
                        self._note_stats(lb_energy, energy, speaker, False)
            except Exception as e:
                log.debug("mixer error: %s", e)
                time.sleep(0.02)
