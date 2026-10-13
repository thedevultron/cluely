"""System-audio capture: computer sound only (WASAPI loopback), mic optional."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("cluely.audio")

TARGET_RATE = 16000
CHUNK_SAMPLES = 1024  # ~64ms @16kHz

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
    this call owns a COM ref (caller must CoUninitialize). Never raises."""
    try:
        import ctypes
        ole = ctypes.windll.ole32
        try:
            ole.CoInitialize.argtypes = [ctypes.c_void_p]
            ole.CoInitialize.restype = ctypes.c_long
        except Exception:
            pass
        rc = ole.CoInitialize(None)
        return int(rc) == 0  # S_OK: we initialized; S_FALSE(1): already init
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
    if data.size == 0:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32).ravel()
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
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 2:
            return arr.mean(axis=1).astype(np.float32)
        return arr.ravel().astype(np.float32)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def mix_audio_frames(mic_chunk: bytes, loopback_chunk: bytes) -> bytes:
    """Mix two int16 mono byte buffers with clipping."""
    try:
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


def classify_speaker(mic_energy: float, lb_energy: float, threshold: float) -> str:
    """Lightweight dual-energy diarization. Returns interviewer/candidate/silence/both."""
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
        vad_threshold_db: int = 300,
        chunk_samples: int = CHUNK_SAMPLES,
        mic_enabled: bool = False,
        audio_mode: str = "system",
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
        self.mic_index = mic_index
        self.loopback_name = (loopback_name or "").strip()
        self.vad_threshold = int(vad_threshold_db)
        self.chunk_samples = int(chunk_samples)
        self.output_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self.speaker_queue: "queue.Queue[str]" = queue.Queue(maxsize=50)
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
        self._latest_loopback = np.zeros(self.chunk_samples, dtype=np.float32)
        self._lb_lock = threading.Lock()
        self._running = False
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
        """Debounce speaker changes (3 consistent frames). Returns marker or None."""
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
            if self._pending_count >= 3:
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
            self.vad_threshold = max(50, min(2000, int(value)))
        except (TypeError, ValueError):
            pass

    def start(self) -> None:
        if self._running:
            return
        self._stop.clear()
        self._running = True
        t_mix = threading.Thread(target=self._mixer_loop, name="audio-mixer", daemon=True)
        t_lb = threading.Thread(target=self._loopback_loop, name="audio-loopback", daemon=True)
        self._threads = [t_mix, t_lb]
        for t in self._threads:
            t.start()
        if self.mic_enabled:
            self._start_mic_stream()
        else:
            log.info("AudioEngine system-only mode: mic disabled, capturing computer sound.")
        log.info("AudioEngine started mode=%s mic_enabled=%s mic=%s loopback=%s vad=%s",
                 self.audio_mode, self.mic_enabled, self.mic_index, self.loopback_name or "<auto>", self.vad_threshold)

    def stop(self) -> None:
        self._stop.set()
        self._running = False
        try:
            if self._mic_stream is not None:
                self._mic_stream.stop()
                self._mic_stream.close()
        except Exception:
            pass
        self._mic_stream = None
        for t in self._threads:
            try:
                t.join(timeout=1.5)
            except Exception:
                pass
        self._threads = []

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

            self._mic_buffer: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)
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
        """Pick loopback device: preferred substring wins, else first (auto)."""
        try:
            if self.loopback_name:
                pref = self.loopback_name.strip().lower()
                for m in loopbacks:
                    try:
                        if pref in str(m.name).lower() or str(m.name).lower() in pref:
                            return m
                    except Exception:
                        continue
                try:
                    return sc.get_microphone(self.loopback_name, include_loopback=True)
                except Exception as e:
                    log.warning("get_microphone(%s, loopback) failed: %s", self.loopback_name, e)
        except Exception:
            pass
        if loopbacks:
            return loopbacks[0]
        return None

    def _loopback_loop(self) -> None:
        if not _HAS_SC:
            self._set_lb_status(False, "soundcard not installed — pip install soundcard")
            log.warning("No soundcard; system-audio capture disabled.")
            return
        # COM must be initialized on this thread (WASAPI/soundcard), else
        # get_microphone/all_microphones fail with CO_E_NOTINITIALIZED (0x800401f0)
        try:
            import ctypes
            ctypes.windll.ole32.CoInitialize(None)
        except Exception:
            pass
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
                    speaker = self._resolve_speaker(loopbacks)
                    if speaker is None:
                        self._set_lb_status(False, "No system-audio device found. Play audio / check output device.")
                        with self._lb_lock:
                            self._latest_loopback = np.zeros(self.chunk_samples, dtype=np.float32)
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
                                    self._latest_loopback = np.zeros(self.chunk_samples, dtype=np.float32)
                                time.sleep(0.5)
                                break
                            try:
                                mono = _to_mono(data)
                                res = _resample_mono(mono * 32767.0, sample_rate, TARGET_RATE)
                                if not first_sound:
                                    try:
                                        if float(np.abs(res).mean()) > 500.0:
                                            first_sound = True
                                            log.info("loopback first sound detected on %s", active_name or "<auto>")
                                    except Exception:
                                        pass
                                with self._lb_lock:
                                    # keep rolling buffer of latest chunk_samples
                                    buf = np.concatenate([self._latest_loopback, res])[-self.chunk_samples * 4:]
                                    if len(buf) >= self.chunk_samples:
                                        self._latest_loopback = buf[-self.chunk_samples:].astype(np.float32)
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
                            self._latest_loopback = np.zeros(self.chunk_samples, dtype=np.float32)
                        except Exception:
                            pass
                    time.sleep(2.0)
                    continue
        finally:
            try:
                import ctypes
                ctypes.windll.ole32.CoUninitialize()
            except Exception:
                pass

    # -- mixer ----------------------------------------------------------
    def _mixer_loop(self) -> None:
        zeros = np.zeros(self.chunk_samples, dtype=np.float32)
        system_only = not self.mic_enabled
        next_t = time.monotonic()
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
                    lb = np.asarray(self._latest_loopback, dtype=np.float32).ravel()
                if len(lb) < self.chunk_samples:
                    lb = np.pad(lb, (0, self.chunk_samples - len(lb)))
                lb = lb[: self.chunk_samples]

                if self._paused.is_set():
                    continue  # context saver: emit nothing while paused

                lb_energy = float(np.abs(lb).mean()) if lb.size else 0.0
                if system_only:
                    speaker = "interviewer" if lb_energy >= float(self.vad_threshold) else "silence"
                    # map to system marker
                    if speaker == "interviewer":
                        speaker = "system"
                else:
                    speaker = classify_speaker(mic_energy, lb_energy, float(self.vad_threshold))
                self._update_speaker(speaker)

                mixed = np.clip(mic_frame + lb, -32768, 32767)
                energy = float(np.abs(mixed).mean()) if mixed.size else 0.0
                if energy < float(self.vad_threshold):
                    # Trailing-silence hangover: after speech ends, keep sending
                    # a few silent frames so the server's endpointing fires fast.
                    # Without this the server may wait tens of seconds for a pause.
                    if getattr(self, "_hangover", 0) > 0:
                        try:
                            self._hangover -= 1
                            pcm0 = np.zeros(self.chunk_samples, dtype=np.int16).tobytes()
                            if self.output_queue.full():
                                try:
                                    self.output_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.output_queue.put_nowait(pcm0)
                            self._note_stats(lb_energy, 0.0, "silence", True)
                        except queue.Full:
                            pass
                        continue
                    self._note_stats(lb_energy, energy, speaker, False)
                    continue  # drop silence to save tokens
                self._hangover = 6  # ~400ms of trailing silence on speech end

                pcm = mixed.astype(np.int16).tobytes()
                try:
                    if self.output_queue.full():
                        try:
                            self.output_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.output_queue.put_nowait(pcm)
                    self._note_stats(lb_energy, energy, speaker, True)
                except queue.Full:
                    self._note_stats(lb_energy, energy, speaker, False)
                    pass
            except Exception as e:
                log.debug("mixer error: %s", e)
                time.sleep(0.02)
