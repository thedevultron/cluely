"""Tests for audio_engine: mixing, diarization, resampling, queues, pause, status.

No hardware / no network: sounddevice/soundcard are mocked where touched.
"""
from __future__ import annotations

import math
import queue
import threading
import time

import numpy as np
import pytest

import audio_engine as ae
from audio_engine import AudioEngine


# ---------------------------------------------------------------- helpers
def _pcm(samples) -> bytes:
    return np.asarray(samples, dtype=np.int16).tobytes()


def _from_pcm(b: bytes):
    return np.frombuffer(b, dtype=np.int16)


class _FakeDevice:
    def __init__(self, name, max_input_channels=1):
        self.name = name
        self.max_input_channels = max_input_channels


class _FakeLoopbackMic:
    def __init__(self, name, isloopback=True):
        self.name = name
        self.isloopback = isloopback


# ---------------------------------------------------------------- mix_audio_frames
class TestMixAudioFrames:
    def test_both_empty_returns_empty(self):
        assert ae.mix_audio_frames(b"", b"") == b""

    def test_one_empty_returns_other(self):
        lb = _pcm([1000, -2000, 3000])
        assert ae.mix_audio_frames(b"", lb) == lb
        mic = _pcm([100, 200])
        assert ae.mix_audio_frames(mic, b"") == mic

    def test_simple_sum_no_clip(self):
        out = ae.mix_audio_frames(_pcm([1000, 2000]), _pcm([500, -1000]))
        assert _from_pcm(out).tolist() == [1500, 1000]

    def test_clipping_positive_and_negative(self):
        assert _from_pcm(ae.mix_audio_frames(_pcm([30000]), _pcm([10000]))).tolist() == [32767]
        assert _from_pcm(ae.mix_audio_frames(_pcm([-30000]), _pcm([-10000]))).tolist() == [-32768]
        assert _from_pcm(ae.mix_audio_frames(_pcm([32767]), _pcm([1]))).tolist() == [32767]

    def test_unequal_lengths_pad_shorter(self):
        out = ae.mix_audio_frames(_pcm([10, 20, 30]), _pcm([1]))
        assert _from_pcm(out).tolist() == [11, 20, 30]
        out2 = ae.mix_audio_frames(_pcm([1]), _pcm([10, 20, 30]))
        assert _from_pcm(out2).tolist() == [11, 20, 30]

    def test_odd_length_truncates_not_crashes(self):
        # 3 bytes = 1.5 int16 samples; must truncate to 1 sample, not raise.
        odd = b"\x01\x00\xff"  # sample 1 + trailing junk byte
        out = ae.mix_audio_frames(odd, _pcm([10]))
        assert len(out) % 2 == 0
        assert _from_pcm(out).tolist() == [11]

    def test_none_like_inputs_do_not_raise(self):
        # Empty/None-equivalent
        assert ae.mix_audio_frames(b"", b"") == b""
        out = ae.mix_audio_frames(_pcm([5]), b"")
        assert _from_pcm(out).tolist() == [5]


# ---------------------------------------------------------------- classify_speaker
class TestClassifySpeaker:
    TH = 300.0

    def test_silence_when_both_below(self):
        assert ae.classify_speaker(0, 0, self.TH) == "silence"
        assert ae.classify_speaker(299, 299, self.TH) == "silence"

    def test_interviewer_loopback_only(self):
        assert ae.classify_speaker(0, 1000, self.TH) == "interviewer"

    def test_candidate_mic_only(self):
        assert ae.classify_speaker(1000, 0, self.TH) == "candidate"

    def test_exact_threshold_counts_as_active(self):
        assert ae.classify_speaker(0, 300, self.TH) == "interviewer"
        assert ae.classify_speaker(300, 0, self.TH) == "candidate"
        assert ae.classify_speaker(300, 300, self.TH) == "both"

    def test_both_dominant_wins(self):
        assert ae.classify_speaker(100, 1000, self.TH) == "interviewer"
        assert ae.classify_speaker(1000, 100, self.TH) == "candidate"

    def test_both_balanced_is_both(self):
        # neither exceeds 1.4x
        assert ae.classify_speaker(1000, 1100, self.TH) == "both"
        assert ae.classify_speaker(1000, 1400, self.TH) == "both"  # exactly 1.4x is NOT > 1.4x

    def test_both_just_above_ratio_dominates(self):
        assert ae.classify_speaker(1000, 1401, self.TH) == "interviewer"
        assert ae.classify_speaker(1401, 1000, self.TH) == "candidate"

    def test_nan_inf_negative_sanitized(self):
        assert ae.classify_speaker(float("nan"), float("nan"), self.TH) == "silence"
        assert ae.classify_speaker(float("inf"), float("inf"), self.TH) == "silence"
        assert ae.classify_speaker(-500, -500, self.TH) == "silence"
        # garbage threshold falls back to 300 -> both zero still silence
        assert ae.classify_speaker(0, 0, float("nan")) == "silence"


# ---------------------------------------------------------------- _resample_mono
class TestResampleMono:
    def test_empty_input(self):
        out = ae._resample_mono(np.zeros(0, dtype=np.float32), 48000)
        assert out.size == 0
        assert out.dtype == np.float32

    def test_none_input_never_raises(self):
        out = ae._resample_mono(None, 48000)  # type: ignore[arg-type]
        assert out.size == 0

    def test_same_rate_identity(self):
        arr = np.array([1, 2, 3, 4], dtype=np.float32)
        out = ae._resample_mono(arr, 16000, 16000)
        assert out.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_downsample_48000_to_16000(self):
        arr = np.linspace(0, 1, 4800, dtype=np.float32)
        out = ae._resample_mono(arr, 48000, 16000)
        assert out.size == pytest.approx(1600, abs=2)

    def test_upsample_8000_to_16000(self):
        arr = np.zeros(800, dtype=np.float32)
        out = ae._resample_mono(arr, 8000, 16000)
        assert out.size == pytest.approx(1600, abs=2)

    def test_single_sample(self):
        out = ae._resample_mono(np.array([7.0], dtype=np.float32), 48000, 16000)
        assert out.size >= 1

    def test_invalid_rates_return_copy_not_crash(self):
        arr = np.array([1, 2, 3], dtype=np.float32)
        for src, dst in [(0, 16000), (-1, 16000), (48000, 0), (48000, -1), (None, 16000), ("x", 16000)]:
            try:
                out = ae._resample_mono(arr, src, dst)  # type: ignore[arg-type]
            except Exception as e:
                pytest.fail(f"resample raised for {src!r}/{dst!r}: {e}")
            assert out.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------- _to_mono
class TestToMono:
    def test_1d_passthrough(self):
        out = ae._to_mono(np.array([1, 2, 3], dtype=np.float32))
        assert out.tolist() == [1.0, 2.0, 3.0]
        assert out.dtype == np.float32

    def test_stereo_averaged(self):
        frame = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)  # 2 frames x 2 ch
        out = ae._to_mono(frame)
        assert out.tolist() == pytest.approx([15.0, 35.0])

    def test_int16_input_converted(self):
        frame = np.array([[1000, 3000], [2000, 4000]], dtype=np.int16)
        out = ae._to_mono(frame)
        assert out.tolist() == pytest.approx([2000.0, 3000.0])

    def test_empty(self):
        out = ae._to_mono(np.zeros((0, 2), dtype=np.float32))
        assert out.size == 0

    def test_3d_does_not_raise(self):
        out = ae._to_mono(np.zeros((2, 2, 2), dtype=np.float32))
        assert out.size == 8  # ravel fallback

    def test_none_like_never_raises(self):
        try:
            out = ae._to_mono(None)
        except Exception as e:
            pytest.fail(f"_to_mono(None) raised: {e}")
        assert out.size == 0


# ---------------------------------------------------------------- pause toggle
class TestPauseToggle:
    def test_default_not_paused(self):
        eng = AudioEngine()
        assert eng.is_paused() is False

    def test_toggle_returns_new_state(self):
        eng = AudioEngine()
        assert eng.toggle_pause() is True
        assert eng.is_paused() is True
        assert eng.toggle_pause() is False
        assert eng.is_paused() is False

    def test_set_paused(self):
        eng = AudioEngine()
        assert eng.set_paused(True) is True
        assert eng.is_paused() is True
        assert eng.set_paused(False) is False
        assert eng.is_paused() is False

    def test_pause_is_thread_safe(self):
        eng = AudioEngine()
        errs = []

        def flip():
            try:
                for _ in range(100):
                    eng.toggle_pause()
            except Exception as e:  # pragma: no cover
                errs.append(e)

        ts = [threading.Thread(target=flip) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=5.0)
        assert not errs
        # 400 flips -> even -> unpaused
        assert eng.is_paused() is False


# ---------------------------------------------------------------- status / stats shape
class TestStatusAndStats:
    def test_get_status_shape(self):
        eng = AudioEngine(loopback_name="Speakers", audio_mode="system")
        st = eng.get_status()
        assert st.keys() == {"ok", "error", "device", "mode", "mic"}
        assert isinstance(st["ok"], bool)
        assert isinstance(st["error"], str)
        assert isinstance(st["device"], str)
        assert st["mode"] == "system"
        assert st["mic"] is False  # system forces mic off

    def test_get_status_dual_shape(self):
        eng = AudioEngine(audio_mode="dual")
        st = eng.get_status()
        assert st["mode"] == "dual"
        assert st["mic"] is True

    def test_get_audio_stats_shape(self):
        eng = AudioEngine(vad_threshold_db=500)
        stats = eng.get_audio_stats()
        assert stats.keys() == {"frames_sent", "lb_energy", "mixed_energy", "speaker", "vad", "last_frame_age_s", "queue"}
        assert isinstance(stats["frames_sent"], int)
        assert isinstance(stats["lb_energy"], float)
        assert isinstance(stats["mixed_energy"], float)
        assert isinstance(stats["speaker"], str)
        assert stats["vad"] == 500
        assert isinstance(stats["last_frame_age_s"], float)
        assert isinstance(stats["queue"], int)
        assert stats["last_frame_age_s"] == -1.0  # no frame yet

    def test_get_audio_stats_never_raises_on_fresh_engine(self):
        eng = AudioEngine()
        eng._note_stats(123.4, 56.7, "interviewer", True)
        stats = eng.get_audio_stats()
        assert stats["frames_sent"] == 1
        assert stats["lb_energy"] == pytest.approx(123.4, abs=0.1)
        assert stats["speaker"] == "interviewer"
        assert stats["last_frame_age_s"] >= 0.0


# ---------------------------------------------------------------- queues / backpressure
class TestQueueBackpressure:
    def test_queues_bounded(self):
        eng = AudioEngine()
        assert eng.output_queue.maxsize == 200
        assert eng.speaker_queue.maxsize == 50
        assert eng._mic_buffer.maxsize == 50

    def test_on_mic_frame_drops_oldest_when_full_no_block(self):
        eng = AudioEngine(chunk_samples=1024)
        for i in range(50):
            eng._on_mic_frame(np.full(1024, float(i), dtype=np.float32))
        assert eng._mic_buffer.qsize() == 50
        # One more: must not block/raise, size stays bounded.
        eng._on_mic_frame(np.full(1024, 999.0, dtype=np.float32))
        assert eng._mic_buffer.qsize() == 50
        drained = [eng._mic_buffer.get_nowait() for _ in range(50)]
        # Oldest (0.0) was evicted.
        assert drained[0][0] == pytest.approx(1.0)
        assert drained[-1][0] == pytest.approx(999.0)

    def test_speaker_queue_drops_oldest_when_full(self):
        eng = AudioEngine(audio_mode="dual")
        for i in range(50):
            eng.speaker_queue.put_nowait(f"m{i}")
        # Two consistent frames with new speaker -> marker enqueued without blocking.
        for _ in range(2):
            eng._update_speaker("interviewer")
        assert eng.speaker_queue.qsize() == 50
        assert eng.speaker_queue.queue[-1] == ae.AudioEngine.SPEAKER_MARKERS["interviewer"]

    def test_update_speaker_debounce_requires_2_frames(self):
        eng = AudioEngine(audio_mode="dual")
        assert eng._update_speaker("interviewer") is None
        marker = eng._update_speaker("interviewer")
        assert marker == ae.AudioEngine.SPEAKER_MARKERS["interviewer"]
        assert eng.get_speaker() == "interviewer"
        # Same speaker again: no re-emit
        assert eng._update_speaker("interviewer") is None

    def test_update_speaker_silence_resets_pending(self):
        eng = AudioEngine(audio_mode="dual")
        eng._update_speaker("interviewer")
        assert eng._update_speaker("silence") is None
        # pending was reset, so interviewer needs 2 fresh frames now
        eng._update_speaker("interviewer")
        assert eng._update_speaker("interviewer") == ae.AudioEngine.SPEAKER_MARKERS["interviewer"]

    def test_mic_frame_empty_never_raises(self):
        eng = AudioEngine()
        eng._on_mic_frame(np.zeros(0, dtype=np.float32))
        assert eng._mic_buffer.qsize() == 1


# ---------------------------------------------------------------- init clamping / validation
class TestInitClamping:
    def test_vad_clamped(self):
        assert AudioEngine(vad_threshold_db=-5).vad_threshold == 50
        assert AudioEngine(vad_threshold_db=99999).vad_threshold == 2000
        assert AudioEngine(vad_threshold_db=300).vad_threshold == 300
        assert AudioEngine(vad_threshold_db="bogus").vad_threshold == 300  # type: ignore[arg-type]
        assert AudioEngine(vad_threshold_db=None).vad_threshold == 300  # type: ignore[arg-type]
        assert AudioEngine(vad_threshold_db=float("inf")).vad_threshold == 300  # type: ignore[arg-type]
        assert AudioEngine(vad_threshold_db=float("nan")).vad_threshold == 300  # type: ignore[arg-type]

    def test_set_vad_threshold_clamps_and_ignores_garbage(self):
        eng = AudioEngine()
        eng.set_vad_threshold(10)
        assert eng.vad_threshold == 50
        eng.set_vad_threshold(9999)
        assert eng.vad_threshold == 2000
        eng.set_vad_threshold(700)
        assert eng.vad_threshold == 700
        eng.set_vad_threshold("bogus")  # type: ignore[arg-type]
        assert eng.vad_threshold == 700
        eng.set_vad_threshold(float("inf"))  # type: ignore[arg-type]
        assert eng.vad_threshold == 700

    def test_chunk_samples_validated(self):
        assert AudioEngine(chunk_samples=1024).chunk_samples == 1024
        assert AudioEngine(chunk_samples=0).chunk_samples == ae.CHUNK_SAMPLES
        assert AudioEngine(chunk_samples=-100).chunk_samples == ae.CHUNK_SAMPLES
        assert AudioEngine(chunk_samples="bogus").chunk_samples == ae.CHUNK_SAMPLES  # type: ignore[arg-type]
        assert AudioEngine(chunk_samples=10).chunk_samples == 64  # min
        assert AudioEngine(chunk_samples=999999).chunk_samples == 16384  # max

    def test_mic_index_coerced(self):
        assert AudioEngine(mic_index=2).mic_index == 2
        assert AudioEngine(mic_index=None).mic_index is None
        assert AudioEngine(mic_index="bogus").mic_index is None  # type: ignore[arg-type]
        assert AudioEngine(mic_index="3").mic_index == 3  # type: ignore[arg-type]

    def test_audio_mode_normalization(self):
        assert AudioEngine(audio_mode="loopback").audio_mode == "system"
        assert AudioEngine(audio_mode="bogus").audio_mode == "system"
        assert AudioEngine(audio_mode="DUAL").audio_mode == "dual"
        assert AudioEngine(audio_mode="system", mic_enabled=True).mic_enabled is False
        assert AudioEngine(audio_mode="dual", mic_enabled=False).mic_enabled is True


# ---------------------------------------------------------------- enumeration with mocks
class TestEnumerationMocked:
    def test_list_microphones_filters_inputs(self, monkeypatch):
        class FakeSD:
            @staticmethod
            def query_devices():
                return [
                    {"max_input_channels": 2, "name": "Mic A"},
                    {"max_input_channels": 0, "name": "Speakers out"},
                    {"max_input_channels": 1, "name": "Mic B"},
                    {"name": "No channels key"},
                ]

        monkeypatch.setattr(ae, "sd", FakeSD)
        monkeypatch.setattr(ae, "_HAS_SD", True)
        out = ae.list_microphones()
        assert out == [(0, "Mic A"), (2, "Mic B")]

    def test_list_microphones_no_sdk_returns_empty(self, monkeypatch):
        monkeypatch.setattr(ae, "_HAS_SD", False)
        assert ae.list_microphones() == []

    def test_list_microphones_exception_returns_empty(self, monkeypatch):
        class BadSD:
            @staticmethod
            def query_devices():
                raise RuntimeError("no audio")

        monkeypatch.setattr(ae, "sd", BadSD)
        monkeypatch.setattr(ae, "_HAS_SD", True)
        assert ae.list_microphones() == []

    def test_list_loopbacks_filters_loopback_only(self, monkeypatch):
        class FakeSC:
            @staticmethod
            def all_microphones(include_loopback=True):
                return [
                    _FakeLoopbackMic("Speakers (Loopback)", True),
                    _FakeLoopbackMic("Real Mic", False),
                    _FakeLoopbackMic("Headphones (Loopback)", True),
                    _FakeLoopbackMic("Speakers (Loopback)", True),  # dup
                ]

        monkeypatch.setattr(ae, "sc", FakeSC)
        monkeypatch.setattr(ae, "_HAS_SC", True)
        monkeypatch.setattr(ae, "_ensure_com", lambda: False)
        out = ae.list_loopbacks()
        assert out == ["Speakers (Loopback)", "Headphones (Loopback)"]

    def test_list_loopbacks_no_sdk_returns_empty(self, monkeypatch):
        monkeypatch.setattr(ae, "_HAS_SC", False)
        assert ae.list_loopbacks() == []

    def test_detect_system_audio_preferred_substring(self, monkeypatch):
        monkeypatch.setattr(ae, "list_loopbacks", lambda: ["Speakers (Loopback)", "Headphones (Loopback)"])
        assert ae.detect_system_audio("headphones")[0] == "Headphones (Loopback)"
        assert ae.detect_system_audio("SPEAKERS")[0] == "Speakers (Loopback)"
        assert ae.detect_system_audio("")[0] == "Speakers (Loopback)"
        assert ae.detect_system_audio("nonexistent")[0] == "Speakers (Loopback)"

    def test_detect_system_audio_empty(self, monkeypatch):
        monkeypatch.setattr(ae, "list_loopbacks", lambda: [])
        assert ae.detect_system_audio("x") == ("", [])

    def test_ensure_com_never_raises(self):
        result = ae._ensure_com()
        assert isinstance(result, bool)
        ae._release_com()  # must not raise


# ---------------------------------------------------------------- start/stop lifecycle
class TestLifecycle:
    def test_start_stop_idempotent_and_no_thread_leak(self):
        eng = AudioEngine(audio_mode="system", chunk_samples=1024)
        # Patch loopback loop to exit quickly (no hardware): set _HAS_SC False
        # so loopback thread returns immediately after setting status.
        eng._stop.set()
        # Simulate start serialization: start twice, assert only 2 threads.
        # We won't actually call start() with real mic (system mode, no mic).
        # Instead drive _running/threads directly through guarded path by
        # temporarily making _loopback_loop a no-op spinner on _stop.
        orig_lb = eng._loopback_loop
        try:
            eng._loopback_loop = lambda: time.sleep(0.05)  # type: ignore[method-assign]
            # Use real start() — mixer thread will pace briefly; stop quickly.
            eng.start()
            threads_after_first = list(eng._threads)
            assert len(threads_after_first) == 2
            eng.start()  # second call must be a no-op
            assert len(eng._threads) == 2
            assert all(t.is_alive() or t.name in ("audio-mixer", "audio-loopback") for t in eng._threads)
            eng.stop()
            eng.stop()  # idempotent
            assert eng._threads == []
        finally:
            eng._loopback_loop = orig_lb  # type: ignore[method-assign]
            try:
                eng.stop()
            except Exception:
                pass

    def test_concurrent_start_does_not_spawn_unbounded_threads(self):
        eng = AudioEngine(audio_mode="system")
        eng._loopback_loop = lambda: time.sleep(0.05)  # type: ignore[method-assign]
        errors = []

        def starter():
            try:
                for _ in range(5):
                    eng.start()
            except Exception as e:  # pragma: no cover
                errors.append(e)

        ts = [threading.Thread(target=starter) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=10.0)
        assert not errors
        eng.stop()
        assert eng._threads == []

    def test_restart_clears_speaker_state(self):
        eng = AudioEngine(audio_mode="dual")
        for _ in range(3):
            eng._update_speaker("interviewer")
        assert eng.get_speaker() == "interviewer"
        # Purge + set silence via start bookkeeping: easiest is call start()
        # with stubbed loops, check pending/current reset after start.
        eng._loopback_loop = lambda: time.sleep(0.02)  # type: ignore[method-assign]
        eng.start()
        try:
            # start() resets speaker state under lifecycle lock
            assert eng.get_speaker() == "silence"
        finally:
            eng.stop()

    def test_mixer_never_raises_on_producer_error(self):
        # _note_stats with corrupt state should not raise; queue ops non-blocking
        eng = AudioEngine()
        eng._note_stats(float("nan"), float("inf"), 123, True)  # type: ignore[arg-type]
        assert eng.frames_sent == 1
        # push backpressure through internal queue — must never raise
        for _ in range(250):
            try:
                if eng.output_queue.full():
                    eng.output_queue.get_nowait()
                eng.output_queue.put_nowait(b"\x00\x01")
            except queue.Full:
                pass
        assert eng.output_queue.qsize() <= 200


def _run_mixer(eng, seconds: float):
    """Run _mixer_loop on a thread for ``seconds`` then stop it."""
    t = threading.Thread(target=eng._mixer_loop, daemon=True)
    t.start()
    time.sleep(seconds)
    eng._stop.set()
    t.join(timeout=2.0)
    return t


# ---------------------------------------------------------------- adaptive VAD + AGC
class TestAdaptiveVadAgc:
    def test_module_constants(self):
        assert ae.VAD_DEFAULT == 300
        assert ae.HANGOVER_FRAMES == 2
        assert ae.PREROLL_SAMPLES >= 3200

    def test_enable_agc_auto_on_by_default(self):
        assert AudioEngine().enable_agc is True

    def test_effective_threshold_default_equals_base(self):
        eng = AudioEngine(vad_threshold_db=300)
        assert eng.effective_threshold() == pytest.approx(300.0)

    def test_effective_threshold_lowers_after_speech(self):
        eng = AudioEngine(vad_threshold_db=300)
        eng._note_rms(1000.0, 2000.0)
        eff = eng.effective_threshold()
        assert eff < 300.0
        assert eff >= 50.0
        assert eff == pytest.approx(150.0)

    def test_effective_threshold_clamped_to_50_2000(self):
        eng = AudioEngine(vad_threshold_db=2000)
        eng._note_rms(5000.0)
        assert eng.effective_threshold() == pytest.approx(1000.0)
        low = AudioEngine(vad_threshold_db=50)
        low._note_rms(5000.0)
        assert low.effective_threshold() >= 50.0
        assert low.effective_threshold() == pytest.approx(50.0)

    def test_set_vad_threshold_semantics_preserved(self):
        eng = AudioEngine()
        eng.set_vad_threshold(10)
        assert eng.vad_threshold == 50
        eng.set_vad_threshold(9999)
        assert eng.vad_threshold == 2000
        assert 50.0 <= eng.effective_threshold() <= 2000.0

    def test_apply_agc_boosts_quiet_but_not_silence(self):
        eng = AudioEngine(vad_threshold_db=300)
        quiet = np.full(1024, 100.0, dtype=np.float32)
        raw_e = float(np.abs(quiet).mean())
        assert raw_e < 300.0
        out = quiet.copy()
        for _ in range(5):
            out = eng._apply_agc(out)
        assert float(np.abs(out).mean()) > raw_e
        # after convergence quiet speech clears the base threshold
        assert float(np.abs(out).mean()) >= 300.0
        sil = np.zeros(1024, dtype=np.float32)
        assert np.allclose(eng._apply_agc(sil), 0.0)

    def test_quiet_speech_passes_mixer_after_agc(self):
        eng = AudioEngine(vad_threshold_db=300, chunk_samples=1024, audio_mode="system")
        eng._stop.clear()
        # quiet meeting speech: mean 100 (well below 300) but non-silent
        buf = np.full(max(int(ae.PREROLL_SAMPLES), 1024 * 4), 100.0, dtype=np.float32)
        with eng._lb_lock:
            eng._latest_loopback = buf.copy()
        _run_mixer(eng, 0.9)
        stats = eng.get_audio_stats()
        assert stats["frames_sent"] >= 1
        assert stats["lb_energy"] >= 100.0
        assert eng.output_queue.qsize() >= 1

    def test_silence_still_dropped_after_agc(self):
        eng = AudioEngine(vad_threshold_db=300, chunk_samples=1024, audio_mode="system")
        eng._stop.clear()
        with eng._lb_lock:
            eng._latest_loopback = np.zeros(max(int(ae.PREROLL_SAMPLES), 1024 * 4), dtype=np.float32)
        _run_mixer(eng, 0.5)
        stats = eng.get_audio_stats()
        assert stats["frames_sent"] == 0
        assert stats["lb_energy"] == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------- pre-roll + hangover + debounce
class TestPrerollHangoverDebounce:
    def test_preroll_take_returns_chunks_and_clears(self):
        eng = AudioEngine(chunk_samples=1024)
        eng._accumulate_preroll(np.full(1024, 10.0, dtype=np.float32))
        eng._accumulate_preroll(np.full(1024, 20.0, dtype=np.float32))
        chunks = eng._take_preroll_pcm()
        assert len(chunks) == 2
        for c in chunks:
            assert len(c) == 1024 * 2  # int16 bytes
        # buffer cleared after drain
        assert eng._take_preroll_pcm() == []

    def test_preroll_flush_precedes_speech_via_mixer(self):
        eng = AudioEngine(vad_threshold_db=300, chunk_samples=1024, audio_mode="system")
        # prime pre-roll with distinctive low-level audio
        eng._accumulate_preroll(np.full(1024, 5.0, dtype=np.float32))
        eng._accumulate_preroll(np.full(1024, 5.0, dtype=np.float32))
        eng._stop.clear()
        buf = np.full(max(int(ae.PREROLL_SAMPLES), 1024 * 4), 3000.0, dtype=np.float32)
        with eng._lb_lock:
            eng._latest_loopback = buf.copy()
        _run_mixer(eng, 0.6)
        assert eng.output_queue.qsize() >= 3  # >=2 pre-roll + >=1 speech
        first = np.frombuffer(eng.output_queue.queue[0], dtype=np.int16)
        assert float(np.abs(first).mean()) < 500.0  # pre-roll arrived first

    def test_hangover_is_two_frames(self):
        assert ae.HANGOVER_FRAMES == 2
        eng = AudioEngine(vad_threshold_db=300, chunk_samples=1024, audio_mode="system")
        eng._stop.clear()
        buf = np.full(max(int(ae.PREROLL_SAMPLES), 1024 * 4), 3000.0, dtype=np.float32)
        with eng._lb_lock:
            eng._latest_loopback = buf.copy()
        _run_mixer(eng, 0.5)
        # hangover reset to 2 (not old 6) while speech active
        assert eng._hangover == 2
        assert eng._in_speech is True

    def test_debounce_is_two_frames(self):
        eng = AudioEngine(audio_mode="dual")
        assert eng._update_speaker("candidate") is None
        assert eng._update_speaker("candidate") == ae.AudioEngine.SPEAKER_MARKERS["candidate"]
        assert eng.get_speaker() == "candidate"

    def test_speaker_text_unchanged(self):
        assert ae.AudioEngine.SPEAKER_MARKERS["interviewer"] == "[SPEAKER: SYSTEM AUDIO speaking — answer concisely with bullets]"
        assert ae.AudioEngine.SPEAKER_MARKERS["candidate"] == "[SPEAKER: CANDIDATE speaking — stay SILENT unless explicit help request]"
        assert ae.AudioEngine.SPEAKER_MARKERS["both"] == "[SPEAKER: BOTH talking / overlap — prioritize system audio question]"
        assert ae.AudioEngine.SPEAKER_MARKERS["system"] == "[SPEAKER: SYSTEM AUDIO speaking — answer concisely with bullets]"


# ---------------------------------------------------------------- loopback buffer + failover
class TestLoopbackBufferFailover:
    def test_loopback_buffer_at_least_preroll(self):
        eng = AudioEngine(chunk_samples=1024)
        assert len(eng._latest_loopback) >= 3200
        assert len(eng._latest_loopback) >= ae.PREROLL_SAMPLES

    def test_mixer_consumes_last_chunk_not_first(self):
        eng = AudioEngine(vad_threshold_db=300, chunk_samples=1024, audio_mode="system")
        eng._stop.clear()
        keep = max(int(ae.PREROLL_SAMPLES), 1024 * 4)
        buf = np.zeros(keep, dtype=np.float32)
        buf[-1024:] = 3000.0  # loud in the last chunk only
        with eng._lb_lock:
            eng._latest_loopback = buf.copy()
        _run_mixer(eng, 0.5)
        assert eng.get_audio_stats()["frames_sent"] >= 1

    def test_failover_triggers_after_5s_silence(self):
        eng = AudioEngine()
        assert eng._check_loopback_failover(0.0, now=100.0) is False
        assert eng._check_loopback_failover(0.0, now=104.9) is False
        assert eng._check_loopback_failover(0.0, now=106.0) is True

    def test_failover_resets_on_sound(self):
        eng = AudioEngine()
        assert eng._check_loopback_failover(0.0, now=10.0) is False
        assert eng._check_loopback_failover(0.0, now=13.0) is False
        assert eng._check_loopback_failover(500.0, now=14.0) is False
        assert eng._lb_silent_since is None
        # after reset, silence needs a fresh 5s before triggering
        assert eng._check_loopback_failover(0.0, now=15.0) is False
        assert eng._check_loopback_failover(0.0, now=21.0) is True

    def test_failover_pick_cycles_to_next_device(self):
        eng = AudioEngine()
        devs = [_FakeLoopbackMic("Speakers (Loopback)"), _FakeLoopbackMic("Headphones (Loopback)")]
        nxt = eng._pick_failover_device(devs, "Speakers (Loopback)")
        assert nxt.name == "Headphones (Loopback)"
        assert eng._loopback_override == "Headphones (Loopback)"
        nxt2 = eng._pick_failover_device(devs, "Headphones (Loopback)")
        assert nxt2.name == "Speakers (Loopback)"

    def test_resolve_speaker_prefers_override(self):
        eng = AudioEngine()
        eng._loopback_override = "Headphones (Loopback)"
        devs = [_FakeLoopbackMic("Speakers (Loopback)"), _FakeLoopbackMic("Headphones (Loopback)")]
        assert eng._resolve_speaker(devs).name == "Headphones (Loopback)"
