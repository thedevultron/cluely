"""Tests for gemini_live_client: config build, clamping, dedup, stats, tool calls.

No network / no hardware: google.genai types and websearch are mocked.
"""
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace

import pytest

import gemini_live_client as glc
from gemini_live_client import GeminiLiveClient, _build_live_config


# ---------------------------------------------------------------- fake SDK types
class _FakeModality:
    TEXT = "TEXT"
    AUDIO = "AUDIO"


class _FakeAudioTranscriptionConfig:
    def __init__(self, *a, **kw):
        self.kw = kw


class _FakeAutomaticActivityDetection:
    def __init__(self, **kw):
        self.kw = kw


class _FakeStartSensitivity:
    START_SENSITIVITY_HIGH = "high"
    START_SENSITIVITY_LOW = "low"


class _FakeEndSensitivity:
    END_SENSITIVITY_HIGH = "high"
    END_SENSITIVITY_LOW = "low"


class _FakeTurnCoverage:
    TURN_INCLUDES_ALL_INPUT = "all"


class _FakeRealtimeInputConfig:
    def __init__(self, automatic_activity_detection=None, turn_coverage=None, **kw):
        self.automatic_activity_detection = automatic_activity_detection
        self.turn_coverage = turn_coverage
        self.kw = kw


class _FakeGenerationConfig:
    def __init__(self, max_output_tokens=None, **kw):
        self.max_output_tokens = max_output_tokens
        self.kw = kw


class _FakeThinkingConfig:
    def __init__(self, thinking_level=None, **kw):
        self.thinking_level = thinking_level
        self.kw = kw


class _FakeFunctionDeclaration:
    def __init__(self, name=None, description=None, parameters=None, **kw):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.kw = kw


class _FakeTool:
    def __init__(self, function_declarations=None, **kw):
        self.function_declarations = function_declarations or []
        self.kw = kw


class _FakeContent:
    def __init__(self, role=None, parts=None, **kw):
        self.role = role
        self.parts = parts or []
        self.kw = kw


class _FakePart:
    def __init__(self, text=None, **kw):
        self.text = text
        self.kw = kw

    @staticmethod
    def from_text(text=None):
        return _FakePart(text=text)


class _FakeFunctionResponse:
    def __init__(self, name=None, id=None, response=None, **kw):
        self.name = name
        self.id = id
        self.response = response
        self.kw = kw


class FakeLiveConnectConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeBlob:
    def __init__(self, data=None, mime_type=None, **kw):
        self.data = data
        self.mime_type = mime_type
        self.kw = kw


def _full_fake_types():
    return SimpleNamespace(
        Modality=_FakeModality,
        AudioTranscriptionConfig=_FakeAudioTranscriptionConfig,
        AutomaticActivityDetection=_FakeAutomaticActivityDetection,
        StartSensitivity=_FakeStartSensitivity,
        EndSensitivity=_FakeEndSensitivity,
        TurnCoverage=_FakeTurnCoverage,
        RealtimeInputConfig=_FakeRealtimeInputConfig,
        GenerationConfig=_FakeGenerationConfig,
        ThinkingConfig=_FakeThinkingConfig,
        FunctionDeclaration=_FakeFunctionDeclaration,
        Tool=_FakeTool,
        Content=_FakeContent,
        Part=_FakePart,
        FunctionResponse=_FakeFunctionResponse,
        LiveConnectConfig=FakeLiveConnectConfig,
        Blob=_FakeBlob,
    )


# ---------------------------------------------------------------- _build_live_config
class TestBuildLiveConfig:
    def test_latency_constants(self):
        assert glc.PREFIX_MS == 80
        assert glc.SILENCE_MS == 150
        assert glc.MAX_OUTPUT_TOKENS == 160

    def test_audio_config_fields(self, monkeypatch):
        monkeypatch.setattr(glc, "types", _full_fake_types())
        cfg = _build_live_config("sys-instr", "HIGH", "AUDIO", enable_web_search=False)
        assert isinstance(cfg, FakeLiveConnectConfig)
        assert cfg.response_modalities == ["AUDIO"]
        assert cfg.system_instruction == "sys-instr"
        assert isinstance(cfg.kwargs.get("output_audio_transcription"), _FakeAudioTranscriptionConfig)
        assert isinstance(cfg.kwargs.get("input_audio_transcription"), _FakeAudioTranscriptionConfig)
        assert isinstance(cfg.kwargs.get("realtime_input_config"), _FakeRealtimeInputConfig)
        assert cfg.realtime_input_config.automatic_activity_detection.kw["disabled"] is False
        assert cfg.realtime_input_config.turn_coverage == "all"
        assert cfg.generation_config.max_output_tokens == 160
        assert cfg.generation_config.max_output_tokens == glc.MAX_OUTPUT_TOKENS
        # temperature/top_p via GenerationConfig kw (fallback-tolerant)
        assert cfg.generation_config.kw.get("temperature") == 0.2
        assert cfg.generation_config.kw.get("top_p") == 0.9
        assert cfg.thinking_config.thinking_level == "HIGH"

    def test_endpointing_delay_fields(self, monkeypatch):
        monkeypatch.setattr(glc, "types", _full_fake_types())
        cfg = _build_live_config("sys-instr", "HIGH", "AUDIO", enable_web_search=False)
        aad = cfg.realtime_input_config.automatic_activity_detection.kw
        assert aad["prefix_padding_ms"] == 80
        assert aad["prefix_padding_ms"] == glc.PREFIX_MS
        assert aad["silence_duration_ms"] == 150
        assert aad["silence_duration_ms"] == glc.SILENCE_MS
        # keep high sensitivity + full turn coverage
        assert aad["start_of_speech_sensitivity"] == "high"
        assert aad["end_of_speech_sensitivity"] == "high"
        assert cfg.realtime_input_config.turn_coverage == "all"

    def test_generation_config_fallback_without_temp(self, monkeypatch):
        # SDK that rejects temperature/top_p must still get max tokens.
        class _StrictGenConfig:
            def __init__(self, max_output_tokens=None, **kw):
                if "temperature" in kw or "top_p" in kw:
                    raise TypeError("unknown keys")
                self.max_output_tokens = max_output_tokens
                self.kw = kw

        types2 = SimpleNamespace(
            Modality=_FakeModality,
            AudioTranscriptionConfig=_FakeAudioTranscriptionConfig,
            AutomaticActivityDetection=_FakeAutomaticActivityDetection,
            StartSensitivity=_FakeStartSensitivity,
            EndSensitivity=_FakeEndSensitivity,
            TurnCoverage=_FakeTurnCoverage,
            RealtimeInputConfig=_FakeRealtimeInputConfig,
            GenerationConfig=_StrictGenConfig,
            ThinkingConfig=_FakeThinkingConfig,
            LiveConnectConfig=FakeLiveConnectConfig,
        )
        monkeypatch.setattr(glc, "types", types2)
        cfg = _build_live_config("s", "HIGH", "AUDIO")
        assert cfg.generation_config.max_output_tokens == 160

    def test_text_modality_forced_to_audio(self, monkeypatch):
        monkeypatch.setattr(glc, "types", _full_fake_types())
        cfg = _build_live_config("s", "HIGH", "TEXT")
        assert cfg.response_modalities == ["AUDIO"]

    def test_web_search_adds_tool(self, monkeypatch):
        monkeypatch.setattr(glc, "types", _full_fake_types())
        cfg = _build_live_config("s", "HIGH", "AUDIO", enable_web_search=True)
        tools = cfg.kwargs.get("tools")
        assert tools is not None and len(tools) == 1
        assert tools[0].function_declarations[0].name == "web_search"

    def test_no_web_search_no_tools(self, monkeypatch):
        monkeypatch.setattr(glc, "types", _full_fake_types())
        cfg = _build_live_config("s", "HIGH", "AUDIO", enable_web_search=False)
        assert "tools" not in cfg.kwargs

    def test_tolerates_missing_sdk_fields(self, monkeypatch):
        minimal = SimpleNamespace(
            Modality=_FakeModality,
            LiveConnectConfig=FakeLiveConnectConfig,
        )
        monkeypatch.setattr(glc, "types", minimal)
        # Must not raise; must return something with response_modalities
        cfg = _build_live_config("s", "bogus", "AUDIO", enable_web_search=True)
        if isinstance(cfg, dict):
            assert "response_modalities" in cfg
        else:
            assert cfg.response_modalities

    def test_types_none_returns_dict_fallback(self, monkeypatch):
        monkeypatch.setattr(glc, "types", None)
        cfg = _build_live_config("s", "HIGH", "AUDIO")
        assert isinstance(cfg, dict)
        assert "response_modalities" in cfg

    def test_empty_system_instruction_omitted(self, monkeypatch):
        monkeypatch.setattr(glc, "types", _full_fake_types())
        cfg = _build_live_config("", "HIGH", "AUDIO")
        assert "system_instruction" not in cfg.kwargs or not cfg.kwargs["system_instruction"]

    def test_thinking_normalization(self, monkeypatch):
        monkeypatch.setattr(glc, "types", _full_fake_types())
        assert _build_live_config("s", "EXTENDED", "AUDIO").thinking_config.thinking_level == "HIGH"
        assert _build_live_config("s", "bogus", "AUDIO").thinking_config.thinking_level == "HIGH"
        assert _build_live_config("s", "LOW", "AUDIO").thinking_config.thinking_level == "LOW"
        assert _build_live_config("s", "MINIMAL", "AUDIO").thinking_config.thinking_level == "MINIMAL"


# ---------------------------------------------------------------- init clamping
class TestInitClamping:
    def test_web_max_results_clamped(self):
        assert GeminiLiveClient("k", web_max_results=0).web_max_results == 1
        assert GeminiLiveClient("k", web_max_results=99).web_max_results == 10
        assert GeminiLiveClient("k", web_max_results=5).web_max_results == 5
        assert GeminiLiveClient("k", web_max_results="bogus").web_max_results == 5  # type: ignore[arg-type]
        assert GeminiLiveClient("k", web_max_results=None).web_max_results == 5  # type: ignore[arg-type]
        assert GeminiLiveClient("k", web_max_results=float("inf")).web_max_results == 5  # type: ignore[arg-type]

    def test_thinking_level_normalized(self):
        assert GeminiLiveClient("k", thinking_level="EXTENDED").thinking_level == "HIGH"
        assert GeminiLiveClient("k", thinking_level="bogus").thinking_level == "HIGH"
        assert GeminiLiveClient("k", thinking_level="LOW").thinking_level == "LOW"
        assert GeminiLiveClient("k", thinking_level="medium").thinking_level == "MEDIUM"
        assert GeminiLiveClient("k", thinking_level="").thinking_level == "HIGH"

    def test_max_session_age_clamped(self):
        assert GeminiLiveClient("k", max_session_minutes=10.0).max_session_age_s == 600.0
        assert GeminiLiveClient("k", max_session_minutes=1.0).max_session_age_s == 120.0  # min
        assert GeminiLiveClient("k", max_session_minutes=-5).max_session_age_s == 120.0
        assert GeminiLiveClient("k", max_session_minutes=1000).max_session_age_s == 3600.0  # max
        assert GeminiLiveClient("k", max_session_minutes="bogus").max_session_age_s == 600.0  # type: ignore[arg-type]
        assert GeminiLiveClient("k", max_session_minutes=float("inf")).max_session_age_s == 600.0  # type: ignore[arg-type]

    def test_api_key_and_model_stripped(self):
        c = GeminiLiveClient("  key123  ", model="  m  ")
        assert c.api_key == "key123"
        assert c.model == "m"
        assert GeminiLiveClient("k", model="").model == "gemini-3.8-live"
        assert GeminiLiveClient(None).api_key == ""  # type: ignore[arg-type]

    def test_summarizer_default(self):
        assert GeminiLiveClient("k").summarizer_model == "gemini-3.5-flash-lite"
        assert GeminiLiveClient("k", summarizer_model="  x  ").summarizer_model == "x"

    def test_fallback_models_dedup_and_primary_excluded(self):
        c = GeminiLiveClient("k", model="main", fallback_models=["a", "main", "a", " b ", "", None])  # type: ignore[list-item]
        assert c.fallback_models == ["a", "b"]

    def test_fallback_models_string_not_split_into_chars(self):
        c = GeminiLiveClient("k", model="main", fallback_models="single-fallback")  # type: ignore[arg-type]
        assert c.fallback_models == ["single-fallback"]

    def test_fallback_models_none_and_empty(self):
        assert GeminiLiveClient("k").fallback_models == []
        assert GeminiLiveClient("k", fallback_models=[]).fallback_models == []


# ---------------------------------------------------------------- fallback dedup via _candidates
class TestCandidates:
    def test_candidates_no_duplicates(self):
        c = GeminiLiveClient("k", model="main", fallback_models=["a", "a", "main"])
        cand = c._candidates()
        assert cand[0] == "main"
        assert len(cand) == len(set(cand))
        assert "gemini-3.8-live" in cand
        assert "gemini-3.1-flash-live-preview" in cand

    def test_candidates_include_custom_fallbacks_first(self):
        c = GeminiLiveClient("k", model="m1", fallback_models=["m2"])
        cand = c._candidates()
        assert cand[:2] == ["m1", "m2"]


# ---------------------------------------------------------------- get_stats shape
class TestGetStats:
    def test_shape_fresh(self):
        c = GeminiLiveClient("k")
        s = c.get_stats()
        assert s.keys() == {
            "sessions", "rx_responses", "rx_chunks", "tx_audio",
            "tx_hellos", "it_chunks", "last_rx_age_s", "working_model", "last_error",
        }
        assert s["sessions"] == 0
        assert s["last_rx_age_s"] == -1.0
        assert s["working_model"] == ""
        assert s["last_error"] == ""

    def test_shape_after_counters(self):
        c = GeminiLiveClient("k")
        c.sessions_opened = 3
        c.rx_responses = 10
        c.rx_chunks = 7
        c.tx_audio_frames = 42
        c.tx_hellos = 1
        c.it_chunks = 2
        c._working_model = "m1"
        c.last_error = "oops" * 100
        s = c.get_stats()
        assert s["sessions"] == 3
        assert s["working_model"] == "m1"
        assert len(s["last_error"]) <= 200
        assert s["last_rx_age_s"] == -1.0


# ---------------------------------------------------------------- tool-call parsing
class _FakeSessionTool:
    def __init__(self):
        self.tool_responses = []
        self.realtime_texts = []

    async def send_tool_response(self, function_responses=None, **kw):
        self.tool_responses.append(function_responses or kw)

    async def send_realtime_input(self, text=None, audio=None, **kw):
        if text is not None:
            self.realtime_texts.append(text)


class TestHandleToolCall:
    def test_dict_form_web_search(self, monkeypatch):
        import websearch as ws

        monkeypatch.setattr(ws, "search_sync", lambda q, n: f"hit:{q}:{n}")
        c = GeminiLiveClient("k", web_max_results=3)
        session = _FakeSessionTool()
        seen = []

        async def on_text(t):
            seen.append(t)

        tc = {"function_calls": [{"name": "web_search", "args": {"query": "  python  "}, "id": "c1"}]}
        asyncio.run(c._handle_tool_call(session, tc, on_text))
        assert session.tool_responses or session.realtime_texts
        assert any("python" in str(s).lower() for s in seen) or session.tool_responses

    def test_object_form_and_json_string_args(self, monkeypatch):
        import json
        import websearch as ws

        monkeypatch.setattr(ws, "search_sync", lambda q, n: "ok")
        c = GeminiLiveClient("k")
        session = _FakeSessionTool()

        async def on_text(t):
            return None

        fc = SimpleNamespace(name="web_search", args=json.dumps({"query": "london"}), id="x")
        tc = SimpleNamespace(function_calls=[fc])
        asyncio.run(c._handle_tool_call(session, tc, on_text))
        assert session.tool_responses or session.realtime_texts

    def test_unknown_function_replies_with_error_not_stall(self, monkeypatch):
        c = GeminiLiveClient("k")
        session = _FakeSessionTool()

        async def on_text(t):
            return None

        tc = {"function_calls": [{"name": "delete_everything", "args": {}, "id": "x"}]}
        asyncio.run(c._handle_tool_call(session, tc, on_text))
        # Must have replied (tool response or realtime), not silently dropped.
        assert session.tool_responses or session.realtime_texts

    def test_missing_query_replies_with_error(self, monkeypatch):
        c = GeminiLiveClient("k")
        session = _FakeSessionTool()

        async def on_text(t):
            return None

        tc = {"function_calls": [{"name": "web_search", "args": {}, "id": "x"}]}
        asyncio.run(c._handle_tool_call(session, tc, on_text))
        assert session.tool_responses or session.realtime_texts

    def test_garbage_never_raises(self, monkeypatch):
        c = GeminiLiveClient("k")
        session = _FakeSessionTool()

        async def on_text(t):
            return None

        for garbage in [None, 123, "str", object(), {"function_calls": None}, {"function_calls": "weird"}]:
            asyncio.run(c._handle_tool_call(session, garbage, on_text))  # must not raise

    def test_search_failure_still_replies(self, monkeypatch):
        import websearch as ws

        def boom(q, n):
            raise RuntimeError("network down")

        monkeypatch.setattr(ws, "search_sync", boom)
        c = GeminiLiveClient("k")
        session = _FakeSessionTool()
        seen = []

        async def on_text(t):
            seen.append(t)

        tc = {"function_calls": [{"name": "web_search", "args": {"query": "x"}, "id": "1"}]}
        asyncio.run(c._handle_tool_call(session, tc, on_text))
        assert any("Search failed" in s for s in seen) or session.tool_responses


# ---------------------------------------------------------------- async helpers
class TestAsyncHelpers:
    def test_emit_swallows_callback_errors(self):
        c = GeminiLiveClient("k")

        def bad(v):
            raise RuntimeError("hud gone")

        asyncio.run(c._emit(bad, "x"))  # never raises

    def test_emit_none_is_noop(self):
        c = GeminiLiveClient("k")
        asyncio.run(c._emit(None, "x"))

    def test_sleep_or_stop_timeout_and_stop(self):
        c = GeminiLiveClient("k")
        # timeout path
        asyncio.run(c._sleep_or_stop(0.01))
        # stop path returns early
        async def go():
            c._stop.set()
            await c._sleep_or_stop(5.0)

        asyncio.run(go())

    def test_watch_stop_deadline_refresh(self):
        c = GeminiLiveClient("k")
        import time as _t

        async def go():
            return await c._watch_stop(deadline=_t.monotonic() + 0.05)

        assert asyncio.run(go()) == "refresh"

    def test_watch_stop_stopped(self):
        c = GeminiLiveClient("k")

        async def go():
            c._stop.set()
            return await c._watch_stop(deadline=None)

        assert asyncio.run(go()) == "stopped"

    def test_stop_sets_event_from_sync_and_async_contexts(self):
        c = GeminiLiveClient("k")
        # no running loop -> direct set
        c.stop()
        assert c._stop.is_set()

    def test_summarize_empty_never_raises(self):
        c = GeminiLiveClient("k")
        assert asyncio.run(c.summarize("")) == ""
        c_no_key = GeminiLiveClient("", model="m")
        assert asyncio.run(c_no_key.summarize("hello")) == ""

    def test_run_missing_key_raises(self):
        c = GeminiLiveClient("", model="m")

        async def on_text(t):
            return None

        with pytest.raises(RuntimeError):
            asyncio.run(c.run(on_text=on_text))


# ---------------------------------------------------------------- producer / marker plumbing
class TestProducerPlumbing:
    def test_produce_drains_speaker_markers(self, monkeypatch):
        c = GeminiLiveClient("k", audio_queue=None)
        c.speaker_queue = queue.Queue()
        c.speaker_queue.put_nowait("[SPEAKER: TEST]")
        sent = []

        class FakeSession:
            async def send_realtime_input(self, text=None, audio=None, **kw):
                if text is not None:
                    sent.append(text)

        session = FakeSession()

        async def run_short():
            task = asyncio.create_task(c._produce(session))
            await asyncio.sleep(0.15)
            c._stop.set()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()

        asyncio.run(run_short())
        assert "[SPEAKER: TEST]" in sent

    def test_produce_never_raises_on_audio_hiccup(self, monkeypatch):
        aq: queue.Queue = queue.Queue()
        aq.put_nowait(b"\x00\x01" * 512)
        c = GeminiLiveClient("k", audio_queue=aq)

        class FlakySession:
            def __init__(self):
                self.calls = 0

            async def send_realtime_input(self, audio=None, text=None, **kw):
                self.calls += 1
                if self.calls <= 2:
                    raise RuntimeError("hiccup")
                raise RuntimeError("closed by peer")  # would be connection-dead; producer re-raises

        async def go():
            # First: hiccup RuntimeError without closed/1007 should NOT kill producer.
            # Use a session that only hiccups, verify producer stays alive briefly.
            class HiccupSession:
                async def send_realtime_input(self, audio=None, text=None, **kw):
                    raise RuntimeError("temporary hiccup")

            task = asyncio.create_task(c._produce(HiccupSession()))
            await asyncio.sleep(0.3)
            assert not task.done()
            c._stop.set()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()

        asyncio.run(go())

    def test_produce_raises_on_closed_connection(self):
        aq: queue.Queue = queue.Queue()
        aq.put_nowait(b"\x00\x01" * 256)
        c = GeminiLiveClient("k", audio_queue=aq)

        class DeadSession:
            async def send_realtime_input(self, audio=None, text=None, **kw):
                raise RuntimeError("websocket closed 1006")

        async def go():
            await c._produce(DeadSession())

        with pytest.raises(RuntimeError):
            asyncio.run(go())


# ---------------------------------------------------------------- fatal classification
class TestFatalClassification:
    def test_1008_transient_not_fatal(self):
        assert glc._is_fatal_model_error("gemini-3.1-flash-live-preview: 1008 None") is False
        assert glc._is_fatal_model_error("1008 policy violation") is False

    def test_1007_fatal(self):
        assert glc._is_fatal_model_error("1007 not supported") is True
        assert glc._is_fatal_model_error("1007") is True

    def test_not_supported_fatal(self):
        assert glc._is_fatal_model_error("model not supported") is True
        assert glc._is_fatal_model_error("unsupported modality") is True

    def test_404_400_fatal(self):
        assert glc._is_fatal_model_error("404 NOT_FOUND model") is True
        assert glc._is_fatal_model_error("400 INVALID_ARGUMENT bad") is True

    def test_transient_other_not_fatal(self):
        assert glc._is_fatal_model_error("connection reset") is False
        assert glc._is_fatal_model_error("") is False

    def test_1008_overrides_other_fatal_words(self):
        # 1008 is transient even if other words present
        assert glc._is_fatal_model_error("1008 not supported temporarily") is False


# ---------------------------------------------------------------- produce non-blocking
class TestProduceLatency:
    def test_produce_source_uses_get_nowait_not_blocking_thread(self):
        import inspect
        src = inspect.getsource(GeminiLiveClient._produce)
        assert "get_nowait" in src
        assert "asyncio.sleep(0.01)" in src
        # No blocking 0.2s queue.get via to_thread on audio queue.
        assert "to_thread(self.audio_queue" not in src
        assert "0.2" not in src or "get_nowait" in src

    def test_produce_polls_fast(self):
        # Proves poll interval is ~10ms: audio sent quickly without 200ms stall.
        aq: queue.Queue = queue.Queue()
        aq.put_nowait(b"\x00\x01" * 256)
        c = GeminiLiveClient("k", audio_queue=aq)

        class Sess:
            async def send_realtime_input(self, audio=None, text=None, **kw):
                return None

        async def go():
            task = asyncio.create_task(c._produce(Sess()))
            await asyncio.sleep(0.1)
            c._stop.set()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()
            return c.tx_audio_frames

        frames = asyncio.run(go())
        assert frames >= 1

    def test_audio_backlog_flush_keeps_latest(self):
        # >8 frames queued: only latest should be sent in one produce pass.
        aq: queue.Queue = queue.Queue()
        for i in range(12):
            aq.put_nowait(bytes([i]) * 64)
        c = GeminiLiveClient("k", audio_queue=aq)
        sent = []

        class Sess:
            async def send_realtime_input(self, audio=None, text=None, **kw):
                sent.append(audio)

        async def go():
            task = asyncio.create_task(c._produce(Sess()))
            await asyncio.sleep(0.15)
            c._stop.set()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()

        asyncio.run(go())
        assert len(sent) >= 1
        # First sent frame should be latest (11), not oldest (0): proves flush.
        first = sent[0]
        data = getattr(first, "data", first.get("data") if isinstance(first, dict) else None)
        if data is not None:
            assert data[0] == 11
        assert aq.empty()

    def test_marker_throttled_and_after_audio(self):
        # Markers throttled to 1 per 500ms; audio never blocked by marker burst.
        aq: queue.Queue = queue.Queue()
        # continuous audio so markers must wait until after audio.
        c = GeminiLiveClient("k", audio_queue=aq)
        c.speaker_queue = queue.Queue()
        for i in range(5):
            c.speaker_queue.put_nowait(f"[SPEAKER: M{i}]")
            aq.put_nowait(b"\x00\x01" * 64)
        order = []

        class Sess:
            async def send_realtime_input(self, audio=None, text=None, **kw):
                if text is not None:
                    order.append(("text", text))
                elif audio is not None:
                    order.append(("audio", None))

        async def go():
            task = asyncio.create_task(c._produce(Sess()))
            await asyncio.sleep(0.2)
            c._stop.set()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()

        asyncio.run(go())
        # In 0.2s, at most 1 marker should have been sent (throttle 500ms).
        markers = [o for o in order if o[0] == "text"]
        assert len(markers) <= 1
        # Audio must have been sent (markers didn't block it).
        assert any(o[0] == "audio" for o in order)

    def test_marker_source_has_throttle(self):
        import inspect
        src = inspect.getsource(GeminiLiveClient._produce)
        assert "MARKER_THROTTLE_S" in src or "_last_marker_time" in src
        assert "pending_marker" in src

    def test_produce_sends_audio_before_marker_when_busy(self):
        # When both audio and marker pending, audio goes first.
        aq: queue.Queue = queue.Queue()
        aq.put_nowait(b"\x00\x01" * 64)
        c = GeminiLiveClient("k", audio_queue=aq)
        c.speaker_queue = queue.Queue()
        c.speaker_queue.put_nowait("[SPEAKER: EARLY]")
        c._last_marker_time = 0.0  # allow immediate marker
        order = []

        class Sess:
            async def send_realtime_input(self, audio=None, text=None, **kw):
                if text is not None:
                    order.append("text")
                elif audio is not None:
                    order.append("audio")

        async def go():
            task = asyncio.create_task(c._produce(Sess()))
            await asyncio.sleep(0.15)
            c._stop.set()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()

        asyncio.run(go())
        assert order and order[0] == "audio"
        assert "text" in order


# ---------------------------------------------------------------- primary reprobe
class TestPrimaryReprobe:
    def test_ordered_candidates_prefers_working_model(self):
        c = GeminiLiveClient("k", model="primary", fallback_models=["fallback"])
        c._working_model = "fallback"
        ordered = c._ordered_candidates(probe_primary=False)
        assert ordered[0] == "fallback"

    def test_ordered_candidates_probe_primary_returns_primary_first(self):
        c = GeminiLiveClient("k", model="primary", fallback_models=["fallback"])
        c._working_model = "fallback"
        ordered = c._ordered_candidates(probe_primary=True)
        assert ordered[0] == "primary"

    def test_run_source_reprobes_every_two(self):
        import inspect
        src = inspect.getsource(GeminiLiveClient.run)
        assert "_session_tries" in src or "probe_primary" in src
        assert "_is_fatal_model_error" in src


# ---------------------------------------------------------------- hello non-blocking
class TestHelloNonBlocking:
    def test_run_session_uses_create_task_for_hello(self):
        import inspect
        src = inspect.getsource(GeminiLiveClient._run_session)
        assert "create_task(self._send_hello" in src
        assert "await self._send_hello" not in src

    def test_selftest_hello_default_off(self):
        c = GeminiLiveClient("k")
        assert c.selftest_hello is False
        assert GeminiLiveClient("k", selftest_hello=True).selftest_hello is True


def test_module_has_no_hardware_import_side_effects():
    # Importing the modules must not open audio devices / fail without SDK.
    import audio_engine
    import gemini_live_client
    assert hasattr(audio_engine, "AudioEngine")
    assert hasattr(gemini_live_client, "GeminiLiveClient")
    assert hasattr(gemini_live_client, "_build_live_config")
