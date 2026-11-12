"""Async Gemini Live client (google-genai SDK), text-only streaming."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, List, Optional

log = logging.getLogger("cluely.gemini")

# Latency tuning: endpointing delay (prefix padding + silence wait).
# PREFIX_MS: audio kept before detected speech start.
# SILENCE_MS: silence required to declare end-of-turn (endpointing wait).
PREFIX_MS = 80
SILENCE_MS = 150
MAX_OUTPUT_TOKENS = 160
MARKER_THROTTLE_S = 0.5
AUDIO_BACKLOG_FLUSH_THRESHOLD = 8

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except Exception as e:  # pragma: no cover
    genai = None  # type: ignore
    types = None  # type: ignore
    _HAS_GENAI = False
    log.warning("google-genai unavailable: %s", e)

TextCallback = Callable[[str], Awaitable[None] | None]
StatusCallback = Callable[[str], Awaitable[None] | None]


def _build_live_config(system_instruction: str, thinking_level: str = "HIGH", modality: str = "TEXT",
                      enable_web_search: bool = False):
    """Build LiveConnectConfig tolerating SDK variants and modality quirks.

    Only AUDIO+transcription works (verified live Oct 2026): TEXT sessions open
    but the server kills them (1007) on first send; AUDIO without explicit
    realtime_input_config never endpoints (model hears nothing); AUDIO + auto
    activity detection + TURN_INCLUDES_ALL_INPUT hears AND answers. UX stays
    text-only: transcription text is displayed, audio is never played.
    """
    if str(modality).upper() == "TEXT":
        log.info("TEXT modality requested but server-rejected (1007); using AUDIO+transcription")
        modality = "AUDIO"
    try:
        mod = types.Modality.TEXT if str(modality).upper() == "TEXT" else types.Modality.AUDIO  # type: ignore[union-attr]
    except Exception:
        mod = str(modality).upper()
    mods = [mod]
    kwargs: dict = {"response_modalities": mods}
    if str(modality).upper() == "AUDIO":
        # audio-native models still give us text via transcription fields
        try:
            kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()  # type: ignore[union-attr]
        except Exception:
            pass
        try:
            kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig()  # type: ignore[union-attr]
        except Exception:
            pass
        # Activity detection + turn coverage: WITHOUT this the model neither
        # endpoints speech nor generates answers (verified experimentally).
        try:
            if types is not None and hasattr(types, "RealtimeInputConfig"):
                _aad_kwargs: dict = {"disabled": False}
                try:
                    if hasattr(types, "StartSensitivity"):
                        _aad_kwargs["start_of_speech_sensitivity"] = types.StartSensitivity.START_SENSITIVITY_HIGH  # type: ignore[union-attr]
                    if hasattr(types, "EndSensitivity"):
                        _aad_kwargs["end_of_speech_sensitivity"] = types.EndSensitivity.END_SENSITIVITY_HIGH  # type: ignore[union-attr]
                except Exception:
                    pass
                try:
                    _aad_kwargs["prefix_padding_ms"] = PREFIX_MS
                    _aad_kwargs["silence_duration_ms"] = SILENCE_MS
                except Exception:
                    pass
                _ric_kwargs: dict = {
                    "automatic_activity_detection": types.AutomaticActivityDetection(**_aad_kwargs),  # type: ignore[union-attr]
                }
                try:
                    if hasattr(types, "TurnCoverage"):
                        _ric_kwargs["turn_coverage"] = types.TurnCoverage.TURN_INCLUDES_ALL_INPUT  # type: ignore[union-attr]
                except Exception:
                    pass
                kwargs["realtime_input_config"] = types.RealtimeInputConfig(**_ric_kwargs)  # type: ignore[union-attr]
        except Exception as e:
            log.debug("realtime_input_config unsupported by SDK: %s", e)
    # Cap reply length: AUDIO answers stream in realtime, so long answers =
    # late answers. Short cap keeps replies fast and on-point.
    # Low temperature/top_p: faster first token, short deterministic replies.
    try:
        if types is not None and hasattr(types, "GenerationConfig"):
            try:
                kwargs["generation_config"] = types.GenerationConfig(  # type: ignore[union-attr]
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0.2,
                    top_p=0.9,
                )
            except TypeError:
                # Older SDK tolerates only a subset of keys; try reduced forms.
                try:
                    kwargs["generation_config"] = types.GenerationConfig(  # type: ignore[union-attr]
                        max_output_tokens=MAX_OUTPUT_TOKENS
                    )
                except Exception as e:
                    log.debug("generation_config unsupported by SDK: %s", e)
    except Exception as e:
        log.debug("generation_config unsupported by SDK: %s", e)
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    # gemini-3.8-live extended thinking: map UI level to SDK ThinkingConfig
    try:
        lvl = str(thinking_level or "").strip().upper()
        if lvl in ("EXTENDED", "MAX", "ULTRA"):
            lvl = "HIGH"
        if lvl and lvl not in ("MINIMAL", "LOW", "MEDIUM", "HIGH"):
            lvl = "HIGH"
        if lvl and types is not None and hasattr(types, "ThinkingConfig"):
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=lvl)  # type: ignore[union-attr]
            except Exception as e:
                log.debug("thinking_config unsupported by SDK/model: %s", e)
    except Exception:
        pass
    if enable_web_search:
        try:
            if types is not None and hasattr(types, "Tool") and hasattr(types, "FunctionDeclaration"):
                fd = types.FunctionDeclaration(  # type: ignore[union-attr]
                    name="web_search",
                    description=(
                        "Free local DuckDuckGo search, no key. "
                        "Use when interviewer asks factual/current questions "
                        "needing outside knowledge."
                    ),
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "query": {"type": "STRING", "description": "search query"}
                        },
                        "required": ["query"],
                    },
                )
                kwargs["tools"] = [types.Tool(function_declarations=[fd])]  # type: ignore[union-attr]
        except Exception as e:
            log.debug("web_search tool unsupported by SDK/model: %s", e)
    try:
        return types.LiveConnectConfig(**kwargs)  # type: ignore[union-attr]
    except Exception:
        # older SDK: generation_config nesting
        try:
            return types.LiveConnectConfig(  # type: ignore[union-attr]
                response_modalities=mods,
                system_instruction=types.Content(parts=[types.Part.from_text(text=system_instruction)]) if system_instruction else None,  # type: ignore[union-attr]
            )
        except Exception:
            return {"response_modalities": ["TEXT"], "system_instruction": system_instruction}


def _is_fatal_model_error(msg: str) -> bool:
    """True if this connect error means the model candidate is unusable.

    Fatal (try next candidate): 1007 / "not supported" / 404 / 400 etc.
    Transient (backoff + retry same candidate): 1008 policy/overload closes.
    1008 must never pin to a flaky fallback when the primary can recover.
    """
    try:
        s = str(msg or "")
    except Exception:
        return False
    if "1008" in s:
        return False
    low = s.lower()
    if "1007" in s:
        return True
    return (
        any(x in s for x in ("404", "NOT_FOUND", "400", "INVALID_ARGUMENT"))
        or any(x in low for x in ("not found", "not supported", "unsupported", "must be specified"))
    )


class GeminiLiveClient:
    """Producer-consumer live session with auto-reconnect + model fallback."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-live",
        system_instruction: str = "",
        audio_queue=None,  # queue.Queue[bytes] (sync, from AudioEngine)
        fallback_models: Optional[List[str]] = None,
        speaker_queue=None,  # queue.Queue[str] speaker markers from AudioEngine
        thinking_level: str = "HIGH",
        summarizer_model: str = "gemini-3.5-flash-lite",
        enable_web_search: bool = False,
        web_max_results: int = 5,
        selftest_hello: bool = False,
        max_session_minutes: float = 10.0,
    ) -> None:
        try:
            self.api_key = str(api_key or "").strip()
        except Exception:
            self.api_key = ""
        try:
            self.model = str(model or "gemini-3.8-live").strip() or "gemini-3.8-live"
        except Exception:
            self.model = "gemini-3.8-live"
        try:
            self.system_instruction = system_instruction or ""
        except Exception:
            self.system_instruction = ""
        self.audio_queue = audio_queue
        self.speaker_queue = speaker_queue
        # Normalize fallbacks: tolerate a single string (not iterable-of-chars),
        # strip whitespace, drop blanks/duplicates and the primary model.
        _fm = fallback_models
        if isinstance(_fm, str):
            _fm = [_fm]
        _seen: set = set()
        _clean: List[str] = []
        try:
            for _m in list(_fm or []):
                try:
                    _s = str(_m or "").strip()
                except Exception:
                    continue
                if _s and _s != self.model and _s not in _seen:
                    _seen.add(_s)
                    _clean.append(_s)
        except Exception:
            _clean = []
        self.fallback_models = _clean
        try:
            self.summarizer_model = str(summarizer_model or "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
        except Exception:
            self.summarizer_model = "gemini-3.5-flash-lite"
        self.enable_web_search = bool(enable_web_search)
        try:
            self.web_max_results = max(1, min(10, int(web_max_results)))
        except (TypeError, ValueError, OverflowError):
            self.web_max_results = 5
        try:
            lvl = str(thinking_level or "HIGH").strip().upper()
        except Exception:
            lvl = "HIGH"
        if lvl in ("EXTENDED", "MAX", "ULTRA"):
            lvl = "HIGH"
        self.thinking_level = lvl if lvl in ("MINIMAL", "LOW", "MEDIUM", "HIGH") else "HIGH"
        self._stop = asyncio.Event()
        self._working_model: Optional[str] = None
        self._audio_mode_models: set = set()
        self.last_error: str = ""
        self.selftest_hello = bool(selftest_hello)
        try:
            _sess = float(max_session_minutes)
            if _sess != _sess or _sess in (float("inf"), float("-inf")):
                _sess = 10.0
            self.max_session_age_s = max(120.0, min(3600.0, _sess * 60.0))
        except (TypeError, ValueError, OverflowError):
            self.max_session_age_s = 600.0
        self._hello_done = False
        self._hello_attempts = 0
        self._hello_time: float = 0.0
        # latency / reconnect bookkeeping (re-probe primary every 2 sessions)
        self._session_tries = 0
        self._last_marker_time: float = 0.0
        # -- diagnostics (console stats ticker) ---------------------------
        self.sessions_opened = 0
        self.rx_responses = 0
        self.rx_chunks = 0
        self.tx_audio_frames = 0
        self.tx_hellos = 0
        self.it_chunks = 0
        self._hello_rx_baseline = 0
        self.last_rx_time: float = 0.0

    def get_stats(self) -> dict:
        try:
            import time as _t
            return {
                "sessions": int(self.sessions_opened),
                "rx_responses": int(self.rx_responses),
                "rx_chunks": int(self.rx_chunks),
                "tx_audio": int(self.tx_audio_frames),
                "tx_hellos": int(self.tx_hellos),
                "it_chunks": int(self.it_chunks),
                "last_rx_age_s": round(_t.monotonic() - float(self.last_rx_time), 1) if self.last_rx_time else -1.0,
                "working_model": str(self._working_model or ""),
                "last_error": str(self.last_error or "")[:200],
            }
        except Exception:
            return {}

    def stop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._stop.set)
        except RuntimeError:
            self._stop.set()

    def _candidates(self) -> List[str]:
        # Live (bidiGenerateContent) capable models only — verified Oct 2026.
        # gemini-2.0-flash-live-001 is retired; gemini-3.5-flash-lite is
        # generateContent-only (summarizer), NOT Live-capable.
        seen, out = set(), []
        for m in [self.model, *self.fallback_models, "gemini-3.8-live", "gemini-3.8-live-extended-thinking", "gemini-3.1-flash-live-preview", "gemini-2.5-flash-native-audio-preview-12-2025"]:
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def _ordered_candidates(self, probe_primary: bool = False) -> List[str]:
        """Candidate order: prefer working model, unless re-probing primary.

        probe_primary=True returns primary-first ordering (used every 2nd
        session so a recovered primary is not permanently shadowed by a
        flaky fallback pinned as _working_model).
        """
        cands = self._candidates()
        if probe_primary:
            return cands
        if self._working_model and self._working_model in cands:
            return [self._working_model] + [m for m in cands if m != self._working_model]
        return cands

    async def run(
        self,
        on_text: TextCallback,
        on_status: Optional[StatusCallback] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not _HAS_GENAI:
            await self._emit(on_status, "disconnected")
            raise RuntimeError("google-genai SDK not installed. pip install google-genai")
        if not self.api_key:
            await self._emit(on_status, "disconnected")
            raise RuntimeError("Missing API key. Set it in the Settings dialog.")

        def _report_err(msg: str) -> None:
            try:
                self.last_error = str(msg or "")[:500]
            except Exception:
                pass
            if on_error is None:
                return
            try:
                r = on_error(self.last_error)
                if asyncio.iscoroutine(r):
                    asyncio.get_running_loop().create_task(r)  # type: ignore[func-returns-value]
            except Exception:
                pass

        self._stop.clear()
        backoff = 2.0
        last_reason = ""
        # Re-probe primary every 2 sessions: never pin to a flaky fallback.
        try:
            self._session_tries = int(getattr(self, "_session_tries", 0) or 0)
        except Exception:
            self._session_tries = 0
        ordered = self._ordered_candidates(probe_primary=(self._session_tries % 2 == 1))

        while not self._stop.is_set():
            self._session_tries += 1
            if self._session_tries % 2 == 0:
                ordered = self._ordered_candidates(probe_primary=True)
            else:
                ordered = self._ordered_candidates(probe_primary=False)
            connected = False
            for candidate in ordered:
                if self._stop.is_set():
                    return
                try:
                    await self._emit(on_status, "connecting")
                    reason = await self._run_session(candidate, on_text, on_status)
                    connected = True
                    last_reason = str(reason or "")
                    self._working_model = candidate
                    backoff = 2.0
                    log.info("session %s ended: %s", candidate, reason)
                    if reason == "refresh":
                        try:
                            r = on_text("\n[refreshing session for speed...]\n")
                            if asyncio.iscoroutine(r):
                                await r
                        except Exception:
                            pass
                    break  # _run_session only returns on stop; reconnect outer if it drops
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if self._stop.is_set():
                        return  # shutting down (e.g. socket closed) — exit quietly
                    msg = str(e)
                    log.warning("model %s failed: %s", candidate, msg[:300])
                    _report_err(f"{candidate}: {msg[:300]}")
                    try:
                        # If the session died within seconds of our self-test hello,
                        # the hello may have disturbed it — retry once on next session.
                        import time as _t
                        if (self._hello_time and self._hello_attempts < 3
                                and _t.monotonic() - self._hello_time < 10.0):
                            self._hello_done = False
                            log.info("session died fast after hello; will retry hello (attempt %d)",
                                     self._hello_attempts + 1)
                    except Exception:
                        pass
                    _fatal = _is_fatal_model_error(msg)
                    if _fatal:
                        continue  # wrong model/capability -> try next candidate
                    break  # transient (e.g. 1008) -> backoff then retry same order
            if self._stop.is_set():
                return
            if connected:
                if last_reason == "refresh":
                    continue  # proactive recycle: reconnect immediately, no pause
                # session ended normally (server closed): reconnect with small pause
                await self._emit(on_status, "disconnected")
                await self._sleep_or_stop(1.5)
                continue
            if not connected:
                await self._emit(on_status, "disconnected")
                await self._sleep_or_stop(min(backoff, 30.0))
                backoff = min(backoff * 1.6, 30.0)
                # refresh order for next loop (re-probe logic at loop top)

    async def _run_session(self, model: str, on_text, on_status) -> str:
        log.info("Connecting Live model=%s thinking=%s web_search=%s", model, self.thinking_level, self.enable_web_search)
        client = genai.Client(api_key=self.api_key)  # type: ignore[union-attr]
        # Config strategy (verified live Oct 2026 against production backends):
        # - ALL models use AUDIO + transcription. TEXT response modality is
        #   rejected server-side (1007, backend rev25): a TEXT session OPENS but
        #   DIES the moment any text/audio is sent. AUDIO sessions survive sends
        #   and deliver model text via output_transcription.
        # - UX stays TEXT-ONLY: transcription text is displayed, audio never played.
        # - thinking_config ONLY on *-extended-thinking (plain models reject it).
        is_extended = "extended-thinking" in str(model).lower()
        use_thinking = is_extended  # ONLY the extended model accepts thinking_config
        mode = "AUDIO"
        self._audio_mode_models.add(model)
        log.info("Live config: model=%s mode=%s thinking=%s tools=%s sys_instr_chars=%d",
                 model, mode, (self.thinking_level if use_thinking else "(off)"),
                 bool(self.enable_web_search), len(self.system_instruction or ""))

        def _cfg(thinking: str, tools_flag: bool, _mode: str = mode):
            return _build_live_config(
                self.system_instruction, thinking, _mode,
                enable_web_search=bool(tools_flag and self.enable_web_search),
            )

        if use_thinking:
            # Extended-thinking runs in AUDIO+transcription mode WITH thinking.
            # Difficulty/thinking_level actually takes effect here.
            if self.enable_web_search:
                configs = [
                    _cfg(self.thinking_level, True),
                    _cfg(self.thinking_level, False),  # retry w/o tools
                    _cfg("", True),
                    _cfg("", False),  # retry w/o thinking
                ]
            else:
                configs = [
                    _build_live_config(self.system_instruction, self.thinking_level, mode),
                    _build_live_config(self.system_instruction, "", mode),  # retry w/o thinking
                ]
        else:
            if self.enable_web_search:
                configs = [_cfg("", True), _cfg("", False)]
            else:
                configs = [_build_live_config(self.system_instruction, "", mode)]
        connect = client.aio.live.connect  # type: ignore[union-attr]
        last_err: Optional[BaseException] = None
        for config in configs:
            try:
                async with connect(model=model, config=config) as session:
                    log.info("Live session open (%s, mode=%s, text-only)", model, mode)
                    self.sessions_opened += 1
                    await self._emit(on_status, "active")
                    try:
                        r = on_text(f"\n[connected → {model}]\n")
                        if asyncio.iscoroutine(r):
                            await r
                    except Exception:
                        pass
                    hello_sent_here = False
                    if self.selftest_hello and not self._hello_done and self._hello_attempts < 3:
                        self._hello_attempts += 1
                        try:
                            import time as _t
                            self._hello_time = _t.monotonic()
                        except Exception:
                            pass
                        # mark done optimistically; cleared below if this session
                        # yields no model text (unanswered hello -> retry next session)
                        self._hello_done = True
                        self._hello_rx_baseline = int(self.rx_chunks)
                        # Non-blocking: never await hello before _session_loop,
                        # so audio streaming starts immediately.
                        asyncio.create_task(self._send_hello(session, on_text))
                        hello_sent_here = True
                    try:
                        reason = await self._session_loop(session, on_text)
                    finally:
                        try:
                            if (hello_sent_here and not self._stop.is_set()
                                    and self._hello_attempts < 3
                                    and int(self.rx_chunks) <= int(self._hello_rx_baseline)):
                                self._hello_done = False
                                log.info("hello went unanswered; will retry on next session (attempt %d)",
                                         self._hello_attempts + 1)
                        except Exception:
                            pass
                return reason
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_err = e
                msg = str(e)
                _low = msg.lower()
                # NOTE: we always start in AUDIO+transcription now (TEXT sessions
                # open but die on first send — 1007/backend rev25), so no
                # modality fallback is needed here. A 1007 here means this
                # candidate is unusable -> run() treats it as fatal -> next model.
                if use_thinking and "thinking level is not supported" in _low:
                    log.info("thinking_config rejected by %s; retrying without it", model)
                    continue
                # Tolerate SDK/model rejection of tools: retry WITHOUT tools.
                _is_tool_err = (
                    "400" in msg
                    or "INVALID_ARGUMENT" in msg
                    or "tool" in _low
                    or "function" in _low
                    or "unknown field" in _low
                    or "unsupported" in _low
                )
                if _is_tool_err and config is not configs[-1]:
                    log.info("config rejected by %s (%s); retrying with fallback config", model, msg[:200])
                    continue
                raise
        if last_err is not None:
            raise last_err

    async def _watch_stop(self, deadline: Optional[float] = None) -> str:
        """Wait for stop() or a session-refresh deadline. Returns why we woke."""
        try:
            if deadline is None:
                await self._stop.wait()
                return "stopped"
            import time as _t
            while True:
                if self._stop.is_set():
                    return "stopped"
                remaining = float(deadline) - _t.monotonic()
                if remaining <= 0:
                    return "refresh"
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=min(remaining, 5.0))
                    return "stopped"
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            raise
        except Exception:
            return "stopped"

    async def _send_hello(self, session, on_text) -> None:
        """One-time pipeline self-test: ask the model for a one-line reply.

        Proves the TEXT path works even in a silent room. Reply arrives via
        the normal _consume path. Never raises (except CancelledError).
        """
        try:
            log.info("self-test: requesting one-line reply from model...")
            try:
                r = on_text("\n[self-test: asking model for a one-line reply…]\n")
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                pass
            hello = "Reply with exactly this line and nothing else: audio pipeline OK"
            sent = False
            # Primary: realtime text (same channel as audio; never disturbs the session).
            if hasattr(session, "send_realtime_input"):
                try:
                    await session.send_realtime_input(text=hello)
                    sent = True
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.debug("hello realtime send failed: %s", e)
            if not sent and hasattr(session, "send_client_content") and types is not None:
                try:
                    turn = types.Content(role="user", parts=[types.Part.from_text(text=hello)])
                    await session.send_client_content(turns=turn, turn_complete=True)
                    sent = True
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.debug("hello client-content send failed: %s", e)
            if sent:
                self.tx_hellos += 1
                log.info("self-test hello sent, waiting for reply...")
            else:
                log.warning("self-test hello could not be sent (no supported method)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.debug("hello error: %s", e)

    async def _session_loop(self, session, on_text) -> str:
        import time as _t
        deadline = _t.monotonic() + float(getattr(self, "max_session_age_s", 600.0))
        producer = asyncio.create_task(self._produce(session))
        consumer = asyncio.create_task(self._consume(session, on_text))
        watcher = asyncio.create_task(self._watch_stop(deadline))
        try:
            while True:
                done, pending = await asyncio.wait(
                    {producer, consumer, watcher}, return_when=asyncio.FIRST_COMPLETED
                )
                if watcher in done and not watcher.cancelled() and watcher.exception() is None:
                    # Clean shutdown requested: drop everything quietly, no reconnect error.
                    # A "refresh" result means proactive session recycle (bounds
                    # server-side context drift -> keeps answers fast + relevant).
                    try:
                        why = watcher.result()
                    except Exception:
                        why = "stopped"
                    for t in pending:
                        t.cancel()
                    return str(why or "stopped")
                for t in done:
                    if not t.cancelled() and t.exception():
                        exc = t.exception()
                        try:
                            name = "producer" if t is producer else "consumer"
                            log.warning("session %s failed: %s", name, str(exc)[:300])
                        except Exception:
                            pass
                        for p in pending:
                            p.cancel()
                        raise exc  # type: ignore[misc]
                if self._stop.is_set():
                    for t in pending:
                        t.cancel()
                    return "stopped"
                # A task ended cleanly without exception (e.g. server closed the
                # stream): session is over, return so run() reconnects.
                for t in pending:
                    t.cancel()
                return "server-closed"
        finally:
            for t in (producer, consumer, watcher):
                try:
                    if not t.done():
                        t.cancel()
                except Exception:
                    pass
            # Reap cancellations so recycles don't leak pending tasks that
            # later log "exception never retrieved" / "task destroyed pending".
            try:
                await asyncio.gather(producer, consumer, watcher, return_exceptions=True)
            except Exception:
                pass

    async def _send_speaker_marker(self, session, marker: str) -> None:
        """Send [SPEAKER: ...] hint so model knows who is talking."""
        try:
            # Primary: realtime text input (standard, never disturbs the session).
            if hasattr(session, "send_realtime_input"):
                try:
                    await session.send_realtime_input(text=marker)
                    return
                except asyncio.CancelledError:
                    raise
                except TypeError:
                    pass
                except Exception as e:
                    log.debug("marker realtime send failed: %s", e)
            # fallback: client content turn
            if hasattr(session, "send_client_content") and types is not None:
                try:
                    turn = types.Content(role="user", parts=[types.Part.from_text(text=marker)])
                    await session.send_client_content(turns=turn, turn_complete=False)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.debug("marker client-content send failed: %s", e)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.debug("speaker marker error: %s", e)

    async def _produce(self, session) -> None:
        import time as _time

        pending_marker: Optional[str] = None
        while not self._stop.is_set():
            try:
                # 1) Poll speaker markers non-blocking into pending (latest
                # wins). Never send before audio: a marker must not delay
                # the next audio frame by more than ~10ms.
                if self.speaker_queue is not None:
                    try:
                        while True:
                            marker = self.speaker_queue.get_nowait()
                            if marker:
                                pending_marker = str(marker)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass  # empty queue
                # 2) Poll audio non-blocking (10ms) instead of a 200ms
                # blocking get + thread hop.
                if self.audio_queue is None:
                    # Audio idle: safe to flush marker (throttled 1/500ms).
                    if pending_marker is not None:
                        now = _time.monotonic()
                        if now - float(getattr(self, "_last_marker_time", 0.0) or 0.0) >= MARKER_THROTTLE_S:
                            self._last_marker_time = now
                            try:
                                await self._send_speaker_marker(session, pending_marker)
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                pass
                            pending_marker = None
                    await asyncio.sleep(0.01)
                    continue
                try:
                    pcm: bytes = self.audio_queue.get_nowait()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Gotten queue empty -> flush marker if idle, then poll.
                    if pending_marker is not None:
                        now = _time.monotonic()
                        if now - float(getattr(self, "_last_marker_time", 0.0) or 0.0) >= MARKER_THROTTLE_S:
                            self._last_marker_time = now
                            try:
                                await self._send_speaker_marker(session, pending_marker)
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                pass
                            pending_marker = None
                    await asyncio.sleep(0.01)
                    continue
                # Flush backlog to latest: keep realtime, drop oldest stale
                # frames when >8 queued (would add >1s latency otherwise).
                try:
                    _qsize = int(self.audio_queue.qsize())
                except Exception:
                    _qsize = 0
                if _qsize > AUDIO_BACKLOG_FLUSH_THRESHOLD:
                    try:
                        while True:
                            pcm = self.audio_queue.get_nowait()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass
                    try:
                        log.debug("audio backlog flush: dropped stale frames, latest kept (qsize=%d)", _qsize)
                    except Exception:
                        pass
                if not pcm:
                    await asyncio.sleep(0.01)
                    continue
                try:
                    blob = types.Blob(data=pcm, mime_type="audio/pcm;rate=16000")  # type: ignore[union-attr]
                except Exception:
                    blob = {"data": pcm, "mimeType": "audio/pcm;rate=16000"}  # type: ignore[assignment]
                try:
                    await session.send_realtime_input(audio=blob)
                    self.tx_audio_frames += 1
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    # Connection actually dead? -> let session recycle.
                    # Anything else (transient send hiccup) -> keep streaming.
                    en = type(e).__name__.lower()
                    em = str(e).lower()
                    if ("closed" in en or "closed" in em or "1007" in em or "1008" in em
                            or "1000" in em or "1001" in em or "websocket" in en):
                        raise
                    log.debug("audio send hiccup (kept alive): %s", str(e)[:200])
                    await asyncio.sleep(0.01)
                    continue
                # 3) After audio is sent, flush pending marker (throttled).
                if pending_marker is not None:
                    now = _time.monotonic()
                    if now - float(getattr(self, "_last_marker_time", 0.0) or 0.0) >= MARKER_THROTTLE_S:
                        self._last_marker_time = now
                        try:
                            await self._send_speaker_marker(session, pending_marker)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            pass
                        pending_marker = None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Do not swallow connection-dead errors raised by the audio
                # send path above (it re-raises on closed/1007/1008/...):
                # those must propagate so _session_loop can recycle.
                _en = type(e).__name__.lower()
                _em = str(e).lower()
                if ("closed" in _en or "closed" in _em or "1007" in _em or "1008" in _em
                        or "1000" in _em or "1001" in _em or "websocket" in _en):
                    raise
                # Never kill a healthy session on a produce-side bug: log + keep going.
                # The consumer task owns session lifetime; connection loss surfaces there.
                log.debug("produce error (kept alive): %s", str(e)[:200])
                await asyncio.sleep(0.01)

    async def _reply_tool(self, session, name: str, call_id, result: str) -> None:
        """Send one FunctionResponse, tolerating SDK variants. Never raises."""
        formatted = str(result or "")
        try:
            if hasattr(session, "send_tool_response"):
                if types is not None:
                    try:
                        try:
                            fr = types.FunctionResponse(name=name, id=call_id, response={"result": formatted})  # type: ignore[call-arg]
                        except TypeError:
                            try:
                                fr = types.FunctionResponse(name=name, response={"result": formatted})  # type: ignore[call-arg]
                            except Exception:
                                fr = {"name": name, "response": {"result": formatted}}
                        await session.send_tool_response(function_responses=[fr])
                        return
                    except asyncio.CancelledError:
                        raise
                    except TypeError:
                        try:
                            await session.send_tool_response([{"name": name, "response": {"result": formatted}}])  # type: ignore[arg-type]
                            return
                        except asyncio.CancelledError:
                            raise
                        except Exception as e2:
                            log.debug("send_tool_response fallback failed: %s", e2)
                    except Exception as e:
                        log.debug("send_tool_response failed: %s", e)
            if hasattr(session, "send_realtime_input"):
                try:
                    await session.send_realtime_input(text=formatted)
                except asyncio.CancelledError:
                    raise
                except TypeError:
                    pass
                except Exception as e:
                    log.debug("tool realtime fallback failed: %s", e)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.debug("tool reply failed: %s", e)

    async def _handle_tool_call(self, session, tc, on_text) -> None:
        """Execute web_search function calls from the Live model. Never raises (except CancelledError)."""
        try:
            try:
                import websearch as _websearch  # lazy to avoid circulars
            except Exception as e:
                log.debug("websearch import failed: %s", e)
                return
            calls = None
            if isinstance(tc, dict):
                calls = tc.get("function_calls") or tc.get("functionCalls") or tc.get("calls")
            else:
                calls = getattr(tc, "function_calls", None)
                if calls is None:
                    calls = getattr(tc, "functionCalls", None)
            if calls is None:
                calls = [tc]
            if not isinstance(calls, (list, tuple)):
                calls = [calls]
            for fc in calls:
                try:
                    if isinstance(fc, dict):
                        name = fc.get("name") or fc.get("function_name") or fc.get("functionName")
                        args = fc.get("args") or fc.get("arguments") or fc.get("parameters") or {}
                        call_id = fc.get("id") or fc.get("function_call_id") or fc.get("call_id")
                    else:
                        name = getattr(fc, "name", None) or getattr(fc, "function_name", None)
                        args = getattr(fc, "args", None)
                        if args is None:
                            args = getattr(fc, "arguments", None)
                        call_id = (
                            getattr(fc, "id", None)
                            or getattr(fc, "function_call_id", None)
                            or getattr(fc, "call_id", None)
                        )
                    if name != "web_search":
                        # Never silently drop a function call: the model blocks
                        # waiting for a function_response. Reply with an error.
                        try:
                            _err = f"Unsupported function {name!r} (only web_search is available)."
                            await self._reply_tool(session, name or "unknown", call_id, _err)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            pass
                        continue
                    # args may be JSON string or dict
                    if isinstance(args, str):
                        try:
                            import json as _json
                            args = _json.loads(args) if args.strip() else {}
                        except Exception:
                            args = {"query": args}
                    if not isinstance(args, dict):
                        args = {"query": str(args or "")}
                    query = str(args.get("query", "") or "").strip()
                    if not query:
                        try:
                            await self._reply_tool(session, "web_search", call_id, "Missing 'query' argument.")
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            pass
                        continue
                    try:
                        result = await asyncio.to_thread(
                            _websearch.search_sync, query, self.web_max_results
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        result = f"Search failed: {e}"
                    formatted = f"WEB RESULT for '{query}':\n{result}"
                    # Reply to model so it can use the facts.
                    try:
                        await self._reply_tool(session, "web_search", call_id, formatted)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        log.debug("tool response send failed: %s", e)
                    # Also forward to HUD so the user sees the facts.
                    try:
                        r = on_text("\n" + formatted + "\n")
                        if asyncio.iscoroutine(r):
                            await r
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        log.debug("tool HUD forward failed: %s", e)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.debug("tool_call entry failed: %s", e)
                    continue
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.debug("handle_tool_call failed: %s", e)

    async def _consume(self, session, on_text) -> None:
        async def _put(text: str) -> None:
            r = on_text(text)
            if asyncio.iscoroutine(r):
                await r
            try:
                import time as _t
                self.rx_chunks += 1
                self.last_rx_time = _t.monotonic()
                if self.rx_chunks == 1:
                    log.info("first text chunk received (%d chars): %r", len(text), text[:80])
            except Exception:
                pass

        async def _put_separator() -> None:
            # Lightweight separator: no rx_chunks/last_rx_time accounting
            # (reduces turn_complete overhead, keeps hello-answer check honest).
            try:
                r = on_text("\n\n---\n\n")
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                pass

        try:
            async for response in session.receive():
                if self._stop.is_set():
                    return
                try:
                    self.rx_responses += 1
                except Exception:
                    pass
                # Function-calling: model requests local web_search.
                try:
                    tc = getattr(response, "tool_call", None)
                    if tc:
                        try:
                            await self._handle_tool_call(session, tc, on_text)
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            log.debug("tool_call handling failed: %s", e)
                        continue
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.debug("tool_call check failed: %s", e)
                try:
                    sc = getattr(response, "server_content", None)
                    if sc is None:
                        # Unknown envelope (no server_content): log shape throttled for diagnosis.
                        try:
                            if self.rx_responses % 20 == 1:
                                log.debug("response without server_content: %s",
                                          [a for a in dir(response) if not a.startswith("_")][:20])
                        except Exception:
                            pass
                        continue
                    # auto-separate answers per turn for readable HUD
                    if getattr(sc, "turn_complete", False):
                        await _put_separator()
                        continue
                    # AUDIO-native models deliver text via transcription fields
                    ot = getattr(sc, "output_transcription", None)
                    if ot is not None:
                        t = getattr(ot, "text", None)
                        if t:
                            await _put(t)
                            continue
                    # Input transcription: what the model HEARS. Log it (console
                    # + app.log) so silence vs speech is diagnosable. Not shown
                    # on HUD to avoid flooding it with transcripts.
                    try:
                        it = getattr(sc, "input_transcription", None)
                        if it is not None:
                            itext = getattr(it, "text", None)
                            if itext:
                                self.it_chunks += 1
                                log.info("heard (%d chars): %r", len(itext), itext[:150])
                                # Fast-path: flush any pending speaker marker
                                # (non-blocking, throttled) so the model knows
                                # who just spoke without delaying the turn.
                                try:
                                    import time as _mt
                                    if self.speaker_queue is not None and (
                                            _mt.monotonic() - float(getattr(self, "_last_marker_time", 0.0) or 0.0) >= MARKER_THROTTLE_S):
                                        try:
                                            _m = self.speaker_queue.get_nowait()
                                        except Exception:
                                            _m = None
                                        if _m:
                                            self._last_marker_time = _mt.monotonic()
                                            asyncio.create_task(self._send_speaker_marker(session, str(_m)))
                                except asyncio.CancelledError:
                                    raise
                                except Exception:
                                    pass
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass
                    turn = getattr(sc, "model_turn", None)
                    if not turn:
                        # interrupted / generation complete still worth separating
                        if getattr(sc, "generation_complete", False):
                            await _put_separator()
                        continue
                    parts = getattr(turn, "parts", []) or []
                    texts = [getattr(part, "text", None) for part in parts]
                    if any(texts):
                        for text in texts:
                            if text:
                                await _put(text)
                    else:
                        # Model turn with no text parts (e.g. audio-only): log shape throttled.
                        try:
                            if self.rx_responses % 20 == 1:
                                kinds = [type(getattr(p, "inline_data", None)).__name__ if getattr(p, "inline_data", None) is not None else "no-inline-data" for p in parts]
                                log.debug("model_turn without text: parts=%d kinds=%s", len(parts), kinds[:4])
                        except Exception:
                            pass
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.debug("consume parse error: %s", e)
                    continue
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Clean shutdown (our own stop() → 1000/1001 close) is not an error.
            em = str(e).lower()
            en = type(e).__name__.lower()
            if self._stop.is_set() and ("1000" in em or "1001" in em or "closed" in en or "closed" in em):
                return
            raise

    async def summarize(self, transcript: str) -> str:
        """Summarize recent transcript with the flash-lite model (context memory)."""
        if not _HAS_GENAI or not self.api_key or not (transcript or "").strip():
            return ""
        try:
            client = genai.Client(api_key=self.api_key)  # type: ignore[union-attr]
            prompt = (
                "Summarize the following live technical-interview assistant output "
                "into 3-5 dense bullets: questions asked, key algorithms/patterns "
                "recommended, edge cases, and open follow-ups. No fluff.\n\n"
            )
            resp = await client.aio.models.generate_content(  # type: ignore[union-attr]
                model=self.summarizer_model,
                contents=prompt + transcript[-6000:],
            )
            return (getattr(resp, "text", "") or "").strip()
        except Exception as e:
            log.warning("summarize failed: %s", e)
            return ""

    @staticmethod
    async def _emit(cb, value: str) -> None:
        if cb is None:
            return
        try:
            r = cb(value)
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            pass

    async def _sleep_or_stop(self, delay: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass
