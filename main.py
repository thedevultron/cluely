"""Entry point: settings dialog -> audio + Gemini Live + HUD (PySide6)."""

from __future__ import annotations

print("[boot] starting...", flush=True)

import asyncio
import logging
import os
import sys
import threading
import traceback
from datetime import datetime

print("[boot] stdlib imports ok", flush=True)

try:
    # Consoles here are often cp1252: never crash on Unicode output (model
    # text, arrows). Unprintable chars become ? instead of raising.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding=sys.stdout.encoding or "utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding=sys.stderr.encoding or "utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
try:
    from logging.handlers import RotatingFileHandler as _RFH
    _fh = _RFH(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log"),
        maxBytes=512 * 1024, backupCount=2, encoding="utf-8",
    )
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    _fh.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(_fh)
except Exception:
    pass
log = logging.getLogger("InterviewCopilot.main")

try:
    def _log_uncaught(exc_type, exc_value, exc_tb) -> None:
        try:
            log.error("UNCAUGHT: %s: %s", getattr(exc_type, "__name__", exc_type), exc_value)
        except Exception:
            pass
    sys.excepthook = _log_uncaught
except Exception:
    pass


def _inject_sys_path() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    if base not in sys.path:
        sys.path.insert(0, base)


_inject_sys_path()

print("[boot] importing config_manager...", flush=True)
from config_manager import load_config  # noqa: E402
print("[boot] importing audio_engine (sounddevice/soundcard)...", flush=True)
from audio_engine import AudioEngine  # noqa: E402
print("[boot] importing gemini_live_client...", flush=True)
from gemini_live_client import GeminiLiveClient  # noqa: E402
print("[boot] all imports ok", flush=True)


def _write_transcript(path: str, text: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass


def _mask_key(key: str) -> str:
    try:
        k = str(key or "").strip()
        if not k:
            return "(empty)"
        if len(k) <= 6:
            return k[0] + "***"
        return f"{k[:4]}...{k[-2:]} (len={len(k)})"
    except Exception:
        return "(?)"


def _log_launch_diagnostics(config: dict) -> None:
    """Console block the user can paste back for diagnosis. Always logged at INFO."""
    try:
        import importlib.metadata as _md
        try:
            _sdk = _md.version("google-genai")
        except Exception:
            _sdk = "?"
        try:
            import concurrent.futures as _cf
            from audio_engine import list_loopbacks as _ll
            _ex = _cf.ThreadPoolExecutor(max_workers=1)
            try:
                _lbs = _ex.submit(_ll).result(timeout=10.0)
            except _cf.TimeoutError:
                _lbs = ["<query TIMED OUT after 10s — audio API hanging>"]
                print("[diag] WARNING: loopback query timed out (audio API hanging)", flush=True)
            finally:
                try:
                    _ex.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
        except Exception as e:
            _lbs = [f"<query failed: {e}>"]
        log.info("=== DIAG launch ===")
        log.info("DIAG key=%s starts_with_AIza=%s", _mask_key(config.get("api_key", "")),
                 str(config.get("api_key", "")).startswith("AIza"))
        log.info("DIAG model=%s fallback=%s ultimate=%s summarizer=%s thinking=%s difficulty=%s",
                 config.get("model"), config.get("fallback_model"),
                 config.get("ultimate_fallback_model"), config.get("summarizer_model"),
                 config.get("thinking_level"), config.get("difficulty_level"))
        log.info("DIAG audio mode=%s mic=%s loopback=%r vad=%s web=%s knowledge=%s sdk=%s",
                 config.get("audio_mode"), config.get("mic_enabled"),
                 config.get("selected_loopback_name"), config.get("vad_threshold_db"),
                 config.get("enable_web_search"), config.get("enable_knowledge"), _sdk)
        log.info("DIAG loopback devices (%d): %s", len(_lbs), _lbs[:5])
        log.info("DIAG sys_instr_chars=%d", len(str(config.get("system_instruction", ""))))
    except Exception as e:
        log.debug("diag block failed: %s", e)


DIFFICULTY_PROMPTS = {
    "EASY": (
        "\n\nDifficulty: EASY — explain simply, short bullets, beginner-friendly, "
        "avoid jargon, give the most obvious working approach first."
    ),
    "MEDIUM": (
        "\n\nDifficulty: MEDIUM — balanced concise bullets: approach, complexity, "
        "edge cases, 2-4 lines of key code when needed."
    ),
    "HARD": (
        "\n\nDifficulty: HARD — dense expert hints: optimal algorithm + proof sketch, "
        "exact complexity, tricky edge cases, trade-offs, minimal but precise code."
    ),
    "EXPERT": (
        "\n\nDifficulty: EXPERT — maximum depth, system-design grade: optimal approach, "
        "alternatives with trade-offs, bottlenecks, failure modes, edge cases, "
        "and production pitfalls. Still concise bullets."
    ),
}


def _enrich_instruction(config: dict) -> str:
    base = str(config.get("system_instruction", ""))
    try:
        mode = str(config.get("audio_mode", "system") or "system").lower()
    except Exception:
        mode = "system"
    if mode == "system":
        marker_help = (
            "\n\nAudio source: ONLY the computer's system audio (WASAPI loopback) is streamed. "
            "There is no microphone. Everything you hear is the interview/meeting audio from the computer. "
            "Speaker protocol: the client interleaves short text hints like "
            "[SPEAKER: SYSTEM AUDIO speaking — answer concisely with bullets]. "
            "When system audio contains a question, answer immediately with high-density concise guidance. "
            "Never reveal the markers. TEXT-ONLY output: never request or produce audio. "
            "Audio-only input: no screen is ever captured or sent."
        )
    else:
        marker_help = (
            "\n\nSpeaker protocol: the client interleaves short text hints like "
            "[SPEAKER: INTERVIEWER speaking — answer concisely with bullets] and "
            "[SPEAKER: CANDIDATE speaking — stay SILENT unless explicit help request]. "
            "Trust these markers over acoustics. Never reveal them. TEXT-ONLY output. "
            "Audio-only input: no screen is ever captured or sent."
        )
    if "Speaker protocol" not in base and "Audio source" not in base:
        base = base + marker_help
    # Difficulty ALWAYS applies via prompt (works on every model).
    try:
        diff = str(config.get("difficulty_level", "MEDIUM") or "MEDIUM").strip().upper()
        if diff not in DIFFICULTY_PROMPTS:
            diff = "MEDIUM"
        if "Difficulty:" not in base:
            base += DIFFICULTY_PROMPTS[diff]
    except Exception:
        pass
    try:
        role = str(config.get("target_role", "") or "").strip()
        company = str(config.get("target_company", "") or "").strip()
        if role or company:
            line = "\n\nCandidate context:"
            if role:
                line += f" target role: {role}."
            if company:
                line += f" target company: {company}."
            base += line
    except Exception:
        pass
    try:
        if bool(config.get("enable_knowledge", False)):
            from knowledge import load_knowledge, list_knowledge_files
            paths = list(config.get("knowledge_paths", []) or [])
            try:
                max_chars = int(config.get("knowledge_max_chars", 12000))
            except (TypeError, ValueError):
                max_chars = 12000
            try:
                files = list_knowledge_files(paths)
                ctx = load_knowledge(paths, max_chars)
            except Exception as e:
                log.warning("load_knowledge failed: %s", e)
                ctx = ""
                files = []
            if ctx and ctx.strip():
                base += (
                    "\n\nLocal knowledge base (candidate's materials "
                    "— prioritize when relevant):\n" + ctx
                )
                log.info("Knowledge injected: %d file(s), %d chars", len(files), len(ctx))
                config["_knowledge_summary"] = f"{len(files)} file(s), {len(ctx)} chars"
            else:
                log.warning("Knowledge enabled but empty (paths=%s)", paths)
                config["_knowledge_summary"] = "enabled but empty — check paths"
    except Exception as e:
        log.debug("knowledge enrich skipped: %s", e)
    return base


def run_hud_session(config: dict) -> int:
    print("[stage] starting HUD session...", flush=True)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    print("[stage] importing overlay_window...", flush=True)
    from overlay_window import OverlayWindow
    print("[stage] creating QApplication...", flush=True)

    app = QApplication.instance() or QApplication(sys.argv)
    print("[stage] creating HUD...", flush=True)

    hud = OverlayWindow(
        font_size=int(config.get("font_size", 14)),
        opacity=float(config.get("hud_opacity", 0.85)),
        auto_resize=bool(config.get("hud_auto_resize", True)),
    )
    hud.show()
    print("[stage] HUD shown, starting hotkeys...", flush=True)
    hud.start_hotkeys()
    print("[stage] creating audio engine...", flush=True)

    engine = AudioEngine(
        mic_index=config.get("selected_mic_index"),
        loopback_name=str(config.get("selected_loopback_name", "")),
        vad_threshold_db=int(config.get("vad_threshold_db", 300)),
        mic_enabled=bool(config.get("mic_enabled", config.get("audio_mode", "system") == "dual")),
        audio_mode=str(config.get("audio_mode", "system") or "system"),
    )

    # Pre-flight: system-audio device must exist, else warn instead of streaming silence.
    # Timeout-guarded: a hanging audio API must never freeze startup.
    print("[stage] pre-flight audio check...", flush=True)
    try:
        import concurrent.futures as _cf
        from audio_engine import list_loopbacks
        _ex = _cf.ThreadPoolExecutor(max_workers=1)
        try:
            _lbs = _ex.submit(list_loopbacks).result(timeout=10.0)
        except _cf.TimeoutError:
            print("[stage] WARNING: audio pre-flight timed out, continuing anyway", flush=True)
            _lbs = ["<timeout>"]
        finally:
            try:
                _ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
    except Exception:
        _lbs = []
    if not _lbs:
        log.warning("No system-audio loopback found at launch.")
        try:
            from PySide6.QtWidgets import QMessageBox
            from PySide6.QtCore import QTimer as _QT
            def _warn():
                try:
                    QMessageBox.warning(
                        hud, "No system audio",
                        "No computer-sound device found.\n\n"
                        "Check your output device is active and soundcard is installed:\n"
                        "pip install soundcard\n\n"
                        "The HUD will still open and keep retrying automatically.",
                    )
                except Exception:
                    pass
            _QT.singleShot(600, _warn)
        except Exception:
            pass

    transcript_path: str | None = None
    if bool(config.get("save_transcript_locally", True)):
        try:
            os.makedirs("transcripts", exist_ok=True)
            transcript_path = os.path.join(
                "transcripts", f"session_{datetime.now():%Y%m%d_%H%M%S}.md"
            )
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(f"# InterviewCopilot session {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        except OSError as e:
            log.warning("transcript file init failed: %s", e)
            transcript_path = None

    # PySide6 signals are thread-safe: emit directly from the asyncio thread.
    def gui_append(text: str) -> None:
        try:
            hud.text_signal.emit(text)
        except Exception:
            pass

    def gui_status(state: str) -> None:
        try:
            hud.status_signal.emit(state)
        except Exception:
            pass

    # gemini callbacks are async; schedule GUI work thread-safely
    recent_text: list[str] = []
    import time as _time
    _last_err_shown = {"t": 0.0}
    live_state = {"status": "connecting"}

    async def on_text(text: str) -> None:
        recent_text.append(text)
        if sum(len(s) for s in recent_text) > 20000:
            while recent_text and sum(len(s) for s in recent_text) > 20000:
                recent_text.pop(0)
        gui_append(text)
        if transcript_path:
            await asyncio.to_thread(_write_transcript, transcript_path, text)

    def on_error_sync(msg: str) -> None:
        # Throttled: show connection problems on the HUD instead of silent recycle.
        try:
            now = _time.monotonic()
            if now - float(_last_err_shown.get("t", 0.0)) < 5.0:
                return
            _last_err_shown["t"] = now
            gui_append(f"\n[connection issue: {msg}]\n")
        except Exception:
            pass

    async def on_status(state: str) -> None:
        try:
            live_state["status"] = str(state)
            if engine.is_paused() and state == "active":
                state = "paused"
            gui_status(state)
        except Exception:
            pass

    def on_pause_toggle() -> None:
        try:
            paused = engine.toggle_pause()
            hud.set_status("paused" if paused else "active")
            log.info("Stream %s", "paused" if paused else "resumed")
        except Exception as e:
            log.warning("pause toggle failed: %s", e)

    hud.signal_pause_toggle.connect(on_pause_toggle)

    def on_quit() -> None:
        try:
            engine.stop()
        except Exception:
            pass
        try:
            hud.stop_hotkeys()
        except Exception:
            pass
        try:
            app.quit()
        except Exception:
            pass

    hud.signal_quit.connect(on_quit)

    print("[stage] starting audio engine...", flush=True)
    try:
        engine.start()
    except Exception as e:
        log.error("AudioEngine start failed: %s", e)
    print("[stage] audio engine started", flush=True)

    hud.set_status("connecting" if not engine.is_paused() else "paused")

    _raw_fallbacks = [config.get("fallback_model", ""), config.get("ultimate_fallback_model", "")]
    _primary = str(config.get("model", "gemini-3.8-live") or "gemini-3.8-live")
    fallbacks: list[str] = []
    for _f in _raw_fallbacks:
        _fs = str(_f or "").strip()
        if _fs and _fs != _primary and _fs not in fallbacks:
            fallbacks.append(_fs)
    if not fallbacks:
        fallbacks = ["gemini-3.1-flash-live-preview"] if _primary != "gemini-3.1-flash-live-preview" else ["gemini-3.8-live"]
    base_instr = _enrich_instruction(config)
    try:
        web_enabled = bool(config.get("enable_web_search", False))
    except Exception:
        web_enabled = False
    try:
        web_max = int(config.get("web_max_results", 5))
    except (TypeError, ValueError):
        web_max = 5

    common_kwargs = dict(
        api_key=str(config.get("api_key", "")),
        model=str(config.get("model", "gemini-3.8-live")),
        system_instruction=base_instr,
        audio_queue=engine.output_queue,
        fallback_models=[f for f in fallbacks if f],
        speaker_queue=engine.speaker_queue,
        thinking_level=str(config.get("thinking_level", "HIGH")),
        summarizer_model=str(config.get("summarizer_model", "gemini-3.5-flash-lite")),
        selftest_hello=bool(config.get("selftest_hello", True)),
    )
    try:
        client = GeminiLiveClient(
            **common_kwargs,
            enable_web_search=web_enabled,
            web_max_results=web_max,
        )
    except TypeError:
        # older client without web-search kwargs
        client = GeminiLiveClient(**common_kwargs)

    stop_event = threading.Event()
    gemini_loop: dict = {}

    def _async_main() -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            gemini_loop["loop"] = loop
            try:
                loop.run_until_complete(client.run(on_text=on_text, on_status=on_status, on_error=on_error_sync))
            finally:
                loop.close()
        except Exception as e:
            log.error("Gemini client exited: %s", e)
            gui_status("disconnected")
            gui_append(f"\n[ERROR] {e}\n")

    t = threading.Thread(target=_async_main, name="gemini-live", daemon=True)
    t.start()
    print("[stage] live client thread started - watch HUD + console DIAG stats", flush=True)

    try:
        _log_launch_diagnostics(config)
    except Exception:
        pass

    # Console stats ticker (for `py main.py` feedback): audio + Live counters every 10s.
    # Plus a ONE-TIME HUD line telling the user whether audio is actually flowing.
    try:
        _flow_reported = {"done": False}
        def _stats_tick():
            try:
                a = engine.get_audio_stats()
                c = client.get_stats()
                log.info("DIAG stats live=%s lb_ok=%s dev=%r frames_sent=%s queue=%s "
                         "lb_e=%s mixed_e=%s vad=%s speaker=%s age_s=%s | "
                         "sessions=%s rx_resp=%s rx_chunks=%s it=%s tx_audio=%s hellos=%s rx_age_s=%s err=%s",
                         live_state.get("status"), engine.get_status().get("ok"),
                         engine.get_status().get("device"), a.get("frames_sent"), a.get("queue"),
                         a.get("lb_energy"), a.get("mixed_energy"), a.get("vad"),
                         a.get("speaker"), a.get("last_frame_age_s"),
                         c.get("sessions"), c.get("rx_responses"), c.get("rx_chunks"),
                         c.get("it_chunks"), c.get("tx_audio"), c.get("tx_hellos"), c.get("last_rx_age_s"),
                         (c.get("last_error") or "")[:150])
                if not _flow_reported["done"] and int(c.get("sessions") or 0) >= 1:
                    _flow_reported["done"] = True
                    try:
                        if int(a.get("frames_sent") or 0) == 0:
                            hud.append_status_line(
                                "No audio frames yet — play computer sound "
                                "(meeting / video / music) so the model hears something. "
                                "Silence = no replies (plus a self-test line above).")
                        else:
                            hud.append_status_line(
                                f"Hearing audio ✔ ({a.get('frames_sent')} frames sent)")
                    except Exception:
                        pass
            except Exception as e:
                log.debug("stats tick failed: %s", e)
        _stats = QTimer()
        _stats.setInterval(10_000)
        _stats.timeout.connect(_stats_tick)
        _stats.start()
        QTimer.singleShot(12_000, _stats_tick)
    except Exception:
        pass

    # HUD welcome line: audio mode + difficulty + knowledge + stealth
    try:
        from PySide6.QtCore import QTimer as _QT2
        def _welcome():
            try:
                mode = str(config.get("audio_mode", "system"))
                diff = str(config.get("difficulty_level", "MEDIUM"))
                model = str(config.get("model", ""))
                think = str(config.get("thinking_level", "HIGH"))
                lines = [f"Ready — {mode} audio → {model} (text-only) • Difficulty {diff} • Reasoning {think}"]
                ks = str(config.get("_knowledge_summary", "") or "")
                if bool(config.get("enable_knowledge", False)):
                    lines.append(f"Docs: {ks or 'enabled'}")
                try:
                    st = engine.get_status()
                    dev = st.get("device") or str(config.get("selected_loopback_name", "") or "auto")
                    lines.append(f"Sound device: {dev or 'auto-detect'}")
                except Exception:
                    pass
                hud.append_status_line(" • ".join(lines))
            except Exception:
                pass
        _QT2.singleShot(400, _welcome)
        # audio health ticker: warn once if loopback is down; never stomp Live status.
        _warned_audio = {"done": False}
        def _health_tick():
            try:
                st = engine.get_status()
                if not st.get("ok") and not _warned_audio["done"]:
                    _warned_audio["done"] = True
                    err = st.get("error") or "waiting for system audio…"
                    hud.append_status_line(f"Audio: {err} (still retrying; Live stays connected)")
                elif st.get("ok"):
                    _warned_audio["done"] = False
            except Exception:
                pass
        _health = QTimer()
        _health.setInterval(3000)
        _health.timeout.connect(_health_tick)
        _health.start()
    except Exception:
        pass

    # periodic flash-lite summarizer: every 5 min distill context to HUD + transcript
    def _summarize_tick() -> None:
        try:
            if not recent_text or "loop" not in gemini_loop:
                return
            if not hasattr(client, "summarize"):
                return
            buf = "".join(recent_text)[-6000:]
            loop = gemini_loop["loop"]
            fut = asyncio.run_coroutine_threadsafe(client.summarize(buf), loop)

            def _done(f):
                try:
                    summary = f.result() or ""
                except Exception:
                    summary = ""
                if summary:
                    block = f"\n\n\u2014 Context summary \u2014\n{summary}\n\n"
                    gui_append(block)
                    if transcript_path:
                        _write_transcript(transcript_path, block)
                    recent_text.clear()

            fut.add_done_callback(_done)
        except Exception as e:
            log.debug("summarize tick failed: %s", e)

    timer = QTimer()
    timer.setInterval(300_000)
    timer.timeout.connect(_summarize_tick)
    timer.start()

    code = app.exec()
    stop_event.set()
    try:
        client.stop()
    except Exception:
        pass
    try:
        engine.stop()
    except Exception:
        pass
    try:
        hud.stop_hotkeys()
    except Exception:
        pass
    return int(code)


async def _diag_live_async(config: dict) -> bool:
    """Headless --diag probe: connect + hello round-trip. Prints everything. Never raises."""
    from gemini_live_client import GeminiLiveClient as _GLC
    texts: list[str] = []
    errors: list[str] = []

    async def on_text(t: str) -> None:
        texts.append(t)
        print(f"[diag] TEXT[{len(t)}]: {t[:150]!r}", flush=True)

    async def on_status(s: str) -> None:
        print(f"[diag] STATUS: {s}", flush=True)

    def on_error(m: str) -> None:
        errors.append(m)
        print(f"[diag] ERROR: {m[:300]}", flush=True)

    try:
        base = _enrich_instruction(config)
    except Exception:
        base = str(config.get("system_instruction", ""))
    try:
        client = _GLC(
            api_key=str(config.get("api_key", "")),
            model=str(config.get("model", "gemini-3.8-live")),
            system_instruction=base,
            thinking_level="",
            summarizer_model=str(config.get("summarizer_model", "gemini-3.5-flash-lite")),
            selftest_hello=True,
        )
    except Exception as e:
        print(f"[diag] client build FAILED: {e}", flush=True)
        return False
    task = asyncio.create_task(client.run(on_text=on_text, on_status=on_status, on_error=on_error))
    got = False
    try:
        for i in range(60):
            await asyncio.sleep(1.0)
            if "audio pipeline OK" in "".join(texts).replace("\n", " "):
                got = True
                break
            if (i + 1) % 15 == 0:
                try:
                    print(f"[diag] ...waiting ({i + 1}s) stats={client.get_stats()}", flush=True)
                except Exception:
                    pass
    finally:
        try:
            client.stop()
        except Exception:
            pass
        try:
            await asyncio.wait_for(task, timeout=10.0)
            print("[diag] shutdown clean", flush=True)
        except asyncio.TimeoutError:
            print("[diag] shutdown TIMEOUT (task hung)", flush=True)
    try:
        print(f"[diag] final stats={client.get_stats()}", flush=True)
    except Exception:
        pass
    return got


def run_diag() -> int:
    """`py main.py --diag`: no GUI. Prints masked config, audio probe, live hello test."""
    print("[diag] headless diagnostics - no GUI", flush=True)
    try:
        config = load_config()
    except Exception as e:
        print(f"[diag] load_config FAILED: {e}", flush=True)
        config = {}
    try:
        _log_launch_diagnostics(config)
    except Exception as e:
        print(f"[diag] diag block FAILED: {e}", flush=True)
    # 1) audio probe: start engine 6s, report what it hears
    print("[diag] audio probe: starting engine for 6s...", flush=True)
    try:
        eng = AudioEngine(
            mic_index=config.get("selected_mic_index"),
            loopback_name=str(config.get("selected_loopback_name", "")),
            vad_threshold_db=int(config.get("vad_threshold_db", 300)),
            mic_enabled=False,
            audio_mode="system",
        )
        eng.start()
        import time as _time
        _time.sleep(6.0)
        print(f"[diag] audio status: {eng.get_status()}", flush=True)
        print(f"[diag] audio stats: {eng.get_audio_stats()}", flush=True)
        eng.stop()
        print("[diag] audio engine stopped", flush=True)
    except Exception as e:
        print(f"[diag] audio probe FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    # 2) live hello round-trip
    print("[diag] live probe: connecting + hello (up to 60s)...", flush=True)
    print("[diag] NOTE: play some computer audio during this test if you can", flush=True)
    try:
        ok = asyncio.run(_diag_live_async(config))
    except Exception as e:
        print(f"[diag] live probe FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return 1
    if ok:
        print("[diag] RESULT: PASS - model text replies work", flush=True)
        return 0
    print("[diag] RESULT: FAIL - no model reply (paste ALL output above for analysis)", flush=True)
    return 1


def main(argv=None) -> int:
    import argparse as _ap
    try:
        _p = _ap.ArgumentParser(description="InterviewCopilot")
        _p.add_argument("--diag", action="store_true",
                        help="headless diagnostics (no GUI): audio probe + live hello test")
        _p.add_argument("--auto", action="store_true",
                        help="skip the settings window, launch straight into the HUD with saved config")
        _args = _p.parse_args(argv)
    except Exception:
        _args = None
    if _args is not None and getattr(_args, "diag", False):
        return run_diag()
    _auto = bool(_args is not None and getattr(_args, "auto", False))

    print("[stage] loading config...", flush=True)
    from PySide6.QtWidgets import QApplication, QMessageBox, QDialog

    app = QApplication.instance() or QApplication(sys.argv)
    try:
        config = load_config()
    except Exception as e:
        log.error("load_config failed: %s", e)
        config = {}
    if _auto:
        print("[stage] --auto: skipping settings window, launching HUD...", flush=True)
        if not str(config.get("api_key", "")).strip():
            print("[stage] WARNING: API key is empty - Live will fail; set it via settings later", flush=True)
        return run_hud_session(config)
    print("[stage] opening settings dialog...", flush=True)

    try:
        print("[stage] importing settings_dialog...", flush=True)
        from settings_dialog import SettingsDialog
        print("[stage] building dialog...", flush=True)
        dlg = SettingsDialog(config)
        print("[stage] dialog ready - click 'Save && Launch' in the Setup window (center of main screen).", flush=True)
        print("[stage] TIP: run 'py main.py --auto' to skip this window next time.", flush=True)
        # capture emitted dict
        emitted: list[dict] = []
        dlg.launch_requested.connect(emitted.append)
        try:
            accepted_val = int(QDialog.DialogCode.Accepted)  # PySide6
        except Exception:
            try:
                accepted_val = int(QDialog.Accepted)  # Qt5/PyQt compat
            except Exception:
                accepted_val = 1
        if dlg.exec() != accepted_val or not emitted:
            print("[stage] dialog closed without launch", flush=True)
            return 0
        print("[stage] launched - starting HUD session...", flush=True)
        config = emitted[0]
    except Exception:
        traceback.print_exc()
        try:
            QMessageBox.critical(None, "InterviewCopilot", f"Settings dialog failed:\n{traceback.format_exc(limit=3)}")
        except Exception:
            pass
        return 1

    # HUD needs a fresh event loop state; reuse same QApplication
    return run_hud_session(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
