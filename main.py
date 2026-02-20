#!/usr/bin/env python3
"""
Realtime microphone transcription on macOS using OpenAI Realtime WebSocket API.

Features:
- 终端模式：保持原有行为
- GUI 模式（默认）：通过 Tkinter 提供开始/停止和实时字幕显示
"""

import argparse
import base64
import codecs
import json
import os
import queue
import urllib.request
import signal
import time
import threading
import sys
from typing import Callable, Optional

import websocket  # websocket-client
import pyaudio


# -----------------------------
# User-tunable audio parameters
# -----------------------------
SAMPLE_RATE = 24000  # Hz; matches common Realtime audio/pcm examples
CHANNELS = 1  # mono
SAMPLE_FORMAT = pyaudio.paInt16  # 16-bit PCM
FRAMES_PER_BUFFER = 1024  # smaller -> lower latency, more overhead


# -----------------------------
# OpenAI Realtime configuration
# -----------------------------
REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"

# You can switch between: "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"

# Optional: set language (ISO-639-1, e.g., "en", "zh") and prompt for domain terms.
TRANSCRIPTION_LANGUAGE = ""  # e.g., "en"
TRANSCRIPTION_PROMPT = (
    ""  # e.g., "The transcript is about computational neuroscience and iEEG."
)
_LANGUAGE_OPTIONS = (
    "自动检测",
    "en",
    "zh",
    "ja",
    "ko",
    "fr",
    "de",
    "es",
    "ru",
    "it",
    "pt",
    "ar",
    "hi",
)
_DEFAULT_TRANSLATION_LANGUAGE_OPTIONS = (
    "中文 (zh)",
    "English (en)",
    "日文 (ja)",
    "韩文 (ko)",
    "法语 (fr)",
    "德语 (de)",
    "西班牙语 (es)",
    "俄语 (ru)",
    "阿拉伯语 (ar)",
    "葡萄牙语 (pt)",
)
_DEFAULT_TRANSLATION_LANGUAGE_MAP = {
    "中文 (zh)": "zh",
    "English (en)": "en",
    "日文 (ja)": "ja",
    "韩文 (ko)": "ko",
    "法语 (fr)": "fr",
    "德语 (de)": "de",
    "西班牙语 (es)": "es",
    "俄语 (ru)": "ru",
    "阿拉伯语 (ar)": "ar",
    "葡萄牙语 (pt)": "pt",
}
_OPENAI_TRANSLATE_URL = "https://api.openai.com/v1/chat/completions"
_TRANSLATION_MODEL = "gpt-4o-mini"

_XDG_CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config"))
_CONFIG_DIR = os.path.join(_XDG_CONFIG_HOME, "realtime-stt-gui")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "config.json")
_CONFIG_KEY = "api_key"


def _ensure_idna_codec() -> None:
    """Ensure the stdlib idna codec exists in packaged runtimes."""
    try:
        codecs.lookup("idna")
    except LookupError:
        try:
            import encodings.idna  # noqa: F401
            codecs.lookup("idna")
        except Exception:
            return


def _load_saved_api_key() -> str:
    """Load saved OpenAI API key from local config."""
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return ""
    except json.JSONDecodeError:
        return ""
    except OSError:
        return ""

    raw_key = payload.get(_CONFIG_KEY) if isinstance(payload, dict) else ""
    if not isinstance(raw_key, str):
        return ""
    return raw_key.strip()


def _normalize_api_key(value: str) -> str:
    cleaned = (value or "").strip().strip('"').strip("'")
    return "".join(ch for ch in cleaned if not ch.isspace())


def _api_key_preview(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "<empty>"
    if len(value) <= 12:
        return f"{value[:3]}...{value[-2:]}" if len(value) > 5 else "********"
    return f"{value[:6]}...{value[-4:]}"


def _api_key_len(value: str) -> int:
    return len((value or "").strip())


def _api_key_source(
    api_key: str, saved_key: str, argv_key_explicit: bool = False
) -> str:
    normalized = _normalize_api_key(api_key)
    env_key = _normalize_api_key(os.environ.get("OPENAI_API_KEY", ""))
    if argv_key_explicit and normalized == _normalize_api_key(_extract_cli_api_key_arg()):
        return "CLI参数"
    if normalized == env_key and env_key:
        return "环境变量"
    if normalized == _normalize_api_key(saved_key):
        return "本地保存"
    if normalized:
        return "手工输入"
    return "未知"


def _extract_cli_api_key_arg() -> str:
    for idx, arg in enumerate(sys.argv):
        if arg == "--api-key" and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith("--api-key="):
            return arg.split("=", 1)[1]
    return ""


def _save_api_key(api_key: str) -> None:
    """Persist API key to local config with restrictive permissions."""
    api_key = (api_key or "").strip()
    if not api_key:
        return

    os.makedirs(_CONFIG_DIR, exist_ok=True)
    try:
        os.chmod(_CONFIG_DIR, 0o700)
    except OSError:
        pass

    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({_CONFIG_KEY: api_key}, f)

    try:
        os.chmod(_CONFIG_FILE, 0o600)
    except OSError:
        pass


def _clear_saved_api_key() -> bool:
    """Delete saved API key from local config. Return True when removed."""
    try:
        os.remove(_CONFIG_FILE)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def _translate_text(api_key: str, source_text: str, target_language: str) -> str:
    """Translate source text to target language using Chat Completions."""
    source_text = (source_text or "").strip()
    target_language = (target_language or "").strip()
    if not api_key or not source_text or not target_language:
        return ""

    prompt = (
        "你是一个准确、直接的翻译助手。"
        f"请将以下语音识别文本翻译成{target_language}，"
        "只返回翻译结果，不要添加注释或说明。"
    )
    payload = {
        "model": _TRANSLATION_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": source_text},
        ],
        "temperature": 0.1,
    }

    request_obj = urllib.request.Request(
        _OPENAI_TRANSLATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request_obj, timeout=30) as response:
        response_data = json.loads(response.read().decode("utf-8"))

    choices = response_data.get("choices")
    if not choices:
        raise RuntimeError("无翻译结果")
    message = choices[0].get("message", {})
    translated = message.get("content", "")
    return str(translated or "").strip()

EventHandler = Callable[[str, str], None]


class RealtimeTranscriber:
    """
    A minimal Realtime transcription client.

    Notes:
    - use transcription_session.update to configure the session
    - continuously send input_audio_buffer.append events with base64 PCM16
    - listen for delta/completed transcript events
    """

    def __init__(
        self,
        api_key: str,
        model: str = TRANSCRIPTION_MODEL,
        language: str = TRANSCRIPTION_LANGUAGE,
        prompt: str = TRANSCRIPTION_PROMPT,
        on_event: Optional[EventHandler] = None,
        debug: bool = False,
    ):
        self.api_key = api_key
        self.model = model
        self.language = language
        self.prompt = prompt
        self.on_event = on_event
        self.debug = debug

        self.ws: Optional[websocket.WebSocketApp] = None
        self._audio = pyaudio.PyAudio()
        self._stream = None

        self._stop_event = threading.Event()
        self._sender_thread: Optional[threading.Thread] = None
        self._thread: Optional[threading.Thread] = None
        self._in_delta_line = False
        self._seen_final_keys: set[str] = set()
        self._last_final_signature: Optional[str] = None
        self._last_final_time: float = 0.0

    def start(self) -> None:
        """
        Start the WebSocket connection and audio streaming in background.
        """
        if self._thread and self._thread.is_alive():
            return

        self._seen_final_keys.clear()
        self._last_final_signature = None
        self._last_final_time = 0.0
        self._stop_event.clear()
        self._sender_thread = None
        self._stream = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait(self) -> None:
        """Block until websocket thread exits (terminal mode helper)."""
        if self._thread:
            self._thread.join()

    def is_running(self) -> bool:
        """Whether the websocket thread is active."""
        return bool(self._thread and self._thread.is_alive())

    def stop(self) -> None:
        """
        Signal threads to stop and close resources.
        """
        self._stop_event.set()

        try:
            if self._sender_thread and self._sender_thread.is_alive():
                self._sender_thread.join(timeout=2)
        except Exception:
            pass

        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
        except Exception:
            pass

        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass

    def _run(self) -> None:
        """Background websocket loop."""
        headers = [
            f"Authorization: Bearer {self.api_key}",
            # Some environments still expect this header.
            "OpenAI-Beta: realtime=v1",
        ]

        self._emit("log", "Connecting to OpenAI Realtime...")
        try:
            self.ws = websocket.WebSocketApp(
                REALTIME_URL,
                header=headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self.ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as exc:
            self._emit("error", f"{type(exc).__name__}: {exc}")
        finally:
            self.ws = None

    def _emit(self, event_type: str, payload: str = "") -> None:
        """Forward events to callback or print in terminal."""
        if self.on_event is not None:
            self.on_event(event_type, payload)
            return

        if event_type == "delta":
            print(payload, end="", flush=True)
            self._in_delta_line = True
        elif event_type == "final":
            if self._in_delta_line:
                print(f"\r\033[K{payload}", flush=True)
            elif payload:
                print(payload, flush=True)
            else:
                print("", flush=True)
            self._in_delta_line = False
        elif event_type == "speech_started":
            pass
        elif event_type == "speech_stopped":
            pass
        elif event_type == "closed":
            # print(f"[closed] {payload}", flush=True)
            pass
        elif event_type == "error":
            print(f"[error] {payload}", flush=True)
        elif event_type == "log":
            if payload:
                print(payload, flush=True)
        elif event_type == "raw" and self.debug:
            print(f"[raw] {payload}", flush=True)

    # -----------------------------
    # WebSocket callbacks
    # -----------------------------
    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        """
        Configure transcription session and start microphone streaming.
        """
        transcription = {
            "model": self.model,
            "prompt": self.prompt,
        }
        if self.language:
            transcription["language"] = self.language

        session_update = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": transcription,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "input_audio_noise_reduction": {
                    "type": "near_field",
                },
            },
        }

        ws.send(json.dumps(session_update))

        # Start microphone capture and sender thread after session config.
        try:
            self._open_mic_stream()
        except Exception as exc:
            self._emit(
                "error", f"Failed to open microphone: {type(exc).__name__}: {exc}"
            )
            ws.close()
            return

        self._sender_thread = threading.Thread(
            target=self._send_audio_loop, daemon=True
        )
        self._sender_thread.start()

        self._emit("log", "Connected. Start speaking...")

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """Handle server events and forward transcript updates."""
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            return

        event_type = event.get("type", "")

        if event_type == "input_audio_buffer.speech_started":
            return
        if event_type == "input_audio_buffer.speech_stopped":
            return

        # Common and fallback transcription events.
        delta = event.get("delta")
        transcript = event.get("transcript")
        event_id = event.get("event_id")
        item_id = event.get("item_id")
        nested = event.get("item")
        if isinstance(nested, dict):
            if delta is None or delta == "":
                delta = nested.get("delta", delta)
            if transcript is None or transcript == "":
                transcript = nested.get("transcript", transcript)
            content = nested.get("content")
            if isinstance(content, list) and (not transcript):
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    ):
                        transcript = part.get("text", "")
                        if transcript:
                            break
        if transcript is None or transcript == "":
            transcript = event.get("text", transcript)
        final_signature = None
        if event_id:
            final_signature = f"event:{event_id}"
        elif item_id:
            final_signature = f"item:{item_id}:{transcript or ''}".strip()
        if (
            event_type == "conversation.item.input_audio_transcription.delta"
            or (isinstance(delta, str) and event_type.endswith(".transcription.delta"))
            or (isinstance(delta, str) and event_type.endswith(".delta") and delta)
        ):
            if delta:
                self._emit("delta", delta)
            return

        if (
            event_type == "conversation.item.input_audio_transcription.completed"
            or (
                isinstance(transcript, str)
                and event_type.endswith(".transcription.completed")
            )
            or (
                isinstance(transcript, str)
                and event_type.endswith(".completed")
                and transcript
            )
        ):
            if transcript:
                normalized_transcript = transcript.strip()
                if final_signature is None:
                    final_signature = f"text:{normalized_transcript}"
                    now = time.time()
                    if (
                        self._last_final_signature == final_signature
                        and now - self._last_final_time < 2.0
                    ):
                        return
                    self._last_final_signature = final_signature
                    self._last_final_time = now
                if final_signature is not None and final_signature in self._seen_final_keys:
                    return
                if final_signature is not None:
                    self._seen_final_keys.add(final_signature)
                self._emit("final", transcript)
            return

        if event_type == "error" or event_type.startswith("error."):
            error_payload = event.get("error")
            if isinstance(error_payload, dict):
                message = (
                    error_payload.get("message")
                    or error_payload.get("code")
                    or str(error_payload)
                )
            else:
                message = str(error_payload or "")
            if not message:
                message = str(event)
            self._emit("error", message)
            return

        if self.debug:
            self._emit("raw", json.dumps(event, ensure_ascii=False))

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """Handle WebSocket errors."""
        self._emit("error", f"{type(error).__name__}: {error}")

    def _on_close(self, ws: websocket.WebSocketApp, status_code, msg) -> None:
        """Handle WebSocket close."""
        self._emit("closed", f"code={status_code} msg={msg}")

    # -----------------------------
    # Audio capture + sending
    # -----------------------------
    def _open_mic_stream(self) -> None:
        """
        Open the default microphone input stream.
        """
        self._stream = self._audio.open(
            format=SAMPLE_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )

    def _send_audio_loop(self) -> None:
        """
        Read PCM frames and append them to OpenAI input audio buffer.
        """
        assert self.ws is not None
        if self._stream is None:
            self._emit("error", "Microphone stream unavailable.")
            return

        while not self._stop_event.is_set():
            try:
                pcm_bytes = self._stream.read(
                    FRAMES_PER_BUFFER, exception_on_overflow=False
                )
            except Exception:
                # If device hiccups, retry shortly.
                threading.Event().wait(0.01)
                continue

            audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
            payload = {"type": "input_audio_buffer.append", "audio": audio_b64}

            try:
                self.ws.send(json.dumps(payload))
            except Exception:
                break


def _run_cli(args: argparse.Namespace) -> None:
    if args.clear_saved_key:
        if _clear_saved_api_key():
            print("[info] 已清除本地保存的 API Key")
        else:
            print("[info] 未找到本地保存的 API Key")
        return

    saved_key = _load_saved_api_key()
    raw_provided_api_key = (args.api_key or "").strip()
    provided_api_key = _normalize_api_key(raw_provided_api_key)
    api_key = provided_api_key or saved_key
    api_key = _normalize_api_key(api_key)
    if raw_provided_api_key and _normalize_api_key(raw_provided_api_key) != raw_provided_api_key:
        print("[warning] API Key 中包含空白字符，已自动去除。")
    cli_api_key_explicit = (
        "--api-key" in sys.argv[1:] or any(
            arg.startswith("--api-key=") for arg in sys.argv[1:]
        )
    )
    key_source = _api_key_source(api_key, saved_key, argv_key_explicit=cli_api_key_explicit)
    if provided_api_key and raw_provided_api_key and provided_api_key != raw_provided_api_key:
        print("[warning] 环境变量/参数中的 API Key 已清理异常字符。")
    if not provided_api_key and api_key:
        print("[info] 已读取本地保存的 API Key")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment or --api-key.")

    if args.save_key:
        _save_api_key(api_key)
    print(
        f"[info] 使用 API Key: {_api_key_preview(api_key)} (len={_api_key_len(api_key)}, 来源:{key_source})"
    )

    translate_language = (args.translate_to or "en").strip()
    running_ui_state = {"in_delta_line": False}
    print_lock = threading.Lock()
    translate_threads: list[threading.Thread] = []

    def _on_translation_done(source_text: str) -> None:
        try:
            translated = _translate_text(
                api_key=api_key,
                source_text=source_text,
                target_language=translate_language,
            )
            if translated:
                with print_lock:
                    print(f"[translation] {translated}", flush=True)
        except Exception as exc:
            with print_lock:
                print(f"[translation error] {type(exc).__name__}: {exc}", flush=True)

    def on_event(event_type: str, payload: str) -> None:
        if event_type == "delta":
            if not payload:
                return
            with print_lock:
                print(payload, end="", flush=True)
                running_ui_state["in_delta_line"] = True
            return

        if event_type == "final":
            with print_lock:
                if running_ui_state["in_delta_line"]:
                    print(f"\r\033[K{payload}", flush=True)
                else:
                    print(payload, flush=True)
                running_ui_state["in_delta_line"] = False

            if args.translate:
                if payload:
                    t = threading.Thread(
                        target=_on_translation_done,
                        args=(payload,),
                        daemon=True,
                    )
                    translate_threads.append(t)
                    t.start()
            return

        if event_type == "log":
            if payload:
                with print_lock:
                    print(f"[info] {payload}", flush=True)
            return

        if event_type == "raw":
            if args.debug:
                with print_lock:
                    print(f"[raw] {payload}", flush=True)
            return

        if event_type == "error":
            with print_lock:
                print(f"[error] {payload}", flush=True)
            return

        if event_type == "closed":
            with print_lock:
                print(f"[closed] {payload}", flush=True)
            return

    transcriber = RealtimeTranscriber(
        api_key=api_key,
        model=args.model,
        language=args.language,
        prompt=args.prompt,
        debug=args.debug,
        on_event=on_event,
    )

    def _handle_sigint(_sig, _frame):
        transcriber.stop()
        # Allow in-flight translation threads to flush their output.
        for worker in list(translate_threads):
            worker.join(timeout=2)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    if args.translate and not translate_language:
        raise SystemExit("--translate requires --translate-to (e.g. --translate-to en).")

    transcriber.start()
    transcriber.wait()

    for worker in list(translate_threads):
        worker.join(timeout=3)


def _run_gui(args: argparse.Namespace) -> None:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk

    class TranscriptionApp(tk.Tk):
        def __init__(self, args: argparse.Namespace):
            super().__init__()
            self.title("Realtime STT")
            self.geometry("920x680")
            self.minsize(780, 560)
            initial_api_key = _normalize_api_key(
                args.api_key.strip() or _load_saved_api_key()
            )

            self._events = queue.Queue()
            self._transcriber: Optional[RealtimeTranscriber] = None
            self._delta_active = False
            self._current_item_id: Optional[str] = None
            self._current_source_text: str = ""

            self._translation_seq = 0
            self._translation_pending: dict[str, int] = {}

            self._api_key_var = tk.StringVar(value=initial_api_key)
            self._remember_api_key_var = tk.BooleanVar(value=bool(initial_api_key))
            self._model_var = tk.StringVar(value=args.model)
            self._language_var = tk.StringVar(
                value="自动检测" if not args.language else args.language
            )
            self._prompt_var = tk.StringVar(value=args.prompt)
            self._debug_var = tk.BooleanVar(value=args.debug)
            self._auto_translate_var = tk.BooleanVar(value=False)
            self._translate_language_var = tk.StringVar(
                value=_DEFAULT_TRANSLATION_LANGUAGE_OPTIONS[0]
            )

            self._build_ui()
            self._poll_events()
            self.protocol("WM_DELETE_WINDOW", self._on_close)

        def _build_ui(self) -> None:
            outer = ttk.Frame(self, padding=12)
            outer.pack(fill=tk.BOTH, expand=True)

            form = ttk.LabelFrame(outer, text="配置")
            form.pack(fill=tk.X, pady=(0, 10))

            ttk.Label(form, text="OpenAI API Key").grid(row=0, column=0, sticky="w")
            key_frame = ttk.Frame(form)
            key_frame.grid(row=0, column=1, sticky="ew", padx=(8, 0))
            self._api_key_entry = ttk.Entry(
                key_frame, textvariable=self._api_key_var, width=58, show="*"
            )
            self._api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._api_key_entry.bind("<FocusIn>", self._select_api_key_all)
            ttk.Checkbutton(
                key_frame, text="记住密钥", variable=self._remember_api_key_var
            ).pack(side=tk.RIGHT, padx=(6, 0))
            ttk.Button(key_frame, text="清除已保存", command=self._clear_saved_api_key_ui).pack(
                side=tk.RIGHT
            )

            ttk.Label(form, text="Model").grid(row=1, column=0, sticky="w", pady=(8, 0))
            ttk.Combobox(
                form,
                textvariable=self._model_var,
                values=("gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"),
                state="readonly",
                width=30,
            ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

            ttk.Label(form, text="Language").grid(
                row=2, column=0, sticky="w", pady=(8, 0)
            )
            ttk.Combobox(
                form,
                textvariable=self._language_var,
                values=_LANGUAGE_OPTIONS,
                state="readonly",
                width=24,
            ).grid(
                row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
            )

            ttk.Label(form, text="Prompt").grid(
                row=3, column=0, sticky="w", pady=(8, 0)
            )
            ttk.Entry(form, textvariable=self._prompt_var, width=70).grid(
                row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
            )

            ttk.Label(form, text="译文语言").grid(
                row=4, column=0, sticky="w", pady=(8, 0)
            )
            self._translate_language_combo = ttk.Combobox(
                form,
                textvariable=self._translate_language_var,
                values=_DEFAULT_TRANSLATION_LANGUAGE_OPTIONS,
                state="disabled",
                width=24,
            )
            self._translate_language_combo.grid(
                row=4, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
            )

            actions = ttk.Frame(form)
            actions.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))
            ttk.Checkbutton(
                actions,
                text="自动翻译",
                variable=self._auto_translate_var,
                command=self._on_toggle_translate,
            ).pack(side=tk.LEFT)
            ttk.Checkbutton(
                actions,
                text="调试日志",
                variable=self._debug_var,
            ).pack(side=tk.RIGHT)
            self.start_button = ttk.Button(actions, text="开始", command=self._start)
            self.start_button.pack(side=tk.LEFT, padx=(10, 6))
            self.stop_button = ttk.Button(
                actions, text="停止", command=self._stop, state=tk.DISABLED
            )
            self.stop_button.pack(side=tk.LEFT)

            self.status_var = tk.StringVar(value="Ready")
            ttk.Label(outer, textvariable=self.status_var).pack(anchor="w")

            out_frame = ttk.LabelFrame(outer, text="字幕（左：原文 | 右：翻译）")
            out_frame.pack(fill=tk.BOTH, expand=True)
            transcript_container = ttk.Frame(out_frame)
            transcript_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

            columns = ("source", "translation")
            self.transcript = ttk.Treeview(
                transcript_container,
                columns=columns,
                show="headings",
                selectmode="none",
            )
            self.transcript.heading("source", text="原文")
            self.transcript.heading("translation", text="翻译")
            self.transcript.column("source", width=430, stretch=True, anchor=tk.W)
            self.transcript.column("translation", width=430, stretch=True, anchor=tk.W)
            self.transcript.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            v_scroll = ttk.Scrollbar(
                transcript_container, orient="vertical", command=self.transcript.yview
            )
            v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.transcript.configure(yscrollcommand=v_scroll.set)

            h_scroll = ttk.Scrollbar(
                transcript_container, orient="horizontal", command=self.transcript.xview
            )
            h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
            self.transcript.configure(xscrollcommand=h_scroll.set)

            log_frame = ttk.LabelFrame(outer, text="日志")
            log_frame.pack(fill=tk.BOTH, expand=False, pady=(8, 0))
            self.log_output = scrolledtext.ScrolledText(
                log_frame, wrap=tk.WORD, state=tk.DISABLED, height=7
            )
            self.log_output.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

            form.columnconfigure(1, weight=1)
            self._on_toggle_translate()

        @staticmethod
        def _sanitize_transcript_text(text: str) -> str:
            return text.replace("\r", " ").replace("\n", " ")

        def _on_toggle_translate(self) -> None:
            state = "readonly" if self._auto_translate_var.get() else tk.DISABLED
            self._translate_language_combo.config(state=state)

        def _append_log(self, text: str) -> None:
            if not text:
                return
            self.log_output.configure(state=tk.NORMAL)
            self.log_output.insert(tk.END, text)
            self.log_output.see(tk.END)
            self.log_output.configure(state=tk.DISABLED)

        def _clear_saved_api_key_ui(self) -> None:
            current_api_key = self._api_key_var.get().strip()
            saved_api_key = _load_saved_api_key()
            if _clear_saved_api_key():
                self._append_log("[info] 已清除本地保存的 API Key。\n")
                if current_api_key == saved_api_key:
                    self._api_key_var.set("")
                self._remember_api_key_var.set(False)
            else:
                self._append_log("[info] 本地尚未保存 API Key。\n")

        def _translation_target_language(self) -> str:
            return _DEFAULT_TRANSLATION_LANGUAGE_MAP.get(
                self._translate_language_var.get().strip(),
                "en",
            )

        def _emit_row(
            self, item_id: str, source: Optional[str], translation: Optional[str]
        ) -> None:
            if not self.transcript.exists(item_id):
                return
            current = self.transcript.item(item_id, "values")
            if not isinstance(current, tuple):
                current = ()
            current_source = (
                source if source is not None else (current[0] if len(current) > 0 else "")
            )
            current_translation = (
                translation
                if translation is not None
                else (current[1] if len(current) > 1 else "")
            )
            self.transcript.item(item_id, values=(current_source, current_translation))
            self.transcript.yview_moveto(1)

        def _append_delta(self, text: str) -> None:
            if not text:
                return
            chunk = self._sanitize_transcript_text(text)

            if not self._delta_active or self._current_item_id is None:
                self._current_source_text = chunk
                self._current_item_id = self.transcript.insert(
                    "", tk.END, values=(self._current_source_text, "")
                )
                self._delta_active = True
                return

            self._current_source_text += chunk
            self._emit_row(self._current_item_id, self._current_source_text, None)

        def _on_final(self, text: str) -> None:
            final_text = self._sanitize_transcript_text(text)
            if not final_text:
                self._delta_active = False
                self._current_item_id = None
                return

            if not self._delta_active or self._current_item_id is None:
                self._current_item_id = self.transcript.insert(
                    "", tk.END, values=(final_text, "")
                )
            else:
                self._emit_row(self._current_item_id, final_text, None)

            self._current_source_text = final_text
            self._delta_active = False

            if self._auto_translate_var.get():
                self._set_translation_pending(self._current_item_id, final_text)

            self.transcript.see(self._current_item_id)

        def _set_translation_pending(self, item_id: str, text: str) -> None:
            if not item_id:
                return
            language = self._translation_target_language()
            self._emit_row(item_id, None, "翻译中...")
            self._translation_seq += 1
            seq = self._translation_seq
            self._translation_pending[item_id] = seq

            if not text:
                self._emit_row(item_id, None, "")
                self._translation_pending.pop(item_id, None)
                return

            if not language:
                self._emit_row(item_id, None, "")
                self._translation_pending.pop(item_id, None)
                return

            threading.Thread(
                target=self._translate_worker,
                args=(item_id, seq, text, language),
                daemon=True,
            ).start()

        def _translate_worker(
            self, item_id: str, seq: int, source_text: str, target_language: str
        ) -> None:
            try:
                translated = self._translate(source_text, target_language)
            except Exception as exc:  # pragma: no cover - network dependent
                self._events.put(
                    (
                        "translation",
                        json.dumps(
                            {
                                "item_id": item_id,
                                "seq": seq,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        ),
                    )
                )
                return

            self._events.put(
                (
                    "translation",
                    json.dumps(
                        {
                            "item_id": item_id,
                            "seq": seq,
                            "text": translated,
                        }
                    ),
                )
            )

        def _translate(self, source_text: str, target_language: str) -> str:
            api_key = _normalize_api_key(self._api_key_var.get())
            translated = _translate_text(
                api_key=api_key,
                source_text=source_text,
                target_language=target_language,
            )
            return self._sanitize_transcript_text(translated)

        def _select_api_key_all(self, _event: object) -> None:
            self._api_key_entry.selection_range(0, tk.END)
            self._api_key_entry.icursor(tk.END)

        def _start(self) -> None:
            raw_api_key = self._api_key_var.get() or ""
            api_key = raw_api_key.strip()
            normalized = _normalize_api_key(api_key)
            if normalized != api_key:
                self._append_log("[warning] API Key 包含空白字符，已自动去除。\n")
            api_key = normalized
            if not api_key:
                messagebox.showerror("Error", "请填写 API Key")
                return
            if self._remember_api_key_var.get():
                try:
                    _save_api_key(api_key)
                except Exception as exc:  # pragma: no cover
                    self._append_log(f"[warning] API Key 保存失败: {type(exc).__name__}: {exc}\n")

            if self._transcriber is not None and self._transcriber.is_running():
                return

            if self._auto_translate_var.get():
                if not self._translate_language_combo.get().strip():
                    messagebox.showerror("Error", "请先选择翻译目标语言")
                    return

            language = self._language_var.get().strip()
            if language == "自动检测":
                language = ""

            self._delta_active = False
            self._current_item_id = None
            self._current_source_text = ""
            self._translation_seq = 0
            self._translation_pending = {}
            self._events = queue.Queue()
            self._set_running_state(True)
            self.status_var.set("Starting...")

            self.transcript.delete(*self.transcript.get_children())
            self.log_output.configure(state=tk.NORMAL)
            self.log_output.delete("1.0", tk.END)
            self.log_output.configure(state=tk.DISABLED)
            self._append_log("[info] 准备连接...\n")
            key_source = _api_key_source(api_key, _load_saved_api_key(), False)
            self._append_log(
                f"[info] 使用 API Key: {_api_key_preview(api_key)} (len={_api_key_len(api_key)}, 来源:{key_source})\n"
            )

            self._transcriber = RealtimeTranscriber(
                api_key=api_key,
                model=self._model_var.get().strip() or TRANSCRIPTION_MODEL,
                language=language,
                prompt=self._prompt_var.get().strip(),
                on_event=lambda event_type, payload: self._events.put(
                    (event_type, payload)
                ),
                debug=self._debug_var.get(),
            )
            self._transcriber.start()

        def _stop(self) -> None:
            if self._transcriber is not None:
                self._transcriber.stop()
            self._set_running_state(False)
            self.status_var.set("Stopped")

        def _on_close(self) -> None:
            if self._transcriber is not None:
                self._transcriber.stop()
            self.destroy()

        def _set_running_state(self, running: bool) -> None:
            self.start_button.config(state=tk.DISABLED if running else tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL if running else tk.DISABLED)
            if running:
                self._translate_language_combo.config(state=tk.DISABLED)
                return
            self._on_toggle_translate()

        def _handle_translation_event(self, payload: str) -> None:
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                self._append_log(f"[translation] {payload}\n")
                return

            item_id = data.get("item_id")
            seq = data.get("seq")
            if not item_id or not isinstance(seq, int):
                self._append_log(f"[translation] malformed: {payload}\n")
                return

            if self._translation_pending.get(item_id) != seq:
                return

            if data.get("error"):
                self._append_log(f"[translation-error] {data.get('error')}\n")
                self._emit_row(item_id, None, "[翻译失败]")
                self._translation_pending.pop(item_id, None)
                return

            translated = (data.get("text") or "").strip()
            self._emit_row(item_id, None, translated)
            self._translation_pending.pop(item_id, None)

        def _poll_events(self) -> None:
            while True:
                try:
                    event_type, payload = self._events.get_nowait()
                except queue.Empty:
                    break

                if event_type == "delta":
                    self._append_delta(payload)
                elif event_type == "final":
                    self._on_final(payload)
                elif event_type == "translation":
                    self._handle_translation_event(payload)
                elif event_type == "log":
                    self._append_log(f"{payload}\n")
                elif event_type == "raw":
                    self._append_log(f"{payload}\n")
                elif event_type == "error":
                    self._append_log(f"[error] {payload}\n")
                    self.status_var.set("Error")
                    self._set_running_state(False)
                elif event_type == "closed":
                    self._append_log(f"[closed] {payload}\n")
                    self._set_running_state(False)
                    self.status_var.set("Disconnected")
                    self._transcriber = None

            self.after(50, self._poll_events)

    app = TranscriptionApp(args)
    app.mainloop()


def main() -> None:
    _ensure_idna_codec()

    parser = argparse.ArgumentParser(description="Realtime STT")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="运行终端模式，而不是 GUI 模式。",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "").strip(),
        help="OpenAI API Key（默认读取 OPENAI_API_KEY 环境变量）",
    )
    parser.add_argument(
        "--save-key",
        action="store_true",
        help="在终端模式下将 API Key 保存到本地（如未提供则读取已保存）",
    )
    parser.add_argument(
        "--clear-saved-key",
        action="store_true",
        help="清除本地保存的 API Key 并退出",
    )
    parser.add_argument(
        "--model",
        default=TRANSCRIPTION_MODEL,
        help="转写模型",
    )
    parser.add_argument(
        "--language",
        default=TRANSCRIPTION_LANGUAGE,
        help='语言代码，如 "en" 或 "zh"，空值表示自动检测',
    )
    parser.add_argument(
        "--prompt",
        default=TRANSCRIPTION_PROMPT,
        help="可选上下文提示词",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="终端模式下输出最终转写的机器翻译结果",
    )
    parser.add_argument(
        "--translate-to",
        default="en",
        help="翻译目标语言（与 --translate 配合使用），如 en / zh / ja",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印原始 Realtime 事件（用于排障）",
    )
    args = parser.parse_args()

    if args.cli:
        _run_cli(args)
    else:
        _run_gui(args)


if __name__ == "__main__":
    main()
