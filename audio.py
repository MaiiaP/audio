
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import wave
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_TITLE = "Автозаполнение листа приёма из аудио "

WHISPER_PRICE_PER_MIN = 0.006
CHUNK_SECONDS = 300

def build_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _ensure_ffmpeg_exists() -> Optional[str]:
    return shutil.which("ffmpeg")


def split_audio_to_chunks(audio_path: Path, chunk_seconds: int = CHUNK_SECONDS) -> List[Path]:
    ffmpeg_bin = _ensure_ffmpeg_exists()
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg не найден в PATH")

    chunk_dir = Path(tempfile.mkdtemp(prefix="visit_chunks_"))
    output_pattern = chunk_dir / "chunk_%03d.wav"

    # Для mixed-форматов (ogg/webm/mp3/...) режем с перекодированием в WAV PCM.
    # Copy-кодек в wav для некоторых контейнеров дает ffmpeg exit code.
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_pattern),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        details = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(f"Ошибка ffmpeg при нарезке: {details}") from e

    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError(f"Не удалось нарезать аудио на части: {audio_path}")
    return chunks


def run_whisper_transcribe(audio_path: str, language: str = "ru") -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не задан")

    if not os.path.exists(audio_path):
        raise RuntimeError(f"Аудиофайл не найден: {audio_path}")

    duration_sec = None
    try:
        with contextlib.closing(wave.open(audio_path, "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration_sec = frames / float(rate)
    except Exception:
        duration_sec = None

    client = build_openai_client()

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="gpt-4o-transcribe",
            language=language if len(language) == 2 else None,
        )

    text = getattr(response, "text", "")
    if not text or not text.strip():
        raise RuntimeError("Пустой результат транскрибации")

    if duration_sec:
        minutes = duration_sec / 60
        cost = minutes * WHISPER_PRICE_PER_MIN
        print(f"[Whisper] duration={minutes:.2f} min | cost≈${cost:.4f}")
    else:
        print("[Whisper] duration=unknown | cost≈unknown")

    return text.strip()


def transcribe_uploaded_audio(audio_path: Path, language: str = "ru") -> str:
    transcript_parts: List[str] = []
    total_start = time.perf_counter()
    audio_start = time.perf_counter()
    total_audio_sec = 0.0
    chunks = split_audio_to_chunks(audio_path)
    for chunk_idx, chunk_path in enumerate(chunks, start=1):
        try:
            with contextlib.closing(wave.open(str(chunk_path), "r")) as f:
                total_audio_sec += f.getnframes() / float(f.getframerate())
        except Exception:
            pass
        part_text = run_whisper_transcribe(str(chunk_path), language=language)
        transcript_parts.append(f"[chunk {chunk_idx}]\n{part_text}")
    audio_elapsed = time.perf_counter() - audio_start
    print(f"[Timing] Загруженное аудио: {audio_elapsed:.2f} sec | {audio_elapsed/60:.2f} min")

    joined = "\n\n".join(transcript_parts).strip()
    if not joined:
        raise RuntimeError("Пустой общий результат транскрибации")

    total_elapsed = time.perf_counter() - total_start
    print(f"[Timing] Все записи (транскрибация): {total_elapsed:.2f} sec | {total_elapsed/60:.2f} min")
    st.session_state.audio_minutes = round(total_audio_sec / 60.0, 2) if total_audio_sec > 0 else None
    return joined


def call_gpt_role_split(transcript: str) -> str:
    client = build_openai_client()

    system_prompt = """
Ты анализируешь расшифровку медицинского приёма.
Определи, кто говорит: врач или пациент.

Верни ТОЛЬКО текст в следующем формате (строго):

врач:
текст врача

пациент:
текст пациента

врач:
...

Правила:
- ничего не выдумывай
- не добавляй комментариев
- не используй markdown
- если подряд говорит один и тот же человек — не объединяй, сохраняй порядок
"""

    response = client.responses.create(
        model="gpt-5-chat-latest",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )

    text = response.output_text
    if not text:
        raise RuntimeError("Пустой ответ при разметке ролей")

    return text.strip()


def parse_role_blocks(text: str):
    blocks = []
    current_role = None
    buffer = []

    for line in text.splitlines():
        l = line.strip().lower()
        if l == "врач:" or l == "пациент:":
            if current_role and buffer:
                blocks.append({"role": current_role, "text": "\n".join(buffer).strip()})
                buffer = []
            current_role = l.replace(":", "")
        else:
            buffer.append(line)

    if current_role and buffer:
        blocks.append({"role": current_role, "text": "\n".join(buffer).strip()})

    return blocks


def safe_date(x: Any) -> Optional[date]:
    if not x:
        return None
    if isinstance(x, date):
        return x
    s = str(x).strip()
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def set_session_defaults():
    defaults = {
        "audio_path": None,
        "uploaded_signature": None,
        "audio_minutes": None,
        "transcript": "",
        "whisper_transcript": "",
        "llm_json": None,
        "role_marked_text": "",
        "role_blocks": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def inject_pretty_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
          :root { color-scheme: light dark; }

          html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
          [data-testid="stHeader"] { background: transparent; border-bottom: 0; }
          [data-testid="stAppViewContainer"] > .main { background: transparent; }

          .stApp {
            --bg: #f4f7fb;
            --bg-accent-a: rgba(34, 168, 132, 0.18);
            --bg-accent-b: rgba(26, 177, 164, 0.14);
            --surface: rgba(255, 255, 255, 0.56);
            --surface-strong: rgba(255, 255, 255, 0.78);
            --surface-border: rgba(255, 255, 255, 0.55);
            --surface-ink-border: rgba(14, 22, 39, 0.09);
            --text: #10192b;
            --text-muted: rgba(16, 25, 43, 0.72);
            --shadow: 0 24px 64px rgba(16, 25, 43, 0.11);
            --inner-glow: inset 0 1px 0 rgba(255, 255, 255, 0.85);
            --brand-gradient: linear-gradient(135deg, #dceff3 0%, #d5ebf1 50%, #d8efe8 100%);
            --brand-ink: #2f5f73;
            background:
              radial-gradient(900px 560px at 8% 0%, var(--bg-accent-a), transparent 60%),
              radial-gradient(900px 560px at 92% 12%, var(--bg-accent-b), transparent 62%),
              var(--bg);
            color: var(--text);
          }

          @media (prefers-color-scheme: dark) {
            .stApp {
              --bg: #090d14;
              --bg-accent-a: rgba(24, 144, 114, 0.24);
              --bg-accent-b: rgba(22, 154, 139, 0.22);
              --surface: rgba(12, 18, 30, 0.58);
              --surface-strong: rgba(18, 26, 43, 0.72);
              --surface-border: rgba(176, 210, 255, 0.28);
              --surface-ink-border: rgba(143, 175, 220, 0.2);
              --text: #eef3ff;
              --text-muted: rgba(226, 235, 255, 0.78);
              --shadow: 0 26px 72px rgba(0, 0, 0, 0.42);
              --inner-glow: inset 0 1px 0 rgba(255, 255, 255, 0.12);
              --brand-gradient: linear-gradient(135deg, #335b68 0%, #356570 50%, #376f67 100%);
              --brand-ink: #e7f5ff;
            }
          }

          html[data-theme="dark"] .stApp,
          body[data-theme="dark"] .stApp,
          .stApp[data-theme="dark"] {
            --bg: #090d14;
            --bg-accent-a: rgba(24, 144, 114, 0.24);
            --bg-accent-b: rgba(22, 154, 139, 0.22);
            --surface: rgba(12, 18, 30, 0.58);
            --surface-strong: rgba(18, 26, 43, 0.72);
            --surface-border: rgba(176, 210, 255, 0.28);
            --surface-ink-border: rgba(143, 175, 220, 0.2);
            --text: #eef3ff;
            --text-muted: rgba(226, 235, 255, 0.78);
            --shadow: 0 26px 72px rgba(0, 0, 0, 0.42);
            --inner-glow: inset 0 1px 0 rgba(255, 255, 255, 0.12);
            --brand-gradient: linear-gradient(135deg, #335b68 0%, #356570 50%, #376f67 100%);
            --brand-ink: #e7f5ff;
          }

          .block-container {
            max-width: 1360px;
            padding-top: clamp(0.8rem, 2vw, 1.4rem);
            padding-bottom: clamp(1rem, 2.6vw, 2rem);
          }

          [data-testid="stSidebar"] {
            background:
              linear-gradient(165deg, rgba(255, 255, 255, 0.62), rgba(255, 255, 255, 0.34)),
              rgba(246, 250, 255, 0.42);
            border-right: 1px solid color-mix(in srgb, var(--surface-border) 84%, transparent);
            backdrop-filter: blur(22px) saturate(130%);
            -webkit-backdrop-filter: blur(22px) saturate(130%);
          }

          [data-testid="stSidebar"] > div:first-child {
            background: transparent;
          }

          .hero-card {
            position: relative;
            overflow: hidden;
            padding: clamp(0.8rem, 1.5vw, 1rem) clamp(1rem, 1.8vw, 1.25rem);
            min-height: clamp(74px, 8vw, 98px);
            display: flex;
            align-items: center;
            border-radius: 16px;
            border: 1px solid color-mix(in srgb, var(--surface-border) 70%, transparent);
            background:
              linear-gradient(135deg, rgba(255, 255, 255, 0.34), rgba(255, 255, 255, 0.08)),
              radial-gradient(120% 130% at 0% 0%, rgba(0, 143, 190, 0.10), transparent 54%),
              var(--surface);
            box-shadow: 0 3px 10px rgba(16, 25, 43, 0.06), var(--inner-glow);
            margin-bottom: 0.55rem;
          }

          .hero-card::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            border-radius: inherit;
            background: linear-gradient(
              135deg,
              rgba(255, 255, 255, 0.36) 0%,
              rgba(255, 255, 255, 0.08) 36%,
              rgba(255, 255, 255, 0) 100%
            );
          }

          .hero-card::after {
            content: "";
            position: absolute;
            left: 0;
            top: 16%;
            bottom: 16%;
            width: 3px;
            border-radius: 0 6px 6px 0;
            background: linear-gradient(180deg, rgba(63, 156, 181, 0.55), rgba(84, 184, 147, 0.45));
            opacity: 0.75;
          }

          .hero-title {
            margin: 0 0 0 0.45rem;
            font-size: clamp(2.5rem, 2.9vw, 3rem);
            font-weight: 800;
            line-height: 1.12;
            letter-spacing: -0.025em;
            color: var(--text);
          }

          .stMarkdown h5, [data-testid="stMarkdownContainer"] h5 {
            margin-top: 0.95rem;
            margin-bottom: 0.5rem;
            letter-spacing: -0.01em;
          }

          .stMarkdown h3, [data-testid="stMarkdownContainer"] h3 {
            margin-top: 0.2rem;
            margin-bottom: 0.7rem;
            letter-spacing: -0.02em;
          }

          hr { margin: 0.9rem 0 0.95rem 0; }

          .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(109, 160, 179, 0.45) !important;
            background: var(--brand-gradient);
            color: var(--brand-ink) !important;
            font-weight: 700;
            letter-spacing: 0.01em;
            box-shadow: 0 6px 14px rgba(66, 122, 146, 0.16);
            min-height: 2.65rem;
            transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
          }

          .stButton > button:hover {
            color: var(--brand-ink) !important;
            border: 1px solid rgba(109, 160, 179, 0.58) !important;
            box-shadow: 0 8px 18px rgba(66, 122, 146, 0.2);
            filter: brightness(1.01);
            transform: translateY(-1px);
          }

          .stButton > button:active {
            color: var(--brand-ink) !important;
            border: 1px solid rgba(109, 160, 179, 0.6) !important;
            transform: translateY(0);
            box-shadow: 0 5px 12px rgba(66, 122, 146, 0.18);
          }

          .stButton > button:focus,
          .stButton > button:focus-visible {
            color: var(--brand-ink) !important;
            border: 1px solid rgba(109, 160, 179, 0.62) !important;
            outline: none !important;
            box-shadow:
              0 0 0 3px rgba(125, 173, 189, 0.24),
              0 6px 14px rgba(66, 122, 146, 0.18) !important;
          }

          .stButton > button p,
          .stButton > button span,
          .stButton > button div {
            color: inherit !important;
          }

          .stTabs [data-baseweb="tab-list"] { gap: 8px; }
          .stTabs [data-baseweb="tab"] {
            border-radius: 12px 12px 0 0;
            border: 1px solid #c8ced6;
            background: linear-gradient(160deg, rgba(255, 255, 255, 0.45), rgba(255, 255, 255, 0.15));
            padding: 8px 14px;
            font-weight: 700;
            color: color-mix(in srgb, var(--text) 88%, #5f6b84);
          }

          .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: var(--brand-ink) !important;
            border-color: #8fc7b8 !important;
            background: linear-gradient(160deg, rgba(255, 255, 255, 0.66), rgba(255, 255, 255, 0.26)) !important;
          }

          .stTabs [data-baseweb="tab-highlight"] {
            background: linear-gradient(90deg, #8bbdcf, #9ccdc2) !important;
            height: 2.5px !important;
            border-radius: 999px;
          }

          .stTextInput > div > div,
          .stTextArea textarea,
          .stNumberInput input,
          .stSelectbox [data-baseweb="select"] > div,
          .stDateInput input {
            border-radius: 10px !important;
            border: 1px solid var(--surface-ink-border) !important;
            background: var(--surface-strong) !important;
            color: var(--text) !important;
          }

          [data-testid="stFileUploaderDropzone"] {
            border-radius: 12px;
            border: 1px dashed var(--surface-ink-border);
            background: linear-gradient(160deg, rgba(255, 255, 255, 0.45), rgba(255, 255, 255, 0.12));
          }

          .stCodeBlock, pre, code {
            border-radius: 12px !important;
          }

          .hint-card {
            border-radius: 12px;
            border: 1px solid rgba(102, 173, 154, 0.42);
            background: linear-gradient(135deg, rgba(214, 240, 231, 0.96), rgba(221, 244, 238, 0.96));
            box-shadow: 0 5px 14px rgba(62, 126, 109, 0.12);
            color: #2f5f52;
            padding: 0.8rem 0.92rem;
            font-weight: 400;
          }

          .ok-card {
            border-radius: 12px;
            border: 1px solid rgba(118, 181, 196, 0.42);
            background: linear-gradient(135deg, rgba(216, 238, 245, 0.96), rgba(224, 243, 247, 0.96));
            box-shadow: 0 5px 14px rgba(59, 112, 130, 0.12);
            color: #2f5f73;
            padding: 0.72rem 0.92rem;
            font-weight: 600;
            margin-bottom: 0.72rem;
          }

          @media (prefers-color-scheme: dark) {
            [data-testid="stSidebar"] {
              background:
                linear-gradient(165deg, rgba(26, 36, 52, 0.74), rgba(15, 22, 35, 0.62)),
                rgba(12, 18, 30, 0.58) !important;
              border-right: 1px solid rgba(124, 160, 187, 0.28) !important;
            }

            .hero-card {
              border: 1px solid rgba(135, 176, 205, 0.24);
              background:
                linear-gradient(135deg, rgba(31, 45, 66, 0.72), rgba(19, 30, 45, 0.64)),
                radial-gradient(120% 130% at 0% 0%, rgba(45, 165, 129, 0.18), transparent 54%),
                var(--surface);
              box-shadow: 0 3px 10px rgba(0, 0, 0, 0.28), var(--inner-glow);
            }

            .hero-card::before {
              background: linear-gradient(
                135deg,
                rgba(67, 98, 124, 0.22) 0%,
                rgba(46, 74, 98, 0.12) 36%,
                rgba(23, 38, 57, 0.04) 100%
              );
            }

            .stTabs [data-baseweb="tab"] {
              border-color: rgba(118, 140, 166, 0.44);
              background: linear-gradient(160deg, rgba(36, 49, 70, 0.7), rgba(24, 36, 54, 0.52));
              color: rgba(226, 235, 255, 0.9);
            }

            .stTabs [data-baseweb="tab"][aria-selected="true"] {
              border-color: rgba(123, 187, 169, 0.72) !important;
              background: linear-gradient(160deg, rgba(42, 63, 83, 0.8), rgba(29, 47, 66, 0.68)) !important;
              color: #dff6ee !important;
            }

            .stTabs [data-baseweb="tab-highlight"] {
              background: linear-gradient(90deg, #66b79b, #75c2b2) !important;
            }

            .stButton > button {
              border-color: rgba(111, 156, 172, 0.62) !important;
              box-shadow: 0 6px 14px rgba(15, 30, 44, 0.34);
            }

            .stButton > button:hover {
              border-color: rgba(134, 184, 201, 0.72) !important;
              box-shadow: 0 8px 18px rgba(15, 30, 44, 0.38);
            }

            .stTextInput > div > div,
            .stTextArea textarea,
            .stNumberInput input,
            .stSelectbox [data-baseweb="select"] > div,
            .stDateInput input {
              border-color: rgba(125, 150, 178, 0.34) !important;
              background: rgba(22, 33, 49, 0.78) !important;
              color: #edf4ff !important;
            }

            [data-testid="stFileUploaderDropzone"] {
              border: 1px dashed rgba(121, 153, 186, 0.36);
              background: linear-gradient(160deg, rgba(34, 48, 69, 0.72), rgba(22, 35, 52, 0.56));
            }

            .hint-card {
              border: 1px solid rgba(99, 165, 145, 0.46);
              background: linear-gradient(135deg, rgba(40, 74, 66, 0.86), rgba(29, 62, 57, 0.84));
              color: #d9f6ea;
              box-shadow: 0 6px 14px rgba(8, 22, 18, 0.34);
            }

            .ok-card {
              border: 1px solid rgba(97, 154, 176, 0.48);
              background: linear-gradient(135deg, rgba(36, 58, 74, 0.88), rgba(31, 52, 66, 0.86));
              color: #d9ecf4;
              box-shadow: 0 6px 14px rgba(8, 18, 24, 0.34);
            }
          }

          html[data-theme="dark"] [data-testid="stSidebar"],
          body[data-theme="dark"] [data-testid="stSidebar"],
          .stApp[data-theme="dark"] [data-testid="stSidebar"] {
            background:
              linear-gradient(165deg, rgba(26, 36, 52, 0.74), rgba(15, 22, 35, 0.62)),
              rgba(12, 18, 30, 0.58) !important;
            border-right: 1px solid rgba(124, 160, 187, 0.28) !important;
          }

          html[data-theme="dark"] .hero-card,
          body[data-theme="dark"] .hero-card,
          .stApp[data-theme="dark"] .hero-card {
            border: 1px solid rgba(135, 176, 205, 0.24);
            background:
              linear-gradient(135deg, rgba(31, 45, 66, 0.72), rgba(19, 30, 45, 0.64)),
              radial-gradient(120% 130% at 0% 0%, rgba(45, 165, 129, 0.18), transparent 54%),
              var(--surface);
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.28), var(--inner-glow);
          }

          html[data-theme="dark"] .hero-card::before,
          body[data-theme="dark"] .hero-card::before,
          .stApp[data-theme="dark"] .hero-card::before {
            background: linear-gradient(
              135deg,
              rgba(67, 98, 124, 0.22) 0%,
              rgba(46, 74, 98, 0.12) 36%,
              rgba(23, 38, 57, 0.04) 100%
            );
          }

          @media (max-width: 768px) {
            .block-container {
              padding-left: 0.7rem;
              padding-right: 0.7rem;
              padding-top: 0.65rem;
            }
            .hero-card { border-radius: 12px; padding: 0.62rem 0.75rem; margin-bottom: 0.45rem; min-height: 64px; }
            .hero-card::after { width: 3px; top: 14%; bottom: 14%; }
            .hero-title { font-size: clamp(1.75rem, 5vw, 2rem); margin-left: 0.3rem; }
            .stTabs [data-baseweb="tab"] { padding: 6px 10px; font-size: 0.86rem; }
            .stButton > button { min-height: 2.45rem; }
            [data-testid="stSidebar"] { min-width: 78vw; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_schema() -> Dict[str, Any]:
    return {
        "name": "visit_note_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "role_protocol": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "doctor": {"type": "string"},
                        "patient": {"type": "string"},
                    },
                    "required": ["doctor", "patient"],
                },
                "teleconsultation": {"type": "boolean"},
                "joint_exam_with_doctor": {"type": "boolean"},
                "joint_exam_with_head": {"type": "boolean"},
                "complaints": {"type": "string"},
                "history_of_present_illness": {"type": "string"},
                "allergic_reactions": {"type": "string"},
                "diseases_list": {"type": "string"},
                "weight_kg": {"type": "number"},
                "height_cm": {"type": "number"},
                "bmi": {"type": "number"},
                "temperature_c": {"type": "number"},
                "bp_systolic": {"type": "integer"},
                "bp_diastolic": {"type": "integer"},
                "heart_rate": {"type": "integer"},
                "resp_rate": {"type": "integer"},
                "creatinine_value": {"type": "number"},
                "creatinine_unit": {
                    "type": "string",
                    "enum": ["мкмоль/л", "мг/дл", "неизвестно"],
                },
                "creatinine_result_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD или пусто",
                },
                "race": {
                    "type": "string",
                    "enum": ["европеоидная", "негроидная", "азиатская", "другое", "неизвестно"],
                },
                "egfr_formula": {
                    "type": "string",
                    "enum": ["CKD-EPI", "MDRD", "другое", "неизвестно"],
                },
                "egfr_value": {"type": "number"},
                "family_history": {"type": "string"},
                "past_diseases": {"type": "string"},
                "past_surgeries": {"type": "string"},
                "chronic_medications": {"type": "string"},
                "alcohol": {
                    "type": "string",
                    "enum": ["нет", "редко", "умеренно", "часто", "неизвестно"],
                },
                "smoking": {
                    "type": "string",
                    "enum": ["нет", "да", "бывший", "неизвестно"],
                },
                "objective_findings": {"type": "string"},
                "diagnoses": {"type": "string"},
                "differential_diagnosis": {"type": "string"},
                "needs_surgery": {"type": "boolean"},
                "orders_studies_consults": {"type": "string"},
                "orders_procedure_room": {"type": "string"},
                "rec_regimen": {"type": "string"},
                "rec_nutrition": {"type": "string"},
                "rec_lifestyle": {"type": "string"},
                "rec_other_treatment": {"type": "string"},
                "med_no_changes_needed": {"type": "boolean"},
                "med_consider_timing": {"type": "boolean"},
                "med_show_inactive": {"type": "boolean"},
                "medications_plan": {"type": "string"},
                "followup_date": {"type": "string", "description": "YYYY-MM-DD или пусто"},
                "followup_not_required": {"type": "boolean"},
                "summary": {
                    "type": "string",
                    "description": "Важная медицинская информация, которая не была отражена в других полях JSON",
                },
            },
            "required": [
                "role_protocol",
                "teleconsultation",
                "joint_exam_with_doctor",
                "joint_exam_with_head",
                "complaints",
                "history_of_present_illness",
                "allergic_reactions",
                "diseases_list",
                "weight_kg",
                "height_cm",
                "bmi",
                "temperature_c",
                "bp_systolic",
                "bp_diastolic",
                "heart_rate",
                "resp_rate",
                "creatinine_value",
                "creatinine_unit",
                "creatinine_result_date",
                "race",
                "egfr_formula",
                "egfr_value",
                "family_history",
                "past_diseases",
                "past_surgeries",
                "chronic_medications",
                "alcohol",
                "smoking",
                "objective_findings",
                "diagnoses",
                "differential_diagnosis",
                "needs_surgery",
                "orders_studies_consults",
                "orders_procedure_room",
                "rec_regimen",
                "rec_nutrition",
                "rec_lifestyle",
                "rec_other_treatment",
                "med_no_changes_needed",
                "med_consider_timing",
                "med_show_inactive",
                "medications_plan",
                "followup_date",
                "followup_not_required",
                "summary",
            ],
        },
    }


def call_gpt_to_fill_fields(transcript: str) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не задан")

    client = build_openai_client()
    schema = build_schema()["schema"]

    system_prompt = f"""
Ты — медицинский ассистент.
Верни ТОЛЬКО валидный JSON на основании записи приема.
Ничего сам не выдумывай.
Сам назначений не делай, исследования выписывай только упомянутые врачом.
Никакого текста, комментариев или markdown.

Структура JSON:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Правила:
- сейчас 2026 год, дату контрольного приема ставь на 2026, если не уточнен год
- если данных нет → пустая строка
- числа → 0
- boolean → false
- enum → ближайшее или "неизвестно"
- поле summary: выпиши только важную медицинскую информацию, которая еще не была отражена в остальных полях JSON
"""

    response = client.responses.create(
        model="gpt-5-chat-latest",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )

    usage = getattr(response, "usage", None)
    if usage:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        cached_input_tokens = getattr(usage, "cached_input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

        PRICE_INPUT = 1.25
        PRICE_CACHED_INPUT = 0.125
        PRICE_OUTPUT = 10.00

        cost_usd = (
            input_tokens / 1_000_000 * PRICE_INPUT
            + cached_input_tokens / 1_000_000 * PRICE_CACHED_INPUT
            + output_tokens / 1_000_000 * PRICE_OUTPUT
        )

        print(
            f"[GPT-5] input={input_tokens} | "
            f"cached_input={cached_input_tokens} | "
            f"output={output_tokens} | "
            f"cost=${cost_usd:.4f}"
        )

    output_text = response.output_text
    if not output_text:
        raise RuntimeError("Пустой ответ от модели")

    return json.loads(output_text)


# -----------------------------
# UI
# -----------------------------
set_session_defaults()
st.set_page_config(page_title=APP_TITLE, layout="wide")
inject_pretty_styles()
st.markdown(
    """
    <div class="hero-card">
      <h1 class="hero-title">Автозаполнение листа приёма</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Загрузить аудиофайл")

    uploaded = st.file_uploader(
        "Загрузить аудиофайл",
        type=["wav", "mp3", "m4a", "ogg", "webm"],
        label_visibility="collapsed",
    )

    st.divider()
    do_transcribe = st.button("1) Транскрибировать", use_container_width=True)
    do_extract = False
    if st.session_state.role_blocks:
        do_extract = st.button("2) Заполнить поля", use_container_width=True)

    st.divider()
    st.caption("Статус")
    if st.session_state.audio_minutes is not None:
        audio_status = f"загружено, {st.session_state.audio_minutes:.2f} мин."
    elif st.session_state.audio_path:
        audio_status = "загружено"
    else:
        audio_status = "не загружено"
    st.write(f"Аудио: {audio_status}")
    st.write(f"Транскрипт: {len(st.session_state.transcript)} символов")
    st.write(f"JSON: {'готов' if st.session_state.llm_json else 'нет'}")

if uploaded:
    file_signature = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"
    if st.session_state.uploaded_signature != file_signature:
        tmpdir = tempfile.mkdtemp(prefix="visit_upload_")
        path = os.path.join(tmpdir, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.audio_path = path
        st.session_state.uploaded_signature = file_signature
        st.session_state.audio_minutes = None
        st.session_state.transcript = ""
        st.session_state.whisper_transcript = ""
        st.session_state.llm_json = None
        st.session_state.role_marked_text = ""
        st.session_state.role_blocks = []

if do_transcribe:
    if not st.session_state.audio_path:
        st.error("Нет загруженного аудио")
    else:
        with st.spinner("Транскрибация аудио…"):
            transcript = transcribe_uploaded_audio(Path(st.session_state.audio_path))
            st.session_state.transcript = transcript
            st.session_state.whisper_transcript = transcript

        with st.spinner("Разметка ролей…"):
            role_text = call_gpt_role_split(transcript)
            st.session_state.role_marked_text = role_text
            st.session_state.role_blocks = parse_role_blocks(role_text)

        # Обновляем интерфейс сразу, чтобы кнопка "Заполнить поля" появилась в sidebar в этот же цикл.
        st.rerun()

if do_extract:
    # В JSON отправляем только сырой текст Whisper, не разметку по ролям.
    whisper_text = st.session_state.whisper_transcript or st.session_state.transcript
    if not whisper_text:
        st.error("Нет транскрипта")
    else:
        json_start = time.perf_counter()
        st.session_state.llm_json = call_gpt_to_fill_fields(whisper_text)
        json_elapsed = time.perf_counter() - json_start
        print("[Info] JSON source: Whisper transcript")
        print(f"[Timing] Транскрипт -> JSON: {json_elapsed:.2f} sec | {json_elapsed/60:.2f} min")
        st.markdown(
            '<div class="ok-card">Лист приёма заполнен</div>',
            unsafe_allow_html=True,
        )


tab1, tab2 = st.tabs(["Протокол", "Лист приёма"])

with tab1:
    if not st.session_state.role_blocks:
        st.markdown(
            '<div class="hint-card">Загрузите файл с записью приёма и нажмите кнопку <b>Транскрибировать</b></div>',
            unsafe_allow_html=True,
        )
    else:
        st.subheader("Размеченный протокол")
        for block in st.session_state.role_blocks:
            safe_text = block["text"].replace("\n", "<br>")
            if block["role"] == "врач":
                st.markdown(
                    f"""
                    <div style="
                        background:linear-gradient(135deg, rgba(216, 238, 245, 0.96), rgba(224, 243, 247, 0.96));
                        border-left:6px solid #7fbfca;
                        padding:12px;
                        margin-bottom:10px;
                        border-radius:10px;
                        color:#2f5f6b;
                    ">
                    <b>Врач</b><br>{safe_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        background:#ffffff;
                        border-left:6px solid #c6ddd5;
                        border:1px solid #dbe8e3;
                        padding:12px;
                        margin-bottom:10px;
                        border-radius:10px;
                        color:#2f3b3a;
                    ">
                    <b>Пациент</b><br>{safe_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def jget(key: str, default: Any) -> Any:
    j = st.session_state.llm_json or {}
    return j.get(key, default)


with tab2:
    st.markdown("##### Флаги приёма")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.checkbox("Дистанционная консультация", value=bool(jget("teleconsultation", False)))
    with c2:
        st.checkbox("Совместный осмотр с врачом", value=bool(jget("joint_exam_with_doctor", False)))
    with c3:
        st.checkbox("Совместный осмотр с заведующим отделением", value=bool(jget("joint_exam_with_head", False)))

    st.divider()

    st.markdown("##### Жалобы и анамнез заболевания")
    st.text_input("Жалобы пациента", value=str(jget("complaints", "")))
    st.text_area("Анамнез заболевания", value=str(jget("history_of_present_illness", "")), height=100)

    st.divider()

    st.markdown("##### Аллергии и заболевания (справочно)")
    c1, c2 = st.columns(2)
    with c1:
        st.text_area("Аллергические реакции", value=str(jget("allergic_reactions", "")), height=100)
    with c2:
        st.text_area("Заболевания", value=str(jget("diseases_list", "")), height=100)

    st.divider()

    st.markdown("##### Антропометрия и витальные показатели")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Вес, кг", value=float(jget("weight_kg", 0.0)), step=0.1)
    with c2:
        st.number_input("Рост, см", value=float(jget("height_cm", 0.0)), step=0.1)
    with c3:
        st.number_input("BMI (расчёт/из протокола)", value=float(jget("bmi", 0.0)), step=0.1, disabled=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.number_input("Температура, °C", value=float(jget("temperature_c", 0.0)), step=0.1)
    with c2:
        st.number_input("АД сист., мм рт.ст.", value=int(jget("bp_systolic", 0)), step=1)
        st.number_input("АД диаст., мм рт.ст.", value=int(jget("bp_diastolic", 0)), step=1)
    with c3:
        st.number_input("ЧСС, уд/мин", value=int(jget("heart_rate", 0)), step=1)
    with c4:
        st.number_input("ЧДД, движений/мин", value=int(jget("resp_rate", 0)), step=1)

    st.divider()

    st.markdown("##### Креатинин / СКФ")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Креатинин", value=float(jget("creatinine_value", 0.0)), step=0.1)
        st.selectbox(
            "Единицы",
            ["мкмоль/л", "мг/дл", "неизвестно"],
            index=["мкмоль/л", "мг/дл", "неизвестно"].index(str(jget("creatinine_unit", "неизвестно"))),
        )
    with c2:
        d = safe_date(jget("creatinine_result_date", ""))
        st.date_input("Дата результата", value=d or date.today())
        st.selectbox(
            "Раса",
            ["европеоидная", "негроидная", "азиатская", "другое", "неизвестно"],
            index=["европеоидная", "негроидная", "азиатская", "другое", "неизвестно"].index(str(jget("race", "неизвестно"))),
        )
    with c3:
        st.selectbox(
            "Формула СКФ",
            ["CKD-EPI", "MDRD", "другое", "неизвестно"],
            index=["CKD-EPI", "MDRD", "другое", "неизвестно"].index(str(jget("egfr_formula", "неизвестно"))),
        )
        st.number_input("СКФ (мл/мин/1.73м²)", value=float(jget("egfr_value", 0.0)), step=0.1)

    st.divider()

    st.markdown("##### Анамнез жизни и привычки")
    st.text_area("Семейный анамнез", value=str(jget("family_history", "")), height=90)
    st.text_area("Перенесённые заболевания (в т.ч. детские инфекции)", value=str(jget("past_diseases", "")), height=90)
    st.text_area("Перенесённые операции", value=str(jget("past_surgeries", "")), height=90)
    st.text_area("Постоянно принимаемые препараты", value=str(jget("chronic_medications", "")), height=90)

    c1, c2 = st.columns(2)
    with c1:
        st.selectbox(
            "Алкоголь",
            ["нет", "редко", "умеренно", "часто", "неизвестно"],
            index=["нет", "редко", "умеренно", "часто", "неизвестно"].index(str(jget("alcohol", "неизвестно"))),
        )
    with c2:
        st.selectbox(
            "Курение",
            ["нет", "да", "бывший", "неизвестно"],
            index=["нет", "да", "бывший", "неизвестно"].index(str(jget("smoking", "неизвестно"))),
        )

    st.divider()

    st.markdown("##### Объективные данные")
    st.text_area(
        "Объективные данные (физикальное исследование, локальный статус)",
        value=str(jget("objective_findings", "")),
        height=140,
    )

    st.divider()

    st.markdown("##### Диагнозы")
    st.text_area("Диагнозы", value=str(jget("diagnoses", "")), height=90)
    st.text_area("Дифференциальный диагноз", value=str(jget("differential_diagnosis", "")), height=90)

    st.divider()

    st.markdown("##### Назначения")
    st.checkbox("Требуется оперативное вмешательство", value=bool(jget("needs_surgery", False)))
    st.text_area("Назначения (исследования, консультации)", value=str(jget("orders_studies_consults", "")), height=90)
    st.text_area("Назначения в процедурный кабинет", value=str(jget("orders_procedure_room", "")), height=90)

    st.divider()

    st.markdown("##### Рекомендации")
    c1, c2 = st.columns(2)
    with c1:
        st.text_area("Рекомендации по режиму", value=str(jget("rec_regimen", "")), height=110)
        st.text_area("Рекомендации по образу жизни", value=str(jget("rec_lifestyle", "")), height=110)
    with c2:
        st.text_area("Рекомендации по питанию", value=str(jget("rec_nutrition", "")), height=110)
        st.text_area("Рекомендации по другим видам лечения", value=str(jget("rec_other_treatment", "")), height=110)

    st.divider()

    st.markdown("##### Медикаментозная терапия")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.checkbox("Не требует изменений", value=bool(jget("med_no_changes_needed", False)))
    with c2:
        st.checkbox("С учётом времени приёма", value=bool(jget("med_consider_timing", False)))
    with c3:
        st.checkbox("Показывать недействующие", value=bool(jget("med_show_inactive", False)))
    st.text_area("Медикаментозная терапия (план/назначения)", value=str(jget("medications_plan", "")), height=110)

    st.divider()

    st.markdown("##### Контроль")
    c1, c2 = st.columns(2)
    with c1:
        fu = safe_date(jget("followup_date", ""))
        st.date_input("Дата контрольного визита", value=fu or date.today())
    with c2:
        st.checkbox("Не требуется", value=bool(jget("followup_not_required", False)))

    st.divider()
    st.markdown("##### Саммари")
    st.text_area(
        "Summary (важная медицинская информация вне остальных полей)",
        value=str(jget("summary", "")),
        height=120,
    )

    st.divider()
    st.markdown("##### JSON от модели (для копирования)")
    st.code(json.dumps(st.session_state.llm_json or {}, ensure_ascii=False, indent=2), language="json")
