# streamlit_app.py
# Streamlit-приложение: запись/загрузка аудио -> Whisper (локально: --model medium)
# -> LLM (GPT-5) возвращает JSON по полям МИС + раздельный протокол по ролям
#
# Установка (пример):
#   pip install -U streamlit openai>=1.40.0
#   pip install -U openai-whisper
#   # (опционально для записи в браузере)
#   pip install -U streamlit-audio-recorder
#
# Требования для whisper:
#   - установлен ffmpeg (в системе)
#   - доступна команда `whisper` в PATH (ставится с openai-whisper)
#
# Запуск:
#   export OPENAI_API_KEY="..."
#   streamlit run streamlit_app.py

import json
import os
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


import streamlit as st
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI Python SDK (Responses API)
# Документация: Structured Outputs / Responses API. :contentReference[oaicite:0]{index=0}
from openai import OpenAI

APP_TITLE = "Автозаполнение листа приёма из аудио "

# -----------------------------
# Helpers
# -----------------------------
def _ensure_whisper_exists() -> Optional[str]:
    return shutil.which("whisper")

import wave
import contextlib
import math

WHISPER_PRICE_PER_MIN = 0.006  # $ / минута


def run_whisper_transcribe(audio_path: str, language: str = "ru") -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Не найден файл, проверьте соединение")

    if not os.path.exists(audio_path):
        raise RuntimeError(f"Аудиофайл не найден: {audio_path}")

    # --- считаем длительность ---
    duration_sec = None
    try:
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration_sec = frames / float(rate)
    except Exception:
        # если не wav — считаем грубо через размер (fallback)
        duration_sec = None

    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="gpt-4o-transcribe",
            language=language if len(language) == 2 else None
        )

    text = getattr(response, "text", "")
    if not text or not text.strip():
        raise RuntimeError("Пустой результат транскрибации")

    # ===== ЛОГИРОВАНИЕ =====
    if duration_sec:
        minutes = duration_sec / 60
        cost = minutes * WHISPER_PRICE_PER_MIN
        print(
            f"[Whisper] duration={minutes:.2f} min | "
            f"cost≈${cost:.4f}"
        )
    else:
        print("[Whisper] duration=unknown | cost≈unknown")

    return text.strip()

def call_gpt_role_split(transcript: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)

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
                blocks.append({
                    "role": current_role,
                    "text": "\n".join(buffer).strip()
                })
                buffer = []
            current_role = l.replace(":", "")
        else:
            buffer.append(line)

    if current_role and buffer:
        blocks.append({
            "role": current_role,
            "text": "\n".join(buffer).strip()
        })

    return blocks

def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

def safe_int(x: Any) -> Optional[int]:
    f = safe_float(x)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None

def safe_date(x: Any) -> Optional[date]:
    if not x:
        return None
    if isinstance(x, date):
        return x
    s = str(x).strip()
    # ожидаем YYYY-MM-DD
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None

def set_session_defaults():
    defaults = {
        "audio_path": None,
        "transcript": "",
        "llm_json": None,
        "doctor_text": "",
        "patient_text": "",
        "role_marked_text": "",
        "role_blocks": [],

        
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# -----------------------------
# LLM JSON schema (Structured Outputs)
# -----------------------------
def build_schema() -> Dict[str, Any]:
    # В строгом режиме лучше делать required для всех полей и разрешать пустые значения. :contentReference[oaicite:1]{index=1}
    return {
        "name": "visit_note_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                # Протокол по ролям
                "role_protocol": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "doctor": {"type": "string"},
                        "patient": {"type": "string"},
                    },
                    "required": ["doctor", "patient"],
                },

                # Флаги приёма
                "teleconsultation": {"type": "boolean"},
                "joint_exam_with_doctor": {"type": "boolean"},
                "joint_exam_with_head": {"type": "boolean"},

                # Жалобы / анамнез заболевания
                "complaints": {"type": "string"},
                "history_of_present_illness": {"type": "string"},

                # Аллергии / заболевания (справочный блок)
                "allergic_reactions": {"type": "string"},
                "diseases_list": {"type": "string"},

                # Антропометрия и витальные
                "weight_kg": {"type": "number"},
                "height_cm": {"type": "number"},
                "bmi": {"type": "number"},

                "temperature_c": {"type": "number"},
                "bp_systolic": {"type": "integer"},
                "bp_diastolic": {"type": "integer"},
                "heart_rate": {"type": "integer"},
                "resp_rate": {"type": "integer"},

                # Креатинин / СКФ
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

                # Анамнез жизни
                "family_history": {"type": "string"},
                "past_diseases": {"type": "string"},
                "past_surgeries": {"type": "string"},
                "chronic_medications": {"type": "string"},

                # Вредные привычки
                "alcohol": {
                    "type": "string",
                    "enum": ["нет", "редко", "умеренно", "часто", "неизвестно"],
                },
                "smoking": {
                    "type": "string",
                    "enum": ["нет", "да", "бывший", "неизвестно"],
                },

                # Объективно
                "objective_findings": {"type": "string"},

                # Диагнозы
                "diagnoses": {"type": "string"},
                "differential_diagnosis": {"type": "string"},

                # Назначения / хирургия
                "needs_surgery": {"type": "boolean"},
                "orders_studies_consults": {"type": "string"},
                "orders_procedure_room": {"type": "string"},

                # Рекомендации
                "rec_regimen": {"type": "string"},
                "rec_nutrition": {"type": "string"},
                "rec_lifestyle": {"type": "string"},
                "rec_other_treatment": {"type": "string"},

                # Медикаментозная терапия
                "med_no_changes_needed": {"type": "boolean"},
                "med_consider_timing": {"type": "boolean"},
                "med_show_inactive": {"type": "boolean"},
                "medications_plan": {"type": "string"},

                # Контроль
                "followup_date": {"type": "string", "description": "YYYY-MM-DD или пусто"},
                "followup_not_required": {"type": "boolean"},
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
            ],
        },
    }

def call_gpt_to_fill_fields(transcript: str) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не задан")

    client = OpenAI(api_key=OPENAI_API_KEY)

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
"""

    response = client.responses.create(
        model="gpt-5-chat-latest",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )

    # ===== ЛОГИРОВАНИЕ =====
    usage = getattr(response, "usage", None)
    if usage:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        cached_input_tokens = getattr(usage, "cached_input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

        PRICE_INPUT = 1.25
        PRICE_CACHED_INPUT = 0.125
        PRICE_OUTPUT = 10.00

        cost_usd = (
            input_tokens / 1_000_000 * PRICE_INPUT +
            cached_input_tokens / 1_000_000 * PRICE_CACHED_INPUT +
            output_tokens / 1_000_000 * PRICE_OUTPUT
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

    try:
        return json.loads(output_text)
    except Exception as e:
        raise RuntimeError(
            f"JSON не распарсился: {e}\n\nОтвет модели:\n{output_text}"
        )


# -----------------------------
# UI
# -----------------------------
set_session_defaults()
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Ввод аудио")

    uploaded = st.file_uploader(
        "Загрузить аудиофайл",
        type=["wav", "mp3", "m4a", "ogg", "webm"],
    )

    st.divider()

    

    do_transcribe = st.button("1) Расшифровать", use_container_width=True)
    do_extract = st.button("2) Заполнить поля", use_container_width=True)

    st.divider()
    st.caption("Статус")
    st.write("Аудио:", "есть" if st.session_state.audio_path else "нет")
    st.write("Транскрипт:", len(st.session_state.transcript))
    st.write("JSON:", "есть" if st.session_state.llm_json else "нет")

# =========================
# FILE UPLOAD → SESSION
# =========================
if uploaded:
    tmpdir = tempfile.mkdtemp(prefix="visit_upload_")
    path = os.path.join(tmpdir, uploaded.name)

    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.session_state.audio_path = path

# =========================
# RECORD MODAL
# =========================


# =========================
# ACTIONS
# =========================
if do_transcribe:
    if not st.session_state.audio_path:
        st.error("Нет аудио для расшифровки")
    else:
        with st.spinner("Расшифровка аудио…"):
            transcript = run_whisper_transcribe(st.session_state.audio_path)
            st.session_state.transcript = transcript

        with st.spinner("Разметка ролей…"):
            role_text = call_gpt_role_split(transcript)
            st.session_state.role_marked_text = role_text
            st.session_state.role_blocks = parse_role_blocks(role_text)

        st.success("Аудио расшифровано и размечено по ролям")


if do_extract:
    if not st.session_state.transcript:
        st.error("Нет транскрипта")
    else:
        st.session_state.llm_json = call_gpt_to_fill_fields(
            st.session_state.transcript
        )
        st.success("Лист приёма заполнен")

# ---------- OUTPUT ----------
tab1, tab2 = st.tabs(["Протокол", "Лист приема"])


# -----------------------------
# Tab 1: role protocol
# -----------------------------
with tab1:
    st.subheader("Размеченный протокол приёма")

    for block in st.session_state.role_blocks:
        safe_text = block["text"].replace("\n", "<br>")

        if block["role"] == "врач":
            st.markdown(
                f"""
                <div style="
                    background-color:#e3f2fd;
                    border-left:6px solid #90caf9;
                    padding:12px;
                    margin-bottom:10px;
                    border-radius:6px;
                ">
                <b>Врач</b><br>{safe_text}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="
                    background-color:#f5f5f5;
                    border-left:6px solid #bdbdbd;
                    padding:12px;
                    margin-bottom:10px;
                    border-radius:6px;
                ">
                <b>Пациент</b><br>{safe_text}
                </div>
                """,
                unsafe_allow_html=True
            )

# -----------------------------
# Tab 2: MIS fields
# -----------------------------
def jget(key: str, default: Any) -> Any:
    j = st.session_state.llm_json or {}
    return j.get(key, default)

with tab2:
    st.subheader("Лист приёма")

    # Верхние флаги + кнопки тут не нужны (кнопки уже в sidebar), оставим поля
    st.markdown("##### Флаги приёма")
    c1, c2, c3 = st.columns(3)
    with c1:
        tele = st.checkbox("Дистанционная консультация", value=bool(jget("teleconsultation", False)))
    with c2:
        joint_doc = st.checkbox("Совместный осмотр с врачом", value=bool(jget("joint_exam_with_doctor", False)))
    with c3:
        joint_head = st.checkbox("Совместный осмотр с заведующим отделением", value=bool(jget("joint_exam_with_head", False)))

    st.divider()

    st.markdown("##### Жалобы и анамнез заболевания")
    complaints = st.text_input("Жалобы пациента", value=str(jget("complaints", "")))
    hpi = st.text_area("Анамнез заболевания", value=str(jget("history_of_present_illness", "")), height=100)

    st.divider()

    st.markdown("##### Аллергии и заболевания (справочно)")
    c1, c2 = st.columns(2)
    with c1:
        allergies = st.text_area("Аллергические реакции", value=str(jget("allergic_reactions", "")), height=100)
    with c2:
        diseases_list = st.text_area("Заболевания", value=str(jget("diseases_list", "")), height=100)

    st.divider()

    st.markdown("##### Антропометрия и витальные показатели")
    c1, c2, c3 = st.columns(3)
    with c1:
        weight = st.number_input("Вес, кг", value=float(jget("weight_kg", 0.0)), step=0.1)
    with c2:
        height = st.number_input("Рост, см", value=float(jget("height_cm", 0.0)), step=0.1)
    with c3:
        bmi_val = float(jget("bmi", 0.0))
        st.number_input("BMI (расчёт/из протокола)", value=bmi_val, step=0.1, disabled=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        temp = st.number_input("Температура, °C", value=float(jget("temperature_c", 0.0)), step=0.1)
    with c2:
        bp_sys = st.number_input("АД сист., мм рт.ст.", value=int(jget("bp_systolic", 0)), step=1)
        bp_dia = st.number_input("АД диаст., мм рт.ст.", value=int(jget("bp_diastolic", 0)), step=1)
    with c3:
        hr = st.number_input("ЧСС, уд/мин", value=int(jget("heart_rate", 0)), step=1)
    with c4:
        rr = st.number_input("ЧДД, движений/мин", value=int(jget("resp_rate", 0)), step=1)

    st.divider()

    st.markdown("##### Креатинин / СКФ")
    c1, c2, c3 = st.columns(3)
    with c1:
        creat = st.number_input("Креатинин", value=float(jget("creatinine_value", 0.0)), step=0.1)
        creat_unit = st.selectbox(
            "Единицы",
            ["мкмоль/л", "мг/дл", "неизвестно"],
            index=["мкмоль/л", "мг/дл", "неизвестно"].index(str(jget("creatinine_unit", "неизвестно"))),
        )
    with c2:
        d = safe_date(jget("creatinine_result_date", ""))
        creat_date = st.date_input("Дата результата", value=d or date.today())
        race = st.selectbox(
            "Раса",
            ["европеоидная", "негроидная", "азиатская", "другое", "неизвестно"],
            index=["европеоидная", "негроидная", "азиатская", "другое", "неизвестно"].index(str(jget("race", "неизвестно"))),
        )
    with c3:
        egfr_formula = st.selectbox(
            "Формула СКФ",
            ["CKD-EPI", "MDRD", "другое", "неизвестно"],
            index=["CKD-EPI", "MDRD", "другое", "неизвестно"].index(str(jget("egfr_formula", "неизвестно"))),
        )
        egfr = st.number_input("СКФ (мл/мин/1.73м²)", value=float(jget("egfr_value", 0.0)), step=0.1)

    st.divider()

    st.markdown("##### Анамнез жизни и привычки")
    fam = st.text_area("Семейный анамнез", value=str(jget("family_history", "")), height=90)
    past_dis = st.text_area("Перенесённые заболевания (в т.ч. детские инфекции)", value=str(jget("past_diseases", "")), height=90)
    past_surg = st.text_area("Перенесённые операции", value=str(jget("past_surgeries", "")), height=90)
    meds_chronic = st.text_area("Постоянно принимаемые препараты", value=str(jget("chronic_medications", "")), height=90)

    c1, c2 = st.columns(2)
    with c1:
        alcohol = st.selectbox(
            "Алкоголь",
            ["нет", "редко", "умеренно", "часто", "неизвестно"],
            index=["нет", "редко", "умеренно", "часто", "неизвестно"].index(str(jget("alcohol", "неизвестно"))),
        )
    with c2:
        smoking = st.selectbox(
            "Курение",
            ["нет", "да", "бывший", "неизвестно"],
            index=["нет", "да", "бывший", "неизвестно"].index(str(jget("smoking", "неизвестно"))),
        )

    st.divider()

    st.markdown("##### Объективные данные")
    obj = st.text_area("Объективные данные (физикальное исследование, локальный статус)", value=str(jget("objective_findings", "")), height=140)

    st.divider()

    st.markdown("##### Диагнозы")
    diag = st.text_area("Диагнозы", value=str(jget("diagnoses", "")), height=90)
    ddx = st.text_area("Дифференциальный диагноз", value=str(jget("differential_diagnosis", "")), height=90)

    st.divider()

    st.markdown("##### Назначения")
    needs_surg = st.checkbox("Требуется оперативное вмешательство", value=bool(jget("needs_surgery", False)))
    orders = st.text_area("Назначения (исследования, консультации)", value=str(jget("orders_studies_consults", "")), height=90)
    proc_orders = st.text_area("Назначения в процедурный кабинет", value=str(jget("orders_procedure_room", "")), height=90)

    st.divider()

    st.markdown("##### Рекомендации")
    c1, c2 = st.columns(2)
    with c1:
        rec_reg = st.text_area("Рекомендации по режиму", value=str(jget("rec_regimen", "")), height=110)
        rec_life = st.text_area("Рекомендации по образу жизни", value=str(jget("rec_lifestyle", "")), height=110)
    with c2:
        rec_food = st.text_area("Рекомендации по питанию", value=str(jget("rec_nutrition", "")), height=110)
        rec_other = st.text_area("Рекомендации по другим видам лечения", value=str(jget("rec_other_treatment", "")), height=110)

    st.divider()

    st.markdown("##### Медикаментозная терапия")
    c1, c2, c3 = st.columns(3)
    with c1:
        med_nochg = st.checkbox("Не требует изменений", value=bool(jget("med_no_changes_needed", False)))
    with c2:
        med_timing = st.checkbox("С учётом времени приёма", value=bool(jget("med_consider_timing", False)))
    with c3:
        med_show_inactive = st.checkbox("Показывать недействующие", value=bool(jget("med_show_inactive", False)))
    med_plan = st.text_area("Медикаментозная терапия (план/назначения)", value=str(jget("medications_plan", "")), height=110)

    st.divider()

    st.markdown("##### Контроль")
    c1, c2 = st.columns(2)
    with c1:
        fu = safe_date(jget("followup_date", ""))
        followup = st.date_input("Дата контрольного визита", value=fu or date.today())
    with c2:
        fu_not_req = st.checkbox("Не требуется", value=bool(jget("followup_not_required", False)))

    st.divider()

    # Покажем текущий JSON (для отладки/копирования)
    st.markdown("##### JSON от модели (для копирования)")
    st.code(json.dumps(st.session_state.llm_json or {}, ensure_ascii=False, indent=2), language="json")
