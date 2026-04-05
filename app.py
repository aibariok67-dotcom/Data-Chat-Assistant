import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import traceback
from openai import OpenAI

st.set_page_config(page_title="Data Chat Assistant", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #fafafa; }
    .chat-msg { padding: 12px 16px; border-radius: 12px; margin-bottom: 8px; }
    .user-msg { background-color: #dbeafe; text-align: right; }
    .ai-msg { background-color: #f0fdf4; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Data Chat Assistant")
    st.markdown("---")

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    uploaded = st.file_uploader("Загрузи CSV файл", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Загружено: {len(df)} строк, {len(df.columns)} колонок")
        st.markdown("**Колонки:**")
        for col in df.columns:
            st.code(f"{col} ({df[col].dtype})", language=None)
    else:
        df = None

    st.markdown("---")
    st.markdown("**Примеры вопросов:**")
    example_questions = [
        "Какой месяц был самым прибыльным?",
        "Покажи топ-5 значений по первой числовой колонке",
        "Есть ли пропуски в данных?",
        "Построй гистограмму распределения",
        "Покажи корреляцию между числовыми колонками",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.prefill = q

# ── Chat state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("💬 Чат с данными")

if df is None:
    st.info("👈 Загрузи CSV файл в боковой панели, чтобы начать")
    st.stop()

if not api_key:
    st.warning("Введи OpenAI API Key в боковой панели")
    st.stop()

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"])
            if "code" in msg:
                with st.expander("Показать код"):
                    st.code(msg["code"], language="python")
            if "figure" in msg:
                st.pyplot(msg["figure"])
        else:
            st.markdown(msg["content"])

# ── Input ──────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Задай вопрос о твоих данных...", key="chat_input")

if not user_input and prefill:
    user_input = prefill

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build context about the dataframe
    df_info = f"""
DataFrame info:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {list(df.columns)}
- Dtypes: {df.dtypes.to_dict()}
- Head (3 rows):
{df.head(3).to_string()}
- Basic stats:
{df.describe().to_string()}
"""

    system_prompt = f"""Ты — AI-ассистент для анализа данных. У тебя есть доступ к DataFrame `df` с следующей структурой:

{df_info}

Когда пользователь задаёт вопрос:
1. Напиши короткий ответ на естественном языке
2. Напиши Python код для анализа данных (используй pandas, matplotlib)
3. Оберни код в тег ```python ... ```
4. Если нужен график — используй matplotlib, сохраняй в plt (не вызывай plt.show())

ВАЖНО: Код должен работать с переменной `df`. Для графиков используй fig, ax = plt.subplots(). Пиши компактный, рабочий код.
"""

    history = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.messages[:-1]:
        history.append({"role": m["role"], "content": m["content"]})
    history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Анализирую..."):
            try:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=history,
                    max_tokens=1000,
                )
                full_response = response.choices[0].message.content

                # Extract code block if present
                code = None
                fig = None
                text_response = full_response

                if "```python" in full_response:
                    parts = full_response.split("```python")
                    text_response = parts[0].strip()
                    code_part = parts[1].split("```")[0].strip()
                    code = code_part

                    # Execute the code safely
                    local_vars = {"df": df.copy(), "pd": pd, "plt": plt}
                    try:
                        exec(code, {}, local_vars)
                        # Check if a figure was created
                        if plt.get_fignums():
                            fig = plt.gcf()
                    except Exception as e:
                        text_response += f"\n\n⚠️ Ошибка выполнения кода: `{e}`"

                st.markdown(text_response)
                if code:
                    with st.expander("Показать код"):
                        st.code(code, language="python")
                if fig:
                    st.pyplot(fig)
                    plt.close()

                msg_entry = {"role": "assistant", "content": text_response}
                if code:
                    msg_entry["code"] = code
                if fig:
                    msg_entry["figure"] = fig
                st.session_state.messages.append(msg_entry)

            except Exception as e:
                error_msg = f"Ошибка: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
