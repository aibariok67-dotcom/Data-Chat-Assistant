import re
import io
import sys
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(page_title="Data Chat Assistant", layout="wide")

matplotlib.rcParams.update({
    'figure.facecolor': '#1e2130',
    'axes.facecolor':   '#1e2130',
    'axes.edgecolor':   '#3a3f55',
    'axes.labelcolor':  '#c0c8d8',
    'text.color':       '#c0c8d8',
    'xtick.color':      '#c0c8d8',
    'ytick.color':      '#c0c8d8',
    'grid.color':       '#2a2f45',
    'figure.figsize':   (8, 4),
    'axes.titlecolor':  '#e0e0e0',
})

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #1a1d27; }
    section[data-testid="stSidebar"] * { color: #c0c8d8 !important; }
    [data-testid="stChatMessage"] { background-color: #1e2130; border-radius: 12px; margin-bottom: 4px; }
    [data-testid="stChatMessageContent"] p { color: #e0e0e0 !important; font-size: 15px; line-height: 1.6; }
    [data-testid="stChatMessageContent"] pre { background: #0d0f18 !important; }
    [data-testid="stChatMessageContent"] code { color: #7dd3fc !important; }
    [data-testid="stChatInput"] textarea { background-color: #1e2130 !important; color: #e0e0e0 !important; border-color: #3a3f55 !important; }
    .stButton button { background-color: #2a2f45 !important; color: #c0c8d8 !important; border: 1px solid #3a3f55 !important; border-radius: 8px !important; font-size: 13px !important; }
    .stButton button:hover { background-color: #363c58 !important; }
    h1 { color: #7dd3fc !important; }
    [data-testid="stExpander"] { background-color: #1a1d27 !important; border: 1px solid #2a2f45 !important; border-radius: 8px; }
    [data-testid="stExpander"] summary { color: #7dd3fc !important; }
</style>
""", unsafe_allow_html=True)


def fix_code(code, df):
    """Auto-fix common GPT code mistakes before execution."""
    def replace_nlargest(m):
        n = m.group(1)
        col = m.group(2)
        str_cols = [c for c in df.columns if df[c].dtype == object and c != col]
        grp = str_cols[0] if str_cols else df.columns[0]
        return f"df.groupby('{grp}')['{col}'].sum().nlargest({n}).reset_index()"
    code = re.sub(r"df\.nlargest\((\d+),\s*['\"](\w+)['\"]\)", replace_nlargest, code)
    return code


with st.sidebar:
    st.title("Data Chat")
    st.markdown("---")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    uploaded = st.file_uploader("Загрузи CSV файл", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"[OK] {len(df)} строк · {len(df.columns)} колонок")
        st.markdown("**Колонки:**")
        for col in df.columns:
            st.code(f"{col}  ({df[col].dtype})", language=None)
    else:
        df = None

    st.markdown("---")
    if st.button("Очистить чат", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("**Примеры вопросов:**")
    examples = [
        "Какой месяц был самым прибыльным?",
        "Покажи топ-5 товаров по продажам",
        "Есть ли пропуски в данных?",
        "Построй гистограмму продаж",
        "Сравни продажи по регионам",
    ]
    for q in examples:
        if st.button(q, use_container_width=True):
            st.session_state.prefill = q

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Чат с данными")

if df is None:
    st.info("Загрузи CSV файл в боковой панели, чтобы начать")
    st.stop()

if not api_key:
    st.warning("Введи OpenAI API Key в боковой панели")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "code" in msg:
            with st.expander("Показать код"):
                st.code(msg["code"], language="python")
        if msg["role"] == "assistant" and "figure" in msg:
            st.pyplot(msg["figure"])

prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Задай вопрос о своих данных...")
if not user_input and prefill:
    user_input = prefill
if not user_input:
    st.stop()

st.session_state.messages.append({"role": "user", "content": user_input})
with st.chat_message("user"):
    st.markdown(user_input)

null_info = df.isnull().sum()
null_str = null_info[null_info > 0].to_string() if null_info.any() else "Propuskov net"

df_info = (
    f"Shape: {df.shape[0]} strok x {df.shape[1]} kolonok\n"
    f"Kolonki i tipy: {df.dtypes.to_dict()}\n"
    f"Pervye 3 stroki:\n{df.head(3).to_string()}\n"
    f"Statistika:\n{df.describe().to_string()}\n"
    f"Propuski: {null_str}"
)

system_prompt = f"""Ty — AI-assistent dlya analiza dannyh. Tebe dostupen pandas DataFrame `df`:

{df_info}

PRAVILA:
1. SNACHALA napishi Python kod v bloke ```python```
2. V kode OBYAZATELNO ispolzui print() dlya vseh rezultatov s ponyatnymi podpisyami
3. Dlya top-N VSEGDA gruppirui: df.groupby('product')['sales'].sum().nlargest(5).reset_index()
4. NIKOGDA ne ispolzui df.nlargest() — eto pokazhet stroki, ne unikalnye produkty
5. Dlya dat: df['date'] = pd.to_datetime(df['date']), zatem df.groupby(df['date'].dt.month)
6. Dlya grafikov: fig, ax = plt.subplots() — NE vyzvai plt.show()
7. pd i plt uzhe dostupny, ne importiruij ih zanovo
8. Posle koda: 1 predlozhenie-vyvod, no BEZ konkretnyh chisel — tolko kod znaet pravdu
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

            code = None
            fig = None
            text_response = full_response

            if "```python" in full_response:
                parts = full_response.split("```python")
                text_response = parts[0].strip()
                code = parts[1].split("```")[0].strip()
                code = fix_code(code, df)

                stdout_capture = io.StringIO()
                base_vars = {"df", "pd", "plt"}
                local_vars = {"df": df.copy(), "pd": pd, "plt": plt}
                old_stdout = sys.stdout

                try:
                    sys.stdout = stdout_capture
                    exec(compile(code, "<string>", "exec"), {}, local_vars)
                    sys.stdout = old_stdout

                    printed = stdout_capture.getvalue().strip()
                    if printed:
                        text_response = ("```\n" + printed + "\n```\n\n" + text_response).strip()
                    else:
                        new_vars = {k: v for k, v in local_vars.items() if k not in base_vars}
                        for k, v in new_vars.items():
                            val_str = str(v).strip()
                            if val_str:
                                text_response = (f"```\n{val_str}\n```\n\n" + text_response).strip()

                    for k, v in local_vars.items():
                        if k not in base_vars and isinstance(v, (pd.DataFrame, pd.Series)):
                            st.dataframe(v, use_container_width=True)

                    if plt.get_fignums():
                        fig = plt.gcf()

                except Exception as e:
                    sys.stdout = old_stdout
                    text_response += f"\n\nOshibka vypolneniya: `{e}`"

            if not text_response:
                text_response = "Анализ выполнен."

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
            err = f"Ошибка: {str(e)}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
