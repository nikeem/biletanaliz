"""
Билет-анализ: какие покупатели ходят на все матчи, на n-1, на n-2.

1 файл xlsx = 1 матч. Каждая строка с email = 1 купленное место.
Запуск локально:  streamlit run app.py
"""
import io
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


# ---------- чтение файлов ----------

def match_name(filename: str) -> str:
    return Path(filename).stem


def match_date(name: str):
    """Пытаемся вытащить дату из имени файла (05_09_26, 05.09.2026, 2026-09-05 ...)."""
    for pattern, fmt in [
        (r"(\d{1,2})[._\-](\d{1,2})[._\-](\d{4})", "%d.%m.%Y"),
        (r"(\d{4})[._\-](\d{1,2})[._\-](\d{1,2})", "%Y.%m.%d"),
        (r"(\d{1,2})[._\-](\d{1,2})[._\-](\d{2})", "%d.%m.%y"),
    ]:
        m = re.search(pattern, name)
        if m:
            try:
                return datetime.strptime(".".join(m.groups()), fmt)
            except ValueError:
                pass
    return None


def read_emails(uploaded_file) -> list[str]:
    """Все email из файла (все листы, все колонки), с повторами = количество мест."""
    data = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".csv"):
        sheets = {"csv": pd.read_csv(io.BytesIO(data), header=None, dtype=str)}
    else:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=str)

    emails = []
    for df in sheets.values():
        for value in df.to_numpy().ravel():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            # в одной ячейке может оказаться несколько адресов
            emails.extend(e.strip().lower() for e in EMAIL_RE.findall(str(value)))
    return emails


@st.cache_data(show_spinner=False)
def build_tables(files_payload: tuple):
    """files_payload: ((match, [emails]), ...) уже в нужном порядке."""
    matches = [m for m, _ in files_payload]
    rows = []
    for match, emails in files_payload:
        for e in emails:
            rows.append((e, match))
    long = pd.DataFrame(rows, columns=["email", "match"])

    # email x матч -> количество мест
    seats = (
        long.groupby(["email", "match"]).size().unstack(fill_value=0)
        .reindex(columns=matches, fill_value=0)
    )
    summary = pd.DataFrame(index=seats.index)
    summary["Матчей"] = (seats > 0).sum(axis=1)
    summary["Всего мест"] = seats.sum(axis=1)
    summary["Пропущены"] = (seats == 0).apply(
        lambda r: ", ".join(c for c, miss in r.items() if miss), axis=1
    )
    result = summary.join(seats).reset_index().rename(columns={"email": "Email"})
    result = result.sort_values(["Матчей", "Всего мест", "Email"], ascending=[False, False, True])
    return result.reset_index(drop=True), matches


def to_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 8), 60)
            ws.freeze_panes = "B2"
    return buf.getvalue()


# ---------- интерфейс ----------

st.set_page_config(page_title="Анализ покупателей билетов", page_icon="🎟️", layout="wide")
st.title("🎟️ Анализ покупателей билетов")
st.caption(
    "Загрузите xlsx-файлы: 1 файл = 1 матч. Каждая строка с email = одно купленное место."
)

uploaded = st.file_uploader(
    "Файлы матчей", type=["xlsx", "xls", "csv"], accept_multiple_files=True
)

if not uploaded:
    st.info("Загрузите файлы, чтобы начать анализ.")
    st.stop()

# читаем и сортируем матчи по дате из имени файла
parsed = []
for f in uploaded:
    name = match_name(f.name)
    try:
        emails = read_emails(f)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось прочитать «{f.name}»: {exc}")
        continue
    if not emails:
        st.warning(f"В файле «{f.name}» не найдено ни одного email — пропущен.")
        continue
    parsed.append((name, emails, match_date(name)))

names = [p[0] for p in parsed]
if len(names) != len(set(names)):
    st.error("Есть файлы с одинаковыми названиями — переименуйте их, чтобы матчи не слились.")
    st.stop()
if not parsed:
    st.stop()

parsed.sort(key=lambda p: (p[2] is None, p[2] or datetime.min, p[0]))
result, matches = build_tables(tuple((n, tuple(e)) for n, e, _ in parsed))
n = len(matches)

# --- сводка по матчам ---
st.subheader(f"Матчей загружено: {n}")
per_match = pd.DataFrame(
    {
        "Матч": matches,
        "Покупателей (email)": [int((result[m] > 0).sum()) for m in matches],
        "Мест": [int(result[m].sum()) for m in matches],
    }
)
c1, c2, c3 = st.columns(3)
c1.metric("Уникальных email", f"{len(result):,}".replace(",", " "))
c2.metric("Всего мест", f"{int(per_match['Мест'].sum()):,}".replace(",", " "))
c3.metric("Были на всех матчах", int((result["Матчей"] == n).sum()))
st.dataframe(per_match, hide_index=True, width="stretch")

# --- распределение ---
dist = (
    result["Матчей"].value_counts().reindex(range(n, 0, -1), fill_value=0)
    .rename_axis("Матчей посещено").reset_index(name="Email")
)
with st.expander("Распределение: сколько email на скольких матчах"):
    st.bar_chart(dist.set_index("Матчей посещено"))
    st.dataframe(dist, hide_index=True, width="stretch")

# --- уровни n, n-1, n-2 ---
depth = st.slider(
    "Сколько уровней показывать (n, n-1, n-2, ...)",
    min_value=1, max_value=n, value=min(3, n),
)

levels = {}
labels = []
for k in range(depth):
    cnt = n - k
    label = "n" if k == 0 else f"n-{k}"
    df = result[result["Матчей"] == cnt].reset_index(drop=True)
    if k == 0:
        df = df.drop(columns=["Пропущены"])
    levels[f"{label} ({cnt} из {n})"] = df
    labels.append(f"{label} = {cnt} матч. ({len(df)})")

tabs = st.tabs(labels)
for tab, (title, df) in zip(tabs, levels.items()):
    with tab:
        st.markdown(f"**{title}** — email, купившие билеты ровно на {title.split('(')[1].split(' ')[0]} матч(а/ей). "
                    "В колонках матчей — количество мест.")
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button(
            "Скачать список email (.txt)",
            "\n".join(df["Email"]),
            file_name=f"emails_{title.split(' ')[0]}.txt",
            key=f"txt_{title}",
        )

# --- выгрузка ---
st.divider()
report = {"Сводка по матчам": per_match, "Распределение": dist}
report.update(levels)
report["Все email"] = result
st.download_button(
    "📥 Скачать полный отчёт (Excel)",
    to_excel(report),
    file_name="analiz_biletov.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
)
