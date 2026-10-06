"""
Анализ покупателей билетов.

1 файл xlsx = 1 матч.
• Простой формат: одна колонка «Email покупателя» (строка = одно место).
• Подробный формат: Сектор | Ряд | Место | Дата/время продажи | Email покупателя.
  Для него дополнительно ищутся признаки перекупов и связанные аккаунты.

Запуск локально:  streamlit run app.py
"""
import hashlib
import io
import itertools
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
COLS = {"sector": "сектор", "row": "ряд", "seat": "место", "ts": "дата", "email": "email"}


# =====================================================================
# Чтение файлов
# =====================================================================

def match_date(name: str):
    """Дата матча из имени файла: 05_09_26, 05_09_2026, 05.09.2026, 2026-09-05 ..."""
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


def _find_header(raw: pd.DataFrame):
    """Номер строки заголовка и сопоставление колонок, если есть колонка email."""
    for i in range(min(len(raw), 30)):
        cells = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
        mapping = {}
        for key, word in COLS.items():
            for j, c in enumerate(cells):
                if word in c and j not in mapping.values():
                    mapping[key] = j
                    break
        if "email" in mapping:
            return i, mapping
    return None, None


def _parse_ts(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, format="%d.%m.%Y %H:%M", errors="coerce")
    rest = ts.isna() & s.notna()
    if rest.any():
        ts[rest] = pd.to_datetime(s[rest], errors="coerce", dayfirst=True)
    return ts


@st.cache_data(show_spinner=False)
def parse_file(data: bytes, filename: str) -> pd.DataFrame:
    """Строки = места. Колонки: email (+ sector, row, seat, ts, если есть)."""
    if filename.lower().endswith(".csv"):
        sheets = {"csv": pd.read_csv(io.BytesIO(data), header=None, dtype=str)}
    else:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=str)

    parts = []
    for raw in sheets.values():
        hdr, mapping = _find_header(raw)
        if hdr is None:
            # нет заголовка — просто собираем все email из всех ячеек
            emails = []
            for v in raw.to_numpy().ravel():
                if isinstance(v, str):
                    emails.extend(EMAIL_RE.findall(v))
            parts.append(pd.DataFrame({"email": emails}))
            continue
        body = raw.iloc[hdr + 1:]
        out = pd.DataFrame({k: body.iloc[:, j].values for k, j in mapping.items()})
        out["email"] = out["email"].astype("string").str.extract(f"({EMAIL_RE.pattern})", expand=False)
        out = out[out["email"].notna()]
        parts.append(out)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["email"])
    df["email"] = df["email"].astype(str).str.strip().str.lower()
    if {"sector", "row", "seat", "ts"}.issubset(df.columns):
        df["sector"] = df["sector"].astype(str).str.strip()
        df["row"] = pd.to_numeric(df["row"], errors="coerce")
        df["seat"] = pd.to_numeric(df["seat"], errors="coerce")
        df["ts"] = _parse_ts(df["ts"])
    return df.reset_index(drop=True)


def is_detailed(df: pd.DataFrame) -> bool:
    return {"sector", "row", "seat", "ts"}.issubset(df.columns) and df["ts"].notna().any()


# =====================================================================
# Посещаемость: n, n-1, n-2
# =====================================================================

def attendance(all_df: pd.DataFrame, matches: list) -> pd.DataFrame:
    seats = (
        all_df.groupby(["email", "match"]).size().unstack(fill_value=0)
        .reindex(columns=matches, fill_value=0)
    )
    res = pd.DataFrame(index=seats.index)
    res["Матчей"] = (seats > 0).sum(axis=1)
    res["Всего мест"] = seats.sum(axis=1)
    res["Пропущены"] = (seats == 0).apply(lambda r: ", ".join(c for c, miss in r.items() if miss), axis=1)
    res = res.join(seats).reset_index().rename(columns={"email": "Email"})
    return res.sort_values(["Матчей", "Всего мест", "Email"], ascending=[False, False, True]).reset_index(drop=True)


# =====================================================================
# Признаки перекупов
# =====================================================================

def base_sector(s: str) -> str:
    """«Сектор C2 (ограниченная видимость)» — часть сектора C2. Мобильные трибуны оставляем отдельными."""
    return re.sub(r"\s*\(ограниченная видимость\)", "", s, flags=re.I)


def count_clusters(g: pd.DataFrame, row_gap: int, seat_gap: int) -> int:
    """Сколько отдельных групп мест: рядом = тот же сектор, ряд ±row_gap, место ±seat_gap."""
    sec = g["sector"].map(base_sector).tolist()
    rows, seats = g["row"].tolist(), g["seat"].tolist()
    n = len(sec)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if (sec[i] == sec[j] and pd.notna(rows[i]) and pd.notna(rows[j])
                    and abs(rows[i] - rows[j]) <= row_gap and abs(seats[i] - seats[j]) <= seat_gap):
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def place_label(g: pd.DataFrame) -> str:
    """Короткая запись мест: «C22 р.11: 5,6,7; B4 р.12: 1»."""
    parts = []
    for (s, r), x in g.groupby(["sector", "row"], sort=True):
        seats = ",".join(str(int(v)) for v in sorted(x["seat"].dropna()))
        parts.append(f"{s.replace('Сектор ', '')} р.{int(r) if pd.notna(r) else '?'}: {seats}")
    return "; ".join(parts)


@st.cache_data(show_spinner=False)
def per_match_stats(det: pd.DataFrame, row_gap: int, seat_gap: int) -> pd.DataFrame:
    rows = []
    for (email, match), g in det.groupby(["email", "match"], sort=False):
        rows.append({
            "email": email, "match": match,
            "seats": len(g),
            "orders": g["ts"].nunique(),
            "clusters": count_clusters(g, row_gap, seat_gap),
            "span_d": (g["ts"].max() - g["ts"].min()).total_seconds() / 86400,
        })
    return pd.DataFrame(rows)


def score_emails(pm: pd.DataFrame, t: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    pm = pm.copy()
    pm["f_blocks"] = pm["clusters"] >= t["blocks"]
    pm["f_span"] = (pm["orders"] >= 2) & (pm["span_d"] >= t["span"])
    pm["f_orders"] = pm["orders"] >= t["orders"]
    pm["f_scatter"] = (pm["seats"] >= t["seats"]) & (pm["clusters"] >= 2)
    e = pm.groupby("email").agg(
        matches=("match", "nunique"), seats=("seats", "sum"),
        orders=("orders", "sum"),
        n_blocks=("f_blocks", "sum"), n_span=("f_span", "sum"),
        n_orders=("f_orders", "sum"), n_scatter=("f_scatter", "sum"),
    ).reset_index()
    any_flag = pm[["f_blocks", "f_span", "f_orders", "f_scatter"]].any(axis=1)
    e = e.merge(pm[any_flag].groupby("email").size().rename("flag_matches"), on="email", how="left")
    e["flag_matches"] = e["flag_matches"].fillna(0).astype(int)
    e["score"] = 2 * (e["n_blocks"] + e["n_span"] + e["n_orders"]) + e["n_scatter"]
    e["seats_per_match"] = (e["seats"] / e["matches"]).round(1)
    return e.sort_values(["score", "seats"], ascending=False).reset_index(drop=True), pm


# =====================================================================
# Связанные аккаунты (справочно, без баллов)
# =====================================================================

def _one_edit(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


@st.cache_data(show_spinner=False)
def linked_accounts(det: pd.DataFrame, emails: tuple, minutes: int, seat_gap: int) -> pd.DataFrame:
    links = []

    # 1) соседние места с разных email, купленные с разницей не больше N минут
    d = det.dropna(subset=["row", "seat", "ts"]).sort_values(["match", "sector", "row", "seat"])
    nxt = d.groupby(["match", "sector", "row"]).shift(-1)
    mask = (
        nxt["email"].notna() & (nxt["email"] != d["email"])
        & ((nxt["seat"] - d["seat"]) <= seat_gap)
        & ((nxt["ts"] - d["ts"]).abs() <= pd.Timedelta(minutes=minutes))
    )
    adj = pd.DataFrame({"a": d.loc[mask, "email"], "b": nxt.loc[mask, "email"], "match": d.loc[mask, "match"]})
    if len(adj):
        adj[["a", "b"]] = pd.DataFrame(
            [sorted(p) for p in adj[["a", "b"]].values], index=adj.index)
        g = adj.groupby(["a", "b"])["match"].agg(lambda s: ", ".join(sorted(set(s)))).reset_index()
        for a, b, ms in g.values:
            links.append((a, b, f"соседние места: {ms}"))

    # 2) похожие имена ящиков
    local = {}
    for e in emails:
        local.setdefault(e.split("@")[0], []).append(e)
    for lp, group in local.items():
        for a, b in itertools.combinations(sorted(group), 2):
            links.append((a, b, "одинаковое имя ящика, разные домены"))
    by_prefix = {}
    for lp in local:
        if len(lp) >= 6:
            by_prefix.setdefault(lp[:3], []).append(lp)
    for group in by_prefix.values():
        for x, y in itertools.combinations(group, 2):
            if _one_edit(x, y):
                for a in local[x]:
                    for b in local[y]:
                        links.append((min(a, b), max(a, b), "имя ящика отличается на 1 символ"))

    if not links:
        return pd.DataFrame(columns=["a", "b", "reason"])
    out = pd.DataFrame(links, columns=["a", "b", "reason"])
    return out.groupby(["a", "b"])["reason"].agg(lambda s: "; ".join(dict.fromkeys(s))).reset_index()


def links_for(email: str, links: pd.DataFrame) -> pd.DataFrame:
    l1 = links[links["a"] == email].rename(columns={"b": "Связанный email"})
    l2 = links[links["b"] == email].rename(columns={"a": "Связанный email"})
    return pd.concat([l1, l2])[["Связанный email", "reason"]].rename(columns={"reason": "Почему"})


# =====================================================================
# Выгрузка
# =====================================================================

def to_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            sn = name[:31]
            df.to_excel(writer, sheet_name=sn, index=False)
            ws = writer.sheets[sn]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 8), 60)
            ws.freeze_panes = "B2"
    return buf.getvalue()


# =====================================================================
# Интерфейс
# =====================================================================

st.set_page_config(page_title="Анализ покупателей билетов", page_icon="🎟️", layout="wide")
st.title("🎟️ Анализ покупателей билетов")
st.caption("1 файл = 1 матч. Если в файлах есть сектор, ряд, место и время продажи — "
           "приложение дополнительно ищет признаки перекупов.")

uploaded = st.file_uploader("Файлы матчей (xlsx)", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
if not uploaded:
    st.info("Загрузите файлы, чтобы начать анализ.")
    st.stop()

# ---------- чтение ----------
files = []
for f in uploaded:
    data = f.getvalue()
    try:
        df = parse_file(data, f.name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось прочитать «{f.name}»: {exc}")
        continue
    if df.empty:
        st.warning(f"В файле «{f.name}» нет ни одного email — файл пропущен.")
        continue
    files.append({"name": Path(f.name).stem, "date": match_date(f.name), "df": df, "detailed": is_detailed(df)})

if not files:
    st.stop()

# один и тот же матч в двух файлах (например, старая и новая выгрузка) — берём подробную
by_key = {}
for fi in files:
    key = fi["date"] or fi["name"]
    if key in by_key:
        keep = max(by_key[key], fi, key=lambda x: (x["detailed"], len(x["df"])))
        st.warning(f"Файлы «{by_key[key]['name']}» и «{fi['name']}» относятся к одному матчу. "
                   f"Используется «{keep['name']}».")
        by_key[key] = keep
    else:
        by_key[key] = fi
files = sorted(by_key.values(), key=lambda x: (x["date"] is None, x["date"] or datetime.min, x["name"]))
for fi in files:
    fi["label"] = fi["date"].strftime("%d.%m.%Y") if fi["date"] else fi["name"]
    fi["df"] = fi["df"].assign(match=fi["label"])

matches = [fi["label"] for fi in files]
n = len(matches)
all_df = pd.concat([fi["df"] for fi in files], ignore_index=True)
det_files = [fi for fi in files if fi["detailed"]]
det = pd.concat([fi["df"] for fi in det_files], ignore_index=True) if det_files else None

# ---------- настройки ----------
with st.sidebar:
    st.header("Пороги признаков")
    st.caption("Признаки считаются по каждому матчу отдельно.")
    t = {
        "blocks": st.number_input("Разброс: групп мест на матч, от", 2, 20, 3,
                                  help="Сколько отдельных, не соседних групп мест купил один email на один матч."),
        "span": st.number_input("Растянутость: дней между первым и последним заказом, от", 0.5, 10.0, 1.0, 0.5),
        "orders": st.number_input("Заказов на матч, от", 2, 20, 3,
                                  help="Заказ = все места, купленные с одного email в одну и ту же минуту."),
        "seats": st.number_input("Мест на матч не рядом, от", 3, 30, 5,
                                 help="5+ мест, разбитых минимум на 2 группы. Компания, сидящая вместе, не считается."),
    }
    st.subheader("Что считать «рядом»")
    row_gap = st.number_input("Соседние ряды: разница до", 0, 3, 1)
    seat_gap = st.number_input("Соседние места: разница номеров до", 1, 5, 2)
    st.subheader("Связанные аккаунты")
    link_min = st.number_input("Соседние места с разных email: разница во времени до, мин", 1, 60, 5)

# ---------- сводка ----------
att = attendance(all_df, matches)
per_match = pd.DataFrame({
    "Матч": matches,
    "Покупателей (email)": [int((att[m] > 0).sum()) for m in matches],
    "Мест": [int(att[m].sum()) for m in matches],
    "Подробные данные": ["да" if fi["detailed"] else "нет" for fi in files],
})
c1, c2, c3 = st.columns(3)
c1.metric("Матчей", n)
c2.metric("Уникальных email", f"{len(att):,}".replace(",", " "))
c3.metric("Всего мест", f"{int(per_match['Мест'].sum()):,}".replace(",", " "))

tab_sus, tab_person, tab_links, tab_att, tab_matches = st.tabs(
    ["🚩 Подозрительные", "🔎 Покупатель", "🔗 Связанные аккаунты", "📅 Посещаемость (n, n-1, n-2)", "📊 Матчи"]
)

with tab_matches:
    st.dataframe(per_match, hide_index=True, width="stretch")
    dist = (att["Матчей"].value_counts().reindex(range(n, 0, -1), fill_value=0)
            .rename_axis("Матчей посещено").reset_index(name="Email"))
    st.markdown("**Сколько email на скольких матчах**")
    st.bar_chart(dist.set_index("Матчей посещено"))

# ---------- посещаемость ----------
levels = {}
with tab_att:
    depth = st.slider("Сколько уровней показывать (n, n-1, n-2, …)", 1, n, min(3, n))
    labels = []
    for k in range(depth):
        cnt = n - k
        tag = "n" if k == 0 else f"n-{k}"
        df_k = att[att["Матчей"] == cnt].reset_index(drop=True)
        if k == 0:
            df_k = df_k.drop(columns=["Пропущены"])
        levels[f"{tag} ({cnt} из {n})"] = df_k
        labels.append(f"{tag} = {cnt} матч. ({len(df_k)})")
    for sub, (title, df_k) in zip(st.tabs(labels), levels.items()):
        with sub:
            st.caption(f"Email, купившие билеты ровно на {title.split('(')[1].split(' ')[0]} матч(а/ей). "
                       "В колонках матчей — количество мест.")
            st.dataframe(df_k, hide_index=True, width="stretch")
            st.download_button("Скачать список email (.txt)", "\n".join(df_k["Email"]),
                               file_name=f"emails_{title.split(' ')[0]}.txt", key=f"txt_{title}")

# ---------- перекупы ----------
report = {"Матчи": per_match}
if det is None:
    for tab in (tab_sus, tab_person, tab_links):
        with tab:
            st.info("Для поиска перекупов нужны файлы с колонками Сектор, Ряд, Место, "
                    "Дата/время продажи, Email покупателя. В загруженных файлах есть только email.")
else:
    if len(det_files) < n:
        st.warning("Признаки перекупов считаются только по матчам с подробными данными: "
                   + ", ".join(fi["label"] for fi in det_files))

    with st.spinner("Считаю признаки…"):
        pm = per_match_stats(det, int(row_gap), int(seat_gap))
        emails_scored, pm = score_emails(pm, t)
        links = linked_accounts(det, tuple(sorted(det["email"].unique())), int(link_min), int(seat_gap))

    link_count = pd.concat([links["a"], links["b"]]).value_counts()
    emails_scored["linked"] = emails_scored["email"].map(link_count).fillna(0).astype(int)

    sus_view = emails_scored.rename(columns={
        "email": "Email", "score": "Балл", "flag_matches": "Матчей с признаками",
        "matches": "Матчей", "seats": "Мест", "seats_per_match": "Мест на матч", "orders": "Заказов",
        "n_blocks": "Разброс по блокам", "n_span": "Растянуто по дням",
        "n_orders": "Много заказов", "n_scatter": "5+ мест не рядом", "linked": "Связанных аккаунтов",
    })[["Email", "Балл", "Матчей с признаками", "Матчей", "Мест", "Мест на матч", "Заказов",
        "Разброс по блокам", "Растянуто по дням", "Много заказов", "5+ мест не рядом", "Связанных аккаунтов"]]

    with tab_sus:
        st.markdown(
            "Балл = 2 × (разброс по блокам + растянуто по дням + много заказов) + 5+ мест не рядом. "
            "В каждой колонке признака — **на скольких матчах** он сработал. "
            "Связанные аккаунты в балл не входят."
        )
        f1, f2 = st.columns(2)
        min_score = f1.number_input("Показывать с баллом от", 0, 200, 4)
        min_fm = f2.number_input("Признаки сработали минимум на скольких матчах", 0, n, 1)
        shown = sus_view[(sus_view["Балл"] >= min_score) & (sus_view["Матчей с признаками"] >= min_fm)]
        st.caption(f"Найдено: {len(shown)}")
        st.dataframe(
            shown, hide_index=True, width="stretch",
            column_config={"Балл": st.column_config.ProgressColumn(
                "Балл", min_value=0, max_value=int(max(sus_view["Балл"].max(), 1)), format="%d")},
        )
        st.download_button("Скачать список email (.txt)", "\n".join(shown["Email"]), file_name="suspicious.txt")

    # ---------- карточка покупателя ----------
    with tab_person:
        options = list(emails_scored["email"])
        default = options.index(shown["Email"].iloc[0]) if len(shown) else 0
        who = st.selectbox("Email", options, index=default,
                           help="Список отсортирован по баллу. Можно начать вводить адрес.")
        row = sus_view[sus_view["Email"] == who].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Балл", int(row["Балл"]))
        m2.metric("Матчей", int(row["Матчей"]))
        m3.metric("Мест", int(row["Мест"]))
        m4.metric("Заказов", int(row["Заказов"]))

        st.markdown("**По матчам**")
        pme = pm[pm["email"] == who].copy()
        pme["order"] = pme["match"].map({m: i for i, m in enumerate(matches)})
        pme = pme.sort_values("order")
        mark = lambda b: "🚩" if b else ""  # noqa: E731
        st.dataframe(pd.DataFrame({
            "Матч": pme["match"], "Мест": pme["seats"], "Заказов": pme["orders"],
            "Групп мест": pme["clusters"], "Дней от 1-го до последнего заказа": pme["span_d"].round(1),
            "Разброс": pme["f_blocks"].map(mark), "Растянуто": pme["f_span"].map(mark),
            "Много заказов": pme["f_orders"].map(mark), "5+ не рядом": pme["f_scatter"].map(mark),
        }), hide_index=True, width="stretch")

        st.markdown("**Все заказы по времени**")
        orders = (det[det["email"] == who].groupby(["ts", "match"])
                  .apply(lambda g: pd.Series({"Мест": len(g), "Места": place_label(g)}), include_groups=False)
                  .reset_index().sort_values("ts"))
        orders["ts"] = orders["ts"].dt.strftime("%d.%m.%Y %H:%M")
        st.dataframe(orders.rename(columns={"ts": "Время покупки", "match": "Матч"}),
                     hide_index=True, width="stretch")

        st.markdown("**Вероятно связанные аккаунты**")
        lf = links_for(who, links)
        if lf.empty:
            st.caption("Не найдено.")
        else:
            lf = lf.merge(sus_view[["Email", "Балл", "Матчей", "Мест"]],
                          left_on="Связанный email", right_on="Email", how="left").drop(columns="Email")
            st.dataframe(lf, hide_index=True, width="stretch")

    # ---------- связанные аккаунты ----------
    with tab_links:
        st.markdown(
            "Справочно, без баллов. Пара попадает сюда, если:\n"
            f"- с разных email куплены соседние места в одном ряду с разницей до {int(link_min)} мин;\n"
            "- или имена ящиков совпадают на разных доменах либо отличаются на 1 символ.\n\n"
            "Это может быть компания, которая покупает с двух телефонов, обход лимита мест в заказе "
            "или один человек с несколькими ящиками."
        )
        lv = links.rename(columns={"a": "Email 1", "b": "Email 2", "reason": "Почему"})
        score_map = emails_scored.set_index("email")["score"]
        lv["Балл 1"] = lv["Email 1"].map(score_map)
        lv["Балл 2"] = lv["Email 2"].map(score_map)
        lv["Макс. балл"] = lv[["Балл 1", "Балл 2"]].max(axis=1)
        thr = st.number_input("Показывать пары, где хотя бы у одного email балл от (0 — все пары)", 0, 200, 4)
        lv = lv[lv["Макс. балл"] >= thr]
        lv = lv.sort_values(["Макс. балл", "Email 1"], ascending=[False, True])
        st.caption(f"Пар: {len(lv)}")
        st.dataframe(lv.drop(columns="Макс. балл"), hide_index=True, width="stretch")

    report["Подозрительные"] = sus_view
    report["Связанные аккаунты"] = lv.drop(columns="Макс. балл")
    report["Признаки по матчам"] = pm.rename(columns={
        "email": "Email", "match": "Матч", "seats": "Мест", "orders": "Заказов", "clusters": "Групп мест",
        "span_d": "Дней между заказами", "f_blocks": "Разброс", "f_span": "Растянуто",
        "f_orders": "Много заказов", "f_scatter": "5+ не рядом"})

report.update(levels)
report["Все email"] = att
st.divider()
st.download_button(
    "📥 Скачать полный отчёт (Excel)", to_excel(report), file_name="analiz_biletov.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
)
