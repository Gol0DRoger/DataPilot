from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agents.data_agent import data_agent
from utils.database import DatabaseUtil, build_url, default_url_from_env
from utils.etl_tools import load_dataframe

ROOT = Path(__file__).resolve().parent
UPLOADS, EXTRACT, OUT = (ROOT / "data" / d for d in ("uploads", "extract", "transform"))
for folder in (UPLOADS, EXTRACT, OUT):
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Data Agent", page_icon="🗄️", layout="wide")
st.title("🗄️ Data Agent")
st.caption("Ask questions about your database (SQL), or extract / transform data from an API or a file (ETL).")

st.session_state.setdefault("history", [])

# ------------------------------------ SIDEBAR ------------------------------------ #

with st.sidebar:

    # ---- Database (used for SQL questions) ----
    st.header("Database (SQL)")
    db_type = st.selectbox("Type", ["PostgreSQL", "SQLite"])
    db_url = ""

    try:
        if db_type == "SQLite":
            uploaded_db = st.file_uploader("SQLite file", type=["db", "sqlite", "sqlite3"])
            db_path = st.text_input("...or path to a local .db file")
            if uploaded_db:
                saved_db = UPLOADS / uploaded_db.name
                saved_db.write_bytes(uploaded_db.getbuffer())
                db_path = str(saved_db)
            if db_path:
                db_url = build_url("sqlite", database=Path(db_path).as_posix())

        elif st.checkbox("Use the settings from .env", value=True):
            db_url = default_url_from_env()

        else:
            host = st.text_input("Host", "localhost")
            port = st.number_input("Port", value=5432, step=1)
            database = st.text_input("Database")
            user = st.text_input("User")
            password = st.text_input("Password", type="password")
            db_url = build_url("postgresql", host, int(port), database, user, password)
    except KeyError as e:
        st.error(f"Missing value in .env: {e}")

    if not db_url:
        st.warning("No database selected - SQL questions will use the PostgreSQL settings from .env.")
    elif st.button("Test connection"):
        try:
            DatabaseUtil(db_url).test()
            st.success("Connected")
        except Exception as e:
            st.error(str(e)[:300])

    # ---- Data file (used for ETL tasks) ----
    st.header("Data file (ETL)")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    existing = {f"{p.parent.name}/{p.name}": p
                for folder in (UPLOADS, EXTRACT, OUT)
                for p in sorted(folder.glob("*")) if p.suffix.lower() in (".csv", ".xlsx", ".json")}
    picked = st.selectbox("...or use an existing file", ["(none)"] + list(existing))

    file_path = None
    if uploaded_file:
        file_path = UPLOADS / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())
    elif picked != "(none)":
        file_path = existing[picked]

    if file_path:
        try:
            with st.expander(f"Preview: {file_path.name}"):
                st.dataframe(load_dataframe(file_path).head(10))
        except Exception as e:
            st.error(f"Could not read the file: {e}")

    out_fmt = st.radio("Export format", ["csv", "xlsx"], horizontal=True)

    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

# ------------------------------------ CHAT HISTORY ------------------------------------ #

for i, turn in enumerate(st.session_state.history):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("route"):
            st.caption(f"Route: {turn['route'].upper()}")
        if turn.get("details"):
            with st.expander("Show SQL / Pandas code"):
                st.code(turn["details"], language="sql" if turn["route"] == "sql" else None)
        for f in turn.get("files", []):
            p = Path(f)
            st.download_button(f"Download {p.name}", p.read_bytes(), file_name=p.name, key=f"download_{i}_{p.name}")

# ------------------------------------ NEW REQUEST ------------------------------------ #

if prompt := st.chat_input("Ask a database question, or describe an ETL task (e.g. Paste an API URL)"):

    # Previous turns (last 6 messages only, to protect the Groq token limit) - built BEFORE adding the new prompt
    past = [(HumanMessage if t["role"] == "user" else AIMessage)(content=t["content"])
            for t in st.session_state.history[-6:]]
    st.session_state.history.append({"role": "user", "content": prompt})

    # Extra info for the ETL agent only (the router never sees it)
    hint = [f"Save transformed files in this folder: {OUT.as_posix()} (format: {out_fmt}).",
            f"Save API extractions in this folder: {EXTRACT.as_posix()} (format: csv)."]
    if file_path:
        hint.insert(0, f"Input file: {file_path.resolve().as_posix()}")

    def snapshot():
        return {p: p.stat().st_mtime for folder in (EXTRACT, OUT) for p in folder.glob("*")}

    before = snapshot()

    reply = {"role": "assistant"}
    with st.spinner("Working..."):
        try:
            result = data_agent.invoke(
                {"messages": past + [HumanMessage(content=prompt)],
                 "route_response": "",
                 "db_url": db_url,
                 "etl_hint": "\n".join(hint)},
                config={"recursion_limit": 25},
            )
            reply.update(content=result["messages"][-1].content,
                         route=result["route_response"],
                         details=result.get("details", ""))
        except Exception as e:
            reply.update(content=f"Something went wrong: {e}", route="", details="")

    after = snapshot()
    reply["files"] = [str(p) for p, mtime in after.items() if mtime > before.get(p, 0)]  # new / updated files
    st.session_state.history.append(reply)
    st.rerun()
