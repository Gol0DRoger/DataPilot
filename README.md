# Data Pilot

A multi-agent data assistant built with **LangGraph**. Ask a question in plain English and the agent decides whether it is a **SQL** task (query a database) or an **ETL** task (extract from an API, or transform a CSV / Excel file), then runs it with a specialised sub-agent. A **Streamlit** chat UI sits on top.

## Working

https://github.com/user-attachments/assets/8a024b2d-f382-49da-ae1d-3ac184be0e93
<img width="959" height="500" alt="Screenshot 2026-11-06 132742" src="https://github.com/user-attachments/assets/7fb52170-5480-433b-af22-9754fbf656df" />
<img width="959" height="421" alt="Screenshot 2026-11-06 122116" src="https://github.com/user-attachments/assets/dba4ae2c-aa53-4249-aed4-69e5ea81311c" />

## Features

- **Smart routing:** a router node classifies each request as `sql` or `etl`.
- **SQL Analyst:** natural language → SQL for **PostgreSQL** and **SQLite**, with a read-only connection and a two-step safety check (rule check + LLM judge) before anything runs.
- **ETL Analyst:** extract data from any JSON API, upload your own **CSV / Excel** files, transform them with LLM-generated Pandas code, and export as **CSV or Excel**.
- **Chat memory:** follow-ups such as "now save that as Excel" work (recent history is passed to the agents).
- **Transparent:** every answer shows its route (SQL / ETL) and the SQL or Pandas code that was executed.
- **Free-tier friendly:** runs on Groq-hosted `gpt-oss` models, with cheaper models picked for simpler steps.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Streamlit UI  (app.py)                  │
│     chat · CSV/Excel upload · database picker · download   │
└──────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────┐
│                Data Agent  (LangGraph router)              │
│          Classifies every request as  sql  or  etl         │
└───────────────┬─────────────────────────────┬──────────────┘
                │ sql                         │ etl
                ▼                             ▼
      ┌──────────────────┐          ┌──────────────────┐
      │   SQL Analyst    │          │   ETL Analyst    │
      └──────────────────┘          └──────────────────┘
                │                             │
                ├─► Query curation            ├─► LLM picks a tool
                ├─► Schema context            │     ├─► Extract & load (API → file)
                ├─► SQL generation            │     └─► Transform & load (file → file)
                ├─► Safety check              │           ├─► Pandas code generation
                │     ├─ safe   → execute     │           ├─► Code guard + execution
                │     └─ unsafe → cancel      │           └─► Save as CSV / Excel
                ├─► Read-only execution       └─► Final answer
                └─► Final answer

        PostgreSQL / SQLite                   data/extract · data/transform
```

## Tech stack

Python · LangGraph · LangChain · Groq (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`) · Pandas · PostgreSQL (`psycopg2`) · SQLite · Streamlit

## Project structure

```
Data_agent/
├── agents/
│   ├── data_agent.py     # parent graph: router → SQL or ETL
│   ├── sql_analyst.py    # SQL sub-agent graph
│   └── etl_analyst.py    # ETL sub-agent graph (tool loop)
├── Models/schema.py      # Pydantic state schemas
├── utils/
│   ├── database.py       # PostgreSQL / SQLite access (read-only)
│   ├── etl_tools.py      # extract, load, transform, code guard
│   └── llm_pick.py       # picks the Groq model per task level
├── data/
│   ├── extract/          # API extractions
│   ├── transform/        # transformed output files
│   └── uploads/          # files uploaded in the UI (git-ignored)
├── app.py                # Streamlit UI
├── main.py               # simple CLI test script
└── feed_db.py            # optional: loads a sample ride-sharing database
```

## Setup

**1. Clone and install** (uses [uv](https://docs.astral.sh/uv/))

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
uv sync
```

**2. Configure environment variables**

Copy `.env.example` to `.env` and fill in your values. A free Groq key is available at [console.groq.com](https://console.groq.com).

```
GROQ_API_KEY=your_groq_api_key
host=localhost
port=5432
database=postgres
user=postgres
password=your_password
```

The database settings are only needed for PostgreSQL. For SQLite, upload a `.db` file in the UI.

**3. Run the app**

```bash
uv run streamlit run app.py
```

## Usage examples

| Type in the chat | What happens |
|---|---|
| `What payment methods do we have in the database?` | Routed to **SQL**. The SQL is generated, checked, run read-only, and answered in plain English. |
| `Extract data from https://pokeapi.co/api/v2/pokemon?limit=100 and save it as csv` | Routed to **ETL**. The API response is saved to `data/extract/extracted_data.csv`. |
| *(select a file in the sidebar)* `Keep only rows where name contains charizard` | Routed to **ETL**. Pandas code is generated and run, and the result is saved and offered for download. |
| `Now save that as Excel` | Uses the chat history for the follow-up. |

Optional: `python feed_db.py` loads a sample ride-sharing database (`users`, `vehicles`, `rides`, `payments`, `ratings`) from CSV files placed in `data/`.

## Safety

- **SQL:** only a single `SELECT` / `WITH` statement is allowed. The connection is read-only (enforced by PostgreSQL and SQLite themselves), and an LLM judge reviews the query as well.
- **ETL:** generated Pandas code runs with restricted builtins, and imports, file access and `eval`-style calls are rejected.
- **Secrets:** credentials live in `.env`, which is git-ignored.

## Limitations

- The ETL code guard is a guard rail, **not a full sandbox**. Run generated code in a container or subprocess before any public deployment.
- Groq's free tier has a tokens-per-minute limit, so rapid repeated requests may be rate-limited.
- A new API extraction overwrites the previous `extracted_data` file.
- Only PostgreSQL and SQLite are supported (MySQL is not implemented).
