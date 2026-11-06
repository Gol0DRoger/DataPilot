import os
import sqlite3
from pathlib import Path
from urllib.parse import quote

import psycopg2


def build_url(db_type, host="localhost", port=5432, database="", user="", password=""):
    """Builds a connection URL: sqlite:///<file path> or postgresql://user:pass@host:port/db"""
    if db_type == "sqlite":
        return f"sqlite:///{database}"
    return (f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
            f"@{host}:{port}/{quote(database, safe='')}")


def default_url_from_env():
    """Builds the PostgreSQL URL from the existing .env variables (host, port, database, user, password)."""
    return build_url(
        "postgresql",
        os.environ.get("host", "localhost"),
        int(os.environ.get("port", 5432)),
        os.environ["database"],
        os.environ["user"],
        os.environ["password"],
    )


class DatabaseUtil:
    """Read-only access to a PostgreSQL or SQLite database. Create one object per use."""

    def __init__(self, db_url):
        self.is_sqlite = db_url.startswith("sqlite:///")

        if self.is_sqlite:
            path = Path(db_url[len("sqlite:///"):]).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"SQLite file not found: {path}")
            # mode=ro -> SQLite itself refuses any write
            self.connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        else:
            self.connection = psycopg2.connect(db_url)
            self.connection.set_session(readonly=True)  # PostgreSQL refuses any write

    def test(self):
        cursor = self.connection.cursor()
        try:
            cursor.execute("SELECT 1")
        finally:
            cursor.close()
            self.connection.close()

    def schema_details(self, schema_name="public"):

        connection = self.connection
        cursor = connection.cursor()

        if self.is_sqlite:
            schema_info_context = "Database type: SQLite\n"
        else:
            schema_info_context = f"Database type: PostgreSQL\nDatabase Schema: {schema_name}\n"

        try:

            if self.is_sqlite:
                cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%';")
            else:
                cursor.execute("SELECT table_name from information_schema.tables where table_schema = %s;", (schema_name,))
            tables_list = cursor.fetchall()

            for table in tables_list:
                table_name = table[0]
                schema_info_context = f"{schema_info_context}\nTable: {table_name}\n"

                # Adding Columns & Data Types
                if self.is_sqlite:
                    cursor.execute(f'PRAGMA table_info("{table_name}");')
                    columns_list = [(c[1], c[2]) for c in cursor.fetchall()]  # (name, type)
                    full_table_name = f'"{table_name}"'
                else:
                    cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = %s AND table_name = %s;", (schema_name, table_name))
                    columns_list = cursor.fetchall()
                    full_table_name = f'{schema_name}."{table_name}"'

                for column in columns_list:
                    column_name = column[0]
                    data_type = column[1]
                    schema_info_context = f"{schema_info_context}  Column: {column_name}, Data Type: {data_type}\n"

                # Adding Sample Data
                cursor.execute(f"SELECT * FROM {full_table_name} LIMIT 5;")
                sample_data = cursor.fetchall()
                schema_info_context = f"{schema_info_context}  Sample Data:\n"
                for row in sample_data:
                    schema_info_context = f"{schema_info_context}    {row}\n"

        except Exception as e:
            print(f"Error fetching schema details: {e}")
            schema_info_context = f"Error fetching schema details: {e}"

        finally:
            cursor.close()
            connection.close()

        return schema_info_context

    def execute_sql(self, query):
        """Runs a query and returns the column names + up to 100 rows as a string.
        Errors are raised (sql_analyst.execute_sql catches them and reports them)."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(100)
            return str({"columns": columns, "rows": rows})[:4000]
        finally:
            cursor.close()
            self.connection.close()