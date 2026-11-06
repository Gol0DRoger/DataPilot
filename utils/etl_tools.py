import ast
import builtins
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

READERS = {".csv": pd.read_csv, ".xlsx": pd.read_excel, ".parquet": pd.read_parquet}

# Generated code only gets these builtins (no open, no __import__, ...)
SAFE_BUILTINS = {name: getattr(builtins, name) for name in (
    "len range str int float bool list dict set tuple min max sum sorted abs round "
    "enumerate zip isinstance any all map filter reversed print "
    "Exception ValueError KeyError TypeError").split()}
BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr",
                 "delattr", "globals", "locals", "vars", "input", "os", "sys", "subprocess"}
BLOCKED_ATTRS = {"eval", "io", "ExcelWriter", "to_csv", "to_excel", "to_json", "to_parquet",
                 "to_pickle", "to_sql", "to_hdf", "to_feather", "to_clipboard"}


def load_dataframe(path) -> pd.DataFrame:
    """Loads a CSV / Excel (.xlsx) / JSON / Parquet file into a DataFrame."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    if ext not in READERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return READERS[ext](path)


def save_dataframe(df, folder, stem, fmt) -> Path:
    """Saves a DataFrame as csv / xlsx / json / parquet. Relative folders are relative to the project root."""
    fmt = fmt.lower().lstrip(".")
    fmt = "xlsx" if fmt == "excel" else fmt
    folder = PROJECT_ROOT / folder  # absolute paths stay unchanged
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{stem}.{fmt}"
    if fmt == "csv":
        df.to_csv(out, index=False)
    elif fmt == "xlsx":
        df.to_excel(out, index=False)
    elif fmt == "json":
        df.to_json(out, orient="records", lines=True)
    elif fmt == "parquet":
        df.to_parquet(out, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return out


def json_to_dataframe(data) -> pd.DataFrame:
    """Turns an API JSON response into a DataFrame (works for lists and for {'results': [...]}-style objects)."""
    if isinstance(data, list):
        return pd.json_normalize(data)
    if isinstance(data, dict):
        for key in ("results", "data", "items", "records"):
            if isinstance(data.get(key), list):
                return pd.json_normalize(data[key])
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return pd.json_normalize(value)
        return pd.json_normalize(data)
    raise ValueError("The API did not return JSON objects or a list of them")


def check_code(code: str):
    """Guard rail for LLM-generated code. NOT a real sandbox - use a container/subprocess before deploying publicly."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed")
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError(f"'{node.id}' is not allowed")
        if isinstance(node, ast.Attribute) and (
                node.attr.startswith(("_", "read_")) or node.attr in BLOCKED_ATTRS):
            raise ValueError(f"'.{node.attr}' is not allowed")


class ETLTools:

    def __init__(self):
        pass

    def extract_load(self, url: str, output_folder: str, format: str):
        """
        This tool extracts the data from the API (url) and loads it into the
        the desired location (output_folder).

        Args:
            url (str): The API endpoint from which to extract data.
            output_folder (str): The folder where the extracted data will be saved.
            format (str): csv, xlsx, json or parquet.

        Returns:
            str: A message indicating the success or failure of the operation.
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            df = json_to_dataframe(response.json())
            out = save_dataframe(df, output_folder, "extracted_data", format)
            return f"Data successfully extracted ({len(df)} rows) and saved to {out}"
        except (requests.exceptions.RequestException, ValueError) as e:
            return f"Failed to extract data: {e}"

    def transform_load_context(self, file_path: str):
        """
        Describes the data in a file (columns, types and the first 3 rows) so the LLM can write the Pandas code.

        Args:
            file_path (str): Path to a CSV / Excel / JSON / Parquet file.
        Returns:
            str: Columns, data types and the first 3 rows.
        """
        df = load_dataframe(file_path)
        return (f"Columns: {list(df.columns)}\n"
                f"Dtypes:\n{df.dtypes.to_string()}\n"
                f"First rows:\n{df.head(3).to_string(max_colwidth=40)}")

    def execute_code(self, code: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the generated Pandas code on a copy of `df` and returns the DataFrame the code stored in `result_df`.
        """
        check_code(code)
        scope = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np, "df": df.copy()}
        exec(code, scope)
        result = scope.get("result_df")
        if not isinstance(result, pd.DataFrame):
            raise ValueError("The code did not produce a DataFrame named result_df")
        return result

    def transform_load(self, input_file_path: str, output_folder: str, output_format: str, code: str):
        """Loads the file, runs the code on it and saves the result (named <input>_transformed_<timestamp>)."""
        try:
            df = load_dataframe(input_file_path)
            result = self.execute_code(code, df)
            stem = f"{Path(input_file_path).stem}_transformed_{time.strftime('%Y%m%d_%H%M%S')}"
            out = save_dataframe(result, output_folder, stem, output_format)
        except Exception as e:
            return f"Transformation failed: {e}\n\nPandas code:\n{code}"
        return f"Saved {len(result)} rows (from {len(df)}) to {out}\n\nPandas code:\n{code}"


if __name__ == "__main__":
    obj = ETLTools()
    #print(obj.extract_load("https://pokeapi.co/api/v2/pokemon", "data/extract", "csv"))
    path = PROJECT_ROOT / "data" / "extract" / "extracted_data.csv"
    print(obj.transform_load_context(path))