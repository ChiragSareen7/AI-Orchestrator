from __future__ import annotations  # allows modern type hint syntax on older Python

import pandas as pd          # pandas: the standard Python library for working with tabular data (like Excel/CSV)
from datasets import Dataset  # HuggingFace datasets library — the format trainers expect


def _read_csv(path: str) -> pd.DataFrame:
    # reads a CSV file into a pandas DataFrame
    # handles encoding issues — CSV files can be saved in different character encodings
    try:
        return pd.read_csv(path, encoding="utf-8")
        # utf-8 is the modern standard — try this first
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")
        # latin-1 (also called ISO-8859-1) is an older encoding
        # some CSV files exported from Excel use this; it can read any byte sequence


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # standardizes column names: lowercase, no spaces, no leading/trailing whitespace
    # why? — CSV column names can be inconsistent ("Question", "question", "Question ")
    df = df.copy()  # work on a copy to avoid modifying the original DataFrame
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    # for each column name:
    #   str(c) = convert to string (in case column names are numbers)
    #   .strip() = remove leading/trailing spaces
    #   .lower() = make lowercase: "Question" → "question"
    #   .replace(" ", "_") = replace spaces with underscores: "chapter no" → "chapter_no"
    return df


def build_qa_from_plain(df: pd.DataFrame) -> pd.DataFrame:
    # builds question-answer pairs from a simple CSV with "question" and "answer" columns
    # used for the Python FAQ dataset (faq_dataset.csv)
    cols = set(df.columns)  # set of column names for quick lookup
    if "question" in cols and "answer" in cols:
        out = df[["question", "answer"]].copy()
        # select only the question and answer columns, drop everything else
    elif "q" in cols and "a" in cols:
        out = df.rename(columns={"q": "question", "a": "answer"})[["question", "answer"]]
        # if columns are named "q" and "a" instead → rename them to standard names
    else:
        raise ValueError(f"Expected question/answer columns, got: {list(df.columns)}")
        # if we can't find question/answer columns, raise an informative error

    out = out.dropna()
    # remove rows where EITHER question OR answer is missing (NaN = Not a Number, means empty)

    out["input_text"] = out["question"].astype(str).str.strip()
    # create "input_text" column: what the model receives as input
    # .astype(str) converts any non-string values to string
    # .str.strip() removes leading/trailing whitespace

    out["target_text"] = out["answer"].astype(str).str.strip()
    # create "target_text" column: what the model should output (the correct answer)

    return out  # returns DataFrame with columns: question, answer, input_text, target_text


def build_qa_organic(df: pd.DataFrame) -> pd.DataFrame:
    # builds question-answer pairs for the organic chemistry dataset
    # the chemistry CSV has compound names and their properties as columns
    df = _normalize_columns(df)  # standardize column names

    name_col = next((c for c in df.columns if "compound" in c or "name" in c), df.columns[0])
    # find the column that contains compound names:
    # generator expression: (c for c in df.columns if "compound" in c or "name" in c)
    # next(...) gets the FIRST matching column
    # if no column has "compound" or "name" in its name → use the first column (df.columns[0])

    prop_cols = [c for c in df.columns if c != name_col]
    # all other columns are "property" columns (boiling point, melting point, toxicity, etc.)

    rows = []  # will build a list of {input_text, target_text} dicts
    for _, row in df.iterrows():
        # df.iterrows() = iterate over each row; _ = row index (we don't need it), row = row data
        name = str(row.get(name_col, "")).strip()
        # get the compound name; if missing, use empty string
        if not name:
            continue  # skip rows with empty compound names

        parts = [f"{c}: {row[c]}" for c in prop_cols if pd.notna(row.get(c))]
        # for each property column: create "property_name: value" string
        # pd.notna() filters out NaN values (empty cells in the CSV)
        # e.g. ["boiling_point: 80.1", "melting_point: 5.5", "toxicity: moderate"]

        ans = "; ".join(parts) if parts else ""
        # join all property strings with "; " separator
        # e.g. "boiling_point: 80.1; melting_point: 5.5; toxicity: moderate"

        q = f"What are the properties of {name}?"
        # generate a standardized question for this compound
        rows.append({"input_text": q, "target_text": ans})
        # add this compound's Q&A pair to our list

    return pd.DataFrame(rows)  # convert list of dicts to a pandas DataFrame


def build_qa_gita(df: pd.DataFrame) -> pd.DataFrame:
    # builds question-answer pairs from the Bhagavad Gita CSV
    # CSV has chapter number, verse number, translation, and explanation columns
    df = _normalize_columns(df)  # standardize column names
    rows = []

    for _, row in df.iterrows():  # iterate over each verse row
        ch = row.get("chapter", row.get("ch", ""))
        # try "chapter" column first, fall back to "ch" column (handles different CSV formats)
        vs = row.get("verse", row.get("v", ""))
        # verse number — same fallback pattern
        trans = row.get("translation", row.get("english", ""))
        # English translation of the verse
        expl = row.get("explanation", row.get("commentary", ""))
        # explanation or commentary on the verse

        q = f"Chapter {ch} Verse {vs}: explain the teaching."
        # standardized question format: "Chapter 2 Verse 47: explain the teaching."
        a = f"Translation: {trans}. Explanation: {expl}"
        # standardized answer format combining translation and explanation

        rows.append({"input_text": str(q).strip(), "target_text": str(a).strip()})
        # str() ensures non-string values are converted; .strip() cleans whitespace

    return pd.DataFrame(rows)


def load_qa_csv(path: str, kind: str) -> Dataset:
    # the main function: reads a CSV and returns a HuggingFace Dataset ready for training
    # 'kind' determines how the CSV is parsed: "python", "organic", or "gita"

    df = _read_csv(path)       # read the CSV file
    df = _normalize_columns(df)  # standardize column names

    if kind == "python":
        qa = build_qa_from_plain(df)   # simple Q&A format
    elif kind == "organic":
        qa = build_qa_organic(df)      # chemistry properties format
    elif kind == "gita":
        qa = build_qa_gita(df)         # verse + translation + explanation format
    else:
        raise ValueError(f"Unknown kind: {kind}")  # fail clearly for unknown types

    return Dataset.from_pandas(qa[["input_text", "target_text"]])
    # convert pandas DataFrame to HuggingFace Dataset object
    # only keep the two columns the trainer needs: input_text and target_text
    # HuggingFace Trainer requires a Dataset object (not a DataFrame)


def split_dataset(ds: Dataset, test_ratio: float = 0.2, seed: int = 42):
    # splits the dataset into training and testing portions
    return ds.train_test_split(test_size=test_ratio, seed=seed)
    # test_size=0.2 = 20% of data goes to test, 80% to training
    # seed=42 = fixed random seed → same split every time (reproducible)
    # returns: {"train": Dataset, "test": Dataset}
