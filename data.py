import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # normalize common column names to canonical names used by agents
    mapping = {
        # heart rate
        "max_heart_rate": "heart_rate",
        "maxhr": "heart_rate",
        "hr": "heart_rate",
        # blood pressure
        "resting_bp_s": "systolic_bp",
        "systolic_bp": "systolic_bp",
        "resting_bp_d": "diastolic_bp",
        "diastolic_bp": "diastolic_bp",
        # temperature
        "temp": "temperature",
        "temperature": "temperature",
        # patient id variants
        "id": "patient_id",
        "patient": "patient_id",
        "patientid": "patient_id",
    }

    # normalize column keys to lower-case without spaces for matching
    cols = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols)

    # apply mapping where available
    rename_map = {}
    for c in df.columns:
        if c in mapping:
            rename_map[c] = mapping[c]
    if rename_map:
        df = df.rename(columns=rename_map)

    # convert timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def load_data(path: str = "dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    return df


def get_patient_groups(df: pd.DataFrame):
    # If a `patient_id` column exists, group by it. Otherwise, treat each row as a separate patient
    groups = {}
    if "patient_id" in df.columns:
        for pid, g in df.groupby("patient_id"):
            groups[pid] = g.sort_values("timestamp") if "timestamp" in g.columns else g
        return groups

    # no patient_id: create per-row groups using the dataframe index
    for idx in df.index:
        groups[f"row_{idx}"] = df.loc[[idx]]
    return groups
