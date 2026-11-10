# app.py
# Model Calibration and Confidence Evaluation System
# AI Evaluation, Observability and Reliability Engineering
#
# Install dependencies without versions:
# pip install streamlit pandas numpy scikit-learn matplotlib pillow pydantic av opencv-python-headless pytest
#
# Run:
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import hashlib
import os
import time
import tempfile
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pydantic import BaseModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PRODUCT CONSTANTS
# =============================================================================

PROJECT_TITLE = "Model Calibration and Confidence Evaluation System"
PROJECT_SUBTITLE = "AI Evaluation, Observability and Reliability Engineering"
SEED = 20261110
EPS = 1e-12
DB_PATH = "model_calibration_demo.sqlite3"
CODE_VERSION = "prototype-1.0.0"

VIDEO_WIDTH = 1200
VIDEO_HEIGHT = 600
VIDEO_FPS = 12
VIDEO_FRAMES = 72

REQUIREMENTS_TEXT = """streamlit
pandas
numpy
scikit-learn
matplotlib
pillow
pydantic
av
opencv-python-headless
pytest
"""

REQUIRED_COLUMNS = [
    "event_id",
    "request_id",
    "timestamp",
    "model_id",
    "model_version",
    "segment",
    "user_tier",
    "feature_latency_ms",
    "feature_missing_flag",
    "input_quality_score",
    "prediction_confidence",
    "predicted_label",
    "true_label",
    "outcome_value",
    "feedback_received",
    "feedback_delay_hours",
    "drift_score",
    "schema_valid",
    "duplicate_flag",
    "late_arrival_flag",
    "provenance_hash",
]

EVENT_DB_COLUMNS = REQUIRED_COLUMNS + ["batch_id", "ingested_at"]

DISPLAY_COLS = [
    "event_id",
    "timestamp",
    "model_version",
    "segment",
    "prediction_confidence",
    "calibrated_confidence",
    "true_label",
    "review_required",
    "decision_status",
    "drift_score",
    "input_quality_score",
    "schema_valid",
]

_SESSION_REQUIRED_KEYS = (
    "initialized",
    "config",
    "active_method",
    "metrics_history",
    "audit",
    "auto_ran",
    "needs_rebuild",
    "proposal",
    "pending_simulation",
    "db_conn",
    "hist_raw",
    "quality_report",
    "calibrators",
    "prior_base_rate",
    "hist_cal",
    "base_eval",
    "reliability_raw",
    "reliability_cal",
    "trend",
    "initial_alerts",
    "demo_video",
    "live_raw",
    "live",
)


# =============================================================================
# CONFIG MODEL
# =============================================================================

class AppConfig(BaseModel):
    seed: int = SEED
    bins: int = 10
    abstain_low: float = 0.35
    abstain_high: float = 0.65
    min_eval_n: int = 200
    max_missing_label_rate: float = 0.25
    max_schema_error_rate: float = 0.05
    drift_alert_threshold: float = 0.65
    safe_mode: bool = False


def model_dict(m: Any) -> Dict[str, Any]:
    if hasattr(m, "model_dump"):
        return m.model_dump()
    return m.dict()


# =============================================================================
# UTILITIES
# =============================================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sha256_obj(obj: Any) -> str:
    try:
        s = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        s = str(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def json_safe(obj: Any) -> Any:
    if obj is None or obj is pd.NA:
        return None

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]

    if isinstance(obj, pd.Series):
        return [json_safe(v) for v in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        return [json_safe(r) for r in obj.to_dict(orient="records")]

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    if isinstance(obj, np.floating):
        f = float(obj)
        if np.isnan(f) or np.isinf(f):
            return None
        return f

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, pd.Timestamp):
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return obj.isoformat()

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def to_json_bytes(obj: Any) -> bytes:
    return json.dumps(json_safe(obj), indent=2, default=str).encode("utf-8")


def to_json_string(obj: Any) -> str:
    return json.dumps(json_safe(obj), indent=2, default=str)


def col_or(df: pd.DataFrame, name: str, default: Any) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def clean_db_value(v: Any) -> Any:
    if v is None or v is pd.NA:
        return None

    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    if isinstance(v, np.integer):
        return int(v)

    if isinstance(v, np.floating):
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f

    if isinstance(v, np.bool_):
        return int(v)

    if isinstance(v, pd.Timestamp):
        return str(v)

    return v


# =============================================================================
# SQLITE PERSISTENCE
# =============================================================================

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            request_id TEXT,
            timestamp TEXT,
            model_id TEXT,
            model_version TEXT,
            segment TEXT,
            user_tier TEXT,
            feature_latency_ms REAL,
            feature_missing_flag INTEGER,
            input_quality_score REAL,
            prediction_confidence REAL,
            predicted_label INTEGER,
            true_label INTEGER,
            outcome_value REAL,
            feedback_received INTEGER,
            feedback_delay_hours REAL,
            drift_score REAL,
            schema_valid INTEGER,
            duplicate_flag INTEGER,
            late_arrival_flag INTEGER,
            provenance_hash TEXT,
            batch_id TEXT,
            ingested_at TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluations (
            run_id TEXT PRIMARY KEY,
            created_at TEXT,
            scope TEXT,
            metrics_json TEXT,
            dataset_hash TEXT,
            code_version TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            actor TEXT,
            action TEXT,
            object_type TEXT,
            object_id TEXT,
            result TEXT,
            evidence_hash TEXT,
            approval_required INTEGER,
            approved_by TEXT,
            note TEXT
        )
        """
    )

    conn.commit()
    return conn


def prepare_df_for_db(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype(str)

        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(int)

        out[c] = out[c].astype(object)
        out[c] = out[c].where(pd.notna(out[c]), None)

    return out


def persist_events(df: pd.DataFrame) -> None:
    if df.empty:
        return

    conn = st.session_state.db_conn
    db_df = prepare_df_for_db(df)

    for c in EVENT_DB_COLUMNS:
        if c not in db_df.columns:
            db_df[c] = None

    db_df = db_df[EVENT_DB_COLUMNS]

    col_sql = ", ".join(EVENT_DB_COLUMNS)
    ph = ", ".join(["?"] * len(EVENT_DB_COLUMNS))
    sql = f"INSERT OR IGNORE INTO events ({col_sql}) VALUES ({ph})"

    rows = []
    for row in db_df.itertuples(index=False, name=None):
        rows.append(tuple(clean_db_value(v) for v in row))

    conn.executemany(sql, rows)
    conn.commit()


def persist_audit(entry: Dict[str, Any]) -> None:
    conn = st.session_state.db_conn
    cols = [
        "ts",
        "actor",
        "action",
        "object_type",
        "object_id",
        "result",
        "evidence_hash",
        "approval_required",
        "approved_by",
        "note",
    ]
    col_sql = ", ".join(cols)
    ph = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO audit_log ({col_sql}) VALUES ({ph})"
    conn.execute(sql, tuple(clean_db_value(entry.get(c, "")) for c in cols))
    conn.commit()


def persist_evaluation(run_id: str, scope: str, metrics: Dict[str, Any], dataset_hash: str) -> None:
    conn = st.session_state.db_conn
    conn.execute(
        """
        INSERT OR REPLACE INTO evaluations
        (run_id, created_at, scope, metrics_json, dataset_hash, code_version)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            now_iso(),
            scope,
            json.dumps(json_safe(metrics), default=str),
            dataset_hash,
            CODE_VERSION,
        ),
    )
    conn.commit()


def add_audit(
    action: str,
    object_type: str,
    object_id: str,
    result: str,
    evidence_hash: str = "",
    approval_required: bool = False,
    approved_by: str = "",
    note: str = "",
) -> None:
    entry = {
        "ts": now_iso(),
        "actor": "system" if not approved_by else approved_by,
        "action": action,
        "object_type": object_type,
        "object_id": object_id,
        "result": result,
        "evidence_hash": evidence_hash,
        "approval_required": 1 if approval_required else 0,
        "approved_by": approved_by,
        "note": note,
    }
    st.session_state.audit.append(entry)
    try:
        persist_audit(entry)
    except Exception as exc:
        st.session_state.audit.append(
            {
                "ts": now_iso(),
                "actor": "system",
                "action": "audit_db_write_failed",
                "object_type": "audit",
                "object_id": object_id,
                "result": str(exc),
                "evidence_hash": "",
                "approval_required": 0,
                "approved_by": "",
                "note": traceback.format_exc(limit=2),
            }
        )


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def add_provenance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["provenance_hash"] = pd.Series(dtype="object")
        return df

    hashes = []
    for rid, ts, conf, lab in zip(
        df["request_id"],
        df["timestamp"].astype(str),
        df["prediction_confidence"].round(6),
        df["true_label"].round(6),
    ):
        payload = f"{rid}|{ts}|{conf}|{lab}"
        hashes.append(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12])

    df["provenance_hash"] = hashes
    return df


def _synthetic_core(
    n: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    model_version_mode: str,
    drift: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    scenario: str = "historical",
    id_prefix: str = "hist",
) -> pd.DataFrame:
    if n <= 0:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in REQUIRED_COLUMNS})

    if rng is None:
        rng = np.random.default_rng(SEED)

    timestamps = pd.date_range(start=start, end=end, periods=n, tz="UTC")
    jitter = pd.to_timedelta(rng.integers(0, 90, n), unit="s")
    timestamps = timestamps + jitter

    segments = rng.choice(
        ["web", "mobile", "enterprise", "partner"],
        size=n,
        p=[0.45, 0.35, 0.12, 0.08],
    )
    user_tier = rng.choice(
        ["free", "plus", "pro", "enterprise"],
        size=n,
        p=[0.50, 0.25, 0.15, 0.10],
    )

    quality = rng.beta(5.2, 2.1, n)
    latency = rng.lognormal(mean=4.15, sigma=0.45, size=n)
    feature_missing = (rng.random(n) < 0.07).astype(int)

    seg_map = {"web": 0.0, "mobile": -0.15, "enterprise": 0.25, "partner": 0.10}
    tier_map = {"free": -0.10, "plus": 0.0, "pro": 0.15, "enterprise": 0.25}

    seg_effect = np.array([seg_map[s] for s in segments])
    tier_effect = np.array([tier_map[t] for t in user_tier])

    if model_version_mode == "mixed":
        split = start + (end - start) * 0.55
        model_version = np.where(timestamps < split, "v1_overconfident", "v2_candidate")
    else:
        model_version = np.full(n, model_version_mode, dtype=object)

    true_logit = (
        -1.10
        + 1.05 * quality
        + seg_effect
        + tier_effect
        - 0.75 * feature_missing
        + 0.00018 * (latency - 70.0)
        + drift * 0.85
        + rng.normal(0.0, 0.35, n)
    )

    true_p = sigmoid(true_logit)
    label = rng.binomial(1, true_p)

    is_v1 = model_version == "v1_overconfident"
    slope = np.where(is_v1, 1.90, 1.25) + drift * 0.65
    noise_sd = 0.24 + 0.30 * drift + np.where(is_v1, 0.05, 0.0)

    raw = sigmoid(slope * true_logit + rng.normal(0.0, noise_sd, n))
    raw = np.array(np.clip(raw, 0.001, 0.999), dtype=float, copy=True)

    predicted_label = (raw >= 0.5).astype(int)

    outcome_value = np.where(
        label == 1,
        rng.gamma(2.0, 18.0, n),
        -rng.gamma(1.5, 7.0, n),
    )

    if scenario == "historical":
        feedback_cutoff = pd.Timestamp("2026-11-07", tz="UTC")
        feedback_prob = np.where(timestamps > feedback_cutoff, 0.78, 0.97)
    else:
        feedback_prob = 0.65

    received = rng.random(n) < feedback_prob
    true_label = np.where(received, label, np.nan)
    feedback_delay = np.where(received, rng.exponential(14.0, n) + 1.0, np.nan)
    late_arrival = np.where(received & (feedback_delay > 48), 1, 0).astype(int)

    drift_score = np.clip(
        0.12
        + drift
        + 0.18 * rng.normal(0.0, 1.0, n)
        + (timestamps > pd.Timestamp("2026-11-05", tz="UTC")).astype(float) * 0.18,
        0.0,
        1.0,
    )

    schema_valid = np.ones(n, dtype=int)
    invalid_count = max(1, int(n * 0.004))
    invalid_idx = rng.choice(n, size=invalid_count, replace=False)
    bad_values = rng.choice([1.25, -0.05, np.nan], size=invalid_count)

    raw[invalid_idx] = bad_values
    schema_valid[invalid_idx] = 0
    predicted_label[invalid_idx] = -1

    duplicate_flag = np.zeros(n, dtype=int)

    event_id = [f"{id_prefix}-{i}" for i in range(n)]
    request_id = [f"req-{id_prefix}-{i}" for i in range(n)]

    df = pd.DataFrame(
        {
            "event_id": event_id,
            "request_id": request_id,
            "timestamp": timestamps,
            "model_id": "risk-binary-v1",
            "model_version": model_version,
            "segment": segments,
            "user_tier": user_tier,
            "feature_latency_ms": latency,
            "feature_missing_flag": feature_missing,
            "input_quality_score": quality,
            "prediction_confidence": raw,
            "predicted_label": predicted_label,
            "true_label": true_label,
            "outcome_value": outcome_value,
            "feedback_received": received.astype(int),
            "feedback_delay_hours": feedback_delay,
            "drift_score": drift_score,
            "schema_valid": schema_valid,
            "duplicate_flag": duplicate_flag,
            "late_arrival_flag": late_arrival,
        }
    )

    df = add_provenance(df)
    return df


def make_historical(n: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    start = pd.Timestamp("2026-10-27", tz="UTC")
    end = pd.Timestamp("2026-11-09 23:45:00", tz="UTC")

    df = _synthetic_core(
        n=n,
        start=start,
        end=end,
        model_version_mode="mixed",
        drift=0.12,
        rng=rng,
        scenario="historical",
        id_prefix="hist",
    )

    if len(df) > 30:
        dup_idx = rng.choice(len(df), size=25, replace=False)
        dups = df.iloc[dup_idx].copy()
        dups["event_id"] = [f"histdup-{i}" for i in range(len(dups))]
        dups["request_id"] = df.iloc[dup_idx]["request_id"].to_numpy()
        dups["duplicate_flag"] = 1
        dups["provenance_hash"] = [
            hashlib.sha1(f"dup|{r}|{t}".encode("utf-8")).hexdigest()[:12]
            for r, t in zip(dups["request_id"], dups["timestamp"].astype(str))
        ]
        df = pd.concat([df, dups], ignore_index=True)

    return df


def generate_stream_batch(
    batch_id: str,
    n: int = 80,
    drift: float = 0.0,
    failure_mode: Optional[str] = None,
    seed_offset: int = 0,
    active_version: str = "v2_candidate",
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 10000 + seed_offset)
    now = pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(minutes=5)
    end = now

    df = _synthetic_core(
        n=n,
        start=start,
        end=end,
        model_version_mode=active_version,
        drift=drift,
        rng=rng,
        scenario="stream",
        id_prefix=batch_id,
    )

    if df.empty:
        return df

    if failure_mode == "labels":
        keep = rng.random(len(df)) < 0.15
        df["true_label"] = np.where(keep, df["true_label"], np.nan)
        df["feedback_received"] = df["true_label"].notna().astype(int)

    elif failure_mode == "schema":
        bad = rng.choice(len(df), size=max(1, len(df) // 3), replace=False)
        df.loc[bad, "prediction_confidence"] = rng.choice(
            [1.4, -0.2, np.nan], size=len(bad)
        )
        df.loc[bad, "schema_valid"] = 0
        df.loc[bad, "predicted_label"] = -1

    elif failure_mode == "dup":
        if len(df) > 8:
            k = len(df) // 4
            dup_rows = df.iloc[:k].copy()
            dup_rows["event_id"] = [f"{batch_id}-dup-{i}" for i in range(len(dup_rows))]
            dup_rows["request_id"] = df.iloc[:k]["request_id"].to_numpy()
            dup_rows["duplicate_flag"] = 1
            df = pd.concat([df, dup_rows], ignore_index=True)

    elif failure_mode == "latency":
        df["feature_latency_ms"] = df["feature_latency_ms"] * 8.0
        df["drift_score"] = np.clip(df["drift_score"] + 0.25, 0.0, 1.0)

    return df


# =============================================================================
# SCHEMA VALIDATION AND INGESTION
# =============================================================================

def validate_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "total_rows": 0,
        "missing_columns": [],
        "error_count": 0,
        "warning_count": 0,
        "duplicate_count": 0,
        "missing_label_rate": 1.0,
        "schema_error_rate": 1.0,
        "invalid_confidence": 0,
        "invalid_quality": 0,
        "outlier_latency": 0,
        "rejected_rows": 0,
        "notes": "",
    }

    if df is None or len(df) == 0:
        report["error_count"] = 1
        report["notes"] = "empty batch"
        return pd.DataFrame(columns=REQUIRED_COLUMNS), report

    report["total_rows"] = len(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report["missing_columns"] = missing
        report["error_count"] = len(missing)
        report["rejected_rows"] = len(df)
        report["notes"] = "missing required columns"
        return pd.DataFrame(columns=REQUIRED_COLUMNS), report

    clean = df.copy()

    numeric_cols = [
        "feature_latency_ms",
        "input_quality_score",
        "prediction_confidence",
        "outcome_value",
        "feedback_delay_hours",
        "drift_score",
    ]
    for c in numeric_cols:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")

    int_cols = [
        "feature_missing_flag",
        "predicted_label",
        "feedback_received",
        "schema_valid",
        "duplicate_flag",
        "late_arrival_flag",
    ]
    for c in int_cols:
        clean[c] = pd.to_numeric(clean[c], errors="coerce").fillna(0).astype(int)

    clean["true_label"] = pd.to_numeric(clean["true_label"], errors="coerce")

    conf_bad = clean["prediction_confidence"].isna() | ~clean["prediction_confidence"].between(0, 1)
    qual_bad = clean["input_quality_score"].isna() | ~clean["input_quality_score"].between(0, 1)
    lat_bad = (
        clean["feature_latency_ms"].isna()
        | (clean["feature_latency_ms"] < 0)
        | (clean["feature_latency_ms"] > 10000)
    )

    clean.loc[conf_bad | qual_bad | lat_bad, "schema_valid"] = 0

    dup_mask = clean.duplicated(subset=["request_id"], keep=False)
    clean.loc[dup_mask, "duplicate_flag"] = 1

    report["invalid_confidence"] = int(conf_bad.sum())
    report["invalid_quality"] = int(qual_bad.sum())
    report["outlier_latency"] = int(lat_bad.sum())
    report["duplicate_count"] = int(dup_mask.sum())

    report["missing_label_rate"] = (
        float(clean["true_label"].isna().mean()) if len(clean) else 1.0
    )
    report["schema_error_rate"] = (
        float(1.0 - clean["schema_valid"].mean()) if len(clean) else 1.0
    )

    report["rejected_rows"] = int((clean["schema_valid"] == 0).sum())
    if report["rejected_rows"] > 0:
        report["error_count"] += 1
    if report["duplicate_count"] > 0:
        report["warning_count"] += 1

    report["notes"] = (
        f"rows={report['total_rows']}, rejected={report['rejected_rows']}, "
        f"duplicates={report['duplicate_count']}, missing_labels={report['missing_label_rate']:.3f}"
    )

    return clean, report


def ingest_and_persist(
    raw_df: pd.DataFrame,
    batch_id: str,
    actor: str = "system",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    canonical, report = validate_schema(raw_df)
    canonical = canonical.copy()
    canonical["batch_id"] = batch_id
    canonical["ingested_at"] = now_iso()

    try:
        persist_events(canonical)
    except Exception as exc:
        report["db_error"] = str(exc)

    evidence = sha256_obj(
        {
            "batch_id": batch_id,
            "rows": len(canonical),
            "report": json_safe(report),
            "actor": actor,
        }
    )

    add_audit(
        action="ingest_batch",
        object_type="batch",
        object_id=batch_id,
        result=(
            f"rows={len(canonical)}; rejected={report['rejected_rows']}; "
            f"duplicates={report['duplicate_count']}"
        ),
        evidence_hash=evidence,
        note=json.dumps(json_safe(report), default=str)[:500],
    )

    return canonical, report


# =============================================================================
# CALIBRATION CORE
# =============================================================================

def labeled_arrays(df: pd.DataFrame, conf_col: str) -> Tuple[np.ndarray, np.ndarray]:
    if df.empty:
        return np.array([], dtype=int), np.array([], dtype=float)

    if conf_col not in df.columns or "true_label" not in df.columns:
        return np.array([], dtype=int), np.array([], dtype=float)

    out = df.copy()

    if "schema_valid" in out.columns:
        out = out[pd.to_numeric(out["schema_valid"], errors="coerce").fillna(0).astype(int) == 1]

    if "duplicate_flag" in out.columns:
        out = out[pd.to_numeric(out["duplicate_flag"], errors="coerce").fillna(0).astype(int) == 0]

    conf = pd.to_numeric(out[conf_col], errors="coerce")
    out = out[conf.notna() & conf.between(0, 1)]

    lab = pd.to_numeric(out["true_label"], errors="coerce")
    out = out[lab.notna()]

    if out.empty:
        return np.array([], dtype=int), np.array([], dtype=float)

    y = pd.to_numeric(out["true_label"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(out[conf_col], errors="coerce").astype(float).to_numpy()
    return y, p


def reliability_bins(
    y: np.ndarray,
    p: np.ndarray,
    bins: int = 10,
) -> Tuple[pd.DataFrame, float, float, int]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    mask = np.isfinite(y) & np.isfinite(p) & (p >= 0) & (p <= 1)
    y = y[mask]
    p = p[mask]
    n = len(y)

    empty = pd.DataFrame(columns=["bin", "bin_lower", "bin_upper", "conf", "acc", "n", "gap", "weighted_gap"])

    if n == 0:
        return empty, float("nan"), float("nan"), 0

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)

    df = pd.DataFrame({"bin": idx, "p": p, "y": y})
    agg = (
        df.groupby("bin")
        .agg(conf=("p", "mean"), acc=("y", "mean"), n=("y", "size"))
        .reset_index()
    )

    agg["bin_lower"] = agg["bin"].apply(lambda i: edges[int(i)])
    agg["bin_upper"] = agg["bin"].apply(lambda i: edges[int(i) + 1])
    agg["gap"] = agg["acc"] - agg["conf"]
    agg["weighted_gap"] = (agg["n"] / n) * agg["gap"].abs()

    ece = float(agg["weighted_gap"].sum())
    mce = float(agg["gap"].abs().max())

    return agg, ece, mce, n


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    bins: int,
    boot: int,
    seed: int,
) -> Dict[str, float]:
    if boot <= 0 or len(y) < 20:
        return {}

    rng = np.random.default_rng(seed)
    eces = []
    briers = []

    for _ in range(boot):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        pp = p[idx]

        _, ece, _, _ = reliability_bins(yy, pp, bins)
        if np.isfinite(ece):
            eces.append(ece)

        briers.append(float(np.mean((pp - yy) ** 2)))

    if not eces:
        return {}

    return {
        "ece_low": float(np.percentile(eces, 2.5)),
        "ece_high": float(np.percentile(eces, 97.5)),
        "brier_low": float(np.percentile(briers, 2.5)),
        "brier_high": float(np.percentile(briers, 97.5)),
    }


def empty_metrics(fallback: bool = True) -> Dict[str, Any]:
    return {
        "n_eval": 0,
        "missing_label_rate": float("nan"),
        "schema_error_rate": float("nan"),
        "fallback_used": fallback,
        "ece": float("nan"),
        "mce": float("nan"),
        "brier": float("nan"),
        "log_loss": float("nan"),
        "auc": float("nan"),
        "auprc": float("nan"),
        "review_rate": float("nan"),
        "ci": {},
    }


def evaluate_calibration(
    df: pd.DataFrame,
    config: AppConfig,
    conf_col: str = "calibrated_confidence",
    boot: int = 0,
) -> Dict[str, Any]:
    input_n = len(df)
    if input_n == 0:
        return empty_metrics(fallback=True)

    labels = pd.to_numeric(col_or(df, "true_label", np.nan), errors="coerce")
    schema_valid = pd.to_numeric(col_or(df, "schema_valid", 1), errors="coerce").fillna(0).astype(int)
    duplicate_flag = pd.to_numeric(col_or(df, "duplicate_flag", 0), errors="coerce").fillna(0).astype(int)
    conf = pd.to_numeric(col_or(df, conf_col, np.nan), errors="coerce")

    missing_rate = float(labels.isna().mean())
    schema_error_rate = float(1.0 - schema_valid.mean())

    valid_mask = (schema_valid == 1) & (duplicate_flag == 0) & conf.notna() & conf.between(0, 1) & labels.notna()
    n = int(valid_mask.sum())

    fallback = (
        n < config.min_eval_n
        or missing_rate > config.max_missing_label_rate
        or schema_error_rate > config.max_schema_error_rate
    )

    if n == 0:
        out = empty_metrics(fallback=True)
        out["missing_label_rate"] = missing_rate
        out["schema_error_rate"] = schema_error_rate
        return out

    y = labels[valid_mask].astype(int).to_numpy()
    p = conf[valid_mask].to_numpy(dtype=float)

    _, ece, mce, _ = reliability_bins(y, p, config.bins)
    brier = float(np.mean((p - y) ** 2))

    pc = np.clip(p, EPS, 1.0 - EPS)
    logloss = float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))

    auc = float("nan")
    auprc = float("nan")

    if len(np.unique(y)) > 1:
        try:
            auc = float(roc_auc_score(y, p))
        except Exception:
            auc = float("nan")

    if y.sum() > 0:
        try:
            auprc = float(average_precision_score(y, p))
        except Exception:
            auprc = float("nan")

    review_series = col_or(df, "review_required", False)
    review_series = pd.to_numeric(review_series, errors="coerce").fillna(0).astype(float)
    review_rate = float(review_series.mean()) if len(review_series) else float("nan")

    ci = bootstrap_ci(y, p, config.bins, boot, config.seed)

    return {
        "n_eval": n,
        "missing_label_rate": missing_rate,
        "schema_error_rate": schema_error_rate,
        "fallback_used": fallback,
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "log_loss": logloss,
        "auc": auc,
        "auprc": auprc,
        "review_rate": review_rate,
        "ci": ci,
    }


def fit_calibrators(val_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    y, p = labeled_arrays(val_df, "prediction_confidence")

    if len(y) < 50:
        return None

    if np.unique(y).size == 1:
        const = float(y[0])

        def const_fn(x: Any, c: float = const) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            return np.full(np.shape(x), c, dtype=float)

        return {
            "isotonic": const_fn,
            "platt": const_fn,
            "fitted_on": len(y),
            "degenerate": True,
        }

    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)

    x = logit(p).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000, C=1.0).fit(x, y)

    def platt(pred: Any) -> np.ndarray:
        pred = np.asarray(pred, dtype=float)
        if pred.size == 0:
            return pred.copy()
        safe = np.where(np.isnan(pred), 0.5, pred)
        safe = np.clip(safe, EPS, 1.0 - EPS)
        return lr.predict_proba(logit(safe).reshape(-1, 1))[:, 1]

    def isotonic(pred: Any) -> np.ndarray:
        pred = np.asarray(pred, dtype=float)
        if pred.size == 0:
            return pred.copy()
        safe = np.where(np.isnan(pred), 0.5, pred)
        safe = np.clip(safe, 0.0, 1.0)
        return iso.predict(safe)

    return {
        "isotonic": isotonic,
        "platt": platt,
        "fitted_on": len(y),
        "degenerate": False,
    }


def apply_calibration(
    df: pd.DataFrame,
    method: str,
    calibrators: Optional[Dict[str, Any]],
    config: AppConfig,
    prior_rate: float = 0.5,
) -> pd.DataFrame:
    out = df.copy()

    if "prediction_confidence" not in out.columns:
        out["prediction_confidence"] = np.nan

    p = np.array(
        pd.to_numeric(out["prediction_confidence"], errors="coerce"),
        dtype=float,
        copy=True,
    )

    if method == "raw" or calibrators is None:
        cal = p
    elif method == "platt":
        cal = calibrators["platt"](p)
    elif method == "isotonic":
        cal = calibrators["isotonic"](p)
    elif method == "fallback_prior":
        cal = np.full_like(p, prior_rate, dtype=float)
    else:
        cal = p

    cal = np.array(cal, dtype=float, copy=True)
    cal[np.isnan(p)] = np.nan

    out["calibrated_confidence"] = np.clip(cal, 0.0, 1.0)
    out["confidence_delta"] = out["calibrated_confidence"] - out["prediction_confidence"]

    low, high = sorted((float(config.abstain_low), float(config.abstain_high)))

    band = (
        out["calibrated_confidence"].notna()
        & (out["calibrated_confidence"] >= low)
        & (out["calibrated_confidence"] <= high)
    )

    quality = pd.to_numeric(col_or(out, "input_quality_score", 1.0), errors="coerce")
    quality_risk = quality < 0.45

    schema = pd.to_numeric(col_or(out, "schema_valid", 1), errors="coerce")
    schema_risk = schema == 0

    out["review_required"] = (
        band.fillna(False)
        | quality_risk.fillna(False)
        | schema_risk.fillna(False)
    )

    if bool(config.safe_mode):
        out["review_required"] = True

    out["decision_status"] = np.where(
        out["review_required"],
        "human_review",
        "auto_accept",
    )

    return out


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.sort_values("timestamp").reset_index(drop=True)
    n = len(out)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return out.iloc[:i1], out.iloc[i1:i2], out.iloc[i2:]


def compute_trend(cal_df: pd.DataFrame, windows: int = 30, bins: int = 10) -> pd.DataFrame:
    df = cal_df.copy()

    if "duplicate_flag" in df.columns:
        df = df[pd.to_numeric(df["duplicate_flag"], errors="coerce").fillna(0).astype(int) == 0]

    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) == 0:
        return pd.DataFrame()

    ws = max(1, len(df) // windows)
    df["window"] = df.index // ws

    all_records = []
    for w, g in df.groupby("window"):
        all_records.append(
            {
                "window": int(w),
                "n": len(g),
                "missing_rate": float(g["true_label"].isna().mean()) if "true_label" in g else float("nan"),
                "avg_drift": float(g["drift_score"].mean()) if "drift_score" in g else float("nan"),
                "review_rate": float(g["review_required"].mean()) if "review_required" in g else float("nan"),
            }
        )

    all_df = pd.DataFrame(all_records).set_index("window")

    if "true_label" not in df.columns:
        return all_df.reset_index()

    lab = df.dropna(subset=["true_label"])

    if lab.empty:
        return all_df.reset_index()

    lab_records = []

    for w, g in lab.groupby("window"):
        y = pd.to_numeric(g["true_label"], errors="coerce").astype(int).to_numpy()

        praw = pd.to_numeric(g["prediction_confidence"], errors="coerce").to_numpy(dtype=float)
        pcal = pd.to_numeric(g["calibrated_confidence"], errors="coerce").to_numpy(dtype=float)

        mask = (
            np.isfinite(y)
            & np.isfinite(praw)
            & np.isfinite(pcal)
            & (praw >= 0)
            & (praw <= 1)
            & (pcal >= 0)
            & (pcal <= 1)
        )

        y = y[mask]
        praw = praw[mask]
        pcal = pcal[mask]

        if len(y) == 0:
            continue

        _, ece_raw, _, _ = reliability_bins(y, praw, bins)
        _, ece_cal, _, _ = reliability_bins(y, pcal, bins)

        lab_records.append(
            {
                "window": int(w),
                "n_labeled": len(y),
                "ece_raw": ece_raw,
                "ece_calibrated": ece_cal,
                "brier_raw": float(np.mean((praw - y) ** 2)),
                "brier_calibrated": float(np.mean((pcal - y) ** 2)),
            }
        )

    if not lab_records:
        return all_df.reset_index()

    lab_df = pd.DataFrame(lab_records).set_index("window")
    trend = all_df.join(lab_df, how="left").reset_index()
    return trend


def segment_table(df: pd.DataFrame, config: AppConfig, conf_col: str = "calibrated_confidence") -> pd.DataFrame:
    if df.empty or "segment" not in df.columns:
        return pd.DataFrame()

    rows = []
    for seg, g in df.groupby("segment"):
        m = evaluate_calibration(g, config, conf_col, boot=0)
        rows.append({"segment": seg, **m})
    return pd.DataFrame(rows)


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def render_video_frame(idx: int, total: int, vd: Dict[str, Any]) -> np.ndarray:
    progress = (idx + 1) / float(total)

    fig, axs = plt.subplots(2, 2, figsize=(12, 6), dpi=100)

    ax = axs[0, 0]
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    rel_raw = vd.get("reliability_raw", pd.DataFrame())
    rel_cal = vd.get("reliability_cal", pd.DataFrame())

    if len(rel_raw) > 0:
        n = max(1, int(len(rel_raw) * progress))
        ax.scatter(
            rel_raw["conf"].iloc[:n],
            rel_raw["acc"].iloc[:n],
            s=rel_raw["n"].iloc[:n] * 4 + 10,
            color="#d62728",
            alpha=0.65,
            label="Raw",
        )

    if len(rel_cal) > 0:
        n = max(1, int(len(rel_cal) * progress))
        ax.scatter(
            rel_cal["conf"].iloc[:n],
            rel_cal["acc"].iloc[:n],
            s=rel_cal["n"].iloc[:n] * 4 + 10,
            color="#2ca02c",
            alpha=0.65,
            label="Calibrated",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Reliability diagram")
    ax.legend(loc="lower right", fontsize=8)

    ax = axs[0, 1]
    trend = vd.get("trend", pd.DataFrame())
    required_trend_cols = {"window", "ece_raw", "ece_calibrated"}
    if len(trend) > 0 and required_trend_cols.issubset(trend.columns):
        n = max(1, int(len(trend) * progress))
        t = trend.iloc[:n]
        ax.plot(t["window"], t["ece_raw"], color="#d62728", marker="o", label="Raw ECE")
        ax.plot(t["window"], t["ece_calibrated"], color="#2ca02c", marker="o", label="Calibrated ECE")
        ax.axhline(0.08, color="gray", ls=":", label="Guardrail 0.08")
        ax.set_xlabel("Time window")
        ax.set_ylabel("ECE")
        ax.set_title("Calibration error trend")
        ax.legend(fontsize=8)
    else:
        ax.text(0.2, 0.5, "No trend data", fontsize=12)
        ax.axis("off")

    ax = axs[1, 0]
    craw = np.asarray(vd.get("conf_raw", []), dtype=float)
    ccal = np.asarray(vd.get("conf_cal", []), dtype=float)

    if len(craw) > 0 and len(ccal) > 0:
        n_raw = max(1, int(len(craw) * progress))
        n_cal = max(1, int(len(ccal) * progress))
        ax.hist(craw[:n_raw], bins=20, range=(0, 1), alpha=0.45, color="#d62728", label="Raw")
        ax.hist(ccal[:n_cal], bins=20, range=(0, 1), alpha=0.45, color="#2ca02c", label="Calibrated")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title("Confidence distribution")
        ax.legend(fontsize=8)
    else:
        ax.text(0.2, 0.5, "No confidence data", fontsize=12)
        ax.axis("off")

    ax = axs[1, 1]
    ax.axis("off")

    lines = [
        "Model Calibration & Confidence Evaluation",
        f"Frame {idx + 1}/{total}",
        f"Timestamp: {vd.get('timestamp', '')}",
        f"Active method: {vd.get('method', 'unknown')}",
        f"Fallback: {vd.get('fallback', False)}",
        "Alerts:",
    ]

    alerts = vd.get("alerts", [])
    if alerts:
        k = max(1, int(len(alerts) * progress))
        lines.extend([f" - {a}" for a in alerts[:k]])
    else:
        lines.append(" - None")

    lines.extend(
        [
            "",
            "Decision rule:",
            "auto-accept outside abstain band;",
            "human review inside band, low quality,",
            "schema failure, or safe mode.",
        ]
    )

    ax.text(0.03, 0.97, "\n".join(lines), va="top", fontsize=9, family="monospace")

    fig.suptitle("Live synthetic demo: raw vs calibrated confidence", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    rgb = np.ascontiguousarray(np.asarray(buf, dtype=np.uint8)[:, :, :3])
    plt.close(fig)

    return rgb


def generate_demo_video(vd: Dict[str, Any]) -> Optional[bytes]:
    try:
        import av

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        path = tmp.name
        tmp.close()

        container = av.open(path, mode="w")
        stream = container.add_stream("libx264", rate=VIDEO_FPS)
        stream.width = VIDEO_WIDTH
        stream.height = VIDEO_HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "30"}

        for i in range(VIDEO_FRAMES):
            rgb = render_video_frame(i, VIDEO_FRAMES, vd)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()

        with open(path, "rb") as f:
            data = f.read()

        try:
            os.unlink(path)
        except Exception:
            pass

        if data:
            return data

    except Exception:
        traceback.print_exc()

    try:
        import cv2

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        path = tmp.name
        tmp.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

        if not writer.isOpened():
            raise RuntimeError("OpenCV VideoWriter could not open output file.")

        for i in range(VIDEO_FRAMES):
            rgb = render_video_frame(i, VIDEO_FRAMES, vd)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

        writer.release()

        with open(path, "rb") as f:
            data = f.read()

        try:
            os.unlink(path)
        except Exception:
            pass

        if data:
            return data

    except Exception:
        traceback.print_exc()

    return None


# =============================================================================
# PLOTS FOR DASHBOARD
# =============================================================================

def reliability_figure(rel_raw: pd.DataFrame, rel_cal: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    if len(rel_raw) > 0:
        ax.scatter(
            rel_raw["conf"],
            rel_raw["acc"],
            s=rel_raw["n"] * 4 + 20,
            alpha=0.65,
            color="#d62728",
            label="Raw",
        )

    if len(rel_cal) > 0:
        ax.scatter(
            rel_cal["conf"],
            rel_cal["acc"],
            s=rel_cal["n"] * 4 + 20,
            alpha=0.65,
            color="#2ca02c",
            label="Calibrated",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Reliability diagram: raw vs calibrated")
    ax.legend()
    fig.tight_layout()
    return fig


def confidence_histogram_figure(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))

    raw = pd.to_numeric(df.get("prediction_confidence", pd.Series(dtype=float)), errors="coerce").dropna()
    cal = pd.to_numeric(df.get("calibrated_confidence", pd.Series(dtype=float)), errors="coerce").dropna()

    raw = raw[(raw >= 0) & (raw <= 1)]
    cal = cal[(cal >= 0) & (cal <= 1)]

    if len(raw) > 0:
        ax.hist(raw, bins=20, range=(0, 1), alpha=0.45, color="#d62728", label="Raw")
    if len(cal) > 0:
        ax.hist(cal, bins=20, range=(0, 1), alpha=0.45, color="#2ca02c", label="Calibrated")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def trend_figure(trend: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    required_cols = {"window", "ece_raw", "ece_calibrated"}

    if len(trend) > 0 and required_cols.issubset(trend.columns):
        ax.plot(trend["window"], trend["ece_raw"], marker="o", color="#d62728", label="Raw ECE")
        ax.plot(trend["window"], trend["ece_calibrated"], marker="o", color="#2ca02c", label="Calibrated ECE")
        ax.axhline(0.08, color="gray", ls=":", label="Guardrail 0.08")
        ax.set_xlabel("Time window")
        ax.set_ylabel("ECE")
        ax.set_title("Calibration error trend")
        ax.legend()
    else:
        ax.text(0.2, 0.5, "No trend data", fontsize=12)
        ax.axis("off")

    fig.tight_layout()
    return fig


# =============================================================================
# SESSION INITIALIZATION
# =============================================================================

def build_video_payload(
    rel_raw: pd.DataFrame,
    rel_cal: pd.DataFrame,
    trend: pd.DataFrame,
    conf_raw: np.ndarray,
    conf_cal: np.ndarray,
    alerts: List[str],
    fallback: bool,
    method: str,
) -> Dict[str, Any]:
    return {
        "reliability_raw": rel_raw,
        "reliability_cal": rel_cal,
        "trend": trend,
        "conf_raw": conf_raw,
        "conf_cal": conf_cal,
        "alerts": alerts,
        "fallback": fallback,
        "method": method,
        "timestamp": now_iso(),
    }


def compute_base_artifacts(
    hist_raw: pd.DataFrame,
    calibrators: Optional[Dict[str, Any]],
    config: AppConfig,
    prior_rate: float,
    boot: int = 120,
) -> Dict[str, Any]:
    train, val, test = chronological_split(hist_raw)

    hist_cal = apply_calibration(hist_raw, "isotonic", calibrators, config, prior_rate=prior_rate)

    test_raw = apply_calibration(test, "raw", calibrators, config, prior_rate=prior_rate)
    test_iso = apply_calibration(test, "isotonic", calibrators, config, prior_rate=prior_rate)
    test_platt = apply_calibration(test, "platt", calibrators, config, prior_rate=prior_rate)

    raw_metrics = evaluate_calibration(test_raw, config, "prediction_confidence", boot=boot)
    iso_metrics = evaluate_calibration(test_iso, config, "calibrated_confidence", boot=boot)
    platt_metrics = evaluate_calibration(test_platt, config, "calibrated_confidence", boot=boot)

    y_raw, p_raw = labeled_arrays(test_raw, "prediction_confidence")
    y_cal, p_cal = labeled_arrays(test_iso, "calibrated_confidence")

    rel_raw, _, _, _ = reliability_bins(y_raw, p_raw, config.bins)
    rel_cal, _, _, _ = reliability_bins(y_cal, p_cal, config.bins)

    trend = compute_trend(hist_cal, windows=30, bins=config.bins)

    alerts = []
    if np.isfinite(iso_metrics["ece"]) and iso_metrics["ece"] > 0.08:
        alerts.append("Calibrated ECE above 0.08 guardrail")
    if iso_metrics["missing_label_rate"] > config.max_missing_label_rate:
        alerts.append("Missing label rate above guardrail")
    if iso_metrics["schema_error_rate"] > config.max_schema_error_rate:
        alerts.append("Schema error rate above guardrail")
    if iso_metrics["fallback_used"]:
        alerts.append("Fallback active due to evaluation quality")

    base_eval = {
        "raw": raw_metrics,
        "isotonic": iso_metrics,
        "platt": platt_metrics,
        "calibrator_fitted_on": 0 if calibrators is None else calibrators.get("fitted_on", 0),
        "degenerate_calibrator": False if calibrators is None else calibrators.get("degenerate", False),
    }

    vd = build_video_payload(
        rel_raw=rel_raw,
        rel_cal=rel_cal,
        trend=trend,
        conf_raw=p_raw,
        conf_cal=p_cal,
        alerts=alerts,
        fallback=iso_metrics["fallback_used"],
        method="isotonic",
    )

    video = generate_demo_video(vd)

    return {
        "hist_cal": hist_cal,
        "base_eval": base_eval,
        "reliability_raw": rel_raw,
        "reliability_cal": rel_cal,
        "trend": trend,
        "alerts": alerts,
        "demo_video": video,
        "conf_raw": p_raw,
        "conf_cal": p_cal,
    }


def init_session() -> None:
    if st.session_state.get("initialized", False) and all(k in st.session_state for k in _SESSION_REQUIRED_KEYS):
        return

    try:
        conn = st.session_state.get("db_conn")
        if conn is not None:
            conn.close()
    except Exception:
        pass

    st.session_state.clear()

    st.session_state.initialized = False
    st.session_state.config = AppConfig()
    st.session_state.active_method = "isotonic"
    st.session_state.metrics_history = []
    st.session_state.audit = []
    st.session_state.auto_ran = False
    st.session_state.needs_rebuild = False
    st.session_state.proposal = None
    st.session_state.pending_simulation = None

    st.session_state.db_conn = init_db()

    hist_raw = make_historical(6000)
    hist_raw, quality_report = validate_schema(hist_raw)
    hist_raw = hist_raw.copy()
    hist_raw["batch_id"] = "historical"
    hist_raw["ingested_at"] = now_iso()

    st.session_state.hist_raw = hist_raw
    st.session_state.quality_report = quality_report

    try:
        persist_events(hist_raw)
    except Exception as exc:
        quality_report["db_error"] = str(exc)

    add_audit(
        action="load_historical_synthetic",
        object_type="dataset",
        object_id="historical",
        result=f"rows={len(hist_raw)}",
        evidence_hash=sha256_obj(json_safe(quality_report)),
        note=f"Synthetic data seed={SEED}; no external dataset used.",
    )

    train, val, test = chronological_split(hist_raw)
    calibrators = fit_calibrators(val)
    st.session_state.calibrators = calibrators

    clean_hist = hist_raw[
        (pd.to_numeric(hist_raw["schema_valid"], errors="coerce").fillna(0).astype(int) == 1)
        & (pd.to_numeric(hist_raw["duplicate_flag"], errors="coerce").fillna(0).astype(int) == 0)
    ]
    labeled_hist = clean_hist.dropna(subset=["true_label"])
    prior_rate = float(labeled_hist["true_label"].mean()) if len(labeled_hist) else 0.5
    st.session_state.prior_base_rate = prior_rate

    artifacts = compute_base_artifacts(
        hist_raw=hist_raw,
        calibrators=calibrators,
        config=st.session_state.config,
        prior_rate=prior_rate,
        boot=120,
    )

    st.session_state.hist_cal = artifacts["hist_cal"]
    st.session_state.base_eval = artifacts["base_eval"]
    st.session_state.reliability_raw = artifacts["reliability_raw"]
    st.session_state.reliability_cal = artifacts["reliability_cal"]
    st.session_state.trend = artifacts["trend"]
    st.session_state.initial_alerts = artifacts["alerts"]
    st.session_state.demo_video = artifacts["demo_video"]

    st.session_state.live_raw = pd.DataFrame(columns=hist_raw.columns)
    st.session_state.live = pd.DataFrame(columns=artifacts["hist_cal"].columns)

    run_id = sha256_obj({"base_eval": json_safe(st.session_state.base_eval), "ts": now_iso()})
    dataset_hash = sha256_obj(
        hist_raw.head(100)[["event_id", "request_id", "prediction_confidence"]].to_dict(orient="records")
    )

    try:
        persist_evaluation(run_id, "historical_holdout_test", st.session_state.base_eval, dataset_hash)
    except Exception:
        pass

    add_audit(
        action="baseline_evaluation",
        object_type="evaluation",
        object_id=run_id,
        result=(
            f"raw_ece={st.session_state.base_eval['raw']['ece']:.3f}; "
            f"iso_ece={st.session_state.base_eval['isotonic']['ece']:.3f}"
        ),
        evidence_hash=run_id,
        note="Chronological holdout evaluation on synthetic data.",
    )

    if st.session_state.demo_video is None:
        add_audit(
            action="video_generation_failed",
            object_type="demo",
            object_id="mp4",
            result="Video backend unavailable",
            evidence_hash="",
            note="Install av or opencv-python-headless for MP4 generation.",
        )

    st.session_state.initialized = True


def rebuild_artifacts() -> None:
    config = st.session_state.config
    hist_raw = st.session_state.hist_raw
    calibrators = st.session_state.calibrators
    prior_rate = st.session_state.prior_base_rate

    artifacts = compute_base_artifacts(
        hist_raw=hist_raw,
        calibrators=calibrators,
        config=config,
        prior_rate=prior_rate,
        boot=80,
    )

    st.session_state.hist_cal = artifacts["hist_cal"]
    st.session_state.base_eval = artifacts["base_eval"]
    st.session_state.reliability_raw = artifacts["reliability_raw"]
    st.session_state.reliability_cal = artifacts["reliability_cal"]
    st.session_state.trend = artifacts["trend"]
    st.session_state.initial_alerts = artifacts["alerts"]
    st.session_state.demo_video = artifacts["demo_video"]
    st.session_state.live = pd.DataFrame(columns=artifacts["hist_cal"].columns)
    st.session_state.live_raw = pd.DataFrame(columns=hist_raw.columns)
    st.session_state.metrics_history = []
    st.session_state.auto_ran = False

    add_audit(
        action="rebuild_artifacts",
        object_type="system",
        object_id="baseline",
        result="recomputed calibration artifacts and demo video",
        evidence_hash=sha256_obj(json_safe(model_dict(config))),
        note="Configuration change rebuild.",
    )


def update_config(approved_by: str, **changes: Any) -> None:
    old = st.session_state.config
    data = model_dict(old)
    data.update(changes)

    low = float(data.get("abstain_low", 0.35))
    high = float(data.get("abstain_high", 0.65))
    if low > high:
        raise ValueError("abstain_low must be <= abstain_high")

    new_config = AppConfig(**data)
    st.session_state.config = new_config
    st.session_state.needs_rebuild = True

    add_audit(
        action="config_change",
        object_type="config",
        object_id="app_config",
        result="applied",
        evidence_hash=sha256_obj(json_safe(model_dict(new_config))),
        approval_required=True,
        approved_by=approved_by,
        note=json.dumps(json_safe(changes), default=str),
    )


# =============================================================================
# LIVE SIMULATION
# =============================================================================

def run_live_simulation(
    ticks: int = 35,
    interval: float = 0.15,
    drift: float = 0.0,
    failure_mode: Optional[str] = None,
    label: str = "Simulation",
) -> None:
    config = st.session_state.config
    calibrators = st.session_state.calibrators
    prior_rate = st.session_state.prior_base_rate
    active_method = st.session_state.active_method

    status_ph = st.empty()
    metrics_ph = st.empty()
    alerts_ph = st.empty()
    table_ph = st.empty()
    progress = st.progress(0.0, text=f"{label}: starting")

    status_ph.markdown(f"**{label}** is ingesting synthetic streaming batches and evaluating calibration online.")

    for i in range(ticks):
        batch_id = f"batch-{int(time.time())}-{i}"
        seed_offset = len(st.session_state.metrics_history) + i

        batch = generate_stream_batch(
            batch_id=batch_id,
            n=80,
            drift=drift,
            failure_mode=failure_mode,
            seed_offset=seed_offset,
            active_version="v2_candidate",
        )

        canonical, report = ingest_and_persist(batch, batch_id)

        cal_batch = apply_calibration(
            canonical,
            method=active_method,
            calibrators=calibrators,
            config=config,
            prior_rate=prior_rate,
        )

        st.session_state.live_raw = (
            pd.concat([st.session_state.live_raw, canonical], ignore_index=True).tail(5000)
        )
        st.session_state.live = (
            pd.concat([st.session_state.live, cal_batch], ignore_index=True).tail(5000)
        )

        rolling = st.session_state.live.tail(1200)

        raw_m = evaluate_calibration(rolling, config, "prediction_confidence", boot=0)
        cal_m = evaluate_calibration(rolling, config, "calibrated_confidence", boot=0)

        avg_drift = float(rolling["drift_score"].mean()) if len(rolling) and "drift_score" in rolling else float("nan")

        point = {
            "tick": i,
            "timestamp": now_iso(),
            "ece_raw": raw_m["ece"],
            "ece_calibrated": cal_m["ece"],
            "brier_calibrated": cal_m["brier"],
            "missing_label_rate": cal_m["missing_label_rate"],
            "schema_error_rate": cal_m["schema_error_rate"],
            "review_rate": cal_m["review_rate"],
            "avg_drift": avg_drift,
            "fallback_used": bool(cal_m["fallback_used"] or raw_m["fallback_used"]),
            "batch_id": batch_id,
        }
        st.session_state.metrics_history.append(point)

        alerts = []

        if point["fallback_used"]:
            alerts.append(("ERROR", "Fallback active: insufficient labeled/sample quality. Routing to human review."))
        if np.isfinite(point["ece_calibrated"]) and point["ece_calibrated"] > 0.08:
            alerts.append(("WARN", f"Calibrated ECE {point['ece_calibrated']:.3f} above guardrail 0.08."))
        if point["missing_label_rate"] > config.max_missing_label_rate:
            alerts.append(("WARN", f"Missing label rate {point['missing_label_rate']:.3f} above guardrail."))
        if point["schema_error_rate"] > config.max_schema_error_rate:
            alerts.append(("ERROR", f"Schema error rate {point['schema_error_rate']:.3f} above guardrail."))
        if np.isfinite(avg_drift) and avg_drift > config.drift_alert_threshold:
            alerts.append(("WARN", f"Drift score {avg_drift:.3f} above alert threshold."))
        if report["duplicate_count"] > 0:
            alerts.append(("INFO", f"{report['duplicate_count']} duplicate request_ids flagged."))

        for level, msg in alerts:
            add_audit(
                action="simulation_alert",
                object_type="batch",
                object_id=batch_id,
                result=msg,
                evidence_hash=sha256_obj({"level": level, "msg": msg, "batch": batch_id}),
                note="Live simulation guardrail.",
            )

        md = f"""
#### Live calibration snapshot — tick {i + 1}/{ticks}

| Metric | Raw | Calibrated | Guardrail |
|---|---:|---:|---:|
| ECE | {point['ece_raw']:.3f} | {point['ece_calibrated']:.3f} | <= 0.08 |
| Brier | {raw_m['brier']:.3f} | {point['brier_calibrated']:.3f} | lower is better |
| Missing label rate | {raw_m['missing_label_rate']:.3f} | {point['missing_label_rate']:.3f} | <= {config.max_missing_label_rate:.2f} |
| Schema error rate | {raw_m['schema_error_rate']:.3f} | {point['schema_error_rate']:.3f} | <= {config.max_schema_error_rate:.2f} |
| Review rate | {raw_m['review_rate']:.3f} | {point['review_rate']:.3f} | monitored |
| Avg drift | {point['avg_drift']:.3f} | {point['avg_drift']:.3f} | <= {config.drift_alert_threshold:.2f} |
| Fallback | {raw_m['fallback_used']} | {point['fallback_used']} | safe failure mode |
"""

        metrics_ph.markdown(md)

        alerts_ph.empty()
        with alerts_ph.container():
            if alerts:
                for level, msg in alerts:
                    if level == "ERROR":
                        st.error(msg)
                    elif level == "WARN":
                        st.warning(msg)
                    else:
                        st.info(msg)
            else:
                st.success("No guardrail breaches in this tick.")

        if len(rolling) > 0:
            show_cols = [c for c in DISPLAY_COLS if c in rolling.columns]
            table_ph.dataframe(rolling.tail(15)[show_cols], use_container_width=True, hide_index=True)
        else:
            table_ph.info("No valid rolling rows yet.")

        if i % 5 == 0 and len(rolling) > 0:
            run_id = sha256_obj({"tick": i, "point": json_safe(point), "ts": now_iso()})
            try:
                persist_evaluation(
                    run_id,
                    "live_rolling_window",
                    {"calibrated": cal_m, "raw": raw_m},
                    sha256_obj(json_safe(point)),
                )
            except Exception:
                pass

        progress.progress((i + 1) / float(ticks), text=f"{label}: tick {i + 1}/{ticks}")
        time.sleep(interval)

    progress.empty()
    status_ph.markdown(f"**{label} complete.** {ticks} synthetic batches ingested, evaluated, audited, and persisted to local SQLite.")


# =============================================================================
# SMOKE TESTS
# =============================================================================

def run_smoke_tests() -> pd.DataFrame:
    results = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append(
            {
                "test": name,
                "status": "PASS" if condition else "FAIL",
                "detail": detail,
            }
        )

    y = np.array([1, 0])
    p = np.array([0.9, 0.1])
    _, ece, _, _ = reliability_bins(y, p, bins=2)
    check("ECE hand calculation", abs(ece - 0.1) < 1e-9, f"expected=0.100, got={ece:.6f}")

    bad = pd.DataFrame({"event_id": ["x"]})
    _, rep = validate_schema(bad)
    check("Schema rejects missing columns", rep["error_count"] > 0, f"errors={rep['error_count']}")

    if "hist_raw" in st.session_state and len(st.session_state.hist_raw) >= 2:
        dup = st.session_state.hist_raw.head(2).copy()
        dup["request_id"] = "duplicate-smoke-test"
        _, rep2 = validate_schema(dup)
        check("Duplicate request_ids detected", rep2["duplicate_count"] >= 2, f"duplicates={rep2['duplicate_count']}")
    else:
        check("Duplicate request_ids detected", False, "historical data unavailable")

    cfg = AppConfig(min_eval_n=100)
    tiny = pd.DataFrame(
        {
            "true_label": [1, 0],
            "prediction_confidence": [0.9, 0.1],
            "schema_valid": [1, 1],
            "duplicate_flag": [0, 0],
        }
    )
    m = evaluate_calibration(tiny, cfg, "prediction_confidence", boot=0)
    check("Fallback triggers on small sample", m["fallback_used"], f"n_eval={m['n_eval']}")

    h1 = make_historical(100)
    h2 = make_historical(100)
    check("Deterministic synthetic generation", h1.equals(h2), "same seed produces identical frame")

    p_det = np.linspace(0.02, 0.98, 200)
    y_det = (np.arange(200) % 2).astype(int)
    det_df = pd.DataFrame(
        {
            "prediction_confidence": p_det,
            "true_label": y_det,
            "schema_valid": 1,
            "duplicate_flag": 0,
        }
    )
    cal = fit_calibrators(det_df)
    check("Calibrator fitting returns usable objects", cal is not None, "fit_calibrators output")

    if cal is not None:
        cal_df = apply_calibration(det_df, "isotonic", cal, AppConfig(), 0.5)
        cm = evaluate_calibration(cal_df, AppConfig(), "calibrated_confidence", boot=0)
        check(
            "Isotonic calibration produces finite metrics",
            np.isfinite(cm["ece"]) and np.isfinite(cm["brier"]) and cm["n_eval"] == 200,
            f"ece={cm['ece']:.6f}, brier={cm['brier']:.6f}",
        )
    else:
        check("Isotonic calibration produces finite metrics", False, "calibrator unavailable")

    safe_cfg = AppConfig(safe_mode=True)
    small = pd.DataFrame(
        {
            "prediction_confidence": [0.10, 0.90],
            "input_quality_score": [0.90, 0.90],
            "schema_valid": [1, 1],
        }
    )
    safe_out = apply_calibration(small, "raw", None, safe_cfg, 0.5)
    check(
        "Safe mode forces human review",
        bool(safe_out["review_required"].all()) and bool((safe_out["decision_status"] == "human_review").all()),
        "safe_mode=True",
    )

    video = st.session_state.get("demo_video")
    check("Live MP4 demo generated", video is not None and len(video) > 1000, f"bytes={0 if video is None else len(video)}")

    check("Audit log initialized", len(st.session_state.get("audit", [])) > 0, f"entries={len(st.session_state.get('audit', []))}")

    return pd.DataFrame(results)


# =============================================================================
# PRODUCT DOCUMENTATION CONTENT
# =============================================================================

PRODUCT_BRIEF_MD = f"""
# {PROJECT_TITLE}
## {PROJECT_SUBTITLE}

### Real operational problem
Production AI systems often expose a raw model score as if it were a probability. That creates unsafe automation:
- a fraud/risk model says “0.92 confidence” but is wrong far more often than 8% of the time;
- a content moderation model auto-rejects borderline cases because thresholds were tuned on ROC, not calibration;
- an industrial vision model reports high confidence on out-of-distribution parts;
- a medical triage assistant prioritizes studies using uncalibrated probabilities, distorting queue order.

The operational problem is: **how do we know when a model’s confidence is trustworthy, when to abstain, when to route to human review, and when to roll back?**

### Why a large technology company would pay to solve it
- Reduces costly false auto-actions and unnecessary human escalations.
- Improves trust in AI automation and supports regulatory/internal audit.
- Prevents silent model degradation after deployment.
- Provides a governed evaluation layer reusable across many models.
- Converts “model quality” into operational SLIs/SLOs: ECE, Brier, review rate, fallback rate, drift alert latency.

### Who uses it
- ML platform engineers
- model owners / applied scientists
- risk and trust & safety operations analysts
- SRE / reliability engineers
- governance, risk, compliance reviewers

### What decisions it improves
- promote / block a model version
- apply recalibration or rollback
- adjust abstain / human-review thresholds
- trigger incident response when calibration degrades
- produce audit evidence for internal or regulated review

---

## Three realistic user journeys

### Journey 1 — ML platform engineer before release
**Legacy workflow:** engineer opens notebooks, computes reliability diagrams manually, compares metrics across branches in spreadsheets, and guesses whether confidence is usable.

**Improved workflow:**
1. Upload or point to a validation dataset with fields:
   `event_id`, `timestamp`, `model_version`, `segment`, `prediction_confidence`, `true_label`, `input_quality_score`, `schema_valid`.
2. System validates schema, detects duplicates and missing labels.
3. System fits Platt and isotonic calibrators on a chronological validation split.
4. System reports holdout ECE, Brier, log loss, AUROC, AUPRC, subgroup gaps, bootstrap CIs.
5. Engineer sees: isotonic reduces ECE from 0.11 to 0.04 on enterprise segment but worsens free-tier gap.
6. Decision: block release until segment-specific recalibration is approved.

**Acceptance check:** release package includes dataset hash, calibrator version, holdout metrics, subgroup table, and approval record.

---

### Journey 2 — Risk operations analyst during live incident
**Legacy workflow:** dashboard shows score distribution and latency, but not whether high scores are actually correct. Analyst manually samples cases and loses hours.

**Improved workflow:**
1. Live stream emits prediction events with delayed labels.
2. System continuously computes rolling calibrated ECE and review rate.
3. At 14:32, mobile segment ECE crosses 0.08 and missing-label rate spikes.
4. System raises alert: “Calibration degraded; fallback to human review recommended.”
5. Analyst drills into evidence: event `evt-8472`, raw confidence 0.94, calibrated 0.69, bin gap +0.18, input quality 0.31.
6. Decision: enable safe mode, route all mobile predictions to human review, page model owner.

**Failure case handled:** label outage causes insufficient labeled sample; system does not trust online ECE and uses deterministic fallback prior plus human review.

**Acceptance check:** alert includes timestamp, batch id, metric value, guardrail, evidence hash, and audit entry.

---

### Journey 3 — Governance lead preparing audit evidence
**Legacy workflow:** metrics are scattered across Confluence, notebooks, and BI dashboards; provenance is unclear.

**Improved workflow:**
1. Governance lead selects model version and date range.
2. System exports:
   - canonical event sample,
   - schema validation report,
   - calibration method and fitted rows,
   - holdout metrics with confidence intervals,
   - subgroup performance,
   - human approval log,
   - fallback incidents.
3. Lead verifies that consequential actions required approval:
   - applying recalibration,
   - changing abstain thresholds,
   - disabling safe mode,
   - exporting data.
4. Decision: approve quarterly model attestation or require remediation.

**Acceptance check:** audit trail is append-only in demo DB and exportable as JSON.

---

## Functional requirements
- FR1: ingest synthetic or licensed prediction events with delayed labels.
- FR2: validate schema, units, ranges, duplicates, late arrivals, malformed input.
- FR3: maintain canonical data model and provenance hash.
- FR4: compute reliability bins, ECE, MCE, Brier, log loss, AUROC, AUPRC.
- FR5: fit and compare transparent calibration baselines: raw, Platt, isotonic.
- FR6: report uncertainty via bootstrap confidence intervals and bin sample sizes.
- FR7: enforce abstain band and human-review routing.
- FR8: detect data-quality and calibration drift failures.
- FR9: provide deterministic fallback when evaluation quality is inadequate.
- FR10: persist events, evaluations, and audit log in SQLite.
- FR11: provide dashboard with filters, drill-downs, trends, explanations, audit history.
- FR12: generate an MP4 live demo video from synthetic data on launch.
- FR13: require human approval for consequential actions.
- FR14: export data, reports, and audit trails.

## Non-functional requirements
- NFR1: core prototype runs on Windows 10/11, CPU-only, no paid APIs.
- NFR2: deterministic reproduction from fixed seed.
- NFR3: single-file Streamlit app for demo; modular boundaries internally.
- NFR4: local SQLite persistence with retention and export.
- NFR5: p95 batch evaluation target under 2 seconds for 1,000 rows on laptop CPU.
- NFR6: memory target under 1 GB for demo workload.
- NFR7: all demo data synthetic or clearly licensed.
- NFR8: tests cover parsing, validation, calibration math, fallback, persistence, video generation.
- NFR9: audit log records actor, action, object, result, evidence hash, approval.
- NFR10: safe failure mode is abstain / human review, not silent auto-accept.

## Assumptions
- Binary classification with probability-like confidence input.
- Labels arrive with delay; not every event is labeled immediately.
- Initial demo uses synthetic data only.
- Calibration layer does not retrain the base model.

## Dependencies
- Python packages listed in embedded requirements.
- Local filesystem for SQLite DB and generated MP4.
- Optional Docker for deployment parity.

## Out of scope for prototype
- Real patient/customer PII.
- Multi-class deep uncertainty estimation.
- Automated model retraining.
- Production Kubernetes serving.
- Paid cloud APIs.
- Legal/regulatory certification.

## Release milestones
- M0: environment and configuration.
- M1: synthetic/open data preparation.
- M2: schema validation and canonical storage.
- M3: baseline reliability analytics.
- M4: Platt/isotonic calibration and holdout evaluation.
- M5: dashboard, filters, evidence inspector.
- M6: live simulation, alerts, fallback, audit.
- M7: MP4 demo video generation.
- M8: tests, packaging, documentation, acceptance.

## Prototype vs production enterprise service
**Prototype:**
- Streamlit single-file app
- SQLite
- synthetic data
- local CPU
- manual approval checkboxes
- generated demo video

**Production service:**
- FastAPI/gRPC ingestion and evaluation APIs
- PostgreSQL/TimescaleDB or lakehouse
- Kafka/queue ingestion with idempotency
- RBAC/SSO, approval workflow, ticketing integration
- model registry linkage
- alerting to PagerDuty/Slack
- horizontal scaling, tenancy, encryption, backup/DR
- CI/CD, IaC, observability with OpenTelemetry
"""

ARCHITECTURE_MD = f"""
# Architecture, Data Contracts, and Algorithm Design

## Modular architecture
1. **Source adapters**
   - synthetic historical generator
   - synthetic stream simulator
   - CSV uploader (future)
   - event bus adapter (production)

2. **Ingestion**
   - batch id, ingestion timestamp
   - schema validation
   - duplicate detection
   - quarantine/reject malformed rows
   - provenance hashing

3. **Canonical data model**
   - one row per prediction event
   - stable field names, types, units
   - quality flags retained, not silently dropped

4. **Feature/signal processing**
   - input quality score
   - latency signal
   - missing-feature flag
   - drift score
   - late label flag

5. **Baseline analytics**
   - reliability bins
   - ECE, MCE, Brier, log loss
   - AUROC/AUPRC for ranking sanity
   - subgroup tables

6. **Model layer**
   - raw confidence
   - Platt scaling
   - isotonic regression
   - fallback prior

7. **Decision/scoring layer**
   - abstain band
   - review_required
   - safe mode
   - fallback activation

8. **API layer**
   - prototype: Streamlit session functions
   - production: FastAPI endpoints for ingest, evaluate, approve, export

9. **Dashboard**
   - live demo video
   - overview metrics
   - filters and drill-downs
   - evidence inspector
   - audit history

10. **Audit history**
   - append-only SQLite table
   - actor, action, object, result, evidence hash, approval

11. **Observability**
   - structured audit entries
   - metric history
   - alert events
   - DB write failures surfaced

## Ownership and failure boundaries
- Ingestion owns schema validity. If schema fails, rows are flagged/rejected and evaluation falls back.
- Calibration layer owns probability transformation. If fitted data is insufficient, it returns None and decision layer uses raw/prior fallback.
- Decision layer owns automation policy. If safe mode or fallback is active, auto-accept is disabled.
- Dashboard is read/write for demo controls but does not bypass approval.
- Audit store is append-only in prototype; production would use immutable log.

## Failure cases and acceptance checks
- **Missing required column:** ingestion returns error report; acceptance: audit entry and rejected row count > 0.
- **Confidence outside [0,1]:** row marked schema_valid=0; acceptance: excluded from calibration metrics.
- **Duplicate request_id:** flagged; acceptance: excluded from metrics but visible in data-quality panel.
- **Late labels:** retained with late_arrival_flag; acceptance: missing-label rate reported.
- **Label outage:** missing_label_rate exceeds guardrail; acceptance: fallback_used=True and review_required=True.
- **Calibration degradation:** ECE > 0.08; acceptance: alert raised and recorded.
"""

DATA_CONTRACT_MD = """
# Data contract

| Field | Type | Unit / Range | Description |
|---|---|---:|---|
| event_id | string | unique | canonical event identifier |
| request_id | string | non-unique allowed | upstream request id; duplicates flagged |
| timestamp | datetime ISO8601 | UTC | prediction time |
| model_id | string |  | model family |
| model_version | string |  | deployed version |
| segment | string | web/mobile/enterprise/partner | business segment |
| user_tier | string | free/plus/pro/enterprise | customer tier |
| feature_latency_ms | float | milliseconds | feature fetch latency |
| feature_missing_flag | int | 0/1 | critical feature missing |
| input_quality_score | float | 0..1 | upstream input quality |
| prediction_confidence | float | 0..1 | raw model confidence |
| predicted_label | int | 0/1/-1 | -1 means invalid/unknown |
| true_label | int/float | 0/1/NaN | delayed ground truth |
| outcome_value | float | currency-like | synthetic business impact |
| feedback_received | int | 0/1 | label returned |
| feedback_delay_hours | float | hours | label latency |
| drift_score | float | 0..1 | synthetic drift signal |
| schema_valid | int | 0/1 | passed validation |
| duplicate_flag | int | 0/1 | duplicate request detected |
| late_arrival_flag | int | 0/1 | label arrived late |
| provenance_hash | string | hex | row lineage hash |
| batch_id | string |  | ingestion batch |
| ingested_at | datetime ISO8601 | UTC | storage time |

## Handling rules
- **Missing values:** numeric fields coerced; invalid confidence marks schema_valid=0.
- **Duplicates:** request_id duplicates flagged and excluded from calibration metrics.
- **Late-arriving records:** retained; late_arrival_flag and feedback_delay_hours preserved.
- **Malformed input:** rejected from canonical evaluation but audited.
- **Outliers:** latency > 10000 ms marked invalid; confidence outside [0,1] marked invalid.
- **Schema changes:** missing required columns cause batch rejection and audit error.
"""

MERMAID_ARCH = """
```mermaid
flowchart LR
  A[Synthetic Generator] --> B[Source Adapter]
  B --> C[Ingestion Validator]
  C --> D[Canonical Store SQLite]
  C --> E[Quality Report]
  D --> F[Reliability Analytics]
  D --> G[Calibration Model Layer]
  G --> H[Platt / Isotonic / Fallback]
  F --> H
  H --> I[Decision Scoring]
  I --> J[Dashboard Streamlit]
  I --> K[Audit Log]
  D --> K
  J --> L[Live MP4 Demo Renderer]
```
"""

MERMAID_SEQ = """
```mermaid
sequenceDiagram
  participant U as User
  participant S as Streamlit App
  participant V as Validator
  participant DB as SQLite
  participant C as Calibration Engine
  participant D as Decision Layer
  participant A as Audit Log

  U->>S: Launch app
  S->>V: Load synthetic historical batch
  V->>DB: Persist canonical events
  S->>C: Fit Platt/isotonic on validation split
  C->>D: Produce calibrated confidence
  D->>A: Record baseline evaluation
  S->>L: Generate MP4 demo video
  U->>S: Run live simulation
  S->>V: Ingest streaming batches
  V->>DB: Persist events and alerts
  S->>C: Rolling calibration evaluation
  C->>D: Apply abstain / fallback policy
  D->>A: Append audit entries
  S->>U: Show metrics, evidence, video
```
"""

SCALE_PLAN_MD = """
# Scale plan without premature complexity

## Initial modular monolith
Keep these in one process:
- ingestion validation
- calibration evaluation
- decision scoring
- dashboard
- SQLite audit/event store

This is appropriate for prototype and small-team pilots.

## Measured evidence that would justify splitting services
- sustained ingest > 500 events/sec with p95 evaluation latency > 2s
- multiple teams needing independent deploy cycles
- DB write contention or SQLite lock errors in production load tests
- need for strict tenant isolation and RBAC
- need for replayable event streaming with exactly-once semantics

## Scaling mechanisms
- **Batching:** evaluate rolling windows of 500–2,000 rows.
- **Caching:** cache fitted calibrators and reliability tables by dataset hash.
- **Queues:** production ingestion via Kafka/PubSub; prototype uses in-memory batches.
- **Retries:** idempotent event_id primary key; safe re-ingest.
- **Idempotency:** INSERT OR IGNORE on event_id; run_id primary key for evaluations.
- **Pagination:** dashboard filters and table tails; production API cursor pagination.
- **Backpressure:** reject or defer non-critical evaluation when queue depth exceeds SLO.
- **Partitioning:** by model_version, tenant, segment.
- **Graceful degradation:** if calibration data insufficient, fallback prior + human review.
"""

TESTING_PROTOCOL_MD = """
# Evaluation protocol

## Train / validation / test separation
- chronological split:
  - train: oldest 60%
  - validation: next 20%
  - test: newest 20%
- calibrators fit only on validation
- final reported metrics on test holdout

## Leakage checks
- no future timestamps in training/validation
- duplicate request_ids excluded from metrics
- missing labels excluded from supervised calibration metrics but reported as quality risk
- raw model probabilities are not refit on test data

## Metrics and formulas
- Reliability bin: `B_k`
- `conf_k = mean(p_i)` for `i in B_k`
- `acc_k = mean(y_i)` for `i in B_k`
- `ECE = sum_k (n_k / N) * |acc_k - conf_k|`
- `MCE = max_k |acc_k - conf_k|`
- `Brier = (1/N) * sum_i (p_i - y_i)^2`
- `LogLoss = -(1/N) * sum_i [y_i log(p_i) + (1-y_i) log(1-p_i)]`
- `AUROC`: ranking probability of positive before negative
- `AUPRC`: area under precision-recall curve
- `Review rate = count(review_required) / N`
- `Subgroup gap = max_segment(ECE) - min_segment(ECE)`

## Confidence intervals
- bootstrap percentile CI for ECE and Brier
- default 120 resamples in baseline, 0 in fast live ticks

## Baseline comparisons
- raw confidence
- Platt scaling
- isotonic regression
- deterministic fallback prior

## What metrics cannot prove
- calibration on labeled subset does not prove calibration on unlabeled future distribution.
- AUROC does not prove probability correctness.
- synthetic demo does not prove production business ROI.
- low aggregate ECE does not prove fairness across subgroups.
"""

SECURITY_MD = """
# Security, reliability, and governance

## Least privilege
- prototype runs as local user
- SQLite file permissions controlled by OS
- no secrets required

## Input validation
- required columns enforced
- numeric ranges enforced
- duplicates and late labels flagged
- malformed rows excluded from decision metrics

## Secrets management
- `.env.example` contains placeholders only
- no API keys used
- no secrets committed

## Encryption and retention
- local demo: OS-level disk encryption assumed
- production: TLS in transit, AES-256 at rest
- retention: configurable; demo documents 180-day audit retention

## Audit logging
- append-only audit table
- records actor, action, object, result, evidence hash, approval

## Threat model
- data poisoning via malicious upload
- label leakage through future timestamps
- overreliance on confidence as authority
- schema drift causing silent failure
- DB lock or disk full
- unauthorized threshold change

## Safe failure mode
- if evaluation quality is inadequate: fallback prior + human review
- if schema error rate high: reject batch and alert
- if calibration guardrail breached: recommend safe mode
- consequential actions require human approval

## Rollback and kill switch
- rollback recalibration to initial historical validator fit
- kill switch enables safe mode: all predictions route to human review
- both actions audited and approval-gated
"""

DEMO_SCRIPT_MD = r"""
# 5–10 minute live demo script

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Expected immediate behavior
- app loads synthetic historical data
- validates schema
- fits isotonic and Platt calibrators
- generates an MP4 demo video automatically
- starts a live simulation in the Live Demo tab
- shows metrics, alerts, latest rows, and audit entries

## Normal scenario
1. Open **Live Demo**.
2. Watch generated MP4: reliability diagram, ECE trend, confidence histogram, status panel.
3. Observe auto simulation ticks and rolling calibrated ECE.

## Difficult edge case
1. Go to **Calibration Workbench**.
2. Filter to `mobile` segment and low input quality.
3. Inspect an event with raw confidence 0.93 but calibrated confidence near abstain band.
4. Evidence panel shows bin gap, provenance hash, schema flags, and audit trail.

## Failure / recovery scenario
1. In sidebar, click **Inject label outage**.
2. Missing label rate rises; fallback activates; alerts appear.
3. Click **Run 12-second recovery burst**.
4. Valid batches restore rolling metrics; fallback clears.

## Explanation of result
1. Open **Testing & Acceptance**.
2. Run smoke tests.
3. Review hand-calculated ECE fixture, fallback trigger, duplicate detection, deterministic generation, and video generation check.
"""

BUSINESS_VALUE_MD = """
# Business value hypothesis

## Hypothesis
A calibrated confidence and abstention layer reduces costly wrong auto-actions and unnecessary human reviews compared with raw-confidence automation.

## Outcome metrics
- reduction in wrong auto-accept rate
- reduction in manual review volume at equal or better error rate
- reduction in time-to-detect calibration drift
- improvement in audit readiness score

## Leading indicators
- calibration job success rate
- label return rate
- schema validation pass rate
- evaluator latency
- alert acknowledgment time

## Quality metrics
- ECE, MCE, Brier, log loss
- AUROC/AUPRC as ranking sanity checks
- subgroup ECE gap
- review rate and precision of reviewed cases

## Reliability metrics
- ingestion error rate
- pipeline p95 latency
- fallback rate
- audit write success rate
- DB availability

## Guardrails
- calibrated ECE <= 0.08 for auto-accept eligibility
- missing label rate <= 0.25
- schema error rate <= 0.05
- subgroup ECE gap <= 0.05
- no auto-accept in safe mode
- human approval required for threshold or recalibration changes

## Baseline
Baseline is current raw-confidence policy:
- raw threshold 0.5 or business threshold
- manual review queue without calibrated abstention
- periodic notebook calibration checks

## Controlled evaluation for incremental value
1. Run system in shadow mode on same traffic.
2. Use chronological holdout or matched segment comparison.
3. Compare:
   - wrong auto-accept rate
   - review volume
   - review precision
   - latency and cost
4. Establish statistical significance with bootstrap CIs.
5. Only claim incremental value if calibrated policy improves outcome metric without violating guardrails.

## Important honesty constraint
The demo uses synthetic data. It proves mechanism and reproducibility, not customer-specific ROI.
"""

RISK_REGISTER_MD = """
# Risk register

| Risk | Category | Likelihood | Impact | Mitigation | Human approval required? |
|---|---|---:|---:|---|---|
| Feedback labels arrive too late or not at all | Data availability | High | High | missing-rate guardrail, fallback prior, human review | Yes for threshold changes |
| Model confidence is overconfident after drift | Model error | High | High | rolling ECE alerts, isotonic recalibration, safe mode | Yes to apply recalibration |
| Synthetic data mistaken for real performance | Misuse | Medium | High | explicit labels: simulated/synthetic; no real ROI claims | Yes for external demo use |
| Privacy exposure if real events uploaded | Privacy | Medium | High | demo uses synthetic only; production requires de-identification and access controls | Yes for data export |
| Fairness: subgroup miscalibration hidden by aggregate ECE | Fairness | Medium | High | segment tables, subgroup gap guardrail | Yes to waive guardrail |
| Malformed batch corrupts metrics | Operational failure | Medium | Medium | schema validation, quarantine, idempotent event_id | No, automatic rejection |
| DB lock / disk full | Operational failure | Low | Medium | SQLite WAL, retention, backup/restore docs | Yes for deletion |
| Users overtrust dashboard explanations | Adoption / misuse | Medium | High | evidence inspector, uncertainty CIs, safe failure mode | Yes for auto-accept policy |
| Compute cost grows with bootstrap evaluation | Cost | Medium | Low | fast live mode boot=0, scheduled deep evaluation | No |
| Unauthorized threshold change | Governance | Low | High | approval checkbox, audit log, kill switch | Yes |
"""

ACCEPTANCE_CHECKLIST_MD = """
# Acceptance checklist

## Product
- [ ] Problem, users, journeys, and decisions are documented.
- [ ] Functional and non-functional requirements are concrete.
- [ ] Prototype vs production scope is explicit.
- [ ] Value hypothesis and guardrails are defined.
- [ ] Risk register identifies human-approval actions.

## Architecture and data
- [ ] Modular architecture with ownership boundaries is documented.
- [ ] Data contract includes field names, types, units, provenance, retention.
- [ ] Missing, duplicate, late, malformed, outlier, schema-change handling is specified.
- [ ] Mermaid architecture and sequence diagrams are included.
- [ ] Transparent baseline and advanced methods are compared.
- [ ] Deterministic fallback is implemented.

## Implementation
- [ ] Python-first, Windows-compatible, CPU-only core demo.
- [ ] No paid API keys required.
- [ ] Streamlit single-file app.
- [ ] Synthetic data auto-loads.
- [ ] MP4 live demo video generated on launch.
- [ ] Simulation starts automatically and remains interactive.
- [ ] SQLite persistence for events, evaluations, audit.
- [ ] Dashboard includes filters, drill-downs, trends, explanations, audit history.

## Testing and reliability
- [ ] Unit/smoke tests for parsing, validation, calibration math, fallback, determinism.
- [ ] Integration/e2e-style tests from ingestion to user-visible result.
- [ ] Chronological split and leakage checks.
- [ ] Metrics formulas documented.
- [ ] Subgroup checks included.
- [ ] Security/threat model and safe failure mode documented.

## Demo and delivery
- [ ] Fresh developer can reproduce from README commands.
- [ ] Demo includes normal, edge, failure/recovery, explanation.
- [ ] License inventory and data provenance documented.
- [ ] Known limitations and roadmap included.
"""


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    with st.spinner("Loading synthetic data, fitting calibration baselines, and generating MP4 demo..."):
        init_session()
except Exception as exc:
    st.error(f"Initialization failed: {exc}")
    st.code(traceback.format_exc())
    st.stop()


# Sidebar
with st.sidebar:
    st.header("Controls")
    st.caption(f"Fixed synthetic seed: **{SEED}**")
    st.caption("No API keys. No external services. CPU-only core demo.")

    st.download_button(
        "Download requirements.txt",
        data=REQUIREMENTS_TEXT.encode("utf-8"),
        file_name="requirements.txt",
        mime="text/plain",
        use_container_width=True,
    )

    cfg = st.session_state.config

    bins = st.slider("Reliability bins", min_value=2, max_value=30, value=int(cfg.bins))
    abstain_low = st.slider("Abstain low", min_value=0.0, max_value=0.60, value=float(cfg.abstain_low), step=0.01)
    abstain_high = st.slider("Abstain high", min_value=0.40, max_value=1.00, value=float(cfg.abstain_high), step=0.01)
    safe_mode = st.checkbox("Safe mode: route all predictions to human review", value=bool(cfg.safe_mode))

    approval = st.checkbox("I approve consequential configuration changes")

    if st.button("Apply configuration", disabled=not approval, use_container_width=True):
        try:
            update_config(
                approved_by="demo_user",
                bins=int(bins),
                abstain_low=float(abstain_low),
                abstain_high=float(abstain_high),
                safe_mode=bool(safe_mode),
            )
            st.success("Configuration applied. Recompute artifacts to refresh baseline video.")
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Current configuration")
    st.json(json_safe(model_dict(st.session_state.config)))

    if st.session_state.get("needs_rebuild", False):
        if st.button("Recompute baseline artifacts and video", use_container_width=True):
            with st.spinner("Recomputing calibration artifacts..."):
                rebuild_artifacts()
            st.session_state.needs_rebuild = False
            st.success("Artifacts rebuilt.")

    st.divider()
    st.subheader("Demo actions")

    if st.button("Run 12-second recovery burst", use_container_width=True):
        st.session_state.pending_simulation = {
            "ticks": 20,
            "interval": 0.15,
            "drift": 0.0,
            "failure_mode": None,
            "label": "Recovery burst",
        }

    if st.button("Inject drift", use_container_width=True):
        st.session_state.pending_simulation = {
            "ticks": 20,
            "interval": 0.15,
            "drift": 0.55,
            "failure_mode": None,
            "label": "Drift injection",
        }

    if st.button("Inject label outage", use_container_width=True):
        st.session_state.pending_simulation = {
            "ticks": 18,
            "interval": 0.15,
            "drift": 0.2,
            "failure_mode": "labels",
            "label": "Label outage",
        }

    if st.button("Force schema failure", use_container_width=True):
        st.session_state.pending_simulation = {
            "ticks": 12,
            "interval": 0.12,
            "drift": 0.1,
            "failure_mode": "schema",
            "label": "Schema failure",
        }

    if st.button("Regenerate MP4 demo video", use_container_width=True):
        with st.spinner("Regenerating MP4..."):
            artifacts = compute_base_artifacts(
                hist_raw=st.session_state.hist_raw,
                calibrators=st.session_state.calibrators,
                config=st.session_state.config,
                prior_rate=st.session_state.prior_base_rate,
                boot=80,
            )
            st.session_state.demo_video = artifacts["demo_video"]
            st.session_state.base_eval = artifacts["base_eval"]
            st.session_state.reliability_raw = artifacts["reliability_raw"]
            st.session_state.reliability_cal = artifacts["reliability_cal"]
            st.session_state.trend = artifacts["trend"]
        if st.session_state.demo_video:
            st.success("MP4 regenerated.")
        else:
            st.warning("Video backend unavailable.")

    st.divider()

    if st.button("Reset demo (clears local SQLite and session)", use_container_width=True):
        conn = st.session_state.get("db_conn")
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

        for suffix in ("", "-wal", "-shm"):
            path = DB_PATH + suffix
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

        st.session_state.clear()
        st.rerun()

    st.divider()
    st.subheader("Downloads")

    st.download_button(
        "Download historical synthetic CSV",
        data=st.session_state.hist_cal.to_csv(index=False).encode("utf-8"),
        file_name="historical_synthetic_calibration.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if not st.session_state.live.empty:
        st.download_button(
            "Download live simulation CSV",
            data=st.session_state.live.to_csv(index=False).encode("utf-8"),
            file_name="live_simulation_calibration.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.download_button(
        "Download audit JSON",
        data=to_json_bytes(st.session_state.audit),
        file_name="audit_log.json",
        mime="application/json",
        use_container_width=True,
    )

    report_payload = {
        "project": PROJECT_TITLE,
        "code_version": CODE_VERSION,
        "seed": SEED,
        "config": model_dict(st.session_state.config),
        "quality_report": st.session_state.quality_report,
        "base_eval": st.session_state.base_eval,
        "initial_alerts": st.session_state.get("initial_alerts", []),
    }

    st.download_button(
        "Download calibration report JSON",
        data=to_json_bytes(report_payload),
        file_name="calibration_report.json",
        mime="application/json",
        use_container_width=True,
    )


# Tabs
tabs = st.tabs(
    [
        "🎬 Live Demo",
        "📄 Product Brief",
        "🏗️ Architecture & Data",
        "📊 Calibration Workbench",
        "💵 Value Calculator",
        "🧪 Testing & Acceptance",
        "🚀 Build, Deploy & License",
        "🔐 Audit & Governance",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1: LIVE DEMO
# -----------------------------------------------------------------------------
with tabs[0]:
    st.title(f"{PROJECT_TITLE} — Live Demo")
    st.caption("Generated MP4, auto-loaded synthetic data, automatic simulation, interactive controls.")

    st.subheader("Generated live demo video (MP4)")
    if st.session_state.get("demo_video"):
        st.video(
            st.session_state.demo_video,
            format="video/mp4",
            autoplay=True,
            muted=True,
            loop=True,
        )
        st.download_button(
            "Download demo MP4",
            data=st.session_state.demo_video,
            file_name="model_calibration_live_demo.mp4",
            mime="video/mp4",
        )
    else:
        st.warning("MP4 backend unavailable. Install `av` or `opencv-python-headless` and regenerate video.")

    st.divider()

    pending = st.session_state.get("pending_simulation")
    if pending is not None:
        st.session_state.pending_simulation = None
        try:
            run_live_simulation(**pending)
        except Exception as exc:
            st.error(f"Pending simulation failed: {exc}")
            st.code(traceback.format_exc())
    elif not st.session_state.auto_ran:
        st.session_state.auto_ran = True
        try:
            run_live_simulation(
                ticks=35,
                interval=0.15,
                drift=0.18,
                failure_mode=None,
                label="Automatic demo simulation",
            )
        except Exception as exc:
            st.error(f"Automatic simulation failed: {exc}")
            st.code(traceback.format_exc())

    st.subheader("Live metric history")
    if st.session_state.metrics_history:
        mh = pd.DataFrame(st.session_state.metrics_history)
        st.line_chart(mh.set_index("tick")[["ece_raw", "ece_calibrated", "review_rate"]])
        st.dataframe(mh.tail(30), use_container_width=True, hide_index=True)
    else:
        st.info("No live simulation ticks yet. Use sidebar actions to run a burst.")

    st.subheader("Latest live rows")
    if not st.session_state.live.empty:
        show_cols = [c for c in DISPLAY_COLS if c in st.session_state.live.columns]
        st.dataframe(st.session_state.live.tail(50)[show_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Live table empty. Run a simulation burst.")

    st.divider()
    st.markdown(DEMO_SCRIPT_MD)


# -----------------------------------------------------------------------------
# TAB 2: PRODUCT BRIEF
# -----------------------------------------------------------------------------
with tabs[1]:
    st.title("Product Requirements and Operational Definition")
    st.markdown(PRODUCT_BRIEF_MD)

    st.divider()
    st.subheader("Measurable value hypothesis")
    st.markdown(BUSINESS_VALUE_MD)

    st.divider()
    st.subheader("Risk register")
    st.markdown(RISK_REGISTER_MD)

    st.divider()
    st.subheader("Actions requiring human approval")
    st.markdown(
        """
- Apply recalibration to live scoring.
- Change abstain thresholds.
- Enable or disable safe mode / kill switch.
- Waive a calibration or fairness guardrail.
- Export data beyond demo scope.
- Delete or retain-reduce audit history.
- Promote a model version based on this evaluation.
        """
    )


# -----------------------------------------------------------------------------
# TAB 3: ARCHITECTURE & DATA
# -----------------------------------------------------------------------------
with tabs[2]:
    st.title("Architecture, Data Contracts, Algorithms")
    st.markdown(ARCHITECTURE_MD)

    st.divider()
    st.subheader("Data contract")
    st.markdown(DATA_CONTRACT_MD)

    st.divider()
    st.subheader("Example canonical records")
    sample = st.session_state.hist_cal.head(3)[
        [
            "event_id",
            "request_id",
            "timestamp",
            "model_version",
            "segment",
            "prediction_confidence",
            "calibrated_confidence",
            "true_label",
            "input_quality_score",
            "schema_valid",
            "duplicate_flag",
            "late_arrival_flag",
            "provenance_hash",
        ]
    ]
    st.dataframe(sample, use_container_width=True, hide_index=True)

    st.download_button(
        "Download example records JSON",
        data=to_json_bytes(sample.to_dict(orient="records")),
        file_name="example_canonical_records.json",
        mime="application/json",
    )

    st.divider()
    st.subheader("Mermaid architecture diagram")
    st.code(MERMAID_ARCH.strip(), language="markdown")

    st.subheader("Mermaid sequence diagram")
    st.code(MERMAID_SEQ.strip(), language="markdown")

    st.divider()
    st.subheader("Algorithm choice and trade-offs")
    st.markdown(
        """
### Transparent baseline
- **Raw confidence**: no transformation; fastest, auditable, but often miscalibrated.
- **Platt scaling**: parametric, stable with smaller data, assumes logistic shape.
- **Isotonic regression**: non-parametric, flexible, needs more data, can overfit small bins.
- **Fallback prior**: deterministic base rate; safe when data quality is inadequate.

### Training vs inference
- Training/fitting: calibrators fit on chronological validation split only.
- Inference: calibrators transform live or holdout prediction_confidence.
- Evaluation: metrics computed on test holdout or rolling labeled window.

### Uncertainty / confidence reporting
- bin sample sizes
- bootstrap CIs for ECE and Brier
- missing-label rate and schema-error rate
- fallback flag

### Deterministic fallback
When:
- labeled evaluation rows < `min_eval_n`
- missing label rate > guardrail
- schema error rate > guardrail
- safe mode enabled

Then:
- use fallback prior or raw confidence
- force `review_required=True`
- emit alert and audit entry

### Acceptance checks
- raw vs isotonic vs Platt reported on same holdout.
- fallback triggers in label-outage simulation.
- duplicate and malformed rows excluded from metrics but visible in quality report.
"""
    )

    st.divider()
    st.subheader("Scale plan")
    st.markdown(SCALE_PLAN_MD)


# -----------------------------------------------------------------------------
# TAB 4: CALIBRATION WORKBENCH
# -----------------------------------------------------------------------------
with tabs[3]:
    st.title("Calibration Workbench")

    hist_cal = st.session_state.hist_cal
    config = st.session_state.config

    st.subheader("Filters")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])

    with c1:
        min_date = hist_cal["timestamp"].min().date()
        max_date = hist_cal["timestamp"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

    with c2:
        versions = sorted(hist_cal["model_version"].dropna().unique().tolist())
        selected_versions = st.multiselect("Model version", options=versions, default=versions)

    with c3:
        segments = sorted(hist_cal["segment"].dropna().unique().tolist())
        selected_segments = st.multiselect("Segment", options=segments, default=segments)

    with c4:
        min_quality = st.slider("Min input quality", 0.0, 1.0, 0.0, step=0.01)

    filtered = hist_cal[
        (hist_cal["timestamp"].dt.date >= start_date)
        & (hist_cal["timestamp"].dt.date <= end_date)
        & (hist_cal["model_version"].isin(selected_versions))
        & (hist_cal["segment"].isin(selected_segments))
        & (pd.to_numeric(hist_cal["input_quality_score"], errors="coerce") >= min_quality)
    ].copy()

    if filtered.empty:
        st.warning("No rows match filters.")
    else:
        st.subheader("Overview metrics")
        raw_m = evaluate_calibration(filtered, config, "prediction_confidence", boot=80)
        iso_m = evaluate_calibration(filtered, config, "calibrated_confidence", boot=80)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Rows", f"{len(filtered):,}")
        m2.metric("Labeled eval rows", f"{iso_m['n_eval']:,}")
        m3.metric("ECE raw", f"{raw_m['ece']:.3f}", delta=f"{raw_m['ece'] - iso_m['ece']:+.3f}", delta_color="inverse")
        m4.metric("ECE calibrated", f"{iso_m['ece']:.3f}")
        m5.metric("Brier calibrated", f"{iso_m['brier']:.3f}")
        m6.metric("Review rate", f"{iso_m['review_rate']:.3f}")

        if iso_m["ci"]:
            st.caption(
                f"Bootstrap 95% CI for calibrated ECE: "
                f"[{iso_m['ci'].get('ece_low', float('nan')):.3f}, {iso_m['ci'].get('ece_high', float('nan')):.3f}]"
            )

        if iso_m["fallback_used"]:
            st.error("Fallback active: evaluation quality is inadequate for trusted auto-accept.")

        st.divider()
        st.subheader("Reliability and confidence")

        y_raw, p_raw = labeled_arrays(filtered, "prediction_confidence")
        y_cal, p_cal = labeled_arrays(filtered, "calibrated_confidence")

        rel_raw, _, _, _ = reliability_bins(y_raw, p_raw, config.bins)
        rel_cal, _, _, _ = reliability_bins(y_cal, p_cal, config.bins)

        cc1, cc2 = st.columns(2)
        with cc1:
            fig = reliability_figure(rel_raw, rel_cal)
            st.pyplot(fig)
            plt.close(fig)

        with cc2:
            fig = confidence_histogram_figure(filtered)
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Trend")
        trend = compute_trend(filtered, windows=24, bins=config.bins)
        fig = trend_figure(trend)
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("Subgroup checks")
        seg_df = segment_table(filtered, config, "calibrated_confidence")
        if not seg_df.empty:
            st.dataframe(
                seg_df[["segment", "n_eval", "ece", "brier", "auc", "review_rate", "missing_label_rate", "fallback_used"]],
                use_container_width=True,
                hide_index=True,
            )

            finite_ece = seg_df["ece"].dropna()
            if len(finite_ece) > 1:
                gap = float(finite_ece.max() - finite_ece.min())
                if gap > 0.05:
                    st.warning(f"Subgroup ECE gap is {gap:.3f}, above fairness guardrail 0.05.")
                else:
                    st.success(f"Subgroup ECE gap is {gap:.3f}, within guardrail.")

        st.divider()
        st.subheader("Evidence inspector")
        selectable = filtered.head(1000)
        selected_event = st.selectbox("Select event_id", selectable["event_id"].tolist())
        row = filtered[filtered["event_id"] == selected_event].iloc[0]

        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("Raw confidence", f"{row['prediction_confidence']:.3f}")
            st.metric("Calibrated confidence", f"{row['calibrated_confidence']:.3f}")
        with e2:
            st.metric("True label", "NaN" if pd.isna(row["true_label"]) else f"{int(row['true_label'])}")
            st.metric("Review required", str(bool(row["review_required"])))
        with e3:
            st.metric("Input quality", f"{row['input_quality_score']:.3f}")
            st.metric("Drift score", f"{row['drift_score']:.3f}")

        p_sel = float(row["calibrated_confidence"]) if pd.notna(row["calibrated_confidence"]) else float("nan")
        if np.isfinite(p_sel):
            edges = np.linspace(0, 1, config.bins + 1)
            bin_idx = int(np.clip(np.digitize([p_sel], edges)[0] - 1, 0, config.bins - 1))
            bin_row = rel_cal[rel_cal["bin"] == bin_idx]
            if len(bin_row) > 0:
                br = bin_row.iloc[0]
                contribution = float(br["weighted_gap"]) if "weighted_gap" in br else float("nan")
                st.markdown(
                    f"""
**Bin explanation**
- Bin index: {bin_idx}
- Bin range: [{br['bin_lower']:.2f}, {br['bin_upper']:.2f}]
- Bin mean confidence: {br['conf']:.3f}
- Bin observed rate: {br['acc']:.3f}
- Bin gap: {br['gap']:+.3f}
- ECE contribution: {contribution:.4f}
- Provenance hash: `{row['provenance_hash']}`
                    """
                )

        st.json(
            json_safe(
                {
                    "event_id": row["event_id"],
                    "request_id": row["request_id"],
                    "timestamp": str(row["timestamp"]),
                    "model_version": row["model_version"],
                    "segment": row["segment"],
                    "prediction_confidence": float(row["prediction_confidence"]) if pd.notna(row["prediction_confidence"]) else None,
                    "calibrated_confidence": float(row["calibrated_confidence"]) if pd.notna(row["calibrated_confidence"]) else None,
                    "true_label": None if pd.isna(row["true_label"]) else int(row["true_label"]),
                    "review_required": bool(row["review_required"]),
                    "decision_status": row["decision_status"],
                    "provenance_hash": row["provenance_hash"],
                }
            )
        )

        event_audits = [a for a in st.session_state.audit if selected_event in str(a.get("object_id", ""))]
        if event_audits:
            st.dataframe(pd.DataFrame(event_audits).tail(20), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Recalibration proposal")
        st.markdown(
            """
This proposal fits temporary calibrators on the selected filtered window using a chronological inner split.
Applying it to live scoring is a consequential action and requires human approval.
            """
        )

        if st.button("Fit proposal on selected window"):
            prop_train, prop_val, prop_test = chronological_split(filtered, 0.6, 0.2)
            prop_cal = fit_calibrators(prop_val)

            if prop_cal is None:
                st.warning("Insufficient validation rows to fit proposal calibrator.")
                st.session_state.proposal = None
            else:
                test_raw = apply_calibration(prop_test, "raw", prop_cal, config, st.session_state.prior_base_rate)
                test_iso = apply_calibration(prop_test, "isotonic", prop_cal, config, st.session_state.prior_base_rate)

                raw_pm = evaluate_calibration(test_raw, config, "prediction_confidence", boot=50)
                iso_pm = evaluate_calibration(test_iso, config, "calibrated_confidence", boot=50)

                st.session_state.proposal = {
                    "calibrators": prop_cal,
                    "raw_metrics": raw_pm,
                    "iso_metrics": iso_pm,
                    "fitted_on": prop_cal.get("fitted_on", 0),
                    "created_at": now_iso(),
                    "filter_summary": {
                        "rows": len(filtered),
                        "versions": selected_versions,
                        "segments": selected_segments,
                        "start_date": str(start_date),
                        "end_date": str(end_date),
                        "min_quality": min_quality,
                    },
                }

                add_audit(
                    action="fit_recalibration_proposal",
                    object_type="proposal",
                    object_id=sha256_obj(json_safe(st.session_state.proposal["filter_summary"])),
                    result=f"raw_ece={raw_pm['ece']:.3f}, iso_ece={iso_pm['ece']:.3f}",
                    evidence_hash=sha256_obj(json_safe(st.session_state.proposal["filter_summary"])),
                    note="Proposal fitted but not applied.",
                )

        if st.session_state.get("proposal"):
            prop = st.session_state.proposal
            st.success(f"Proposal fitted on {prop['fitted_on']} validation rows at {prop['created_at']}.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"method": "raw", **prop["raw_metrics"]},
                        {"method": "proposal_isotonic", **prop["iso_metrics"]},
                    ]
                )[["method", "n_eval", "ece", "brier", "log_loss", "auc", "review_rate", "fallback_used"]],
                use_container_width=True,
                hide_index=True,
            )

            apply_approval = st.checkbox("I approve applying this recalibration proposal to live scoring")
            if st.button("Apply proposal to live scoring", disabled=not apply_approval):
                st.session_state.calibrators = prop["calibrators"]
                st.session_state.active_method = "isotonic"
                st.session_state.needs_rebuild = True
                add_audit(
                    action="apply_recalibration_proposal",
                    object_type="calibrator",
                    object_id=sha256_obj(json_safe(prop["filter_summary"])),
                    result="applied to session live scoring",
                    evidence_hash=sha256_obj(json_safe(prop["filter_summary"])),
                    approval_required=True,
                    approved_by="demo_user",
                    note="Consequential action approved in demo.",
                )
                st.success("Proposal applied. Recompute artifacts to refresh baseline dashboard/video.")

        st.divider()
        rollback_approval = st.checkbox("I approve rollback to initial calibrator")
        if st.button("Rollback to initial historical calibrator", disabled=not rollback_approval):
            train, val, test = chronological_split(st.session_state.hist_raw)
            st.session_state.calibrators = fit_calibrators(val)
            st.session_state.active_method = "isotonic"
            st.session_state.needs_rebuild = True
            add_audit(
                action="rollback_calibrator",
                object_type="calibrator",
                object_id="initial_historical_val",
                result="rolled back",
                evidence_hash=sha256_obj({"rollback": now_iso()}),
                approval_required=True,
                approved_by="demo_user",
                note="Refit from historical validation split.",
            )
            st.success("Rolled back. Recompute artifacts to refresh baseline.")
        elif not rollback_approval:
            st.caption("Approval required for rollback.")


# -----------------------------------------------------------------------------
# TAB 5: VALUE CALCULATOR
# -----------------------------------------------------------------------------
with tabs[4]:
    st.title("Transparent Business Value Calculator")
    st.markdown(
        """
This calculator is **illustrative**. It uses user-entered assumptions and does not fabricate customer results.
Synthetic demo data proves mechanism, not production ROI.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        monthly_volume = st.number_input("Monthly predictions", min_value=1_000, max_value=100_000_000, value=1_000_000, step=1_000)
        current_review_rate = st.number_input("Current manual review rate", min_value=0.0, max_value=1.0, value=0.12, step=0.01)
        target_review_rate = st.number_input("Target review rate with calibrated abstention", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
    with c2:
        cost_per_review = st.number_input("Cost per manual review (USD)", min_value=0.0, max_value=1000.0, value=2.50, step=0.10)
        wrong_auto_accept_cost = st.number_input("Cost per wrong auto-accept (USD)", min_value=0.0, max_value=100000.0, value=25.00, step=1.00)
        current_wrong_rate = st.number_input("Current wrong auto-accept rate", min_value=0.0, max_value=1.0, value=0.015, step=0.001)
    with c3:
        projected_wrong_rate = st.number_input("Projected wrong auto-accept rate", min_value=0.0, max_value=1.0, value=0.010, step=0.001)
        annual_platform_cost = st.number_input("Annual platform cost (USD)", min_value=0.0, max_value=10_000_000.0, value=60_000.0, step=1000.0)
        confidence_discount = st.slider("Confidence discount for assumptions", 0.0, 1.0, 0.30, step=0.05)

    reviews_saved_month = monthly_volume * max(0.0, current_review_rate - target_review_rate)
    review_savings_month = reviews_saved_month * cost_per_review

    error_reduction_month = monthly_volume * max(0.0, current_wrong_rate - projected_wrong_rate)
    avoided_error_month = error_reduction_month * wrong_auto_accept_cost

    gross_value_month = review_savings_month + avoided_error_month
    gross_value_year = gross_value_month * 12.0

    discounted_year = gross_value_year * (1.0 - confidence_discount)
    net_year = discounted_year - annual_platform_cost
    payback_months = (annual_platform_cost / gross_value_month) if gross_value_month > 0 else float("inf")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Monthly review savings", f"${review_savings_month:,.0f}")
    k2.metric("Monthly avoided error cost", f"${avoided_error_month:,.0f}")
    k3.metric("Discounted annual value", f"${discounted_year:,.0f}")
    k4.metric("Net annual value", f"${net_year:,.0f}", delta="after platform cost")

    if payback_months == float("inf"):
        st.warning("No positive value under current assumptions.")
    else:
        st.info(f"Simple payback: {payback_months:.1f} months before confidence discount.")

    st.subheader("Sensitivity analysis")
    rows = []
    for vol_mult in [0.5, 1.0, 1.5]:
        for review_delta in [-0.02, 0.0, 0.02]:
            adj_target = min(1.0, max(0.0, target_review_rate + review_delta))
            rev_saved = monthly_volume * vol_mult * max(0.0, current_review_rate - adj_target)
            rev_value = rev_saved * cost_per_review * 12.0
            err_value = monthly_volume * vol_mult * max(0.0, current_wrong_rate - projected_wrong_rate) * wrong_auto_accept_cost * 12.0
            total = rev_value + err_value
            rows.append(
                {
                    "volume_multiplier": vol_mult,
                    "target_review_rate": adj_target,
                    "annual_review_value": rev_value,
                    "annual_error_avoided": err_value,
                    "annual_gross_value": total,
                    "net_after_platform": total - annual_platform_cost,
                }
            )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        """
### Guardrails for using this calculator
- Do not present synthetic demo outputs as customer results.
- Require human approval before using assumptions in business cases.
- Validate review cost, error cost, and volumes with finance/domain owners.
- Re-evaluate after real shadow-mode data is available.
        """
    )


# -----------------------------------------------------------------------------
# TAB 6: TESTING & ACCEPTANCE
# -----------------------------------------------------------------------------
with tabs[5]:
    st.title("Testing, Evaluation, Security, Reliability")

    st.subheader("Smoke tests")
    if st.button("Run smoke tests"):
        with st.spinner("Running tests..."):
            test_df = run_smoke_tests()
        st.dataframe(test_df, use_container_width=True, hide_index=True)

        passes = int((test_df["status"] == "PASS").sum())
        total = len(test_df)
        if passes == total:
            st.success(f"All smoke tests passed: {passes}/{total}")
        else:
            st.error(f"Smoke test failures: {total - passes}/{total}")
    else:
        st.info("Click the button to run in-app smoke tests.")

    st.divider()
    st.subheader("Evaluation protocol")
    st.markdown(TESTING_PROTOCOL_MD)

    st.divider()
    st.subheader("Security and reliability")
    st.markdown(SECURITY_MD)

    st.divider()
    st.subheader("Acceptance checklist")
    st.markdown(ACCEPTANCE_CHECKLIST_MD)

    st.divider()
    st.subheader("Example pytest commands for a fuller repo")
    st.code(
        """
pytest tests/unit -q
pytest tests/integration -q
pytest tests/e2e -q
pytest -q --maxfail=1
        """.strip(),
        language="powershell",
    )


# -----------------------------------------------------------------------------
# TAB 7: BUILD, DEPLOY & LICENSE
# -----------------------------------------------------------------------------
with tabs[6]:
    st.title("Technology Stack, Folder Structure, Build Steps")

    st.subheader("Windows PowerShell setup")
    st.code(
        r"""
New-Item -ItemType Directory -Force -Path C:\src\model-calibration-confidence-eval
cd C:\src\model-calibration-confidence-eval

# Place app.py in this folder.
# Download requirements.txt from the sidebar, or create it manually.

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
        """.strip(),
        language="powershell",
    )

    st.subheader("Recommended repository layout for productionization")
    st.code(
        """
/model-calibration-confidence-eval
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── /src
│   ├── ingestion/
│   ├── validation/
│   ├── canonical/
│   ├── calibration/
│   ├── decisioning/
│   ├── audit/
│   └── observability/
├── /api
│   └── future_fastapi_service.py
├── /domain
│   ├── models.py
│   └── contracts.py
├── /data
│   ├── synthetic/
│   └── licensed/
├── /schemas
│   └── event_contract.json
├── /models
│   └── calibration_artifacts/
├── /services
│   ├── evaluation_service.py
│   └── simulation_service.py
├── /dashboard
│   └── streamlit_components/
├── /configs
│   └── default.yaml
├── /scripts
│   ├── generate_synthetic.py
│   └── export_audit.py
├── /migrations
│   └── 001_init.sql
├── /docs
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── license_inventory.md
│   └── demo_script.md
└── /tests
    ├── /unit
    ├── /integration
    └── /e2e
        """.strip(),
        language="text",
    )

    st.markdown(
        """
### Directory purposes
- `/src`: core domain logic separated from UI.
- `/api`: future service endpoints; not required for prototype.
- `/domain`: Pydantic/dataclass contracts.
- `/data`: synthetic or licensed datasets with provenance.
- `/schemas`: JSON Schema / data contracts.
- `/models`: calibration artifacts and version manifests.
- `/services`: orchestration for ingestion, evaluation, simulation.
- `/dashboard`: Streamlit components if split from single-file app.
- `/configs`: environment-specific settings.
- `/scripts`: reproducibility utilities.
- `/migrations`: SQLite/Postgres schema migrations.
- `/docs`: architecture, data dictionary, licenses, demo script.
- `/tests`: unit, integration, e2e.
        """
    )

    st.divider()
    st.subheader("Milestone implementation plan")
    st.markdown(
        """
1. **Environment and configuration**
   - venv, requirements, fixed seed, SQLite path.
   - Acceptance: app starts without API keys.

2. **Synthetic/open data preparation**
   - generate historical and stream batches.
   - Acceptance: deterministic frame from seed.

3. **Schema validation**
   - required columns, ranges, duplicates, late labels.
   - Acceptance: malformed batch rejected and audited.

4. **Canonical storage**
   - SQLite events table with idempotent event_id.
   - Acceptance: re-ingest does not duplicate event_id.

5. **Baseline analytics**
   - reliability bins, ECE, MCE, Brier.
   - Acceptance: hand-calculated fixture passes.

6. **Core algorithm**
   - Platt, isotonic, fallback prior.
   - Acceptance: holdout comparison reported.

7. **API / service boundary**
   - prototype uses internal functions; production would expose FastAPI.
   - Acceptance: request/response contract documented.

8. **Dashboard**
   - filters, drill-down, evidence inspector.
   - Acceptance: reviewer can trace result to row and bin.

9. **Audit trail**
   - append-only audit log.
   - Acceptance: consequential actions include approval flag.

10. **Tests**
   - smoke tests, integration-style ingestion-to-result.
   - Acceptance: all required tests pass locally.

11. **Packaging and documentation**
   - README, license inventory, demo script.
   - Acceptance: fresh developer reproduces demo from README.
        """
    )

    st.divider()
    st.subheader(".env.example")
    st.code(
        """
# Local demo configuration. No secrets required.
DEMO_SEED=20261110
CALIB_DB_PATH=model_calibration_demo.sqlite3
LOG_LEVEL=INFO
RETENTION_DAYS=180
SAFE_MODE_DEFAULT=false
        """.strip(),
        language="text",
    )

    st.markdown("Never commit real secrets. This demo uses no API keys.")

    st.divider()
    st.subheader("License inventory")
    license_rows = [
        {"package": "streamlit", "license": "Apache-2.0", "commercial_use": "Yes", "attribution": "Retain license notices"},
        {"package": "pandas", "license": "BSD-3-Clause", "commercial_use": "Yes", "attribution": "Retain copyright notice"},
        {"package": "numpy", "license": "BSD-3-Clause", "commercial_use": "Yes", "attribution": "Retain copyright notice"},
        {"package": "scikit-learn", "license": "BSD-3-Clause", "commercial_use": "Yes", "attribution": "Retain copyright notice"},
        {"package": "matplotlib", "license": "PSF/BSD-style", "commercial_use": "Yes", "attribution": "Retain license files"},
        {"package": "pillow", "license": "MIT-CMU", "commercial_use": "Yes", "attribution": "Retain license"},
        {"package": "pydantic", "license": "MIT", "commercial_use": "Yes", "attribution": "Retain license"},
        {"package": "av", "license": "BSD-3-Clause", "commercial_use": "Yes", "attribution": "Retain license; FFmpeg components may be LGPL"},
        {"package": "opencv-python-headless", "license": "Apache-2.0", "commercial_use": "Yes", "attribution": "Retain license and 3rdparty notices"},
        {"package": "pytest", "license": "MIT", "commercial_use": "Yes", "attribution": "Retain license"},
        {"package": "SQLite", "license": "Public Domain", "commercial_use": "Yes", "attribution": "None required"},
        {"package": "Python", "license": "PSF", "commercial_use": "Yes", "attribution": "Retain notices"},
    ]
    st.dataframe(pd.DataFrame(license_rows), use_container_width=True, hide_index=True)

    st.markdown(
        """
### Transitive dependency inventory command
For a full direct + transitive inventory in a production repo, run:

```powershell
pip install pip-licenses
pip-licenses --format=csv --output-file=license_inventory.csv --with-authors --with-urls
```

### Dataset and model provenance
- Demo dataset: synthetic, generated by `make_historical()` and `generate_stream_batch()`.
- Seed: fixed.
- No external dataset used.
- No pretrained model weights used.
- Calibration models are fitted locally from synthetic probabilities.
        """
    )

    st.divider()
    st.subheader("Optional Docker Compose route")
    st.code(
        """
# Dockerfile example
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker-compose.yml example
services:
  calibration-demo:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - CALIB_DB_PATH=/app/data/model_calibration_demo.sqlite3
        """.strip(),
        language="yaml",
    )

    st.divider()
    st.subheader("Known limitations and roadmap")
    st.markdown(
        """
### Known limitations
- Prototype uses synthetic data only.
- Streamlit app is not a multi-tenant production service.
- SQLite is suitable for demo/local persistence, not large-scale concurrency.
- Video generation depends on available encoding backend.
- Calibration is binary-classification focused.
- No automated model retraining.

### Roadmap
1. Add CSV/Parquet upload with licensed data governance.
2. Add FastAPI service and async ingestion queue.
3. Add PostgreSQL/TimescaleDB and migrations.
4. Add RBAC/SSO and formal approval workflow.
5. Add multi-class and selective prediction / conformal risk control.
6. Add model registry integration.
7. Add production observability: OpenTelemetry, Prometheus, Grafana.
8. Add continuous benchmark suite and regression gates.
        """
    )


# -----------------------------------------------------------------------------
# TAB 8: AUDIT & GOVERNANCE
# -----------------------------------------------------------------------------
with tabs[7]:
    st.title("Audit, Governance, Safe Operation")

    st.subheader("Audit log")
    audit_df = pd.DataFrame(st.session_state.audit)
    if audit_df.empty:
        st.info("No audit entries yet.")
    else:
        actions = sorted(audit_df["action"].dropna().unique().tolist())
        selected_actions = st.multiselect("Filter actions", options=actions, default=actions)
        view = audit_df[audit_df["action"].isin(selected_actions)]
        st.dataframe(view.tail(200), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Human approval and kill switch")
    st.markdown(
        """
Consequential actions in this prototype require explicit approval:
- apply recalibration
- rollback calibrator
- change abstain thresholds
- enable/disable safe mode
- export data
- delete or retain-reduce audit history
        """
    )

    kill_approval = st.checkbox("I approve enabling safe mode (all predictions to human review)")
    if st.button("Enable safe mode / kill switch", disabled=not kill_approval):
        try:
            update_config(approved_by="demo_user", safe_mode=True)
            st.success("Safe mode enabled. Recompute artifacts to refresh dashboard.")
        except Exception as exc:
            st.error(str(exc))

    disable_approval = st.checkbox("I approve disabling safe mode")
    if st.button("Disable safe mode", disabled=not disable_approval):
        try:
            update_config(approved_by="demo_user", safe_mode=False)
            st.success("Safe mode disabled. Recompute artifacts to refresh dashboard.")
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Retention, backup, restore, rollback")
    st.markdown(
        r"""
### Retention
- Demo default: retain events and audit for 180 days.
- Production: retention policy per data class and regulation.

### Backup
```powershell
Copy-Item model_calibration_demo.sqlite3 backups\calibration_demo_$(Get-Date -Format yyyyMMdd_HHmmss).sqlite3
```

### Restore
```powershell
Copy-Item backups\calibration_demo_20261110_120000.sqlite3 model_calibration_demo.sqlite3
```

### Rollback
- Rollback calibrator via Calibration Workbench.
- Rollback config by applying previous approved values.
- Kill switch forces human review.

### Troubleshooting
- If SQLite is locked, close other app processes and reset demo.
- If MP4 does not play, regenerate video or install `av`.
- If tests fail, check deterministic seed and schema fixtures.
- If missing labels are high, expected fallback activates; this is safe behavior.
        """
    )

    st.divider()
    st.subheader("Structured log preview")
    log_preview = {
        "ts": now_iso(),
        "level": "INFO",
        "service": "model-calibration-confidence-eval",
        "event": "baseline_evaluation_complete",
        "ece_raw": st.session_state.base_eval["raw"]["ece"],
        "ece_isotonic": st.session_state.base_eval["isotonic"]["ece"],
        "fallback_used": st.session_state.base_eval["isotonic"]["fallback_used"],
        "seed": SEED,
        "code_version": CODE_VERSION,
    }
    st.code(to_json_string(log_preview), language="json")


st.caption(
    "All demo data is synthetic. Observed, estimated, simulated, and predicted values are labeled in the UI. "
    "This prototype is not a clinical, legal, financial, or production authorization system."
)