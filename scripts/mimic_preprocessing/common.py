"""
Common utilities for MIMIC preprocessing scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import duckdb
import pandas as pd
import numpy as np


@dataclass
class MIMICPaths:
    db_path: str = "data/mimic/mimiciii.duckdb"
    vital_meta_path: str = "data/mimic/vital_metadata.csv"
    labs_meta_path: str = "data/mimic/labs_metadata.csv"


def connect_db(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


def load_metadata(paths: MIMICPaths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vital_meta = pd.read_csv(paths.vital_meta_path)
    labs_meta = pd.read_csv(paths.labs_meta_path)
    return vital_meta, labs_meta


def chunk_list(values: List[int], chunk_size: int) -> Iterable[List[int]]:
    for i in range(0, len(values), chunk_size):
        yield values[i:i + chunk_size]


def fetch_labevents_timeseries(
    conn: duckdb.DuckDBPyConnection,
    hadm_ids: List[int],
    itemids: List[int],
    value_col: str,
    min_val: float,
    max_val: float,
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    Load labevents time series for given itemids and hadm_ids in batches.
    """
    if not hadm_ids or not itemids:
        return pd.DataFrame()

    frames = []
    for batch in chunk_list(hadm_ids, batch_size):
        query = f"""
        SELECT
            le.hadm_id::INTEGER AS hadm_id,
            le.charttime::TIMESTAMP AS charttime,
            EXTRACT(EPOCH FROM (le.charttime::TIMESTAMP - a.admittime::TIMESTAMP)) / 86400.0 AS time_days,
            CAST(le.valuenum AS REAL) AS {value_col}
        FROM labevents le
        INNER JOIN admissions a ON le.hadm_id = a.hadm_id
        WHERE
            le.itemid IN {tuple(itemids)}
            AND le.valuenum::REAL IS NOT NULL
            AND le.valuenum::REAL BETWEEN {min_val} AND {max_val}
            AND le.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.dischtime::TIMESTAMP
            AND le.hadm_id::INTEGER IN {tuple(batch)}
        """
        df = conn.execute(query).fetchdf()
        if len(df) > 0:
            df.columns = df.columns.str.lower()
            frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_vitals_labs(
    conn: duckdb.DuckDBPyConnection,
    hadm_ids: List[int],
    vital_items: List[int],
    lab_items: List[int],
    batch_size: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not hadm_ids:
        return pd.DataFrame(), pd.DataFrame()

    if not vital_items:
        vital_items = []
    if not lab_items:
        lab_items = []

    vitals_frames = []
    labs_frames = []

    for batch in chunk_list(hadm_ids, batch_size):
        if vital_items:
            vitals_query = f"""
            SELECT chartevents.subject_id::INTEGER AS subject_id
                , chartevents.hadm_id::INTEGER AS hadm_id
                , chartevents.charttime::TIMESTAMP AS charttime
                , chartevents.itemid::INTEGER AS itemid
                , chartevents.valuenum::DOUBLE AS valuenum
                , admissions.admittime::TIMESTAMP AS admittime
            FROM chartevents
            INNER JOIN admissions
                ON chartevents.subject_id = admissions.subject_id
                AND chartevents.hadm_id = admissions.hadm_id
                AND chartevents.charttime::TIMESTAMP BETWEEN
                    (admissions.admittime::TIMESTAMP)
                    AND (admissions.dischtime::TIMESTAMP)
                AND itemid::INTEGER IN {tuple(vital_items)}
                AND chartevents.hadm_id::INTEGER IN {tuple(batch)}
            WHERE chartevents.error::INTEGER IS DISTINCT FROM 1
            """

            vdf = conn.execute(vitals_query).fetchdf()
            if len(vdf) > 0:
                vdf.columns = vdf.columns.str.lower()
                vitals_frames.append(vdf)

        if lab_items:
            labs_query = f"""
            SELECT labevents.subject_id::INTEGER AS subject_id
                , labevents.hadm_id::INTEGER AS hadm_id
                , labevents.charttime::TIMESTAMP AS charttime
                , labevents.itemid::INTEGER AS itemid
                , labevents.valuenum::DOUBLE AS valuenum
                , admissions.admittime::TIMESTAMP AS admittime
            FROM labevents
            INNER JOIN admissions
                ON labevents.subject_id = admissions.subject_id
                AND labevents.hadm_id = admissions.hadm_id
                AND labevents.charttime::TIMESTAMP BETWEEN
                    (admissions.admittime::TIMESTAMP)
                    AND (admissions.dischtime::TIMESTAMP)
                AND itemid::INTEGER IN {tuple(lab_items)}
                AND labevents.hadm_id::INTEGER IN {tuple(batch)}
            """

            ldf = conn.execute(labs_query).fetchdf()
            if len(ldf) > 0:
                ldf.columns = ldf.columns.str.lower()
                labs_frames.append(ldf)

    vitals_df = pd.concat(vitals_frames, ignore_index=True) if vitals_frames else pd.DataFrame()
    labs_df = pd.concat(labs_frames, ignore_index=True) if labs_frames else pd.DataFrame()
    return vitals_df, labs_df


def ensure_time_day(df: pd.DataFrame) -> pd.DataFrame:
    if 'time_days' in df.columns:
        df['time_day'] = np.floor(df['time_days']).astype(int)
    return df


def aggregate_vitals_labs(vitals_df: pd.DataFrame, labs_df: pd.DataFrame,
                          vital_meta: pd.DataFrame, labs_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate vitals and labs to daily min/max/mean.
    Mirrors the notebook_utils implementation to keep preprocessing consistent.
    """
    vitals_df = vitals_df.merge(vital_meta, on='itemid', how='left')
    vitals_df = vitals_df[vitals_df['valuenum'].between(vitals_df['min'], vitals_df['max'], inclusive='both')]

    labs_df = labs_df.merge(labs_meta, on='itemid', how='left')
    labs_df = labs_df[labs_df['valuenum'].between(labs_df['min'], labs_df['max'], inclusive='both')]

    if 'units' in vitals_df.columns:
        vitals_df.loc[vitals_df['units'] == 'F', 'valuenum'] = (
            vitals_df.loc[vitals_df['units'] == 'F', 'valuenum'] - 32
        ) * 5.0/9.0
        vitals_df.loc[vitals_df['units'] == 'F', 'units'] = 'C'
        vitals_df.loc[vitals_df['feature name'] == 'TempF', 'feature name'] = 'TempC'

    vitals_labs = pd.concat([vitals_df, labs_df], ignore_index=True)

    vitals_labs_pivot = vitals_labs.pivot_table(
        index=['hadm_id', 'admittime', pd.Grouper(freq='1D', key='charttime')],
        columns='feature name',
        values='valuenum',
        aggfunc=['min', 'max', 'mean']
    )

    vitals_labs_pivot.columns = [f'{col[1]}_{col[0]}' for col in vitals_labs_pivot.columns]
    return vitals_labs_pivot


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.7, exclude_cols=None):
    """
    Drop columns with missing values above threshold after forward fill.
    """
    if exclude_cols is None:
        exclude_cols = ['hadm_id', 'time_day', 'target']

    missing = df.isna().mean()
    drop_cols = [c for c, frac in missing.items() if frac > threshold and c not in exclude_cols]
    cleaned = df.drop(columns=drop_cols)
    return cleaned, drop_cols
