"""tooling.py

S3/MinIO data access utilities.

IMPORTANT DESIGN CHOICE (project constraint):
- The ONLY source of benchmark input data is the real-time flushed LOG files
  stored in S3 under:  {S3_DATA_PREFIX}{symbol}/flush_*.parquet
- Sensitivity sweeps (alpha/W/tau) must NEVER be treated as data files.
  They are experiment metadata and artifact-selection rules only.
"""

from __future__ import annotations

from io import BytesIO
from typing import List, Optional, Sequence, Union
import re

import numpy as np
import pandas as pd
import boto3

import config as cfg


class MinioHandler:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=cfg.MINIO_ENDPOINT,
            aws_access_key_id=cfg.MINIO_ACCESS_KEY,
            aws_secret_access_key=cfg.MINIO_SECRET_KEY,
        )

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _list_all_objects(self, prefix: str) -> List[dict]:
        """List all objects under a prefix (handles pagination)."""
        out: List[dict] = []
        token: Optional[str] = None

        while True:
            kwargs = {
                "Bucket": cfg.BUCKET_NAME,
                "Prefix": prefix,
                "MaxKeys": 1000,
            }
            if token:
                kwargs["ContinuationToken"] = token

            resp = self.client.list_objects_v2(**kwargs)
            out.extend(resp.get("Contents", []))

            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
                if not token:
                    break
            else:
                break

        return out

    def _read_parquet_from_s3(self, key: str) -> pd.DataFrame:
        resp = self.client.get_object(Bucket=cfg.BUCKET_NAME, Key=key)
        raw = resp["Body"].read()
        return pd.read_parquet(BytesIO(raw))

    @staticmethod
    def _flush_index_from_key(key: str) -> Optional[int]:
        """
        Try to parse 'flush_123.parquet' -> 123.
        If not possible, return None.
        """
        m = re.search(r"flush_(\d+)\.parquet$", key)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def load_historical_features(
        self,
        *,
        symbol: str = cfg.SYMBOL,
        n_rows: Optional[int] = None,
        cols: Optional[Sequence[str]] = None,
        allow_dummy: bool = False,
        return_df: bool = False,
        enforce_time_sort: bool = False,
        time_col: str = "ts",
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Load real-time flushed feature logs from S3/MinIO.

        Reads multiple parquet flush files under:
            {cfg.S3_DATA_PREFIX}{symbol}/flush_*.parquet

        Strategy:
        - List all flush parquet files under prefix
        - Sort files (prefer parsing flush index from filename; fallback to LastModified)
        - Read newest backwards until >= n_rows accumulated
        - Concatenate and return last n_rows

        Args:
          n_rows:
            required number of rows to return. If None, defaults to cfg.N_STEPS
            BUT NOTE: cfg.N_STEPS is benchmark-oriented, so caller should pass explicitly.
          cols:
            columns to return. If None -> cfg.FEATURE_COLS.
            You can pass something like ["ts"] + cfg.FEATURE_COLS for analyses.
          enforce_time_sort:
            If True and time_col exists, sorts final df by time_col before tail(n_rows).
            Use this if your flush files may arrive out of order.
          allow_dummy:
            Only for local smoke tests. In experiments, keep False.

        Returns:
          - if return_df=True -> DataFrame
          - else -> np.ndarray float32
        """

        if n_rows is None:
            # defaulting is allowed, but we print a warning because this is easy to misuse.
            n_rows = int(getattr(cfg, "N_STEPS", 5000))
            print(f"[!] n_rows not provided; defaulting to cfg.N_STEPS={n_rows}. "
                  f"Consider passing n_rows explicitly for reproducibility.")

        if n_rows <= 0:
            raise ValueError(f"n_rows must be positive, got {n_rows}")

        use_cols = list(cfg.FEATURE_COLS) if cols is None else list(cols)

        prefix = f"{cfg.S3_DATA_PREFIX}{symbol}/"
        print(f"[*] Fetching historical data from S3: {cfg.BUCKET_NAME}/{prefix} (need {n_rows} rows)")

        try:
            objs = self._list_all_objects(prefix)
            parquet_objs = [
                o for o in objs
                if o.get("Key", "").endswith(".parquet") and "flush_" in o.get("Key", "")
            ]

            if not parquet_objs:
                raise FileNotFoundError(
                    f"No parquet flush files found under prefix '{prefix}'. "
                    f"Expected keys like '{prefix}flush_0.parquet'."
                )

            # Sort: prefer flush index if parsable, else LastModified
            def sort_key(o: dict):
                key = o.get("Key", "")
                idx = self._flush_index_from_key(key)
                lm = o.get("LastModified")
                # idx first; if None, push it behind parsable ones
                return (idx is None, idx if idx is not None else 0, lm)

            parquet_objs.sort(key=sort_key)

            # Read newest backwards until enough rows are collected
            parts: List[pd.DataFrame] = []
            rows_acc = 0

            for o in reversed(parquet_objs):
                key = o["Key"]
                df_part = self._read_parquet_from_s3(key)

                missing = [c for c in use_cols if c not in df_part.columns]
                if missing:
                    raise ValueError(
                        f"Flush file '{key}' missing columns: {missing}. "
                        f"Found: {list(df_part.columns)}"
                    )

                df_part = df_part[use_cols]
                parts.append(df_part)
                rows_acc += len(df_part)

                if rows_acc >= n_rows:
                    break

            parts = list(reversed(parts))
            df = pd.concat(parts, axis=0, ignore_index=True)

            if enforce_time_sort and (time_col in df.columns):
                df = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)

            if len(df) < n_rows:
                raise RuntimeError(
                    f"Insufficient rows in S3 logs under '{prefix}'. "
                    f"Have {len(df)} rows across {len(parts)} files, need {n_rows}."
                )

            df = df.tail(n_rows).reset_index(drop=True)

            if return_df:
                return df
            return df.to_numpy(dtype=np.float32, copy=True)

        except Exception as e:
            if allow_dummy:
                print(f"[!] S3 Load Error: {e}. Falling back to dummy data (allow_dummy=True).")
                dummy = np.random.randn(n_rows, len(use_cols)).astype(np.float32)
                if return_df:
                    return pd.DataFrame(dummy, columns=use_cols)
                return dummy
            raise