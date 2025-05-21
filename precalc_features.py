from datetime import datetime
import glob
import zipfile
from pathlib import Path
import requests
import shutil

import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
from textwrap import wrap
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score

from calculate_stats import prepare_train, prepare_test, precalc_stats, calculate_tstemp_stats


class Featurizer:
    START_DATE = datetime(2022, 12, 30)
    SEP_NAMES = "__"
    SEP_ALL = "___"

    def __init__(self, train, reload_feat=False):
        self.out_dir = Path("data/daily")
        self.unique_days = list(np.array(train.select("day_index").unique().sort("day_index")).reshape(-1))

        if reload_feat:
            assert False, "Note: if doing reloading, you need the full training set (to make preds later)."
            self.calculate_and_save_daily(train)

    @staticmethod
    def add_tstemps(df):
        df = df.with_columns(
            pl.from_epoch("timestamp", time_unit="s").alias("datetime")
        )
        return df.with_columns(
            pl.from_epoch("timestamp", time_unit="s").dt.weekday().alias("weekday"),
            ((pl.col("datetime").cast(pl.Date) - Featurizer.START_DATE).dt.total_days()).alias("day_index")
        ).drop('datetime')

    def get_optimal_day(self, df_train, df_test):
        max_train = df_train["day_index"].max()
        min_test = df_test["day_index"].min()
        assert min_test > max_train
        return max_train

    def calculate_and_save_daily(self, train, window_feat=(3, 7, 30, 400)):
        shutil.rmtree(self.out_dir)
        self.out_dir.mkdir()
        for day in tqdm(self.unique_days, desc="generating features"):
            day_dir = self.out_dir / f"leq_{day}"
            day_dir.mkdir(exist_ok=True)
            dfs = []
            for window_size in window_feat:
                df = train.filter((pl.col("day_index") <= day) & (pl.col("day_index") >= day - window_size))
                window_dfs = []
                for df_, keys in precalc_stats(df):
                    replace_names = list(set(df_.columns) - set(keys))
                    replace_mapping = {k: k + f"_w{window_size}" for k in replace_names}
                    assert len(replace_names) == len(df_.columns) - len(keys)
                    window_dfs.append((df_.rename(replace_mapping), keys))
                dfs.extend(window_dfs)

            all_names = set()
            for df_, keys in dfs:
                df_name = self.SEP_NAMES.join(df_.columns) + self.SEP_ALL + self.SEP_NAMES.join(keys) + ".parquet"
                assert df_name not in all_names, f'{df_name} already in {all_names}'
                all_names.add(df_name)
                assert df_name.count(self.SEP_ALL) == 1, f'{df_name}'
                df_.with_columns(pl.lit(day).alias("info_day_index")).write_parquet(day_dir / df_name)

    def load_daily(self, day):
        dfs = []
        day_dir = self.out_dir / f"leq_{day}"
        assert day_dir.exists(), f'{day_dir}'
        for df_pth in glob.glob(f"{str(day_dir)}/*.parquet"):
            df = pl.read_parquet(df_pth)
            df_name = Path(df_pth).stem
            dfs.append((df, df_name.split(self.SEP_ALL)[-1].split(self.SEP_NAMES)))
        return dfs

    def add_daily_features_train(self, all_df, verbose):
        rng = tqdm(all_df["info_day_index"].unique(), desc="loading train") if verbose else all_df["info_day_index"].unique()
        result_dfs = []
        for day in rng:
            df = all_df.filter(pl.col("info_day_index") == day)
            day_dfs = self.load_daily(day)
            for df_info, keys in day_dfs:
                assert "info_day_index" not in keys
                if len(df_info) == 0:
                    continue
                assert df_info["info_day_index"][0] == day, f'{df_info["info_day_index"][0]}'
                df = df.join(df_info, on=["info_day_index"] + keys, how='left')
            result_dfs.append(df)
        return pl.concat(result_dfs)

    def add_daily_features_test(self, all_df, min_index_day=2, verbose=True, log_steps=10):
        df = all_df.clone()
        last_day = all_df["info_day_index"][0]
        assert len(all_df["info_day_index"].unique().to_numpy().reshape(-1)) == 1, f"{all_df['info_day_index'].unique()}"

        step, reduce_days = -1, 30
        rng = range(last_day, last_day - reduce_days, step)
        if verbose:
            print(f"Running test loading with range days: {rng}")
        rng = rng if not verbose else tqdm(rng, desc="loading test")
        prev_counts = None
        for idx_, current_day in enumerate(rng):
            day_dfs = self.load_daily(current_day)

            for df_info, keys in day_dfs:
                renamed_cols = {
                    col: f"{col}_new" for col in df_info.columns if col not in keys
                }
                temp_df_info = df_info.rename(renamed_cols)
                joined_df = df.join(temp_df_info, on=keys, how="left")

                cols_to_update = [
                    col for col in df_info.columns if col not in keys
                ]
                exprs = []
                for col in cols_to_update:
                    new_col_name = f"{col}_new"
                    if col in df.columns:
                        exprs.append(
                            pl.coalesce([pl.col(col), pl.col(new_col_name)]).alias(col)
                        )
                    else:
                        exprs.append(pl.col(new_col_name).alias(col))

                df = joined_df.select([
                    pl.all().exclude([f"{col}_new" for col in cols_to_update] + cols_to_update + ["info_day_index"])
                ] + exprs)

            if prev_counts is not None and verbose and (idx_ + 1) % log_steps == 0:
                cur_counts = df.null_count().row(0, named=True)
                changes = {k: prev_counts[k] - cur_counts[k] for k in prev_counts}
                changes_val = list(changes.values())
                mean, mn, mx = np.mean(changes_val), np.min(changes_val), np.max(changes_val)
                negs = {k: v for k, v in changes.items() if v < 0}
                mn_cnt = np.mean(list(cur_counts.values()))
                st = - log_steps * step
                print(f"(over {st}d) drop on {current_day}: mean={mean}, min={mn}, max={mx}. Neg: {negs}. Mean count: {int(mn_cnt)}")
                prev_counts = df.null_count().row(0, named=True)

            if prev_counts is None:
                prev_counts = df.null_count().row(0, named=True)

        return df

    def create_causal_data(self, df, lag_type="random", min_index_day=2, verbose=True):
        if lag_type == "random":
            lags = np.random.randint(1, 31, size=len(df))
        elif lag_type.startswith("constant"):
            lags = np.ones(len(df)).astype(int) * int(lag_type.split("_")[-1])
        elif lag_type.startswith("test"):
            lags = df["day_index"].to_numpy() - int(lag_type.split("_")[-1])

        info_day_index = df["day_index"].to_numpy() - lags
        remaining_ids = info_day_index >= min_index_day
        df = df.with_columns(pl.Series("info_day_index", info_day_index))
        if lag_type.startswith("test"):
            assert remaining_ids.sum() == len(df), f'Expected {remaining_ids.sum()} == {len(df)} for the test case'
        if verbose:
            print(f"Creating df (lag={lag_type}) with {remaining_ids.sum()}/{len(df)} remaining, {remaining_ids.sum() / len(df) * 100:.2f}%")
        df = df.filter(remaining_ids)

        if not lag_type.startswith("test"):
            final_df = self.add_daily_features_train(df, verbose=verbose)
        else:
            final_df = self.add_daily_features_test(df, min_index_day=min_index_day, verbose=verbose)

        if verbose:
            print(f"Get final df with shape: {final_df.shape}")
        return calculate_tstemp_stats(final_df)


if __name__ == "__main__":
    all_train = Featurizer.add_tstemps(pl.read_parquet("train.parquet"))
    featurizer = Featurizer(all_train, reload_feat=True)
