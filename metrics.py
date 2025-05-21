from functools import reduce
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table
import sklearn
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score

from precalc_features import Featurizer


def pred_ranking(model, val_data):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(val_data)[:, 1]
    return model.predict(val_data)


def calc_ndcg(model, val_data, df_val, verbose):
    catboost_predicts = df_val.with_columns(
        predict=pred_ranking(model, val_data)
    )
    true = []
    pred = []
    for i in catboost_predicts.group_by('request_id'):
        value = i[1].sort('predict', descending=True)[:10]
        if sum(value['target']) == 0:
            continue
        l = [0] * (10 - len(value['target']))
        true.append(value['target'].to_list() + l)
        pred.append(value['predict'].to_list() + l)

    sklearn_ndcg = ndcg_score(true, pred, k=10, ignore_ties=True)
    print(f"NDCG@10: {sklearn_ndcg:.4f}")
    return sklearn_ndcg


def calc_probs_and_importance(model, val_data, df_val):
    y_pred_proba = pred_ranking(model, val_data)

    roc_auc = roc_auc_score(df_val['target'].to_list(), y_pred_proba)
    print(f"ROC AUC: {roc_auc:.4f}")

    logloss = log_loss(df_val['target'].to_list(), y_pred_proba)
    print(f"LogLoss: {logloss:.4f}")

    if model.feature_importances_.ndim > 0:
        return dict(zip(model.feature_names_, model.feature_importances_))


def save_result(model, kaggle_data, test, save_path=None):
    if save_path is None:
        test_data = test['index', 'request_id']
        test_scored = test_data.with_columns(
            predict=pred_ranking(model, kaggle_data)
        ).sort(
            'predict',
            descending=True
        ).select(
            'index',
            'request_id'
        ).write_csv('cb_submit.csv')
        print("\n\nSaved test data predictions...")
        return

    test['index', 'request_id', 'day_index'].with_columns(
        predict=pred_ranking(model, kaggle_data)
    ).write_parquet(save_path)
    print(f"\n\nSaved test data scores in {save_path}")


def process_results(model, val_data, df_val, test_data, df_test, save_path=None, verbose=True, is_ranker=False):
    if verbose:
        save_result(model, test_data, df_test, save_path)
    if verbose and not is_ranker:
        d_imp = calc_probs_and_importance(model, val_data, df_val)
        return d_imp, calc_ndcg(model, val_data, df_val, verbose)
    return None, calc_ndcg(model, val_data, df_val, verbose)


def print_stats(df, column_names):
    assert "predict_random" in column_names
    rnd_col = df["predict_random"]
    stats_dict = df.select([
        pl.col(col).mean().alias(f"{col}_mean") for col in column_names
    ] + [
        pl.col(col).std().alias(f"{col}_std") for col in column_names
    ]).to_dict(as_series=False)

    console = Console()
    table = Table(title="Column stats")
    table.add_column("Column", justify="left")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Mean diff with rand", justify="right")
    table.add_column("Std diff with rnd", justify="right")
    for col in column_names:
        mean = round(stats_dict[f"{col}_mean"][0], 4)
        std = round(stats_dict[f"{col}_std"][0], 4)
        diff_rnd = round((df[col] - rnd_col).abs().mean(), 4)
        std_rnd = round((df[col] - rnd_col).abs().std(), 4)
        table.add_row(col, str(mean), str(std), str(diff_rnd), str(std_rnd))
    console.print(table)

    return stats_dict


def process_ensemble_results(
    save_path,
    lag_range=range(1, 31),
    merge_type="mean_const_and_random",
    normalize=True,
):
    d = {}
    for lag_type in ["random"] + [f"constant_{lag}" for lag in lag_range]:
        d[f"predict_{lag_type}"] = pl.read_parquet(
            Path(save_path) / f"out_{lag_type}.parquet").rename({"predict": f"predict_{lag_type}"}
        )

    column_names = list(d.keys())
    df = reduce(
        lambda left, right: left.join(right, on=list(set(d[column_names[0]].columns) - {column_names[0]}), how="inner"),
        d.values()
    )
    assert len(df) == len(d[column_names[0]]), f'{df.shape}[0] != {d[column_names[0]].shape}[0]'

    print("Initial prediction stats:")
    stats_dict = print_stats(df, column_names)

    if normalize:
        normalized_cols = [
            ((pl.col(col) - stats_dict[f"{col}_mean"][0]) / stats_dict[f"{col}_std"][0])
            .alias(f"{col}_normalized")
            for col in column_names
        ]
        df = df.with_columns(normalized_cols)
        cols_to_keep = ["day_index", "index", "request_id"] + [f"{col}_normalized" for col in column_names]
        df = df.select(cols_to_keep).rename({f"{col}_normalized": col for col in column_names})

    constants_cols = {i: f"predict_constant_{i}" for i in lag_range}
    df = df.with_columns(
        (pl.col("day_index") - pl.col("day_index").min() + 1).alias("day_index")
    )
    assert len(df['day_index'].unique()) == len(constants_cols), f"{df['day_index'].unique()}"

    if merge_type == "mean_const_and_random":
        def compute_predict(row):
            const_col = constants_cols[row['day_index']]
            return (row['predict_random'] + row[const_col]) / 2
    elif merge_type == "mean_all_and_random":
        def compute_predict(row):
            const_col_values = [row[const_col] for const_col in constants_cols.values()]
            return sum([row['predict_random']] + const_col_values) / (len(const_col_values) + 1)
    elif merge_type == "only_const":
        def compute_predict(row):
            const_col = constants_cols[row['day_index']]
            return row[const_col]
    else:
        raise RuntimeError(f"Unexpected merge_type = {merge_type}")

    predicts = [compute_predict(row) for row in df.iter_rows(named=True)]
    df = df.with_columns(pl.Series("predict", predicts))

    print(f"Saving ensemble results into {merge_type}.csv")
    df['index', 'request_id', 'predict'].sort(
        'predict',
        descending=True,
    ).select(
        'index',
        'request_id'
    ).write_csv(f'{merge_type}.csv')


if __name__ == "__main__":
    process_ensemble_results(
        save_path="ensemble_results",
        merge_type="mean_const_and_random",
    )
