import numpy as np
import polars as pl
from sklearn.utils.extmath import randomized_svd
import torch



def calculate_pos_stat(dataset, agg_names):
    pos_suf = "_" + "_".join(agg_names)
    return dataset.group_by(agg_names).agg(
        pl.col("position_in_request").mean().alias(f"mean_position{pos_suf}"),
        pl.col("position_in_request").median().alias(f"median_position{pos_suf}"),
        pl.col("position_in_request").std().alias(f"std_position{pos_suf}"),
    )


def calculate_tgt_stat(dataset, tgt_vals, agg_names):
    df = None
    for tgt_val in tgt_vals:
        tgt_name = tgt_val.split("_")[-1]
        stat_df = dataset.filter(
                pl.col('action_type') == tgt_val
            ).group_by(*agg_names).agg(
                np.log1p(pl.len().alias(
                    f'log_count_{tgt_name}_by_{"_".join(map(lambda el: el.split("_")[0], agg_names))}'
                ))
            )
        df = stat_df if df is None else df.join(stat_df, on=agg_names, how='inner')
    return df


def calculate_ctr(dataset: pl.DataFrame, tgt_vals: list, col_names: list) -> pl.DataFrame:
    df = None
    data = dataset.group_by('action_type', *col_names).agg(pl.len())
    for click_name in tgt_vals:
        clicks = data.filter(pl.col('action_type') == click_name)
        views = data.filter(pl.col('action_type') == "AT_View")

        col_name = "ctr_" + click_name.split("_")[-1] + "_" + "_".join(map(lambda el: el.split("_")[0], col_names))
        ctr = clicks.join(
            views, on=col_names
        ).with_columns(
            (pl.col('len') / pl.col('len_right')).alias(col_name)
        ).select(
            *col_names, col_name
        )
        df = ctr if df is None else df.join(ctr, on=col_names, how='inner')

    return df


def calculate_user_item_matrix(df: pl.DataFrame):
    user_ids = list(np.array(df.select("user_id").unique().sort("user_id")).reshape(-1))
    id_to_user = {el: idx for idx, el in enumerate(user_ids)}
    item_ids = list(np.array(df.select("product_id").unique().sort("product_id")).reshape(-1))
    id_to_item = {el: idx for idx, el in enumerate(item_ids)}

    m = np.zeros([len(user_ids), len(item_ids)])
    for user_id, item_id in np.array(df.select("user_id", "product_id")):
        m[id_to_user[user_id], id_to_item[item_id]] += 1

    return user_ids, item_ids, m


def calculate_jaccard_t(m):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = torch.tensor(m, device=device)
    m_binary = (m > 0).float()
    intersection = torch.mm(m_binary.t(), m_binary)  # |U_i ∩ U_j|
    item_sums = m_binary.sum(dim=0)
    union = item_sums.unsqueeze(1) + item_sums.unsqueeze(0) - intersection  # |U_i| + |U_j| - |U_i ∩ U_j|
    jaccard_items = intersection / (union + 0.05)
    jaccard_items.fill_diagonal_(1)
    user_interactions = m_binary.sum(dim=1)
    jaccard_user_item = torch.zeros_like(m, dtype=torch.float32, device=device)

    for u in range(m.shape[0]):
        interacted_items = torch.nonzero(m_binary[u, :]).squeeze(1)
        if len(interacted_items) == 0:
            jaccard_user_item[u, :] = 0
        else:
            jaccard_user_item[u, :] = jaccard_items[:, interacted_items].mean(dim=1)

    return jaccard_user_item.cpu().numpy()


def calculate_npmi(interaction_matrix, epsilon=1e-10):
    mat = interaction_matrix.copy()
    total = mat.sum() + epsilon
    p_ui = mat / total
    p_u = mat.sum(axis=1) / total
    p_i = mat.sum(axis=0) / total
    pmi = np.log((p_ui + epsilon) / ((p_u[:, None] * p_i[None, :]) + epsilon))
    npmi = pmi / (-np.log(p_ui + epsilon) + epsilon)
    npmi[p_ui == 0] = 0
    return npmi


def calculate_svd(interaction_matrix, n_factors=50):
    if interaction_matrix.shape == (0, 0):
        return interaction_matrix
    U, sigma, Vt = randomized_svd(
        interaction_matrix,
        n_components=n_factors,
        n_iter=10,
        random_state=0,
    )
    reconstructed_matrix = np.dot(U * sigma, Vt)
    return reconstructed_matrix


def calculate_collab_stats(df: pl.DataFrame):
    df = df.filter(pl.col("action_type") == "AT_CartUpdate")

    user_ids, item_ids, m = calculate_user_item_matrix(df)
    npmi_col = calculate_npmi(m).ravel()
    jac_col = calculate_jaccard_t(m).ravel()
    svd_col = calculate_svd(m).ravel()

    user_ids_col = np.repeat(user_ids, len(item_ids))
    product_ids_col = np.tile(item_ids, len(user_ids))

    return pl.DataFrame({
        "user_id": user_ids_col,
        "product_id": product_ids_col,
        "npmi": npmi_col,
        "jaccard": jac_col,
        "svd": svd_col,
    })


def calculate_tstemp_stats(df):
    df = df.with_columns(
        pl.from_epoch("timestamp", time_unit="s").alias("datetime")
    )

    df = df.with_columns(
        pl.col("datetime").dt.month().alias("month"),
        pl.col("datetime").dt.hour().alias("hour"),
        pl.col("datetime").dt.weekday().alias("weekday"),
        pl.col("datetime").dt.weekday().is_between(5, 6).alias("is_weekend"),
        pl.col("datetime").dt.hour().cut(
            breaks=[6, 12, 17, 22],
            labels=["night", "morning", "afternoon", "evening", "night"],
        ).alias("time_of_day")
    )

    return df.drop("timestamp", "datetime")


def precalc_stats(df):
    tgt_vals = ["AT_View", "AT_Purchase"]
    ctr_tgt_vals = ["AT_Purchase", "AT_CartUpdate", "AT_Click"]
    # ["AT_View", "AT_Purchase", "AT_CartUpdate", "AT_Click"]
    agg_vals = [
        ["user_id"], ["product_id"], ["store_id"], ["city_name"],
        ["user_id", "product_id"], ["product_id", "store_id"], ["product_id", "city_name"],
    ]
    agg_vals_pos = [
        ["product_id"], ["user_id", "product_id"], ["product_id", "city_name"]
    ]

    dfs = []
    for agg_names in agg_vals:
        dfs.append((
            calculate_tgt_stat(df, tgt_vals, agg_names),
            agg_names,
        ))
        dfs.append((
            calculate_ctr(df, ctr_tgt_vals, agg_names),
            agg_names,
        ))

    for agg_names in agg_vals_pos:
        dfs.append((
            calculate_pos_stat(df, agg_names),
            agg_names,
        ))

    dfs.append((
        calculate_collab_stats(df),
        ["user_id", "product_id"],
    ))

    return dfs


def apply_all_stats(df, dfs):
    df = calculate_tstemp_stats(df)
    for df_, keys in dfs:
        df = df.join(df_, on=keys, how='left')
    return df


def prepare_train(df, dfs):
    df = df.filter(
        pl.col('action_type').is_in(["AT_View", "AT_CartUpdate"])
    ).with_columns(
        target=pl.when(pl.col("action_type") == "AT_View")
                .then(0)
                .when(pl.col("action_type") == "AT_Click")
                .then(1)
                .when(pl.col("action_type") == "AT_CartUpdate")
                .then(2)
                .when(pl.col("action_type") == "AT_Purchase")
                .then(3)
    ).drop(
        'product_image',
        'product_name',
        'position_in_request',
        'action_type',
    )

    return apply_all_stats(df, dfs)


def prepare_test(df, dfs):
    df = df.drop(
        'product_image',
        'product_name',
    )
    return apply_all_stats(df, dfs)
