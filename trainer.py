from catboost import CatBoostClassifier, CatBoostRanker, Pool
from pathlib import Path
import shutil
import polars as pl

from precalc_features import Featurizer
from preprocess import Preprocessor
from metrics import process_results


class Trainer:
    def __init__(self, lag_type="random", use_ranker=False, model_params={}, verbose=True, reduce_val_days=0):
        self.cache_dir = Path("data") / "trainer"
        self.cache_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        self.use_ranker = use_ranker
        self.reduce_val_days = reduce_val_days

        all_train = Featurizer.add_tstemps(pl.read_parquet("train.parquet"))
        test = Featurizer.add_tstemps(pl.read_parquet("test.parquet")).drop(
            'product_name', 'product_image',
        )

        prepr = Preprocessor(all_train)
        self.initial_train, self.initial_val = prepr.run()

        self.featurizer = Featurizer(all_train)
        self.test = self._call_cached_feat(test, "test", f"test_{self.featurizer.get_optimal_day(all_train, test)}")
        self._init_train_and_val(lag_type if lag_type != "ensemble" else "random")

        cond = lambda el: "_w1" in el or "_w3" in el or ("store" in el and "product_store" not in el and el != "store_id") or \
            ("product_id_city_name" in el)
        self.train = self.train.drop(col for col in self.train.columns if cond(col))
        self.val = self.val.drop(col for col in self.val.columns if cond(col))
        self.test = self.test.drop(col for col in self.test.columns if cond(col))

        self.lag_type = lag_type
        if self.lag_type == "ensemble":
            assert False, "Do you really want to run ensemble? Note: it will remove the prev results"
            self.save_results = "ensemble_results"
            shutil.rmtree(self.save_results)
            Path(self.save_results).mkdir()
        self.model_params = model_params

    def _init_train_and_val(self, lag_type):
        assert lag_type != "ensemble"
        self.train, self.val = None, None

        train, val, train_name_suff = self.initial_train, self.initial_val, ""
        if self.reduce_val_days > 0:
            train_name_suff = f"_plus_{self.reduce_val_days}"
            split_day = self.initial_val["day_index"].min() + self.reduce_val_days
            train = pl.concat([train, val.filter(pl.col("day_index") < split_day)])
            val = val.filter(pl.col("day_index") >= split_day)

        self.train = self._call_cached_feat(train, "train" + train_name_suff, lag_type)
        self.train = self._filter_large_req(self.train)

        self.val = self._call_cached_feat(val, "val", f"test_{self.featurizer.get_optimal_day(train, val)}")
        self.val = self._filter_large_req(self.val)

        self._clear_sets()

    def _filter_large_req(self, df):
        requests = df.with_row_index(name="index").select(["request_id", "index"]).group_by("request_id") \
            .agg([pl.len()]).filter(pl.col('len') < 1000).select(["request_id"])
        df = df.filter(pl.col("request_id").is_in(requests["request_id"]))
        if not self.use_ranker:
            # replace targets [0, 1, 2, ...] -> [0, ..., 1] for classifier
            df = df.filter(pl.col("target").is_in([0, 2])).rename({"target": "target_"}).with_columns(
                target=pl.when(pl.col("target_") == 0).then(0).when(pl.col("target_") == 2).then(1)
            ).drop("target_")
        return df

    def _clear_sets(self):
        if self.verbose:
            for nm, df in zip(["Train", "Val", "Test"], [self.train, self.val, self.test]):
                print(f"{nm} split borders: {[df['day_index'].min(), df['day_index'].max()]} with shape {df.shape}")
                num_nans, total = sum(df.null_count()).item(), df.width * df.height
                print(f"{nm} split nulls: {num_nans / total * 100:.2f}% or {num_nans} / {total}")
        self.train, self.val, self.test = self.train.fill_null(0), self.val.fill_null(0), self.test.fill_null(0)

    def _call_cached_feat(self, df, df_name, lag_type):
        df_path = self.cache_dir / f"{df_name}_{lag_type}.parquet"
        if df_path.exists():
            return pl.read_parquet(df_path)
        df = self.featurizer.create_causal_data(df, lag_type)
        df.write_parquet(df_path)
        return df

    def _prepare_data(self):
        del_columns = [
            'target', 'position_in_request', 'request_id', 'product_id', 'day_index', 'info_day_index', 'month',
        ]
        sel_columns = None
        cat_features = [
            'city_name', 'source_type', 'store_id', 'month', 'hour', 'weekday','is_weekend', 'time_of_day',
            'user_id', 'product_id', 'product_category',
        ]
        cat_features = list(set(cat_features) - set(del_columns))

        if self.use_ranker:
            assert len(self.train.filter(pl.col("request_id").is_null())) == 0
            self.train = self.train.sort("request_id")
            self.val = self.val.sort("request_id")
            self.test = self.test.sort("request_id")

        df_train, df_val, df_test = self.train, self.val, self.test
        if sel_columns is not None:
            cat_features = list(set(cat_features) & set(sel_columns))
            train_sel = df_train.select(sel_columns)
            val_sel = df_val.select(sel_columns)
            test_sel = df_test.select(sel_columns)
        else:
            train_sel = df_train.drop(del_columns)
            val_sel = df_val.drop(del_columns)
            test_sel = df_test.drop(del_columns[2:])

        if self.verbose:
            print(f"Remaining columns: {train_sel.columns}", flush=True)

        train_data = Pool(
            data=train_sel.to_pandas(),
            label=df_train['target'].to_list(),
            cat_features=cat_features,
            group_id=df_train['request_id'].cast(pl.Utf8).to_list() if self.use_ranker else None,
        )

        val_data = Pool(
            data=val_sel.to_pandas(),
            label=df_val['target'].to_list(),
            cat_features=cat_features,
            group_id=df_val['request_id'].cast(pl.Utf8).to_list() if self.use_ranker else None,
        )

        test_data = Pool(
            data=test_sel.to_pandas(),
            cat_features=cat_features,
            group_id=df_test['request_id'].cast(pl.Utf8).to_list() if self.use_ranker else None,
        )

        return train_data, val_data, test_data

    def _init_model(self):
        common_params = {
            "iterations": self.model_params.get("iters", 300),
            "learning_rate": self.model_params.get("lr", 0.02),
            "depth": self.model_params.get("depth", 3),
            "score_function": "Cosine",
            "early_stopping_rounds": 50,
            "verbose": self.verbose,
            "task_type": "GPU",
            "devices": "0:1",
        }

        if not self.use_ranker:
            return CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                **common_params,
            )

        return CatBoostRanker(
            loss_function="YetiRankPairwise",
            eval_metric="NDCG:top=10",
            **common_params,
        )

    def _process_results(self, model, val_data, test_data, lag_value=None):
        save_path = None if lag_value is None else f"{self.save_results}/out_{lag_value}.parquet"
        assert (self.lag_type == "ensemble") ^ (lag_value is None)
        d_imp, ndcg = process_results(
            model, val_data, self.val, test_data, self.test, verbose=self.verbose, save_path=save_path, is_ranker=self.use_ranker,
        )
        if not self.use_ranker and self.verbose:
            top_imp = sorted(list(d_imp.items()), key=lambda el: -el[1])
            print(top_imp)
            print([el[0] for el in top_imp if el[1] > 0.5])
        return ndcg

    def run(self):
        if self.lag_type != "ensemble":
            train_data, val_data, test_data = self._prepare_data()
            model = self._init_model()
            model.fit(train_data, eval_set=val_data)
            return self._process_results(model, val_data, test_data)

        ndcg = {}
        for lag_type in ["random"] + [f"constant_{lag}" for lag in range(1, 31)]:
            self._init_train_and_val(lag_type)
            train_data, val_data, test_data = self._prepare_data()
            model = self._init_model()
            model.fit(train_data, eval_set=val_data)
            ndcg[lag_type] = self._process_results(model, val_data, test_data, lag_value=lag_type)

        return ndcg


if __name__ == "__main__":
    model_params = {
        "iters": 1500, "lr": 0.04, "depth": 6,
    }
    trainer = Trainer(model_params=model_params, use_ranker=True, lag_type="random")
    print(f'Final metrics: {trainer.run()}')
