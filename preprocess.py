import polars as pl


class ActionType:
    VIEW = 'AT_View'
    CLICK = 'AT_Click'
    CART_UPDATE = 'AT_CartUpdate'
    PURCHASE = 'AT_Purchase'


class Preprocessor:
    mapping_action_types = {
        ActionType.VIEW: 0,
        ActionType.CART_UPDATE: 1,
        ActionType.CLICK: 2,
        ActionType.PURCHASE: 3
    }
    def __init__(
        self,
        train_df: pl.DataFrame,
    ):
        self.train_df = train_df

    def run(self):
        self.train_df = self.train_df.with_columns(
            pl.col("source_type").fill_null("").alias("source_type")
        )

        self.train_df = self.train_df.with_columns(
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
            'action_type',
        )

        requests_with_cartupdate_and_view = (
            self.train_df
            .select(["request_id", "timestamp"])
            .group_by("request_id")
            .agg([pl.len()])
            .filter(pl.col('len') >= 10)
            .select(["request_id"])
        )

        self.train_df = self.train_df.join(requests_with_cartupdate_and_view, on="request_id", how="inner")

        self.day_valid_end = self.train_df["day_index"].max()
        self.day_valid_start = self.day_valid_end - 30
        self.day_train_end = self.day_valid_start - 1
        self.day_train_start = self.train_df["day_index"].min()

        self.train_history = self.train_df.filter(pl.col('day_index') <= self.day_train_end)
        self.valid_history = self.train_df.filter(pl.col('day_index') > self.day_train_end)

        return self.train_history, self.valid_history
