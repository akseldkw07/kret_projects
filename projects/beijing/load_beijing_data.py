from functools import cache
from pathlib import Path

import pandas as pd
from kret_sklearn.custom_transformers import DateTimeSinCosNormalizer
from kret_sklearn.pd_pipeline import PipelinePD
from kret_utils.constants_kret import KretConstants
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer
from ucimlrepo import fetch_ucirepo

DATA_DIR = KretConstants.DATA_DIR / "beijing"

BEIJING_AIR_REPO_ID = 381
COLUMNS_BEIJING = ["year", "month", "day", "hour", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]


@cache
def _load_beijing_air_quality_data(id: int = BEIJING_AIR_REPO_ID):
    beijing_pm2_5 = fetch_ucirepo(id=id)

    # The data (features and target) are now pandas DataFrames
    X = beijing_pm2_5.data.features  # type: ignore
    y = beijing_pm2_5.data.targets  # type: ignore

    assert isinstance(X, pd.DataFrame)  # Features (Meteorological data)
    assert isinstance(y, pd.DataFrame)  # Target (PM2.5 concentration)

    return X.copy(deep=True), y.copy(deep=True)


def load_beijing_air_quality_data_reload(id: int = BEIJING_AIR_REPO_ID):
    X, y = _load_beijing_air_quality_data(id=id)
    return X.copy(deep=True), y.copy(deep=True)


def load_beijing_air_quality_data(data_dir: Path = DATA_DIR):
    raw = pd.read_csv(data_dir / "beijing_air.csv")
    Y = raw[["pm2.5"]]
    X = raw.drop(columns=["No"])
    return X.copy(deep=True), Y.copy(deep=True)


def get_beijing_pipeline():
    float_cols = ["pm2.5", "year", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    date_cols = ["month", "day", "hour"]
    wind_cols = ["cbwd"]

    date_time_normalizer = DateTimeSinCosNormalizer(
        datetime_cols={"month": 12, "day": 31, "hour": 24}
    )  # Normalize 'month' and 'hour' columns
    power_transformer = PowerTransformer(method="yeo-johnson", standardize=True)

    wind_encoder = OrdinalEncoder()

    column_transform = ColumnTransformer(
        transformers=[
            ("datetime", date_time_normalizer, date_cols),
            ("scaler", power_transformer, float_cols),
            ("windlabel", wind_encoder, wind_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
        verbose=True,
    )
    pipeline_x = PipelinePD(steps=[("column_transform", column_transform)])
    pipeline_y = PipelinePD(steps=[("scaler", power_transformer)])

    return pipeline_x, pipeline_y
