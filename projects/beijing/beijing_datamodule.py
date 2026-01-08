import typing as t
from pathlib import Path

from kret_lightning.data_module_custom import CustomDataModule, LoadedDfTuple, PandasInputMixin, SplitTuple
from kret_np_pd.np_pd_nb_imports import *
from kret_sklearn.pd_pipeline import PipelinePD
from kret_torch_utils.tensor_ds_custom import TensorDatasetCustom
from kret_torch_utils.UTILS_torch import KRET_TORCH_UTILS

from projects.beijing.load_beijing_data import load_beijing_air_quality_data

if t.TYPE_CHECKING:
    pass


class BeijingDataModule(CustomDataModule, PandasInputMixin):
    col_order = {"start": ["year"], "end": ["pm2.5", "cbwd"]}

    def __init__(
        self,
        data_dir: Path | str,
        sequence_length: int = 24,
        pipeline_pd: tuple[PipelinePD, PipelinePD] | None = None,
        split: SplitTuple | None = None,
    ) -> None:
        super().__init__(data_dir=data_dir, split=split, pipeline_pd=pipeline_pd)
        self.sequence_length = sequence_length

    def prepare_data(self) -> None:
        # Implement data downloading or preprocessing if needed
        pass

    def load_df(self) -> LoadedDfTuple:
        x, y = load_beijing_air_quality_data()
        return LoadedDfTuple(X=x, y=y)

    def setup(self, stage: str) -> None:
        # Load and preprocess the Beijing air quality data
        # TODO implement splitting if test or predict > 0
        no_nans = self.load_and_strip_nans()

        train_split_idx = int(self.data_split.train * len(no_nans.X))

        X_train_raw = no_nans.X.iloc[:train_split_idx]
        y_train_raw = no_nans.y.iloc[:train_split_idx]
        X_val_raw = no_nans.X.iloc[train_split_idx:]
        y_val_raw = no_nans.y.iloc[train_split_idx:]

        # TODO make sure pipeline is fit just once, and is stored

        X_train_cleaned = UKS_NP_PD.move_columns(self.PipelineX.fit_transform_df(X_train_raw), **self.col_order)
        y_train_cleaned = self.PipelineY.fit_transform_df(y_train_raw)

        X_val_cleaned = UKS_NP_PD.move_columns(self.PipelineX.transform_df(X_val_raw), **self.col_order)
        y_val_cleaned = self.PipelineY.transform_df(y_val_raw)

        train1d = TensorDatasetCustom.from_pd_xy(X_train_cleaned, y_train_cleaned)
        val1d = TensorDatasetCustom.from_pd_xy(X_val_cleaned, y_val_cleaned)
        if stage == "fit":
            train_temporal = KRET_TORCH_UTILS.create_sequence(
                train1d, sequence_length=self.sequence_length, target_offset=0
            )
            self._train = train_temporal

            val_temporal = KRET_TORCH_UTILS.create_sequence(
                val1d, sequence_length=self.sequence_length, target_offset=0
            )
            self._val = val_temporal
