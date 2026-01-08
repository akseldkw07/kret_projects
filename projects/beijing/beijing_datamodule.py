import typing as t
from pathlib import Path

from kret_lightning.datamodule.data_module_custom import (
    STAGE_LITERAL,
    CustomDataModule,
    LoadedDfTuple,
    PandasInputMixin,
    SplitTuple,
)
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
        super().__init__(data_dir=data_dir, split=split, pipeline_pd=pipeline_pd, sequence_length=sequence_length)
        self.sequence_length = sequence_length

    def prepare_data(self) -> None:
        # Implement data downloading or preprocessing if needed
        pass

    def load_df(self) -> LoadedDfTuple:
        x, y = load_beijing_air_quality_data(self.data_dir)
        return LoadedDfTuple(X=x, y=y)

    def setup(self, stage: STAGE_LITERAL) -> None:  # type: ignore[override]
        print(f"Setting up data for stage: {stage}")

        self.data_preprocess()

        match stage:
            case "fit":
                eff_split = self.SplitIdx.train
            case "validate":
                eff_split = self.SplitIdx.val
            case "test":
                assert self.SplitIdx.test is not None, f"Test split indices not defined."
                eff_split = self.SplitIdx.test
            case "predict":
                assert self.SplitIdx.predict is not None, f"Predict split indices not defined."
                eff_split = self.SplitIdx.predict
            case _:
                raise ValueError(f"Unknown stage: {stage!r}")

        tensor1d = TensorDatasetCustom.from_pd_xy(
            self.x_y_processed.X.iloc[eff_split], self.x_y_processed.y.iloc[eff_split]
        )
        tensor_temporal = KRET_TORCH_UTILS.create_sequence(
            tensor1d, sequence_length=self.sequence_length, target_offset=0
        )
        match stage:
            case "fit":
                self._train = tensor_temporal
                self.setup("validate")  # Also setup val set
            case "validate":
                self._val = tensor_temporal
            case "test":
                self._test = tensor_temporal
            case "predict":
                self._predict = tensor_temporal
