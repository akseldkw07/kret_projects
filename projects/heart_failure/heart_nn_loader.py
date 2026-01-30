import pandas as pd
from kret_lightning.datamodule.data_module_custom import (
    STAGE_LITERAL,
    CustomDataModule,
    LoadedDfTuple,
    PandasInputMixin,
)
from kret_np_pd.UTILS_np_pd import NP_PD_Utils as UKS_NP_PD
from kret_sklearn.pd_pipeline import PipelinePD
from kret_torch_utils.tensor_ds_custom import TensorDatasetCustom
from sklearn.preprocessing import FunctionTransformer


class HeartNNLoader(CustomDataModule, PandasInputMixin):
    def prepare_data(self) -> None:
        # Implement data downloading or preprocessing if needed
        pass

    def load_df(self) -> LoadedDfTuple:
        df_load = FunctionTransformer(func=pd.read_csv, validate=False, kw_args={})
        custom_cleanup = FunctionTransformer(func=UKS_NP_PD.data_cleanup, validate=False, kw_args={"ret": True})
        pipeline_load_and_clean = PipelinePD(
            steps=[
                ("df_load", df_load),
                ("cleanup_custom", custom_cleanup),
            ]
        )
        file_sub_path = "datasets/heart-failure-prediction/versions/1/heart.csv"
        df = pipeline_load_and_clean.fit_transform_df(self.data_dir / file_sub_path)
        features, target = UKS_NP_PD.pop_label_and_drop(df, label_col="HeartDisease", label_ret_type="df")
        return LoadedDfTuple(X=features, y=target)

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
        match stage:
            case "fit":
                self._train = tensor1d
                self.setup("validate")  # Also setup val set
            case "validate":
                self._val = tensor1d
            case "test":
                self._test = tensor1d
            case "predict":
                self._predict = tensor1d
