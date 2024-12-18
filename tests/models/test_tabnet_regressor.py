from pytorch_tabnet.tab_model import TabNetRegressor
from src.enums.objective import Objective
from src.models.tabnet_regressor import TabNetRegressorWrapper
from tests.data_load import load_regression_data
from tests.models.model_wrapper_base import ModelWrapperBase


class TestTabnetRegressor(ModelWrapperBase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_regression_data()
        cls.model = TabNetRegressorWrapper(early_stopping_rounds=1)
        cls.base_model = TabNetRegressor
        cls.objective = Objective.REGRESSION
