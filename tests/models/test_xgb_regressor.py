from xgboost import XGBRegressor

from src.enums.objective import Objective
from src.models.xgb_regressor import XGBRegressorWrapper
from tests.data_load import load_regression_data
from tests.models.model_wrapper_base import ModelWrapperBase


class TestXGBRegresor(ModelWrapperBase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_regression_data()
        cls.model = XGBRegressorWrapper(early_stopping_rounds=1)
        cls.base_model = XGBRegressor
        cls.objective = Objective.REGRESSION

