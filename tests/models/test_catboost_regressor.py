
from catboost import CatBoostRegressor

from src.enums.objective import Objective
from src.models.catboost_regressor import CatBoostRegressorWrapper
from tests.data_load import load_regression_data
from tests.models.model_wrapper_base import ModelWrapperBase


class TestCatboostRegresor(ModelWrapperBase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_regression_data()
        cls.model = CatBoostRegressorWrapper(early_stopping_rounds=1)
        cls.base_model = CatBoostRegressor
        cls.objective = Objective.REGRESSION
