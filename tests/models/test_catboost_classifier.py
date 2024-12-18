from catboost import CatBoostClassifier

from src.enums.objective import Objective
from src.models.catboost_classifier import CatBoostClassifierWrapper
from tests.data_load import load_classification_data
from tests.models.model_wrapper_base import ModelWrapperBase


class TestCatboostClassifier(ModelWrapperBase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_classification_data()
        cls.model = CatBoostClassifierWrapper(early_stopping_rounds=1)
        cls.base_model = CatBoostClassifier
        cls.objective = Objective.CLASSIFICATION
