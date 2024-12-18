from xgboost import XGBClassifier

from src.enums.objective import Objective
from src.models.xgb_classifier import XGBClassifierWrapper
from tests.data_load import load_classification_data
from tests.models.model_wrapper_base import ModelWrapperBase


class TestXGBClassifier(ModelWrapperBase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_classification_data()
        cls.model = XGBClassifierWrapper(early_stopping_rounds=1)
        cls.base_model = XGBClassifier
        cls.objective = Objective.CLASSIFICATION
