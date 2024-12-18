from lightgbm import LGBMClassifier

from src.enums.objective import Objective
from src.models.lgbm_classifier import LGBMClassifierWrapper
from tests.data_load import load_classification_data
from tests.models.model_wrapper_base import ModelWrapperBase


class TestLgbmCatboostClassifier(ModelWrapperBase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_classification_data()
        cls.model = LGBMClassifierWrapper(early_stopping_rounds=1)
        cls.base_model = LGBMClassifier
        cls.objective = Objective.CLASSIFICATION
