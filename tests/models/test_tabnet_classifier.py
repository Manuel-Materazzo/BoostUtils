from pytorch_tabnet.tab_model import TabNetClassifier
from src.enums.objective import Objective
from src.models.tabnet_classifier import TabNetClassifierWrapper
from tests.data_load import load_classification_data
from tests.models.model_wrapper_base import ModelWrapperBase


class TestTabnetClassifier(ModelWrapperBase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_classification_data()
        cls.model = TabNetClassifierWrapper(early_stopping_rounds=1)
        cls.base_model = TabNetClassifier
        cls.objective = Objective.CLASSIFICATION
