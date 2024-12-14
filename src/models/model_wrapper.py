from abc import abstractmethod

from src.enums.objective import Objective
from src.models.model_inference_wrapper import ModelInferenceWrapper


class ModelWrapper(ModelInferenceWrapper):
    """
    Interface for wrapping a Model in order to standardize methods and properties names
    """

    def __init__(self, early_stopping_rounds=10):
        self.model = None
        self.early_stopping_rounds = early_stopping_rounds

    @abstractmethod
    def get_objective(self) -> Objective:
        """
        Returns the model default objective.
        :return:
        """

    @abstractmethod
    def get_base_model(self, iterations, params) -> any:
        """
        Trains the model for the given number of iterations.
        :param iterations:
        :param params:
        :return:
        """

    @abstractmethod
    def get_starter_params(self) -> dict:
        """
        Gets a dictionary of parameters that are considered a "starting point" for optimization.
        :return:
        """

    @abstractmethod
    def get_grid_space(self) -> list[dict]:
        """
        Gets the parameter space for gridsearch model optimization.
        :return:
        """

    @abstractmethod
    def get_bayesian_space(self) -> dict:
        """
        Gets the parameter space for bayesian model optimization.
        :return:
        """

    @abstractmethod
    def train_until_optimal(self, train_X, validation_X, train_y, validation_y, params) -> int:
        """
        Trains the model until the loss function stops improving. Returns the number of iterations.
        :param params:
        :param train_X:
        :param validation_X:
        :param train_y:
        :param validation_y:
        :return:
        """

    @abstractmethod
    def fit(self, X, y, iterations, params):
        """
        Trains the model for the given number of iterations.
        :param params:
        :param X:
        :param y:
        :param iterations:
        :return:
        """

    @abstractmethod
    def predict(self, X):
        """
        Predicts the target for the given input data.
        :param X:
        :return:
        """

    @abstractmethod
    def predict_proba(self, X):
        """
        Predicts the target probability for the given input data.
        :param X:
        :return:
        """

    @abstractmethod
    def get_best_iteration(self) -> int:
        """
        Gets the best iteration for the validation run.
        :return:
        """

    @abstractmethod
    def get_loss(self) -> dict[str, dict[str, list[float]]]:
        """
        Returns the training loss function as a dictionary.
        :return:
        """

    @abstractmethod
    def get_feature_importance(self, features):
        """
        Returns the feature importance of the provided columns.
        :return:
        """
