from abc import ABC, abstractmethod


class ModelWrapper(ABC):
    """
    Interface for wrapping a Model in order to standardize methods and properties names
    """

    def __init__(self):
        self.model = None

    @abstractmethod
    def get_base_model(self, params) -> any:
        """
        Trains the model for the given number of iterations.
        :param params:
        :return:
        """
        pass

    @abstractmethod
    def get_starter_params(self) -> dict:
        """
        Gets a dictionary of parameters that are considered a "starting point" for optimization.
        :return:
        """
        pass

    @abstractmethod
    def get_grid_space(self) -> list[dict]:
        """
        Gets the parameter space for gridsearch model optimization.
        :return:
        """
        pass

    @abstractmethod
    def get_bayesian_space(self) -> dict:
        """
        Gets the parameter space for bayesian model optimization.
        :return:
        """
        pass

    @abstractmethod
    def space_to_params(self, space: dict) -> dict:
        """
        Converts a parameter space output back to a param dictionary.
        :param space:
        :return:
        """
        pass

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
        pass

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
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_best_iteration(self) -> int:
        pass

    @abstractmethod
    def get_loss(self) -> dict[str, dict[str, list[float]]]:
        """
        Returns the training loss function as a dictionary.
        :return:
        """
        pass

    @abstractmethod
    def get_feature_importance(self, features):
        """
        Returns the feature importance of the provided columns.
        :return:
        """
        pass
