from enum import Enum


class AccuracyMetric(Enum):
    # Regression
    MAE = "MAE"
    MSE = "MSE"
    RMSE = "RMSE"
    # Classification
    AUC = "AUC"
    Accuracy = "accuracy"
    QWK = "QWK"
