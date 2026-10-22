from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_ridge_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))