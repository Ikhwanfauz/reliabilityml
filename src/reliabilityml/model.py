from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_ridge_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def make_random_forest_model():
    return RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )


def make_boosted_model():
    return HistGradientBoostingRegressor(
        max_iter=150,
        max_leaf_nodes=15,
        learning_rate=0.08,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )