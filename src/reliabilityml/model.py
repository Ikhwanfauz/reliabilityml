from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


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

def make_xgboost_model():
    return XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )