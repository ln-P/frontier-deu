import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

palette = sns.color_palette("Set2")


def create_feature_matrix_and_target(df, selected_features, target):
    """Define problem"""
    X = df[selected_features]
    y = df[target]
    return X, y


def perform_cross_validation(model, X, y, kf):
    """Use cross validation to calculate error rate"""
    neg_mse_scores = cross_val_score(
        model, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=-1
    )
    mse_scores = -neg_mse_scores
    rmse_scores = np.sqrt(mse_scores)
    return rmse_scores


def fit_linear_models_and_get_coefficients(X, y, models):
    """Save linear models coefficients"""
    coefficients = pd.DataFrame({"Feature": X.columns})
    for model_name, model in models.items():
        model.fit(X, y)
        if hasattr(model, "coef_"):
            coefficients[model_name] = model.coef_
    return coefficients


def plot_coefficients(coefficients, prefix=""):
    coefficients_melted = coefficients.melt(
        id_vars="Feature", var_name="Model", value_name="Coefficient"
    )
    plt.figure(figsize=(20, 14))
    sns.barplot(
        x="Coefficient",
        y="Feature",
        hue="Model",
        data=coefficients_melted,
        palette=palette[::2],
    )
    plt.title("Feature Coefficients for Linear and Ridge Regression")
    plt.tight_layout()
    plt.savefig(f"./output/analysis/{prefix}coefficients_ols_ridge.png")


def plot_feature_importance(model, X, y, prefix=""):
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importances = pd.DataFrame(
        {"Feature": X.columns, "Importance": importances}
    )
    plt.figure(figsize=(20, 14))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importances.sort_values(by="Importance", ascending=False),
        color=palette[2],
    )
    plt.title("Feature Importance in Random Forest")
    plt.tight_layout()
    plt.savefig(f"./output/analysis/{prefix}importance_rf.png")


def grid_search_ridge(X, y):
    """Use grid search to find best alpha"""
    alpha_range = np.logspace(-3, 3, 50)
    param_grid = {"alpha": alpha_range}
    ridge = Ridge()
    grid_search = GridSearchCV(
        ridge, param_grid, cv=10, scoring="neg_mean_squared_error", n_jobs=-1
    )
    grid_search.fit(X, y)
    best_ridge = grid_search.best_estimator_
    print(f"Best alpha: {grid_search.best_params_['alpha']}")
    return best_ridge


def plot_rmse_comparison(results_df, prefix=""):
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=results_df, palette="Set2")
    plt.title("RMSE Comparison of Three Models")
    plt.ylabel("RMSE")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./output/analysis/{prefix}rmse_comparison.png")


def run_modeling(df, selected_features, target, prefix=""):
    X, y = create_feature_matrix_and_target(df, selected_features, target)

    # Grid search for Ridge regression
    best_ridge = grid_search_ridge(X, y)

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": best_ridge,
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=123),
    }

    # Calculate RMSE
    kf = KFold(n_splits=10, shuffle=True, random_state=123)
    results = {}
    for model_name, model in models.items():
        results[model_name] = perform_cross_validation(model, X, y, kf)

    results_df = pd.DataFrame(results)

    # RMSE comparison
    plot_rmse_comparison(results_df, prefix)

    # Get coefficients and feature importances
    coefficients = fit_linear_models_and_get_coefficients(X, y, models)
    plot_coefficients(coefficients, prefix)
    plot_feature_importance(models["Random Forest"], X, y, prefix)
