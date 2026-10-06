"""
Cross-validation by hand on the diabetes data.

Model A is linear regression on bmi and sex_dummy.
Model B is an inverse-MSE weighted average of two small trees.
Folds are built manually; nothing is taken from sklearn's CV helpers.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_text


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data():
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame
    df["sex_category"] = df["sex"].apply(lambda x: "Moonbeam" if x < 0 else "Thunderpaws")
    df["sex_dummy"] = (df["sex_category"] == "Thunderpaws").astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Model functions
# ---------------------------------------------------------------------------

def fit_linear(train_df):
    model = LinearRegression()
    model.fit(train_df[["bmi", "sex_dummy"]], train_df["target"])
    return model


def fit_ensemble(train_df):
    """Fit both trees and their weights on train_df.

    Weights use each tree's MSE on these training rows only. Building the
    weights from test rows would leak the data we are trying to evaluate,
    and the test error would look better than it is.
    """
    y = train_df["target"]
    tree_bmi = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_sex = DecisionTreeRegressor(max_depth=1, random_state=42)
    tree_bmi.fit(train_df[["bmi"]], y)
    tree_sex.fit(train_df[["sex_dummy"]], y)

    mse_bmi = mse(y, tree_bmi.predict(train_df[["bmi"]]))
    mse_sex = mse(y, tree_sex.predict(train_df[["sex_dummy"]]))
    inv_sum = (1.0 / mse_bmi) + (1.0 / mse_sex)
    w_bmi = (1.0 / mse_bmi) / inv_sum
    w_sex = (1.0 / mse_sex) / inv_sum
    return tree_bmi, tree_sex, w_bmi, w_sex


def predict_ensemble(model_tuple, test_df):
    tree_bmi, tree_sex, w_bmi, w_sex = model_tuple
    pred_bmi = tree_bmi.predict(test_df[["bmi"]])
    pred_sex = tree_sex.predict(test_df[["sex_dummy"]])
    return w_bmi * pred_bmi + w_sex * pred_sex


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return ((y_true - y_pred) ** 2).mean()


def _internal_thresholds(tree):
    """Split thresholds only. Leaves store a -2.0 placeholder, so skip them."""
    structure = tree.tree_
    internal = structure.children_left != -1
    return structure.threshold[internal]


def _leaf_values(tree):
    """Leaf predictions in increasing node-id order (left to right)."""
    structure = tree.tree_
    leaves = structure.children_left == -1
    return structure.value[leaves, 0, 0]


# ---------------------------------------------------------------------------
# 3. Fold construction
# ---------------------------------------------------------------------------

def make_folds(n, k=5, seed=42):
    """Shuffle row indices, then cut them into k consecutive blocks.

    Block i has size n // k, and the first n % k blocks get one extra row.
    For n=442 and k=5 that is 89, 89, 88, 88, 88.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    base = n // k
    extra = n % k

    folds = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        folds.append(shuffled[start:start + size])
        start += size

    covered = np.concatenate(folds)
    assert np.array_equal(np.sort(covered), np.arange(n)), (
        "folds must be disjoint and cover every row"
    )
    return folds


# ---------------------------------------------------------------------------
# 4. K-fold cross-validation
# ---------------------------------------------------------------------------

def k_fold_cv(df, k=5, seed=42):
    """Test MSE of each model. Every fold refits from scratch on its train rows."""
    folds = make_folds(len(df), k=k, seed=seed)
    scores = {name: [] for name in ("linear", "tree_bmi", "tree_sex", "ensemble")}

    for test_idx in folds:
        test_mask = np.zeros(len(df), dtype=bool)
        test_mask[test_idx] = True
        train_df = df.iloc[~test_mask]
        test_df = df.iloc[test_idx]
        y_test = test_df["target"]

        linear = fit_linear(train_df)
        scores["linear"].append(mse(y_test, linear.predict(test_df[["bmi", "sex_dummy"]])))

        # New trees and new weights on this fold's training rows only.
        # Reusing a previous fold's weights would mix in rows that are now test data.
        fitted = fit_ensemble(train_df)
        tree_bmi, tree_sex, _, _ = fitted
        scores["tree_bmi"].append(mse(y_test, tree_bmi.predict(test_df[["bmi"]])))
        scores["tree_sex"].append(mse(y_test, tree_sex.predict(test_df[["sex_dummy"]])))
        scores["ensemble"].append(mse(y_test, predict_ensemble(fitted, test_df)))

    return pd.DataFrame(scores)


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------

def _print_tree(title, tree, feature_name, train_df):
    y = train_df["target"]
    pred = tree.predict(train_df[[feature_name]])
    thresholds = ", ".join(f"{value:.6f}" for value in _internal_thresholds(tree))
    leaves = ", ".join(f"{value:.4f}" for value in _leaf_values(tree))

    print(f"\n{title}")
    print(f"  learned thresholds: {thresholds}")
    print(f"  leaf values (left to right): {leaves}")
    print(f"  training MSE: {mse(y, pred):.4f}")
    print("  export_text:")
    print(export_text(tree, feature_names=[feature_name], decimals=6))


def _print_cv_table(scores):
    cols = ["linear", "tree_bmi", "tree_sex", "ensemble"]
    print(f"{'fold':>4} | {'linear':>10} | {'tree_bmi':>10} | {'tree_sex':>10} | {'ensemble':>10}")
    print("-" * 75)
    for i, row in scores.iterrows():
        print(
            f"{i + 1:4d} | {row['linear']:10.2f} | {row['tree_bmi']:10.2f} | "
            f"{row['tree_sex']:10.2f} | {row['ensemble']:10.2f}"
        )

    means = " | ".join(f"{np.mean(scores[col]):10.2f}" for col in cols)
    sds = " | ".join(f"{np.std(scores[col].to_numpy(), ddof=1):10.2f}" for col in cols)
    print(f"{'mean':>4} | {means}")
    print(f"{'sd':>4} | {sds}")


def main():
    df = load_data()
    y = df["target"].to_numpy()
    n = len(df)
    baseline = mse(y, np.full(n, y.mean()))

    print("Part 1: fit on the full dataset")
    linear = fit_linear(df)
    pred_linear = linear.predict(df[["bmi", "sex_dummy"]])
    sse = float(np.sum((y - pred_linear) ** 2))

    print("\nModel A (LinearRegression)")
    print(f"  intercept: {linear.intercept_:.4f}")
    print(f"  bmi coefficient: {linear.coef_[0]:.4f}")
    print(f"  sex_dummy coefficient: {linear.coef_[1]:.4f}")
    print(f"  SSE: {sse:.4f}")
    print(f"  MSE (SSE / n): {sse / n:.4f}")

    # Same fit path as CV: trees and weights both come from the training rows.
    fitted = fit_ensemble(df)
    tree_bmi, tree_sex, w_bmi, w_sex = fitted

    _print_tree("Tree on bmi (max_depth=2)", tree_bmi, "bmi", df)
    _print_tree("Tree on sex_dummy (max_depth=1)", tree_sex, "sex_dummy", df)

    print("Ensemble weights (from training MSE only)")
    print(f"  w_bmi: {w_bmi:.3f}")
    print(f"  w_sex: {w_sex:.3f}")
    print(f"  sum check: {w_bmi + w_sex:.3f}")
    print(f"  ensemble MSE: {mse(y, predict_ensemble(fitted, df)):.4f}")
    print(f"  baseline MSE (always predict mean y): {baseline:.4f}")

    print("\nPart 2: 5-fold cross-validation")
    folds = make_folds(n, k=5, seed=42)
    print("Fold sizes:", [len(fold) for fold in folds])
    print()
    _print_cv_table(k_fold_cv(df, k=5, seed=42))
    print(f"\nBaseline MSE (always predict mean y): {baseline:.4f}")


if __name__ == "__main__":
    main()
