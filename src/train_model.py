"""Experimental band-gap prediction and generalisation experiment.

Run from the project root:
    python src/train_model.py

This script deliberately reports metrics from the actual run rather than
assuming a particular model performance.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42


def load_data() -> pd.DataFrame:
    """Load the matminer experimental band-gap dataset and validate columns."""
    df = load_dataset("expt_gap")
    print("Dataset columns:", list(df.columns))

    formula_col = next((c for c in ("formula", "composition", "Composition") if c in df.columns), None)
    target_col = next((c for c in ("gap expt", "gap_expt", "expt_gap", "band_gap") if c in df.columns), None)

    if formula_col is None or target_col is None:
        raise ValueError(
            "Could not identify expected formula/experimental-band-gap columns. "
            f"Available columns: {list(df.columns)}. Check matminer dataset documentation "
            "and update formula_col/target_col in src/train_model.py."
        )

    result = df[[formula_col, target_col]].copy()
    result.columns = ["formula", "band_gap"]
    result["band_gap"] = pd.to_numeric(result["band_gap"], errors="coerce")
    result = result.dropna(subset=["formula", "band_gap"])
    result = result[np.isfinite(result["band_gap"])]
    result = result[result["band_gap"] >= 0].copy()

    # Canonicalise compositions; multiple rows with the same reduced formula are
    # retained in the main data, but the grouped evaluation keeps formulas together.
    canonical = []
    keep = []
    for idx, formula in result["formula"].items():
        try:
            comp = Composition(str(formula))
            canonical.append(comp.reduced_formula)
            keep.append(idx)
        except Exception:
            continue
    result = result.loc[keep].copy()
    result["group_formula"] = canonical
    return result.reset_index(drop=True)


def featurize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Convert formulae to Magpie elemental-property descriptors."""
    compositions = []
    groups = []
    targets = []
    for formula, group_formula, target in zip(df["formula"], df["group_formula"], df["band_gap"]):
        try:
            comp = Composition(str(formula))
            if comp.num_atoms <= 0:
                continue
            compositions.append(comp)
            groups.append(group_formula)
            targets.append(float(target))
        except Exception:
            continue

    featurizer = ElementProperty.from_preset("magpie")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_features = featurizer.featurize_many(
            compositions, ignore_errors=True, return_errors=False, pbar=False
        )

    X = pd.DataFrame(raw_features, columns=featurizer.feature_labels())
    y = pd.Series(targets, name="band_gap")
    g = pd.Series(groups, name="group_formula")
    valid = X.notna().any(axis=1) & y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), g.loc[valid].reset_index(drop=True)


def make_model(kind: str) -> Pipeline:
    if kind == "baseline":
        estimator = DummyRegressor(strategy="mean")
    elif kind == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {kind}")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def evaluate_split(X, y, train_idx, test_idx, split_name: str):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    results = {}
    prediction_frame = pd.DataFrame({"actual_band_gap": y_test.to_numpy()})
    fitted_rf = None

    for model_name in ("baseline", "random_forest"):
        model = make_model(model_name)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[model_name] = {
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": float(r2_score(y_test, pred)) if len(y_test) > 1 else None,
        }
        prediction_frame[f"predicted_{model_name}"] = pred
        if model_name == "random_forest":
            fitted_rf = model

    prediction_frame.to_csv(OUT / f"predictions_{split_name}.csv", index=False)

    rf_pred = prediction_frame["predicted_random_forest"].to_numpy()
    actual = prediction_frame["actual_band_gap"].to_numpy()
    lo = float(min(actual.min(), rf_pred.min()))
    hi = float(max(actual.max(), rf_pred.max()))
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, rf_pred, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual experimental band gap (eV)")
    plt.ylabel("Predicted band gap (eV)")
    plt.title(f"Actual vs predicted: {split_name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(OUT / f"{split_name}_actual_vs_predicted.png", dpi=160)
    plt.close()

    if fitted_rf is not None and split_name == "random_split":
        forest = fitted_rf.named_steps["model"]
        importances = pd.Series(forest.feature_importances_, index=X.columns)
        top = importances.sort_values(ascending=False).head(15).sort_values()
        plt.figure(figsize=(9, 6))
        top.plot(kind="barh")
        plt.xlabel("Random Forest feature importance")
        plt.title("Top composition descriptors (model-specific)")
        plt.tight_layout()
        plt.savefig(OUT / "feature_importance.png", dpi=160)
        plt.close()

    return results, len(train_idx), len(test_idx)


def main():
    print("Loading matminer 'expt_gap' dataset...")
    df = load_data()
    print(f"Rows after basic cleaning: {len(df)}")

    print("Creating Magpie composition descriptors...")
    X, y, groups = featurize(df)
    if len(X) < 50:
        raise RuntimeError(f"Only {len(X)} usable rows were featurized; check dataset/library compatibility.")
    print(f"Usable rows: {len(X)} | features: {X.shape[1]} | unique reduced formulae: {groups.nunique()}")

    all_indices = np.arange(len(X))
    random_train, random_test = train_test_split(
        all_indices, test_size=0.2, random_state=SEED
    )

    # Keep reduced formula groups disjoint between train and test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    grouped_train, grouped_test = next(splitter.split(X, y, groups=groups))

    random_results, random_n_train, random_n_test = evaluate_split(
        X, y, random_train, random_test, "random_split"
    )
    grouped_results, grouped_n_train, grouped_n_test = evaluate_split(
        X, y, grouped_train, grouped_test, "grouped_split"
    )

    report = {
        "dataset": "matminer: expt_gap",
        "target": "experimental band gap (eV, as represented in source dataset)",
        "n_usable_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_unique_reduced_formulae": int(groups.nunique()),
        "seed": SEED,
        "random_split": {
            "train_size": int(random_n_train),
            "test_size": int(random_n_test),
            "metrics": random_results,
        },
        "formula_grouped_split": {
            "train_size": int(grouped_n_train),
            "test_size": int(grouped_n_test),
            "metrics": grouped_results,
        },
        "limitations": [
            "Composition-only features omit crystal structure and measurement conditions.",
            "A grouped split by reduced formula is more challenging but does not prove broad chemical novelty.",
            "Experimental band gaps may be collected using different methods and sources.",
            "Feature importance is model-specific and not causal evidence.",
            "Predictions are not experimental measurements or new materials discoveries.",
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nResults from this run:")
    for split_name, metrics in (("Random split", random_results), ("Formula-grouped split", grouped_results)):
        print(f"\n{split_name}:")
        for model_name, values in metrics.items():
            print(f"  {model_name}: MAE={values['MAE']:.4f}, RMSE={values['RMSE']:.4f}, R2={values['R2']}")
    print(f"\nSaved outputs to: {OUT}")
    print("Read README.md and metrics.json before reporting results.")


if __name__ == "__main__":
    main()
