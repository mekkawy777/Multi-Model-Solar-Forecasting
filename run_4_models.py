# run_4_models.py
# =========================================================
# Run 4 regression models safely on all datasets:
# SVR + MLP + LSTM + KNN
# Handles datetime columns correctly and avoids dummy-encoding timestamps.
# =========================================================

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR

# --- Optional TF (for LSTM) ---
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    TF_AVAILABLE = False


# -----------------------------
# Utils: safer data cleaning
# -----------------------------

def _coerce_object_to_numeric(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """
    Try to convert object columns to numeric if most values are numeric-like.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            conv = pd.to_numeric(df[col], errors="coerce")
            # if most values become numeric, accept conversion
            if conv.notna().mean() >= threshold:
                df[col] = conv
    return df


def _add_datetime_features(df: pd.DataFrame, min_parse_rate: float = 0.80) -> pd.DataFrame:
    """
    Detect datetime-like columns and replace them with engineered numeric features.
    """
    df = df.copy()
    for col in list(df.columns):
        # only try for object / datetime columns
        if df[col].dtype == object or np.issubdtype(df[col].dtype, np.datetime64):
            dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if dt.notna().mean() >= min_parse_rate:
                df[f"{col}_year"] = dt.dt.year
                df[f"{col}_month"] = dt.dt.month
                df[f"{col}_day"] = dt.dt.day
                df[f"{col}_dayofweek"] = dt.dt.dayofweek
                df[f"{col}_hour"] = dt.dt.hour
                df[f"{col}_minute"] = dt.dt.minute
                df.drop(columns=[col], inplace=True)
    return df


def _pick_target(df: pd.DataFrame, target_hint: str | None = None) -> str:
    """
    Pick target column robustly.
    """
    if target_hint and target_hint in df.columns:
        return target_hint
    if "AC_POWER" in df.columns:
        return "AC_POWER"
    # common names in your files
    for cand in ["SystemProduction", "generated_power_kw"]:
        if cand in df.columns:
            return cand
    return df.columns[-1]


def _build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return pre


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# -----------------------------
# LSTM (uses same preprocessed X)
# -----------------------------

def _run_lstm(preprocessor: ColumnTransformer,
             X_train: pd.DataFrame, X_test: pd.DataFrame,
             y_train: np.ndarray, y_test: np.ndarray,
             seed: int = 42) -> dict:
    """
    Train and evaluate a small LSTM on tabular data (timesteps=1).
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow غير متوفر. ثبّت tensorflow لتشغيل LSTM.")

    # reproducibility (best-effort)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    # preprocess -> dense numeric matrix
    Xtr = preprocessor.fit_transform(X_train).astype(np.float32)
    Xte = preprocessor.transform(X_test).astype(np.float32)

    # scale y
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)
    ytr = ((y_train - y_mean) / y_std).astype(np.float32)
    yte = ((y_test - y_mean) / y_std).astype(np.float32)

    # reshape for LSTM: (samples, timesteps=1, features)
    Xtr = Xtr.reshape((Xtr.shape[0], 1, Xtr.shape[1]))
    Xte = Xte.reshape((Xte.shape[0], 1, Xte.shape[1]))

    model = Sequential([
        Input(shape=(1, Xtr.shape[2])),
        LSTM(16),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        Xtr, ytr,
        validation_split=0.2,
        epochs=25,
        batch_size=256,
        callbacks=[es],
        verbose=0
    )

    pred_scaled = model.predict(Xte, verbose=0).ravel()
    pred = pred_scaled * y_std + y_mean
    y_true = yte * y_std + y_mean

    return _reg_metrics(y_true, pred)


# -----------------------------
# Main experiment runner
# -----------------------------

def run_all(base_dir: Path) -> pd.DataFrame:
    datasets = {
        "A": ("Solar Power Plant Data.csv", None),
        "B": ("Plant_1_Generation_Data.csv", "AC_POWER"),
        "C": ("Plant_2_Generation_Data.csv", "AC_POWER"),
        "E": ("spg.csv", "generated_power_kw"),
    }

    results = []

    for ds_name, (file_name, target_hint) in datasets.items():
        path = base_dir / file_name
        if not path.exists():
            results.append({
                "Dataset": ds_name, "File": file_name, "Model": "ALL",
                "MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan,
                "Seconds": 0.0, "Error": f"File not found: {path}"
            })
            continue

        # load
        df = pd.read_csv(path)
        df = df.dropna(axis=1, how="all")
        df = _coerce_object_to_numeric(df)
        df = _add_datetime_features(df)

        target = _pick_target(df, target_hint)

        # target must be numeric
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[target])

        X = df.drop(columns=[target])
        y = df[target].astype(np.float32).to_numpy()

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pre = _build_preprocessor(X_train)

        # --- Models ---
        # SVR can be very slow with large datasets -> auto fallback to LinearSVR
        if len(X_train) > 12000:
            svr = Pipeline([("pre", pre), ("svr", LinearSVR(random_state=42, max_iter=5000))])
            svr_name = "SVR (LinearSVR fallback)"
        else:
            svr = Pipeline([("pre", pre), ("svr", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1))])
            svr_name = "SVR (RBF)"

        mlp = TransformedTargetRegressor(
            regressor=Pipeline([
                ("pre", pre),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    max_iter=300,
                    early_stopping=True,
                    n_iter_no_change=10,
                    random_state=42
                ))
            ]),
            transformer=StandardScaler()
        )

        knn = Pipeline([
            ("pre", pre),
            ("knn", KNeighborsRegressor(
                n_neighbors=7,
                weights="distance",
                n_jobs=-1
            ))
        ])

        model_specs = [
            (svr_name, svr, "sklearn"),
            ("MLPRegressor", mlp, "sklearn"),
            ("KNN", knn, "sklearn"),
            ("LSTM", None, "lstm"),
        ]

        for model_name, model, kind in model_specs:
            t0 = time.time()
            err = ""
            try:
                if kind == "sklearn":
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    mets = _reg_metrics(y_test, pred)

                else:  # LSTM
                    mets = _run_lstm(pre, X_train, X_test, y_train, y_test)

            except Exception as e:
                mets = {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
                err = str(e)

            dt = time.time() - t0
            results.append({
                "Dataset": ds_name,
                "File": file_name,
                "Target": target,
                "Model": model_name,
                **mets,
                "Seconds": float(dt),
                "Error": err
            })

    return pd.DataFrame(results)

def _superscript_int(n: int) -> str:
    sup = str(n).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
    return sup

def fmt_mse(x: float) -> str:
    if pd.isna(x):
        return "NaN"
    ax = abs(float(x))
    # scientific for big/small values
    if ax >= 1e4 or (0 < ax < 1e-2):
        s = f"{x:.3g}"  # e.g. 7.34e+05
        base, exp = s.split("e")
        exp = int(exp)
        return f"{base} × 10{_superscript_int(exp)}"
    return f"{x:.4f}"

def fmt_rmse(x: float) -> str:
    return "NaN" if pd.isna(x) else f"{float(x):.2f}"

def fmt_r2(x: float) -> str:
    return "NaN" if pd.isna(x) else f"{float(x):.4f}"

def model_group(name: str) -> str:
    if str(name).startswith("SVR"):
        return "SVR"
    if str(name).startswith("MLP"):
        return "MLP"
    if str(name).startswith("KNN"):
        return "KNN"
    if str(name).startswith("LSTM"):
        return "LSTM"
    return str(name)

def print_model_tables(df_results: pd.DataFrame) -> None:
    df = df_results.copy()
    df["ModelGroup"] = df["Model"].apply(model_group)

    for mg in ["SVR", "MLP", "KNN", "LSTM"]:
        t = df[df["ModelGroup"] == mg][["Dataset", "MSE", "RMSE", "R2"]].copy()
        t = t.sort_values("Dataset")
        t["MSE"] = t["MSE"].apply(fmt_mse)
        t["RMSE"] = t["RMSE"].apply(fmt_rmse)
        t["R²"] = t["R2"].apply(fmt_r2)
        t = t.drop(columns=["R2"])

        print(f"\n===== {mg} Results =====")
        print(t.to_string(index=False))

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    df_results = run_all(base_dir)

    # Print each model as its own table
    print_model_tables(df_results)

    # Save full results
    out_path = base_dir / "results_4_models.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

