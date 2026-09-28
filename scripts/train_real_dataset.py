from pathlib import Path
import json
import warnings
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "real"
MODEL_DIR = ROOT / "models"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# UCI DATASET
# ============================================================

DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/421/"
    "aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip"
)

ZIP_PATH = DATA_DIR / "scania_uci.zip"

TRAIN_FILE = DATA_DIR / "aps_failure_training_set.csv"


# ============================================================
# DOWNLOAD DATASET
# ============================================================
def download_dataset():
    """
    Download and extract the official UCI APS Failure at Scania Trucks
    dataset.
    """

    import time
    import urllib.request
    import zipfile

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Check whether the actual training file already exists
    # ---------------------------------------------------------
    if TRAIN_FILE.exists():
        print("=" * 70)
        print("SCANIA DATASET ALREADY EXTRACTED")
        print("=" * 70)
        print(f"Training file: {TRAIN_FILE}")
        return

    # ---------------------------------------------------------
    # Remove incomplete ZIP
    # ---------------------------------------------------------
    if ZIP_PATH.exists():
        size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)

        print(
            f"Existing ZIP found: {size_mb:.1f} MB"
        )

        # A valid archive should be substantially larger than
        # the interrupted 47 MB file.
        if size_mb < 50:
            print("Existing ZIP appears incomplete.")
            print("Removing incomplete ZIP...")
            ZIP_PATH.unlink()

    # ---------------------------------------------------------
    # Download
    # ---------------------------------------------------------
    if not ZIP_PATH.exists():

        print("=" * 70)
        print("DOWNLOADING OFFICIAL UCI SCANIA DATASET")
        print("=" * 70)
        print()
        print("Downloading approximately 54 MB...")
        print()

        for attempt in range(1, 6):

            print(f"Download attempt {attempt}/5")

            try:

                request = urllib.request.Request(
                    DOWNLOAD_URL,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 "
                            "(Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 "
                            "Chrome/153.0 Safari/537.36"
                        )
                    },
                )

                with urllib.request.urlopen(
                    request,
                    timeout=180
                ) as response:

                    total_size = int(
                        response.headers.get(
                            "Content-Length",
                            0
                        )
                    )

                    downloaded = 0
                    chunk_size = 1024 * 1024

                    with open(
                        ZIP_PATH,
                        "wb"
                    ) as file:

                        start_time = time.time()

                        while True:

                            chunk = response.read(
                                chunk_size
                            )

                            if not chunk:
                                break

                            file.write(chunk)
                            downloaded += len(chunk)

                            if total_size:

                                percent = (
                                    downloaded /
                                    total_size
                                ) * 100

                                downloaded_mb = (
                                    downloaded /
                                    (1024 * 1024)
                                )

                                total_mb = (
                                    total_size /
                                    (1024 * 1024)
                                )

                                elapsed = max(
                                    time.time() -
                                    start_time,
                                    0.001
                                )

                                speed = (
                                    downloaded_mb /
                                    elapsed
                                )

                                print(
                                    f"\rProgress: "
                                    f"{percent:6.2f}% | "
                                    f"{downloaded_mb:6.1f}/"
                                    f"{total_mb:.1f} MB | "
                                    f"{speed:5.2f} MB/s",
                                    end="",
                                    flush=True
                                )

                print()
                print()

                # -------------------------------------------------
                # Validate ZIP
                # -------------------------------------------------
                if not zipfile.is_zipfile(
                    ZIP_PATH
                ):
                    raise RuntimeError(
                        "Downloaded file is not a valid ZIP archive."
                    )

                size_mb = (
                    ZIP_PATH.stat().st_size /
                    (1024 * 1024)
                )

                print(
                    f"Download completed: "
                    f"{size_mb:.1f} MB"
                )

                break

            except KeyboardInterrupt:
                print()
                print(
                    "Download interrupted."
                )
                raise

            except Exception as e:

                print()
                print(
                    f"Download failed: {e}"
                )

                if ZIP_PATH.exists():
                    ZIP_PATH.unlink()

                if attempt < 5:
                    print(
                        "Retrying in 5 seconds..."
                    )
                    time.sleep(5)

                else:
                    raise RuntimeError(
                        "Unable to download the "
                        "UCI dataset after 5 attempts."
                    )

    # ---------------------------------------------------------
    # Extract ZIP
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("EXTRACTING SCANIA DATASET")
    print("=" * 70)

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as zip_ref:

        members = zip_ref.namelist()

        print(
            f"Files inside ZIP: {len(members)}"
        )

        for member in members:

            print(
                f"  {member}"
            )

        zip_ref.extractall(DATA_DIR)

    # ---------------------------------------------------------
    # Locate training file
    # ---------------------------------------------------------
    if TRAIN_FILE.exists():
        print()
        print(
            f"Training dataset extracted:"
        )
        print(
            f"{TRAIN_FILE}"
        )
        return

    # Sometimes UCI archives contain an additional
    # directory level. Search recursively.
    matches = list(
        DATA_DIR.rglob(
            "aps_failure_training_set.csv"
        )
    )

    if matches:

        source = matches[0]

        if source != TRAIN_FILE:

            TRAIN_FILE.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            source.replace(
                TRAIN_FILE
            )

        print()
        print(
            f"Training dataset found:"
        )
        print(
            f"{TRAIN_FILE}"
        )

        return

    raise FileNotFoundError(
        "aps_failure_training_set.csv "
        "was not found after extraction."
    )



# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    """
    Load the official UCI APS Failure at Scania Trucks dataset.

    File structure:
        Lines 1-20  -> license/description
        Line 21     -> CSV header
        Line 22+    -> data
    """

    import zipfile
    import shutil

    print("=" * 70)
    print("LOADING SCANIA DATASET")
    print("=" * 70)

    # ---------------------------------------------------------
    # Make sure training file exists
    # ---------------------------------------------------------
    if not TRAIN_FILE.exists():

        print()
        print("Training CSV not found.")
        print("Extracting downloaded UCI ZIP...")
        print()

        if not ZIP_PATH.exists():
            raise FileNotFoundError(
                f"Dataset ZIP not found: {ZIP_PATH}"
            )

        with zipfile.ZipFile(
            ZIP_PATH,
            "r"
        ) as zip_ref:

            zip_ref.extractall(DATA_DIR)

        matches = list(
            DATA_DIR.rglob(
                "aps_failure_training_set.csv"
            )
        )

        if not matches:
            raise FileNotFoundError(
                "aps_failure_training_set.csv "
                "was not found in the downloaded ZIP."
            )

        TRAIN_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if matches[0] != TRAIN_FILE:
            shutil.copy2(
                matches[0],
                TRAIN_FILE
            )

        print(
            f"Training file extracted:"
        )
        print(
            TRAIN_FILE
        )

    else:

        print(
            f"Training CSV already exists: "
            f"{TRAIN_FILE}"
        )

    # ---------------------------------------------------------
    # Read the actual CSV
    # ---------------------------------------------------------
    print()
    print("Reading training data...")

    df = pd.read_csv(
        TRAIN_FILE,
        skiprows=20,
        sep=",",
        na_values=["na"],
        low_memory=False
    )

    # ---------------------------------------------------------
    # Clean column names
    # ---------------------------------------------------------
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Verify structure
    # ---------------------------------------------------------
    print()
    print(
        f"Dataset shape: "
        f"{df.shape[0]:,} rows x "
        f"{df.shape[1]:,} columns"
    )

    if df.shape[1] != 171:
        raise ValueError(
            f"Expected 171 columns but found "
            f"{df.shape[1]} columns."
        )

    if "class" not in df.columns:
        raise ValueError(
            "Target column 'class' was not found."
        )

    # ---------------------------------------------------------
    # Target distribution BEFORE encoding
    # ---------------------------------------------------------
    print()
    print("Target distribution:")

    print(
        df["class"].value_counts()
    )

    print()
    print("Target percentage:")

    print(
        df["class"]
        .value_counts(normalize=True)
        .mul(100)
        .round(3)
    )

    # ---------------------------------------------------------
    # Encode target
    # ---------------------------------------------------------
    df["class"] = (
        df["class"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "neg": 0,
            "pos": 1
        })
    )

    if df["class"].isna().any():
        raise ValueError(
            "Unexpected target value found."
        )

    df["class"] = df["class"].astype(int)

    print()
    print("Target encoding:")
    print("0 = NEGATIVE")
    print("1 = POSITIVE")

    return df
# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):
    """
    Clean the Scania dataset and prepare X and y.

    Target encoding has already been performed in load_data():
        0 = NEGATIVE
        1 = POSITIVE
    """

    print("=" * 70)
    print("CLEANING DATA")
    print("=" * 70)

    data = df.copy()

    # ---------------------------------------------------------
    # Separate target
    # ---------------------------------------------------------
    if "class" not in data.columns:
        raise ValueError(
            "Target column 'class' not found."
        )

    y = data["class"].copy()

    X = data.drop(
        columns=["class"]
    ).copy()

    # ---------------------------------------------------------
    # Verify target
    # ---------------------------------------------------------
    unique_targets = sorted(
        y.dropna().unique().tolist()
    )

    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(
            f"Unexpected target values found: "
            f"{unique_targets}"
        )

    y = y.astype(int)

    # ---------------------------------------------------------
    # Convert all features to numeric
    # ---------------------------------------------------------
    print()
    print("Converting sensor features to numeric...")

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # ---------------------------------------------------------
    # Remove completely empty columns
    # ---------------------------------------------------------
    empty_columns = [
        column
        for column in X.columns
        if X[column].isna().all()
    ]

    if empty_columns:

        print()
        print(
            f"Removing {len(empty_columns)} "
            f"completely empty columns."
        )

        X = X.drop(
            columns=empty_columns
        )

    # ---------------------------------------------------------
    # Replace infinite values
    # ---------------------------------------------------------
    X = X.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    # ---------------------------------------------------------
    # Dataset information
    # ---------------------------------------------------------
    print()
    print(
        f"Features after cleaning: "
        f"{X.shape[1]}"
    )

    print(
        f"Samples: "
        f"{X.shape[0]:,}"
    )

    print()
    print(
        f"Positive samples: "
        f"{int((y == 1).sum()):,}"
    )

    print(
        f"Negative samples: "
        f"{int((y == 0).sum()):,}"
    )

    print()
    print(
        f"Missing values: "
        f"{int(X.isna().sum().sum()):,}"
    )

    return X, y

# ============================================================
# COST FUNCTION
# ============================================================

def calculate_cost(
    y_true,
    predictions
):

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    # Official Scania challenge costs:
    #
    # False Positive = unnecessary workshop check = 10
    # False Negative = missed faulty truck = 500

    false_positive_cost = 10
    false_negative_cost = 500

    total_cost = (
        fp * false_positive_cost
        + fn * false_negative_cost
    )

    return (
        int(total_cost),
        int(fp),
        int(fn),
        int(tp),
        int(tn)
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.50
    best_cost = float("inf")

    threshold_results = []

    # Because missing a real failure is much more expensive,
    # evaluate a wider range of thresholds.

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        cost, fp, fn, tp, tn = (
            calculate_cost(
                y_true,
                predictions
            )
        )

        threshold_results.append(
            {
                "threshold": round(
                    float(threshold),
                    2
                ),
                "cost": cost,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "true_negatives": tn,
            }
        )

        if cost < best_cost:

            best_cost = cost
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_cost,
        threshold_results
    )


# ============================================================
# MODEL CREATION
# ============================================================

def build_models(
    scale_pos_weight
):

    models = {}

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    models["Logistic Regression"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear"
                )
            ),
        ]
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    models["Random Forest"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1
                )
            ),
        ]
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    models["XGBoost"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "classifier",
                XGBClassifier(
                    n_estimators=350,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist"
                )
            ),
        ]
    )

    return models


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    download_dataset()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    X, y = clean_data(df)

    # --------------------------------------------------------
    # SAVE CLEAN DATA
    # --------------------------------------------------------

    clean_df = X.copy()

    clean_df["failure"] = y.values

    clean_path = (
        DATA_DIR /
        "scania_clean.csv"
    )

    clean_df.to_csv(
        clean_path,
        index=False
    )

    print()
    print(
        f"Clean dataset saved:\n{clean_path}"
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION / TEST
    # --------------------------------------------------------

    X_train_full, X_test, y_train_full, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=42,
            stratify=y
        )
    )

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.1765,
            random_state=42,
            stratify=y_train_full
        )
    )

    print()
    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_validation)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    # --------------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------------

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        negative_count /
        max(
            positive_count,
            1
        )
    )

    print()
    print(
        "XGBoost scale_pos_weight:",
        round(
            scale_pos_weight,
            2
        )
    )

    # --------------------------------------------------------
    # BUILD MODELS
    # --------------------------------------------------------

    models = build_models(
        scale_pos_weight
    )

    results = {}

    best_model_name = None
    best_model = None
    best_validation_cost = float("inf")
    best_threshold = 0.50

    # ========================================================
    # TRAIN EACH MODEL
    # ========================================================

    for name, model in models.items():

        print()
        print("=" * 70)
        print(
            f"TRAINING {name}"
        )
        print("=" * 70)

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # VALIDATION PROBABILITIES
        # ----------------------------------------------------

        validation_probabilities = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        (
            threshold,
            validation_cost,
            threshold_results
        ) = find_best_threshold(
            y_validation,
            validation_probabilities
        )

        validation_predictions = (
            validation_probabilities
            >= threshold
        ).astype(int)

        # ----------------------------------------------------
        # VALIDATION METRICS
        # ----------------------------------------------------

        validation_accuracy = accuracy_score(
            y_validation,
            validation_predictions
        )

        validation_precision = precision_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )

        validation_recall = recall_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )

        validation_f1 = f1_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )

        validation_roc_auc = roc_auc_score(
            y_validation,
            validation_probabilities
        )

        validation_pr_auc = average_precision_score(
            y_validation,
            validation_probabilities
        )

        # ----------------------------------------------------
        # TEST PROBABILITIES
        # ----------------------------------------------------

        test_probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        test_predictions = (
            test_probabilities
            >= threshold
        ).astype(int)

        # ----------------------------------------------------
        # TEST METRICS
        # ----------------------------------------------------

        test_accuracy = accuracy_score(
            y_test,
            test_predictions
        )

        test_precision = precision_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        test_recall = recall_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        test_f1 = f1_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        test_roc_auc = roc_auc_score(
            y_test,
            test_probabilities
        )

        test_pr_auc = average_precision_score(
            y_test,
            test_probabilities
        )

        (
            test_cost,
            fp,
            fn,
            tp,
            tn
        ) = calculate_cost(
            y_test,
            test_predictions
        )

        matrix = confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1]
        )

        report = classification_report(
            y_test,
            test_predictions,
            target_names=[
                "Normal",
                "Failure"
            ],
            zero_division=0
        )

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        results[name] = {

            "threshold":
                round(
                    float(threshold),
                    4
                ),

            "validation_cost":
                int(validation_cost),

            "validation_accuracy":
                round(
                    float(validation_accuracy),
                    4
                ),

            "validation_precision":
                round(
                    float(validation_precision),
                    4
                ),

            "validation_recall":
                round(
                    float(validation_recall),
                    4
                ),

            "validation_f1":
                round(
                    float(validation_f1),
                    4
                ),

            "validation_roc_auc":
                round(
                    float(validation_roc_auc),
                    4
                ),

            "validation_pr_auc":
                round(
                    float(validation_pr_auc),
                    4
                ),

            "test_cost":
                int(test_cost),

            "test_accuracy":
                round(
                    float(test_accuracy),
                    4
                ),

            "test_precision":
                round(
                    float(test_precision),
                    4
                ),

            "test_recall":
                round(
                    float(test_recall),
                    4
                ),

            "test_f1":
                round(
                    float(test_f1),
                    4
                ),

            "test_roc_auc":
                round(
                    float(test_roc_auc),
                    4
                ),

            "test_pr_auc":
                round(
                    float(test_pr_auc),
                    4
                ),

            "false_positives":
                int(fp),

            "false_negatives":
                int(fn),

            "true_positives":
                int(tp),

            "true_negatives":
                int(tn),

            "confusion_matrix":
                matrix.tolist(),

            "classification_report":
                report,
        }

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print()

        print(
            f"Threshold : {threshold:.2f}"
        )

        print(
            f"Accuracy  : {test_accuracy:.4f}"
        )

        print(
            f"Precision : {test_precision:.4f}"
        )

        print(
            f"Recall    : {test_recall:.4f}"
        )

        print(
            f"F1        : {test_f1:.4f}"
        )

        print(
            f"ROC-AUC   : {test_roc_auc:.4f}"
        )

        print(
            f"PR-AUC    : {test_pr_auc:.4f}"
        )

        print(
            f"Test cost : {test_cost}"
        )

        print()
        print(
            "Confusion Matrix:"
        )

        print(matrix)

        # ----------------------------------------------------
        # SELECT MODEL BY VALIDATION COST
        # ----------------------------------------------------

        if validation_cost < best_validation_cost:

            best_validation_cost = (
                validation_cost
            )

            best_model_name = name
            best_model = model
            best_threshold = threshold

    # ========================================================
    # REFIT BEST MODEL ON TRAIN + VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("REFITTING BEST MODEL")
    print("=" * 70)

    X_final_train = pd.concat(
        [
            X_train,
            X_validation
        ],
        axis=0
    )

    y_final_train = pd.concat(
        [
            y_train,
            y_validation
        ],
        axis=0
    )

    best_model.fit(
        X_final_train,
        y_final_train
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    model_path = (
        MODEL_DIR /
        "scania_failure_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    threshold_path = (
        MODEL_DIR /
        "scania_failure_threshold.joblib"
    )

    joblib.dump(
        best_threshold,
        threshold_path
    )

    feature_path = (
        MODEL_DIR /
        "scania_feature_names.joblib"
    )

    joblib.dump(
        list(X.columns),
        feature_path
    )

    # ========================================================
    # XGBOOST FEATURE IMPORTANCE
    # ========================================================

    xgb_model = models["XGBoost"]

    xgb_classifier = (
        xgb_model
        .named_steps["classifier"]
    )

    feature_importances = (
        xgb_classifier
        .feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": feature_importances
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_path = (
        MODEL_DIR /
        "scania_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison_rows = []

    for name, metrics in results.items():

        comparison_rows.append(
            {
                "model": name,

                "threshold":
                    metrics["threshold"],

                "test_cost":
                    metrics["test_cost"],

                "accuracy":
                    metrics["test_accuracy"],

                "precision":
                    metrics["test_precision"],

                "recall":
                    metrics["test_recall"],

                "f1":
                    metrics["test_f1"],

                "roc_auc":
                    metrics["test_roc_auc"],

                "pr_auc":
                    metrics["test_pr_auc"],
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        MODEL_DIR /
        "scania_model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    # ========================================================
    # SAVE JSON RESULTS
    # ========================================================

    results_payload = {

        "dataset":
            "APS Failure at Scania Trucks",

        "source":
            "UCI Machine Learning Repository",

        "uci_dataset_id":
            421,

        "samples":
            int(len(X)),

        "features":
            int(X.shape[1]),

        "normal_samples":
            int((y == 0).sum()),

        "failure_samples":
            int((y == 1).sum()),

        "failure_rate":
            round(
                float(y.mean()),
                6
            ),

        "best_model":
            best_model_name,

        "selected_threshold":
            round(
                float(best_threshold),
                4
            ),

        "model_selection_metric":
            "validation_total_cost",

        "false_positive_cost":
            10,

        "false_negative_cost":
            500,

        "models":
            results,
    }

    results_path = (
        MODEL_DIR /
        "scania_model_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results_payload,
            file,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("REAL DATASET TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        "Best model:",
        best_model_name
    )

    print(
        "Selected threshold:",
        round(
            best_threshold,
            4
        )
    )

    print()
    print(
        "Saved model:"
    )

    print(model_path)

    print()
    print(
        "Saved threshold:"
    )

    print(threshold_path)

    print()
    print(
        "Saved feature importance:"
    )

    print(importance_path)

    print()
    print(
        "Saved comparison:"
    )

    print(comparison_path)

    print()
    print(
        "Saved results:"
    )

    print(results_path)

    print()
    print(
        "Your existing 3-class predictive-maintenance "
        "application has NOT been modified."
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()