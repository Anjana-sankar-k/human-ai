import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from feature_extractor import extract_features


# --------------------------------------------------
# 1. LOAD AND CLEAN DATA
# --------------------------------------------------

df = pd.read_csv("data/hc3_processed.csv")

df = df.dropna(subset=["text", "label", "question_id"])

df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].str.lower().str.strip()

df = df[df["label"].isin(["human", "ai"])]

df["word_count"] = df["text"].apply(
    lambda x: len(x.split())
)

df = df[
    (df["word_count"] >= 100) &
    (df["word_count"] <= 500)
]

df = df.drop_duplicates(subset=["text"])

df["label"] = df["label"].map({
    "human": 0,
    "ai": 1
})

df = df.reset_index(drop=True)


# --------------------------------------------------
# 2. RECREATE THE SAME TEST SPLIT
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df,
        df["label"],
        groups=df["question_id"]
    )
)

test_df = df.iloc[test_idx]

X_test_text = test_df["text"].tolist()
y_test = test_df["label"]


# --------------------------------------------------
# 3. LOAD BOTH MODELS
# --------------------------------------------------

ml_model = joblib.load("models/texttrace_model.pkl")

statistical_data = joblib.load(
    "models/statistical_model.pkl"
)

statistical_model = statistical_data["model"]
feature_columns = statistical_data["feature_columns"]


# --------------------------------------------------
# 4. GET PREDICTIONS
# --------------------------------------------------

print("Predicting with ML model...")

ml_predictions = ml_model.predict(X_test_text)

print("Extracting statistical features...")

X_test_features = pd.DataFrame(
    [extract_features(text) for text in X_test_text]
)[feature_columns]

print("Predicting with statistical model...")

statistical_predictions = statistical_model.predict(
    X_test_features
)


# --------------------------------------------------
# 5. COMPARE RESULTS
# --------------------------------------------------

def get_metrics(y_true, predictions):
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions),
        "Recall": recall_score(y_true, predictions),
        "F1 Score": f1_score(y_true, predictions)
    }


results = pd.DataFrame({
    "Statistical Model": get_metrics(
        y_test, statistical_predictions
    ),

    "TF-IDF + Logistic Regression": get_metrics(
        y_test, ml_predictions
    )
})

print("\nMODEL COMPARISON")
print("----------------")

print(results.round(4))

results.to_csv(
    "models/model_comparison.csv"
)

print("\nResults saved to models/model_comparison.csv")