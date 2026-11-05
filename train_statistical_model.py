import os
import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from feature_extractor import extract_features


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/hc3_processed.csv")

df = df.dropna(subset=["text", "label", "question_id"])

df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].str.lower().str.strip()

# Keep only human and AI samples
df = df[df["label"].isin(["human", "ai"])]

# Keep texts between 100 and 500 words
df["word_count"] = df["text"].apply(
    lambda x: len(x.split())
)

df = df[
    (df["word_count"] >= 100) &
    (df["word_count"] <= 500)
]

# Remove duplicate texts
df = df.drop_duplicates(subset=["text"])

# Convert labels
df["label"] = df["label"].map({
    "human": 0,
    "ai": 1
})

df = df.reset_index(drop=True)


# --------------------------------------------------
# 2. TRAIN-TEST SPLIT
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

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))


# --------------------------------------------------
# 3. EXTRACT STATISTICAL FEATURES
# --------------------------------------------------

feature_columns = [
    "average_word_length",
    "average_sentence_length",
    "sentence_length_std",
    "type_token_ratio",
    "hapax_ratio",
    "stopword_ratio",
    "punctuation_per_100_words"
]

print("\nExtracting training features...")

X_train = pd.DataFrame(
    train_df["text"].apply(extract_features).tolist()
)[feature_columns]

y_train = train_df["label"]

print("Extracting testing features...")

X_test = pd.DataFrame(
    test_df["text"].apply(extract_features).tolist()
)[feature_columns]

y_test = test_df["label"]


# --------------------------------------------------
# 4. TRAIN GAUSSIAN NAIVE BAYES
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

print("\nTraining statistical model...")

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. EVALUATE MODEL
# --------------------------------------------------

predictions = model.predict(X_test)

print("\nSTATISTICAL MODEL RESULTS")
print("-------------------------")

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Human", "AI"]
    )
)


# --------------------------------------------------
# 6. CALCULATE AVERAGE FEATURES BY CLASS
# --------------------------------------------------

train_features = X_train.copy()
train_features["label"] = y_train.values

class_averages = train_features.groupby("label").mean()

class_averages.index = [
    "Human" if label == 0 else "AI"
    for label in class_averages.index
]


# --------------------------------------------------
# 7. SAVE MODEL
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "class_averages": class_averages
    },
    "models/statistical_model.pkl"
)

print("\nModel saved to models/statistical_model.pkl")