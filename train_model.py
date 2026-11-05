
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# -----------------------------------------
# 1. LOAD EXISTING DATASET
# -----------------------------------------

df = pd.read_csv("data/hc3_processed.csv")

print("Original dataset:", df.shape)
print(df.head())

# -----------------------------------------
# 2. CLEAN DATA
# -----------------------------------------

df = df.dropna(subset=["text", "label", "question_id"])

df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].str.lower().str.strip()

# Keep only valid labels
df = df[df["label"].isin(["human", "ai"])]

# Count words again to ensure accuracy
df["word_count"] = df["text"].apply(
    lambda x: len(x.split())
)

# Keep samples between 100 and 500 words
df = df[
    (df["word_count"] >= 100) &
    (df["word_count"] <= 500)
]

# Convert labels to numbers
df["label"] = df["label"].map({
    "human": 0,
    "ai": 1
})

# Remove duplicate texts
df = df.drop_duplicates(subset=["text"])

df = df.reset_index(drop=True)

print("\nCleaned dataset:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

# -----------------------------------------
# 3. TRAIN-TEST SPLIT
# -----------------------------------------

X = df["text"]
y = df["label"]
groups = df["question_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------------------
# 4. FEATURE EXTRACTION
# -----------------------------------------

word_features = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=20000,
    sublinear_tf=True,
    min_df=2
)

character_features = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=20000,
    sublinear_tf=True,
    min_df=2
)

combined_features = FeatureUnion([
    ("word", word_features),
    ("character", character_features)
])

# -----------------------------------------
# 5. CLASSIFIER
# -----------------------------------------

model = Pipeline([
    ("features", combined_features),

    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# -----------------------------------------
# 6. TRAIN
# -----------------------------------------

print("\nTraining TextTrace...")

model.fit(X_train, y_train)

print("Training complete!")

# -----------------------------------------
# 7. EVALUATE
# -----------------------------------------

predictions = model.predict(X_test)

print("\nAccuracy:",
      accuracy_score(y_test, predictions))

print("Precision:",
      precision_score(y_test, predictions))

print("Recall:",
      recall_score(y_test, predictions))

print("F1 Score:",
      f1_score(y_test, predictions))

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    target_names=["Human", "AI"]
))

# -----------------------------------------
# 8. CONFUSION MATRIX
# -----------------------------------------

os.makedirs("models", exist_ok=True)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Human", "AI"]
)

plt.title("TextTrace Confusion Matrix")
plt.savefig("models/confusion_matrix.png")
plt.show()

# -----------------------------------------
# 9. SAVE MODEL
# -----------------------------------------

joblib.dump(model, "models/texttrace_model.pkl")

print("\nTextTrace model saved successfully!")
