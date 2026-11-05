import streamlit as st
import pandas as pd
import joblib

from feature_extractor import extract_features


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="TextTrace",
    page_icon="🔎",
    layout="wide"
)

st.title("TextTrace")
st.subheader("Human vs AI Text Analysis")

st.write(
    "Analyze writing patterns and compare statistical "
    "and machine learning approaches to text classification."
)


# --------------------------------------------------
# 2. LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    ml_model = joblib.load(
        "models/texttrace_model.pkl"
    )

    statistical_data = joblib.load(
        "models/statistical_model.pkl"
    )

    return (
        ml_model,
        statistical_data["model"],
        statistical_data["feature_columns"],
        statistical_data["class_averages"]
    )


try:
    ml_model, statistical_model, feature_columns, class_averages = (
        load_models()
    )

except FileNotFoundError:
    st.error(
        "Model files not found. Please train both models first."
    )
    st.stop()


# --------------------------------------------------
# 3. TEXT INPUT
# --------------------------------------------------

st.markdown("### Enter your text")

text_input = st.text_area(
    "Paste a piece of text below:",
    height=250,
    placeholder="Paste your text here..."
)

word_count = len(text_input.split())

st.write(f"Word count: **{word_count}**")


# --------------------------------------------------
# 4. ANALYZE BUTTON
# --------------------------------------------------

if st.button("Analyze Text", type="primary"):

    if not text_input.strip():
        st.warning("Please enter some text.")

    elif word_count < 100 or word_count > 500:
        st.warning(
            "Please enter text containing between 100 and 500 words."
        )

    else:

        # ------------------------------------------
        # EXTRACT FEATURES
        # ------------------------------------------

        features = extract_features(text_input)

        X_input = pd.DataFrame(
            [features]
        )[feature_columns]


        # ------------------------------------------
        # STATISTICAL MODEL PREDICTION
        # ------------------------------------------

        statistical_prediction = statistical_model.predict(
            X_input
        )[0]

        statistical_probabilities = (
            statistical_model.predict_proba(X_input)[0]
        )

        statistical_confidence = max(
            statistical_probabilities
        ) * 100


        # ------------------------------------------
        # ML MODEL PREDICTION
        # ------------------------------------------

        ml_prediction = ml_model.predict(
            [text_input]
        )[0]

        ml_probabilities = ml_model.predict_proba(
            [text_input]
        )[0]

        ml_confidence = max(ml_probabilities) * 100


        # ------------------------------------------
        # DISPLAY PREDICTIONS
        # ------------------------------------------

        labels = {
            0: "Human-written",
            1: "AI-generated"
        }

        st.markdown("---")
        st.header("Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Statistical Model")

            st.metric(
                "Prediction",
                labels[int(statistical_prediction)]
            )

            st.metric(
                "Model confidence",
                f"{statistical_confidence:.2f}%"
            )

        with col2:
            st.subheader("TF-IDF + Logistic Regression")

            st.metric(
                "Prediction",
                labels[int(ml_prediction)]
            )

            st.metric(
                "Model confidence",
                f"{ml_confidence:.2f}%"
            )


        # ------------------------------------------
        # FEATURE COMPARISON
        # ------------------------------------------

        st.markdown("---")
        st.header("Writing Style Analysis")

        st.write(
            "The following features describe the writing style "
            "of the submitted text. Compare them with the average "
            "values observed in the HC3 training dataset."
        )

        feature_names = {
            "average_word_length": "Average Word Length",
            "average_sentence_length": "Average Sentence Length",
            "sentence_length_std": "Sentence Length Variation",
            "type_token_ratio": "Type-Token Ratio",
            "hapax_ratio": "Hapax Ratio",
            "stopword_ratio": "Stopword Ratio",
            "punctuation_per_100_words": "Punctuation per 100 Words"
        }

        comparison_data = []

        for feature in feature_columns:

            comparison_data.append({
                "Feature": feature_names[feature],
                "Submitted Text": features[feature],
                "Human Average": class_averages.loc["Human", feature],
                "AI Average": class_averages.loc["AI", feature]
            })

        comparison_df = pd.DataFrame(comparison_data)

        st.dataframe(
            comparison_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Feature Comparison Chart")

        chart_df = comparison_df.set_index("Feature")[
            ["Submitted Text", "Human Average", "AI Average"]
        ]

        st.bar_chart(chart_df)

        st.markdown("---")

        st.caption(
            "TextTrace is an experimental classifier trained on the "
            "HC3 dataset. Predictions and confidence scores are not "
            "proof of authorship. Results may differ on text from "
            "other sources, writing styles, and AI systems."
        )