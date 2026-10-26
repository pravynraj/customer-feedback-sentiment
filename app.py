import io
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Customer Feedback Analytics", page_icon="📊", layout="wide")

st.title("Customer Feedback Sentiment Analysis")
st.caption("NLP • Statistics • Probability • Data Visualization")

SAMPLE_DATA = [
    ("The product is excellent and works perfectly", "positive"),
    ("Very happy with the quality and quick delivery", "positive"),
    ("Customer support was helpful and friendly", "positive"),
    ("Easy to use and worth the price", "positive"),
    ("The experience was wonderful", "positive"),
    ("Great value and reliable performance", "positive"),
    ("I love this product and recommend it", "positive"),
    ("Delivery was fast and packaging was good", "positive"),
    ("The app is smooth and convenient", "positive"),
    ("The team resolved my issue quickly", "positive"),
    ("The product is terrible and stopped working", "negative"),
    ("Very disappointed with the poor quality", "negative"),
    ("Customer support was rude and unhelpful", "negative"),
    ("Delivery was late and the package was damaged", "negative"),
    ("Not worth the money", "negative"),
    ("The app crashes constantly and is frustrating", "negative"),
    ("I received the wrong item", "negative"),
    ("The service was slow and unreliable", "negative"),
    ("I want a refund because this does not work", "negative"),
    ("Poor experience and no response from support", "negative"),
    ("The product is okay, nothing special", "neutral"),
    ("It arrived yesterday and I have tried it once", "neutral"),
    ("The service has standard features", "neutral"),
    ("The package contained the item and a manual", "neutral"),
    ("It works as described", "neutral"),
    ("The price is similar to other products", "neutral"),
    ("I have mixed feelings about this purchase", "neutral"),
    ("The update changed the layout of the app", "neutral"),
    ("The order status says it is being processed", "neutral"),
    ("The product is available in two colours", "neutral"),
]

@st.cache_resource
def build_model():
    train_df = pd.DataFrame(SAMPLE_DATA, columns=["text", "sentiment"])
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(train_df["text"], train_df["sentiment"])
    return model, train_df

model, training_data = build_model()

with st.sidebar:
    st.header("Project controls")
    st.write("Upload a CSV containing a text column. An optional sentiment column can be used for evaluation.")
    uploaded = st.file_uploader("Upload customer feedback CSV", type=["csv"])
    st.markdown("**Expected columns**")
    st.code("text\nsentiment  # optional")
    st.markdown("Or select the built-in demonstration dataset.")

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must contain a column named `text`.")
            st.stop()
        df["text"] = df["text"].fillna("").astype(str)
        df = df[df["text"].str.strip().ne("")].copy()
        if df.empty:
            st.error("No non-empty feedback was found in the uploaded CSV.")
            st.stop()
        data_source = "Uploaded dataset"
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    df = pd.DataFrame(SAMPLE_DATA, columns=["text", "sentiment"])
    data_source = "Built-in demonstration dataset"

df["predicted_sentiment"] = model.predict(df["text"])
probabilities = model.predict_proba(df["text"])
df["confidence"] = probabilities.max(axis=1)

if "sentiment" not in df.columns:
    df["sentiment"] = ""
else:
    df["sentiment"] = df["sentiment"].fillna("").astype(str).str.lower().str.strip()

st.info(f"Data source: {data_source}. The included model is trained on a small demonstration dataset; validate it on representative, labelled data before using it for business decisions.")

total = len(df)
positive_pct = (df["predicted_sentiment"].eq("positive").mean() * 100) if total else 0
negative_pct = (df["predicted_sentiment"].eq("negative").mean() * 100) if total else 0
avg_conf = df["confidence"].mean() * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Feedback records", f"{total:,}")
c2.metric("Positive feedback", f"{positive_pct:.1f}%")
c3.metric("Negative feedback", f"{negative_pct:.1f}%")
c4.metric("Average model confidence", f"{avg_conf:.1f}%")

left, right = st.columns(2)
with left:
    st.subheader("Sentiment distribution")
    counts = df["predicted_sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("Number of reviews")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Sentiment share")
    fig, ax = plt.subplots()
    counts.plot(kind="pie", autopct="%1.1f%%", ax=ax, ylabel="")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Statistical summary")
summary = pd.DataFrame({
    "Metric": ["Total records", "Positive count", "Neutral count", "Negative count", "Mean confidence", "Median confidence"],
    "Value": [
        total,
        int(counts.get("positive", 0)),
        int(counts.get("neutral", 0)),
        int(counts.get("negative", 0)),
        f"{df['confidence'].mean():.3f}" if total else "N/A",
        f"{df['confidence'].median():.3f}" if total else "N/A",
    ]
})
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Explore feedback")
sentiment_filter = st.multiselect(
    "Filter predicted sentiment",
    options=["positive", "neutral", "negative"],
    default=["positive", "neutral", "negative"]
)
search = st.text_input("Search feedback text")
view = df[df["predicted_sentiment"].isin(sentiment_filter)]
if search.strip():
    view = view[view["text"].str.contains(re.escape(search.strip()), case=False, na=False)]
st.dataframe(view[["text", "predicted_sentiment", "confidence"] + (["sentiment"] if "sentiment" in view.columns and view["sentiment"].ne("").any() else [])],
             use_container_width=True, hide_index=True)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download analyzed results (CSV)", data=csv_bytes, file_name="sentiment_analysis_results.csv", mime="text/csv")

if "sentiment" in df.columns:
    labelled = df[df["sentiment"].isin(["positive", "neutral", "negative"])]
    if len(labelled) > 0:
        st.subheader("Evaluation on labelled rows")
        accuracy = accuracy_score(labelled["sentiment"], labelled["predicted_sentiment"])
        st.metric("Accuracy on labelled rows", f"{accuracy:.1%}")
        st.caption("This is a simple holdout-style check only when the uploaded labels are independent of the demonstration training data. For a valid evaluation, use a representative labelled test set that was not used for training.")
        with st.expander("Show classification report"):
            st.text(classification_report(labelled["sentiment"], labelled["predicted_sentiment"], zero_division=0))

st.subheader("How this project uses statistics and probability")
st.markdown("""
- **Descriptive statistics:** counts, percentages, mean and median model confidence.
- **Probability:** the classifier estimates class probabilities; the highest probability determines the displayed prediction.
- **NLP:** TF-IDF converts text into numeric features and a logistic-regression model classifies sentiment.
- **Analytics:** charts and filters help explore feedback patterns and possible areas for follow-up.
""")
