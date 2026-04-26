import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# Page config
st.set_page_config(page_title="Reddit Insight Engine", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Controls")

suggestions = [
    "Samsung TV",
    "iPhone",
    "Gaming Laptop",
    "Headphones",
    "Starbucks",
    "Hostel",
    "Gym",
    "Netflix",
    "Goa",
    "Online classes"
]

keyword = st.sidebar.selectbox("Choose product/category", suggestions)

custom_input = st.sidebar.text_input("Or type your own keyword")

if custom_input:
    keyword = custom_input

st.sidebar.markdown("---")
st.sidebar.info("Built using NLP + Sentiment Analysis")

# Title
st.title("🔍 Reddit Insight Engine")
st.caption("Turning Reddit-style discussions into actionable insights")

# Optional suggestion line
st.info("💡 Try topics like: iPhone, Gym, Hostel, Netflix, Goa")

# Load dataset
df = pd.read_csv("data.csv")

# Run analysis
if keyword:
    with st.spinner("Analyzing discussions..."):
        time.sleep(1)

        filtered_df = df[df["text"].str.lower().str.contains(keyword.lower())]

        if len(filtered_df) == 0:
            st.warning("No data found for this keyword.")
        else:
            # Clean text
            def clean_text(text):
                text = text.lower()
                text = re.sub(r'[^a-zA-Z ]', '', text)
                return text

            filtered_df["cleaned"] = filtered_df["text"].apply(clean_text)

            # Sentiment analysis
            analyzer = SentimentIntensityAnalyzer()

            def get_sentiment(text):
                score = analyzer.polarity_scores(text)["compound"]
                if score > 0.05:
                    return "Positive"
                elif score < -0.05:
                    return "Negative"
                else:
                    return "Neutral"

            filtered_df["sentiment"] = filtered_df["cleaned"].apply(get_sentiment)

            total = len(filtered_df)
            pos = (filtered_df["sentiment"] == "Positive").sum()
            neg = (filtered_df["sentiment"] == "Negative").sum()
            neu = (filtered_df["sentiment"] == "Neutral").sum()

            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"{pos/total*100:.1f}%")
            col2.metric("Negative", f"{neg/total*100:.1f}%")
            col3.metric("Neutral", f"{neu/total*100:.1f}%")

            # Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Analysis", "👍 Pros & Cons", "🧠 Insights"])

            # TAB 1 — Charts
            with tab1:
                colA, colB = st.columns(2)

                # Bar chart
                with colA:
                    fig, ax = plt.subplots()
                    filtered_df["sentiment"].value_counts().reindex(["Positive", "Negative", "Neutral"]).fillna(0).plot(kind="bar", ax=ax)
                    ax.set_title("Sentiment Distribution")
                    st.pyplot(fig)

                # Pie chart
                with colB:
                    fig2, ax2 = plt.subplots()
                    filtered_df["sentiment"].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax2)
                    ax2.set_ylabel("")
                    st.pyplot(fig2)

            # Stopwords
            stopwords = ["the","is","and","has","are","for","very","this","with","was"]

            def get_top_words(text_series):
                words = " ".join(text_series).split()
                words = [w for w in words if w not in stopwords and len(w) > 2]
                return Counter(words).most_common(5)

            # TAB 2 — Pros & Cons
            with tab2:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("👍 Strengths")
                    pos_words = get_top_words(filtered_df[filtered_df["sentiment"] == "Positive"]["cleaned"])
                    for w, c in pos_words:
                        st.write(f"✔ {w} ({c})")

                with col2:
                    st.subheader("👎 Weaknesses")
                    neg_words = get_top_words(filtered_df[filtered_df["sentiment"] == "Negative"]["cleaned"])
                    if len(neg_words) == 0:
                        st.write("No major issues detected")
                    else:
                        for w, c in neg_words:
                            st.write(f"✖ {w} ({c})")

            # TAB 3 — Insights
            with tab3:
                if pos > neg:
                    st.success("Overall perception is positive. Users highlight strong performance and quality.")
                elif neg > pos:
                    st.error("Users report notable issues affecting satisfaction.")
                else:
                    st.info("Mixed opinions observed.")

                st.markdown("### Key Takeaways")
                st.write("- User-generated discussions reflect real-world experiences")
                st.write("- Sentiment analysis helps identify trends quickly")
                st.write("- Useful for customers, researchers, and companies")
