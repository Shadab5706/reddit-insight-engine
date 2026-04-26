import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Page config
st.set_page_config(page_title="Reddit Insight Engine", layout="centered")

# Premium CSS
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: white;
}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;'>🔍 Reddit Insight Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered Reddit opinion analysis</p>", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("data.csv")

# Input
keyword = st.text_input("🔎 Enter product/topic")

if keyword:
    filtered_df = df[df["text"].str.lower().str.contains(keyword.lower())]

    if len(filtered_df) == 0:
        st.warning("No matching data found")
    else:
        # Clean text
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z ]', '', text)
            return text

        filtered_df["cleaned"] = filtered_df["text"].apply(clean_text)

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

        # Sentiment counts
        total = len(filtered_df)
        pos = (filtered_df["sentiment"]=="Positive").sum()
        neg = (filtered_df["sentiment"]=="Negative").sum()
        neu = (filtered_df["sentiment"]=="Neutral").sum()

        # Card: Sentiment
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Sentiment Overview")
        st.write(f"🟢 Positive: {pos/total*100:.1f}%")
        st.write(f"🔴 Negative: {neg/total*100:.1f}%")
        st.write(f"⚪ Neutral: {neu/total*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        # Graph
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        filtered_df["sentiment"].value_counts().reindex(["Positive","Negative","Neutral"]).fillna(0).plot(kind="bar", ax=ax)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        # Stopwords
        stopwords = ["the","is","and","has","are","for","very","this","with","was"]

        def get_top_words(text_series):
            words = " ".join(text_series).split()
            words = [w for w in words if w not in stopwords and len(w) > 2]
            return Counter(words).most_common(5)

        # Pros
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("👍 Top Pros")
        pos_words = get_top_words(filtered_df[filtered_df["sentiment"]=="Positive"]["cleaned"])
        for w,c in pos_words:
            st.write(f"✔ {w} ({c})")
        st.markdown("</div>", unsafe_allow_html=True)

        # Cons
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("👎 Top Cons")
        neg_words = get_top_words(filtered_df[filtered_df["sentiment"]=="Negative"]["cleaned"])
        if len(neg_words) == 0:
            st.write("No major negative patterns found")
        else:
            for w,c in neg_words:
                st.write(f"✖ {w} ({c})")
        st.markdown("</div>", unsafe_allow_html=True)

        # Summary
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧠 Final Insight")
        if pos > neg:
            st.success("Users generally have a positive opinion.")
        elif neg > pos:
            st.error("Users report several issues.")
        else:
            st.info("Mixed opinions observed.")
        st.markdown("</div>", unsafe_allow_html=True)
