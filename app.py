import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title("🔍 Reddit Insight Engine")
st.write("Analyze opinions from Reddit-style discussions")

data = [
    "Samsung TV has amazing picture quality and great display",
    "The sound quality of this TV is very bad",
    "I love the brightness and colors of Samsung TVs",
    "The interface is slow and laggy",
    "Great value for money and good performance",
    "Not worth the price, very disappointing",
    "Best TV I have used so far",
    "Remote control is confusing and hard to use",
    "Display is crisp and clear",
    "Software updates are terrible"
]

df = pd.DataFrame(data, columns=["text"])

keyword = st.text_input("Enter product/topic")

if keyword:
    filtered_df = df[df["text"].str.lower().str.contains(keyword.lower())]

    if len(filtered_df) == 0:
        st.warning("No matching data found")
    else:
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

        total = len(filtered_df)
        pos = (filtered_df["sentiment"]=="Positive").sum()
        neg = (filtered_df["sentiment"]=="Negative").sum()
        neu = (filtered_df["sentiment"]=="Neutral").sum()

        st.subheader("Sentiment")
        st.write(f"Positive: {pos/total*100:.1f}%")
        st.write(f"Negative: {neg/total*100:.1f}%")
        st.write(f"Neutral: {neu/total*100:.1f}%")

        fig, ax = plt.subplots()
        filtered_df["sentiment"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        def get_top_words(text_series):
            words = " ".join(text_series).split()
            return Counter(words).most_common(5)

        st.subheader("Top Pros")
        for w,c in get_top_words(filtered_df[filtered_df["sentiment"]=="Positive"]["cleaned"]):
            st.write(f"{w} ({c})")

        st.subheader("Top Cons")
        for w,c in get_top_words(filtered_df[filtered_df["sentiment"]=="Negative"]["cleaned"]):
            st.write(f"{w} ({c})")

        if pos > neg:
            st.success("Overall Positive")
        elif neg > pos:
            st.error("Overall Negative")
        else:
            st.info("Mixed Opinions")
