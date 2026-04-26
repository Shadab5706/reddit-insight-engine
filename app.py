import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Reddit Insight Engine", layout="wide")

# Premium Styling
st.markdown("""
<style>
body {background-color: #0E1117;}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔍 Reddit Insight Engine")
st.caption("AI-powered Reddit opinion intelligence system")

# Load Data
df = pd.read_csv("data.csv")

# Suggestions
suggestions = ["Samsung TV", "iPhone", "Gaming Laptop", "Headphones"]
keyword = st.selectbox("🔎 Select or type product", suggestions)

if keyword:
    filtered_df = df[df["text"].str.lower().str.contains(keyword.lower())]

    if len(filtered_df) == 0:
        st.warning("No data found")
    else:
        # Clean
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

        # TOP METRICS
        col1, col2, col3 = st.columns(3)

        col1.metric("Positive", f"{pos/total*100:.1f}%")
        col2.metric("Negative", f"{neg/total*100:.1f}%")
        col3.metric("Neutral", f"{neu/total*100:.1f}%")

        # TABS
        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "👍 Pros & Cons", "🧠 Insights"])

        # TAB 1 — VISUALS
        with tab1:
            st.subheader("Sentiment Distribution")

            colA, colB = st.columns(2)

            # Bar Chart
            with colA:
                fig, ax = plt.subplots()
                filtered_df["sentiment"].value_counts().reindex(["Positive","Negative","Neutral"]).fillna(0).plot(kind="bar", ax=ax)
                st.pyplot(fig)

            # Pie Chart
            with colB:
                fig2, ax2 = plt.subplots()
                filtered_df["sentiment"].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax2)
                ax2.set_ylabel("")
                st.pyplot(fig2)

        # STOPWORDS
        stopwords = ["the","is","and","has","are","for","very","this","with","was"]

        def get_top_words(text_series):
            words = " ".join(text_series).split()
            words = [w for w in words if w not in stopwords and len(w) > 2]
            return Counter(words).most_common(5)

        # TAB 2 — PROS CONS
        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("👍 Top Pros")
                for w,c in get_top_words(filtered_df[filtered_df["sentiment"]=="Positive"]["cleaned"]):
                    st.write(f"✔ {w} ({c})")

            with col2:
                st.subheader("👎 Top Cons")
                neg_words = get_top_words(filtered_df[filtered_df["sentiment"]=="Negative"]["cleaned"])
                if len(neg_words) == 0:
                    st.write("No major issues found")
                else:
                    for w,c in neg_words:
                        st.write(f"✖ {w} ({c})")

        # TAB 3 — INSIGHT
        with tab3:
            st.subheader("Final Insight")

            if pos > neg:
                st.success("Users generally have a POSITIVE opinion.")
            elif neg > pos:
                st.error("Users report significant issues.")
            else:
                st.info("Mixed opinions observed.")

            st.write("### Key Observations:")
            st.write("- Product perception is based on user-generated discussions")
            st.write("- Sentiment patterns highlight strengths and weaknesses")
            st.write("- Useful for decision-making and product research")
