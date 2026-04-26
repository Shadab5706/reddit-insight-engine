import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

st.set_page_config(page_title="Reddit Insight Engine", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Select a product to analyze")

suggestions = ["Samsung TV", "iPhone", "Gaming Laptop", "Headphones"]
keyword = st.sidebar.selectbox("Choose product", suggestions)

st.sidebar.markdown("---")
st.sidebar.info("Built using NLP + Sentiment Analysis")

# Logo + Title
col1, col2 = st.columns([1, 6])

with col1:
    try:
        st.image("logo.png", width=60)
    except:
        pass

with col2:
    st.title("Reddit Insight Engine")
    st.caption("Turning Reddit discussions into actionable insights")

# Load Data
df = pd.read_csv("data.csv")

if keyword:
    with st.spinner("Analyzing Reddit discussions..."):
        time.sleep(1)

        filtered_df = df[df["text"].str.lower().str.contains(keyword.lower())]

        if len(filtered_df) == 0:
            st.warning("No data found")
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

            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"{pos/total*100:.1f}%")
            col2.metric("Negative", f"{neg/total*100:.1f}%")
            col3.metric("Neutral", f"{neu/total*100:.1f}%")

            # Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Analysis", "👍 Pros & Cons", "🧠 Insights"])

            # Charts
            with tab1:
                colA, colB = st.columns(2)

                with colA:
                    fig, ax = plt.subplots()
                    filtered_df["sentiment"].value_counts().reindex(["Positive","Negative","Neutral"]).fillna(0).plot(kind="bar", ax=ax)
                    st.pyplot(fig)

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

            # Pros & Cons
            with tab2:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("👍 Strengths")
                    for w,c in get_top_words(filtered_df[filtered_df["sentiment"]=="Positive"]["cleaned"]):
                        st.write(f"✔ {w} ({c})")

                with col2:
                    st.subheader("👎 Weaknesses")
                    neg_words = get_top_words(filtered_df[filtered_df["sentiment"]=="Negative"]["cleaned"])
                    if len(neg_words) == 0:
                        st.write("No major issues detected")
                    else:
                        for w,c in neg_words:
                            st.write(f"✖ {w} ({c})")

            # Insights
            with tab3:
                if pos > neg:
                    st.success("Overall perception is positive. Users highlight strong performance and quality.")
                elif neg > pos:
                    st.error("Users report notable issues affecting satisfaction.")
                else:
                    st.info("Opinions are mixed with both positives and concerns.")

                st.markdown("### Key Takeaways")
                st.write("- User-generated discussions reveal real-world performance")
                st.write("- Sentiment analysis helps identify trends quickly")
                st.write("- Useful for customers and product teams")
