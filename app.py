import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

st.title("🛒 Amazon Reviews Sentiment Dashboard")

st.write("Analyze customer reviews easily with graphs and word clouds!")

# --- Single Review Input ---
review_text = st.text_area("Enter a review:")

if st.button("Analyze Review"):
    if review_text.strip():
        blob = TextBlob(review_text)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            sentiment = "😊 Positive"
        elif polarity < 0:
            sentiment = "😡 Negative"
        else:
            sentiment = "😐 Neutral"

        st.success(f"Sentiment: {sentiment}")
        st.write(f"Polarity Score: {polarity:.2f}")

# --- Multiple Reviews Upload ---
uploaded_file = st.file_uploader("Upload a CSV file of reviews", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    sentiments = []
    for review in data['review_text']:
        blob = TextBlob(str(review))
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiments.append("Positive")
        elif polarity < 0:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")

    data['sentiment'] = sentiments
    st.write("### Sentiment Results")
    st.write(data.head())

    # Graph: Sentiment distribution
    st.write("### Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='sentiment', data=data, palette="Set2", ax=ax)
    st.pyplot(fig)

    # Word Cloud
    st.write("### Word Cloud of Reviews")
    text = " ".join(str(review) for review in data['review_text'])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
