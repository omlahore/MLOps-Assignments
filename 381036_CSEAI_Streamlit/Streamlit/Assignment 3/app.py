import streamlit as st
from textblob import TextBlob

st.title("Sentiment Analysis")

user_text = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_text:
        analysis = TextBlob(user_text)
        polarity = analysis.sentiment.polarity

        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Polarity:** {polarity}")
    else:
        st.warning("Please enter some text for analysis!")
