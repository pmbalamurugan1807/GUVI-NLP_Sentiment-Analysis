import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

nltk.download('stopwords')
from nltk.corpus import stopwords
import re

# TITLE 

st.set_page_config(page_title="ChatGPT Reviews Sentiment Dashboard", layout="wide")
st.title("ChatGPT Reviews Sentiment Analysis Dashboard")
st.markdown("### Interactive NLP and EDA with Streamlit")

# LOAD DATA

DATA_PATH = r"c:\Users\BALA\Downloads\chatgpt_style_reviews_dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['review_length'] = df['review'].astype(str).apply(len)
    return df

df = load_data()

# DATA CLEANING & SENTIMENT ANALYSIS

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

df['clean_review'] = df['review'].astype(str).apply(clean_text)

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['clean_review'].apply(get_sentiment)

# SIDEBAR

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Exploratory Data Analysis (EDA)",
        "Sentiment Analysis Insights",
        "Model Training and Evaluation",
        "Try Your Own Review"
    ]
)

# EDA SHOWCASE

if page == "Exploratory Data Analysis (EDA)":
    st.header("Exploratory Data Analysis (EDA)")

    eda_option = st.selectbox(
        "Choose an EDA Visualization:",
        [
            "Distribution of Review Ratings",
            "Helpful Reviews (Votes > 10)",
            "Word Clouds (Positive vs Negative)",
            "Average Rating Over Time",
            "Ratings by Location",
            "Platform Comparison (Web vs Mobile)",
            "Verified vs Non-Verified Ratings",
            "Review Length vs Rating",
            "Keywords in 1-Star Reviews",
            "Average Rating per ChatGPT Version",
        ]
    )

    if eda_option == "Distribution of Review Ratings":
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='rating', palette='coolwarm', ax=ax)
        st.pyplot(fig)

    elif eda_option == "Helpful Reviews (Votes > 10)":
        df['helpful_label'] = np.where(df['helpful_votes'] > 10, 'Helpful', 'Not Helpful')
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='helpful_label', palette='viridis', ax=ax)
        st.pyplot(fig)

    elif eda_option == "Word Clouds (Positive vs Negative)":
        col1, col2 = st.columns(2)
        with col1:
            wc_pos = WordCloud(width=500, height=300, background_color='white').generate(' '.join(df[df['rating'] >= 4]['clean_review']))
            st.image(wc_pos.to_array(), caption="Positive Reviews")
        with col2:
            wc_neg = WordCloud(width=500, height=300, background_color='white').generate(' '.join(df[df['rating'] <= 2]['clean_review']))
            st.image(wc_neg.to_array(), caption="Negative Reviews")

    elif eda_option == "Average Rating Over Time":
        trend = df.groupby(df['date'].dt.to_period('M'))['rating'].mean()
        fig, ax = plt.subplots()
        trend.plot(kind='line', ax=ax)
        ax.set_ylabel("Average Rating")
        st.pyplot(fig)

    elif eda_option == "Ratings by Location":
        top_locs = df['location'].value_counts().head(10).index
        loc_data = df[df['location'].isin(top_locs)]
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(data=loc_data, x='location', y='rating', estimator=np.mean, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    elif eda_option == "Platform Comparison (Web vs Mobile)":
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='platform', y='rating', estimator=np.mean, ax=ax, palette='mako')
        st.pyplot(fig)

    elif eda_option == "Verified vs Non-Verified Ratings":
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='verified_purchase', y='rating', estimator=np.mean, ax=ax, palette='Set2')
        st.pyplot(fig)

    elif eda_option == "Review Length vs Rating":
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='rating', y='review_length', ax=ax, palette='coolwarm')
        st.pyplot(fig)

    elif eda_option == "Keywords in 1-Star Reviews":
        wc_one_star = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['rating'] == 1]['clean_review']))
        st.image(wc_one_star.to_array())

    elif eda_option == "Average Rating per ChatGPT Version":
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='version', y='rating', estimator=np.mean, ax=ax, palette='crest')
        st.pyplot(fig)

# SENTIMENT ANALYSIS QUESTIONS

elif page == "Sentiment Analysis Insights":
    st.header("Sentiment Analysis Insights")

    q_option = st.selectbox(
        "Select a Sentiment Question:",
        [
            "Overall Sentiment Distribution",
            "Sentiment vs Rating",
            "Keywords by Sentiment Class",
            "Sentiment Over Time",
            "Verified vs Non-Verified Sentiment",
            "Review Length vs Sentiment",
            "Sentiment by Location",
            "Sentiment by Platform",
            "Sentiment by Version",
            "Negative Feedback Themes"
        ]
    )

    if q_option == "Overall Sentiment Distribution":
        fig, ax = plt.subplots()
        df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen','lightgrey','salmon'], ax=ax)
        st.pyplot(fig)

    elif q_option == "Sentiment vs Rating":
        cross_tab = pd.crosstab(df['rating'], df['sentiment'], normalize='index') * 100
        st.dataframe(cross_tab.style.background_gradient(cmap='Blues'))

    elif q_option == "Keywords by Sentiment Class":
        col1, col2, col3 = st.columns(3)
        for sentiment, col in zip(['Positive','Neutral','Negative'], [col1, col2, col3]):
            text = ' '.join(df[df['sentiment']==sentiment]['clean_review'])
            wc = WordCloud(width=400, height=300, background_color='white').generate(text)
            col.image(wc.to_array(), caption=f"{sentiment} Reviews")

    elif q_option == "Sentiment Over Time":
        trend = df.groupby(df['date'].dt.to_period('M'))['sentiment'].value_counts().unstack().fillna(0)
        trend.plot(kind='line')
        st.pyplot(plt.gcf())

    elif q_option == "Verified vs Non-Verified Sentiment":
        sns.countplot(data=df, x='verified_purchase', hue='sentiment', palette='Set3')
        st.pyplot(plt.gcf())

    elif q_option == "Review Length vs Sentiment":
        sns.boxplot(data=df, x='sentiment', y='review_length', palette='coolwarm')
        st.pyplot(plt.gcf())

    elif q_option == "Sentiment by Location":
        top_locs = df['location'].value_counts().head(10).index
        sns.countplot(data=df[df['location'].isin(top_locs)], x='location', hue='sentiment')
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())

    elif q_option == "Sentiment by Platform":
        sns.countplot(data=df, x='platform', hue='sentiment', palette='mako')
        st.pyplot(plt.gcf())

    elif q_option == "Sentiment by Version":
        sns.countplot(data=df, x='version', hue='sentiment', palette='crest')
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())

    elif q_option == "Negative Feedback Themes":
        neg_reviews = ' '.join(df[df['sentiment']=='Negative']['clean_review'])
        wc_neg = WordCloud(width=800, height=400, background_color='white').generate(neg_reviews)
        st.image(wc_neg.to_array(), caption="Most Frequent Negative Words")

# MODEL TRAINING

elif page == "Model Training and Evaluation":
    st.header("Train a Basic Sentiment Classifier (TF-IDF + Logistic Regression)")

    # Automatically train when page loads
    with st.spinner("Training model..."):
        X = df['clean_review']
        y = df['sentiment']
        tfidf = TfidfVectorizer(max_features=5000)
        X_vec = tfidf.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted')
        recall = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')

    st.success("Model training completed successfully!")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
        st.metric("Precision", f"{precision:.3f}")
    with col2:
        st.metric("Recall", f"{recall:.3f}")
        st.metric("F1 Score", f"{f1:.3f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# PREDICTION

elif page == "Try Your Own Review":
    st.header("Predict Sentiment of Your Own Review")
    user_input = st.text_area("Enter a ChatGPT review:")
    if st.button("Predict Sentiment"):
        clean_input = clean_text(user_input)
        sentiment = get_sentiment(clean_input)
        st.success(f"Predicted Sentiment: {sentiment}")
