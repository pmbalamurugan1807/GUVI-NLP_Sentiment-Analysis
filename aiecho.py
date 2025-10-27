import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pickle
import os
import re

# Embedding model
from sentence_transformers import SentenceTransformer

nltk.download('stopwords')
from nltk.corpus import stopwords

# ================= TITLE =================
st.set_page_config(page_title="ChatGPT Reviews Sentiment", layout="wide")
st.title("ChatGPT Reviews Sentiment Analysis")
st.markdown("### EDA + Sentiment Insights + Multiple Models")

# ================= LOAD DATA =================
DATA_PATH = r"C:\Users\BALA\Downloads\chatgpt_style_reviews_dataset.csv"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['review','title','helpful_votes','location','platform','version','verified_purchase']:
        if col not in df.columns:
            df[col] = np.nan
    df['review'] = df['review'].fillna('')
    df['title'] = df['title'].fillna('')
    df['helpful_votes'] = df['helpful_votes'].fillna(0)
    df['review_length'] = (df['title'] + ' ' + df['review']).apply(len)
    return df

df = load_data()

# ================= DATA CLEANING =================
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

df['clean_review'] = (df['title'] + ' ' + df['review']).apply(clean_text)

# ================= SENTIMENT BASED ON RATING =================
def rating_to_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating <= 2:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['rating'].apply(rating_to_sentiment)

# ================= SIDEBAR =================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Exploratory Data Analysis (EDA)",
     "Sentiment Analysis Insights",
     "Model Training and Evaluation",
     "Try Your Own Review"]
)

# ================= EDA =================
if page == "Exploratory Data Analysis (EDA)":
    st.header("Exploratory Data Analysis (EDA)")

    eda_option = st.selectbox(
        "Choose an EDA Visualization:",
        [
            "Distribution of Review Ratings",
            "Helpful Reviews (Votes > 10)",
            "Word Clouds (Positive vs Neutral vs Negative)",
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
        plt.close(fig)

    elif eda_option == "Helpful Reviews (Votes > 10)":
        df['helpful_label'] = np.where(df['helpful_votes'] > 10, 'Helpful', 'Not Helpful')
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='helpful_label', palette='viridis', ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    elif eda_option == "Word Clouds (Positive vs Neutral vs Negative)":
        col1, col2, col3 = st.columns(3)
        for sentiment, col in zip(['Positive','Neutral','Negative'], [col1, col2, col3]):
            text = ' '.join(df[df['sentiment']==sentiment]['clean_review'])
            if text.strip():
                wc = WordCloud(width=400, height=300, background_color='white').generate(text)
                col.image(wc.to_array(), caption=f"{sentiment} Reviews")

    elif eda_option == "Average Rating Over Time":
        trend = df.groupby(df['date'].dt.to_period('M'))['rating'].mean()
        fig, ax = plt.subplots()
        trend.plot(kind='line', ax=ax)
        ax.set_ylabel("Average Rating")
        st.pyplot(fig)
        plt.close(fig)

    elif eda_option == "Ratings by Location":
        if not df['location'].dropna().empty:
            top_locs = df['location'].value_counts().head(10).index
            loc_data = df[df['location'].isin(top_locs)]
            fig, ax = plt.subplots(figsize=(10,5))
            sns.barplot(data=loc_data, x='location', y='rating', estimator=np.mean, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
            plt.close(fig)

    elif eda_option == "Platform Comparison (Web vs Mobile)":
        if not df['platform'].dropna().empty:
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='platform', y='rating', estimator=np.mean, ax=ax, palette='mako')
            st.pyplot(fig)
            plt.close(fig)

    elif eda_option == "Verified vs Non-Verified Ratings":
        if not df['verified_purchase'].dropna().empty:
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='verified_purchase', y='rating', estimator=np.mean, ax=ax, palette='Set2')
            st.pyplot(fig)
            plt.close(fig)

    elif eda_option == "Review Length vs Rating":
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='rating', y='review_length', ax=ax, palette='coolwarm')
        st.pyplot(fig)
        plt.close(fig)

    elif eda_option == "Keywords in 1-Star Reviews":
        text_one_star = ' '.join(df[df['rating'] == 1]['clean_review'])
        if text_one_star.strip():
            wc_one_star = WordCloud(width=800, height=400, background_color='white').generate(text_one_star)
            st.image(wc_one_star.to_array())

    elif eda_option == "Average Rating per ChatGPT Version":
        if not df['version'].dropna().empty:
            top_versions = df['version'].value_counts().head(15).index
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(data=df[df['version'].isin(top_versions)], y='version', x='rating', estimator=np.mean, ax=ax, palette='crest')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ================= SENTIMENT INSIGHTS =================
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
            "Negative Feedback Themes"
        ]
    )

    if q_option == "Overall Sentiment Distribution":
        fig, ax = plt.subplots()
        df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen','lightgrey','salmon'], ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    elif q_option == "Sentiment vs Rating":
        cross_tab = pd.crosstab(df['rating'], df['sentiment'], normalize='index') * 100
        st.dataframe(cross_tab.style.background_gradient(cmap='Blues'))

    elif q_option == "Keywords by Sentiment Class":
        col1, col2, col3 = st.columns(3)
        for sentiment, col in zip(['Positive','Neutral','Negative'], [col1, col2, col3]):
            text = ' '.join(df[df['sentiment']==sentiment]['clean_review'])
            if text.strip():
                wc = WordCloud(width=400, height=300, background_color='white').generate(text)
                col.image(wc.to_array(), caption=f"{sentiment} Reviews")

    elif q_option == "Sentiment Over Time":
        trend = df.groupby(df['date'].dt.to_period('M'))['sentiment'].value_counts().unstack().fillna(0)
        trend.plot(kind='line')
        st.pyplot(plt.gcf())
        plt.close()

    elif q_option == "Verified vs Non-Verified Sentiment":
        if not df['verified_purchase'].dropna().empty:
            sns.countplot(data=df, x='verified_purchase', hue='sentiment', palette='Set3')
            st.pyplot(plt.gcf())
            plt.close()

    elif q_option == "Review Length vs Sentiment":
        sns.boxplot(data=df, x='sentiment', y='review_length', palette='coolwarm')
        st.pyplot(plt.gcf())
        plt.close()

    elif q_option == "Sentiment by Location":
        if not df['location'].dropna().empty:
            top_locs = df['location'].value_counts().head(10).index
            sns.countplot(data=df[df['location'].isin(top_locs)], x='location', hue='sentiment')
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
            plt.close()

    elif q_option == "Sentiment by Platform":
        if not df['platform'].dropna().empty:
            sns.countplot(data=df, x='platform', hue='sentiment', palette='mako')
            st.pyplot(plt.gcf())
            plt.close()

    elif q_option == "Negative Feedback Themes":
        neg_reviews = ' '.join(df[df['sentiment']=='Negative']['clean_review'])
        if neg_reviews.strip():
            wc_neg = WordCloud(width=800, height=400, background_color='white').generate(neg_reviews)
            st.image(wc_neg.to_array(), caption="Most Frequent Negative Words")

# ================= MODEL TRAINING =================
elif page == "Model Training and Evaluation":
    st.header("Train Multiple Multiclass Sentiment Classifiers")

    save_path = r"C:\Users\BALA\Downloads"

    # Balance classes
    df_pos = df[df['sentiment']=='Positive']
    df_neu = df[df['sentiment']=='Neutral']
    df_neg = df[df['sentiment']=='Negative']
    max_len = max(len(df_pos), len(df_neu), len(df_neg))
    df_balanced = pd.concat([
        resample(df_pos, replace=True, n_samples=max_len, random_state=42),
        resample(df_neu, replace=True, n_samples=max_len, random_state=42),
        resample(df_neg, replace=True, n_samples=max_len, random_state=42)
    ])

    X_text = df_balanced['clean_review'].tolist()
    X_numeric = df_balanced[['review_length','helpful_votes']].values
    y = df_balanced['sentiment']

    st.info("Computing embeddings (may take some time)...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    X_emb = embed_model.encode(X_text, show_progress_bar=True)
    X_final = np.hstack([X_emb, X_numeric])
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_final)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42),
        "Naive Bayes": GaussianNB()
    }

    trained_models = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        preds = model.predict(X_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average='weighted'),
            "recall": recall_score(y_test, preds, average='weighted'),
            "f1": f1_score(y_test, preds, average='weighted'),
            "conf_matrix": confusion_matrix(y_test, preds)
        }

        # Save each model
        with open(os.path.join(save_path, f"{name.replace(' ','_').lower()}.pkl"), 'wb') as f:
            pickle.dump(model, f)

    # Save scaler
    with open(os.path.join(save_path, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    st.success("All models trained and saved!")

    # Display metrics
    for name in models.keys():
        st.subheader(name)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metrics[name]['accuracy']:.3f}")
            st.metric("Precision", f"{metrics[name]['precision']:.3f}")
        with col2:
            st.metric("Recall", f"{metrics[name]['recall']:.3f}")
            st.metric("F1 Score", f"{metrics[name]['f1']:.3f}")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(metrics[name]['conf_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

# ================= PREDICTION =================
elif page == "Try Your Own Review":
    st.header("Predict Sentiment of Your Own Review")

    user_input = st.text_area("Enter a ChatGPT review:")

    save_path = r"C:\Users\BALA\Downloads"

    # Model selection
    model_choice = st.selectbox(
        "Select Model for Prediction:",
        ["Logistic Regression", "Random Forest", "Naive Bayes"]
    )

    model_path = os.path.join(save_path, f"{model_choice.replace(' ','_').lower()}.pkl")
    scaler_file = os.path.join(save_path, "scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_file) and st.button("Predict Sentiment") and user_input.strip():
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)

        clean_input = clean_text(user_input)
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        emb = embed_model.encode([clean_input])
        X_input = np.hstack([emb, [[len(user_input),1]]])
        X_input = scaler.transform(X_input)

        try:
            proba = model.predict_proba(X_input)[0]
            for label, p in zip(model.classes_, proba):
                st.write(f"{label}: {p:.2f}")
        except:
            pass

        prediction = model.predict(X_input)[0]
        st.success(f"Predicted Sentiment ({model_choice}): {prediction}")








