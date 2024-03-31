import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from PIL import Image
import nltk

# Download NLTK Vader lexicon
nltk.download('vader_lexicon')

# Load the pre-trained sentiment analysis model
sia = SentimentIntensityAnalyzer()

header_image_url = "https://raw.githubusercontent.com/MR-REDDY-19/Sentiment-Analysis/blob/main/twitter-sentiment-analysis1.jpg"
pos_image_url = "https://raw.githubusercontent.com/MR-REDDY-19/Sentiment-Analysis/blob/main/Happy.jpg"
neg_image_url = "https://raw.githubusercontent.com/MR-REDDY-19/Sentiment-Analysis/blob/main/Sad.jpg"
neu_image_url = "https://raw.githubusercontent.com/MR-REDDY-19/Sentiment-Analysis/blob/main/Neutral.jpg"

def get_sentiment_status(score):
    if score <= -0.3:
        return 'Negative'
    elif -0.3 < score < 0.3:
        return 'Neutral'
    else:
        return 'Positive'

# Title 
st.title('Sentiment Analysis Web-App')

# Display header image
header_image = Image.open(header_image_url)
st.image(header_image, use_column_width=True)

# Sidebar for selecting the mode of input
st.sidebar.markdown("<h1 style='text-align: center;'>Choose Input Method:</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border-top: 2px solid white;'>", unsafe_allow_html=True)
option = st.sidebar.radio("", ("Enter Review Manually","Upload CSV"))

# Function to analyze sentiment for manually entered review
def analyze_manual_review():
    manual_review = st.text_area('Enter your review or text 📝')
    if st.button('Analyze Sentiment 🚀'):
        if manual_review:
            start = time.time()
            # Predict sentiment for the manually entered review
            sentiment_score = sia.polarity_scores(manual_review)['compound']
            end = time.time()

            # Determine the sentiment status based on the sentiment score
            sentiment_status = get_sentiment_status(sentiment_score)

            # Display prediction time
            st.write('Time taken for Analysis:', round(end - start, 2), 'seconds')

            # Create two columns for displaying the values
            col1, col2 = st.columns(2)

            # Display the predicted sentiment score in the first column
            col1.metric(label="Sentiment Score", value=sentiment_score)

            # Display the sentiment status in the second column
            col2.metric(label="Sentiment Status", value=sentiment_status)

            # Display corresponding sentiment image
            if sentiment_status == 'Positive':
                st.image(pos_image_url, width=200)
            elif sentiment_status == 'Negative':
                st.image(neg_image_url, width=200)
            else:
                st.image(neu_image_url, width=200)
            
        else:
            st.warning("Please enter a review or text.")

# Function to analyze sentiment for uploaded CSV
def analyze_uploaded_csv():
    uploaded_file = st.file_uploader("Upload CSV file containing reviews 📂", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        st.write("## Original DataFrame")
        st.write(df.head(5))

        # Select column containing reviews
        review_column = st.selectbox("Select column containing reviews 👇", df.columns)

        if st.button('Analyze Sentiment 🚀'):
            start = time.time()
            # Predict sentiment for each review in the selected column
            sentiment_scores = df[review_column].apply(lambda review: sia.polarity_scores(str(review))['compound'])
            end = time.time()

            # Display prediction time
            st.write('Time taken for Analysis:', round(end - start, 2), 'seconds')

            # Display the sentiment status for each review
            df['Sentiment Status'] = sentiment_scores.apply(get_sentiment_status)
            st.write("## DataFrame with Sentiment Status")
            st.write(df.head(20))
            
            # Plot bar plot based on sentiment status
            sentiment_counts = df['Sentiment Status'].value_counts()
            fig, ax = plt.subplots()
            bars = ax.bar(sentiment_counts.index, sentiment_counts.values)
            ax.set_xlabel("Sentiment Status")
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Analysis Results")
            
            # Add labels to the bars
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
            
            st.pyplot(fig)

# Execute corresponding function based on selected option
if option == "Enter Review Manually":
    analyze_manual_review()
elif option == "Upload CSV":
    analyze_uploaded_csv()
