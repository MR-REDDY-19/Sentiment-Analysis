import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from PIL import Image

header_image = Image.open("twitter-sentiment-analysis1.jpg")
st.image(header_image, use_column_width=True)
  
# Load the pre-trained sentiment analysis model
sia = SentimentIntensityAnalyzer()

def get_sentiment_status(score):
    if score <= -0.3:
        return 'Negative'
    elif -0.3 < score < 0.3:
        return 'Neutral'
    else:
        return 'Positive'

# Title 
st.title('Sentiment Analysis Web-App')

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

            try:
                if sentiment_status == 'Positive':
                    pos = Image.open("Happy.jpg")
                    st.image(pos, width=200)
                elif sentiment_status == 'Negative':
                    sad = Image.open("Sad.jpg")
                    st.image(sad, width=200)
                else:
                    neu = Image.open("Neutral.jpg")
                    st.image(neu, width=200)
            except Exception as e:
                st.error(f"Error occurred while opening image: {e}")
            
        else:
            st.warning("Please enter a review or text.")


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
            plt.bar_label(bars, label_type='center')
            
            st.pyplot(fig)

# Execute corresponding function based on selected option
if option == "Enter Review Manually":
    analyze_manual_review()
elif option == "Upload CSV":
    analyze_uploaded_csv()
