import streamlit as st
import pickle
import string
import os
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Set page config
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱", layout="centered")

@st.cache_resource
def load_model_and_vectorizer():
    try:
        # Get the absolute path to the model files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, 'model')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        model_path = os.path.join(model_dir, 'model.pkl')

        # Check if files exist
        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model files not found in {model_dir}")

        # Load the files
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise e

# Load NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    ps = PorterStemmer()
except Exception as e:
    st.error(f"Error downloading NLTK data: {str(e)}")
    st.stop()

# Load model and vectorizer
try:
    model, vectorizer = load_model_and_vectorizer()
except Exception as e:
    st.error("Failed to load the model. Please check if model files exist and are not corrupted.")
    st.stop()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("SMS/Email Spam Classifier")
st.markdown("""
This app classifies whether a message is spam or not spam (ham).
Enter your message below to check!
""")

input_sms = st.text_area("Enter the message:", height=100)

if st.button('Classify'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Preprocess the input
        transformed_sms = transform_text(input_sms)
        st.write("Transformed text:", transformed_sms)  # Debug info
        
        # Vectorize the text
        vector_input = vectorizer.transform([transformed_sms])
        st.write("Vector shape:", vector_input.shape)  # Debug info
        
        # Make prediction
        result = model.predict(vector_input)[0]
        probabilities = model.predict_proba(vector_input)[0]
        st.write("Raw probabilities:", probabilities)  # Debug info
        
        # Show results with probability
        if result == 1:
            st.error("This message is SPAM! 🚨")
            prob = probabilities[1]
            st.write(f"Probability of being spam: {prob:.2%}")
        else:
            st.success("This message is NOT spam (HAM) ✅")
            prob = probabilities[0]
            st.write(f"Probability of being ham: {prob:.2%}")

# Add info about the model
with st.expander("About this classifier"):
    st.write("""
    This SMS Spam Classifier uses:
    - NLTK for text preprocessing
    - TF-IDF vectorization
    - A machine learning model trained on SMS spam dataset
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
