# SMS Spam Classifier

A machine learning-based web application that classifies SMS messages as spam or not spam (ham).

## Features
- Text preprocessing using NLTK
- TF-IDF vectorization
- Machine learning classification
- Interactive web interface using Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run src/app.py
```

2. Open your web browser and go to http://localhost:8501

## Model Training

The model was trained on the SMS Spam Collection Dataset using various machine learning algorithms including Naive Bayes, SVM, and Random Forest.

## Directory Structure
```
sms-spam-classifier/
├── src/
│   ├── app.py
│   └── model/
│       ├── model.pkl
│       └── vectorizer.pkl
├── requirements.txt
└── README.md
```

## License
MIT License