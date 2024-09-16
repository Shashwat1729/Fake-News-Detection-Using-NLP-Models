import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Function to train the model (can be run separately for model training)
def train_model():
    # Load dataset
    df = pd.read_csv('data/news.csv')

    # Preprocess dataset
    X = df['text']
    y = df['label']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a pipeline with TF-IDF and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('lr', LogisticRegression(solver='lbfgs', max_iter=1000))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Save the model to disk
    with open('models/fake_news_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

# Function to predict fake news (loads the model and makes predictions)
def predict_fake_news(text):
    # Load pre-trained model
    with open('models/fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Predict if the news is fake or real
    prediction = model.predict([text])[0]
    return "Fake" if prediction == 1 else "Real"

# Uncomment the line below to train the model
# train_model()
