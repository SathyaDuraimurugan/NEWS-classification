from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Global model variable
model = None

def preprocess_text(text):
    """
    Preprocess text by:
    - Converting to lowercase
    - Removing punctuation
    - Removing digits
    - Removing stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    return text

def create_sample_dataset():
    """Create a sample dataset for training if BBC dataset is not available"""
    data = {
        'text': [
            # Tech samples
            "Apple announces new iPhone with revolutionary camera technology and faster processor. The device features AI-powered photography and enhanced battery life.",
            "Google releases advanced machine learning framework for developers. The new tools promise to accelerate AI development and deployment.",
            "Tesla unveils new autonomous driving features powered by artificial intelligence. The technology uses advanced neural networks for navigation.",
            "Microsoft launches cloud computing platform with quantum computing capabilities. The service targets enterprise customers.",
            "Meta introduces virtual reality headset with improved graphics and user experience. The device supports immersive gaming and social applications.",
            
            # Business samples
            "Stock markets rose today as investors welcomed news of strong quarterly earnings from major technology companies. The Dow Jones gained 1.2%.",
            "Amazon reports record revenue growth in latest quarterly earnings. The company's cloud services division showed particularly strong performance.",
            "Federal Reserve announces interest rate decision affecting mortgage and loan rates. Economic analysts predict market volatility.",
            "Major retail chain announces store closures amid changing consumer shopping patterns. The company cites increased online competition.",
            "Cryptocurrency prices surge following institutional investment announcements. Bitcoin reaches new monthly high.",
            
            # Sports samples
            "Manchester United defeated Chelsea 2-1 in yesterday's Premier League match. The winning goal was scored in the 89th minute by Marcus Rashford.",
            "Olympic champion breaks world record in swimming competition. The athlete's performance exceeded previous records by significant margin.",
            "Tennis tournament concludes with upset victory in final match. The unseeded player defeated three-time champion.",
            "Basketball team advances to playoffs after decisive victory. The team's strong defensive performance secured the win.",
            "Football season kicks off with exciting matchups and surprising results. Several underdogs achieved unexpected victories.",
            
            # Politics samples
            "The Prime Minister announced new policies regarding healthcare reform during today's parliamentary session. Opposition parties criticized the proposed changes.",
            "Congressional hearing addresses cybersecurity concerns and data protection measures. Lawmakers questioned technology executives about privacy.",
            "International trade negotiations continue as deadlines approach. Both sides express cautious optimism about reaching agreement.",
            "Local election results show voter turnout increased compared to previous years. Several incumbent candidates face challenges.",
            "Supreme Court ruling affects voting rights legislation across multiple states. Legal experts analyze potential implications.",
            
            # Entertainment samples
            "The latest Marvel movie broke box office records during its opening weekend, earning over $200 million globally. Critics praised the special effects.",
            "Grammy Awards ceremony celebrates outstanding musical achievements. Several surprise wins and memorable performances highlighted the evening.",
            "Streaming service announces new original series featuring popular actors. The show promises to deliver compelling storytelling.",
            "Film festival showcases independent movies from emerging directors. Several productions received standing ovations from audiences.",
            "Music album debuts at number one on charts worldwide. The artist's latest release features collaborations with multiple genres."
        ],
        'category': [
            # Tech categories
            'tech', 'tech', 'tech', 'tech', 'tech',
            # Business categories
            'business', 'business', 'business', 'business', 'business',
            # Sports categories
            'sport', 'sport', 'sport', 'sport', 'sport',
            # Politics categories
            'politics', 'politics', 'politics', 'politics', 'politics',
            # Entertainment categories
            'entertainment', 'entertainment', 'entertainment', 'entertainment', 'entertainment'
        ]
    }
    
    return pd.DataFrame(data)

def train_model():
    """Train the news classification model"""
    print("Training model...")
    
    # Try to load BBC dataset, if not available use sample data
    try:
        df = pd.read_csv('bbc-text.csv', encoding='latin-1')
        print(f"Loaded BBC dataset with {len(df)} samples")
    except FileNotFoundError:
        print("BBC dataset not found. Using sample dataset...")
        df = create_sample_dataset()
        print(f"Created sample dataset with {len(df)} samples")
    
    print(f"Categories: {df['category'].unique()}")
    
    # Preprocess text
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Remove rows with empty text
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Define features and target
    X = df['cleaned_text']
    y = df['category']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline with TF-IDF and Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = pipeline.predict(X_test)
    
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    # Save the model
    joblib.dump(pipeline, 'news_classifier_model.pkl')
    print("Model saved as 'news_classifier_model.pkl'")
    
    return pipeline

def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('news_classifier_model.pkl')
        print("Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("Model not found. Training new model...")
        return train_model()

def predict_category(text, model):
    """Predict the category of input text"""
    if not text.strip():
        return {"error": "Please enter some text to classify."}
    
    # Preprocess the input text
    processed_text = preprocess_text(text)
    
    if not processed_text.strip():
        return {"error": "Text contains no meaningful content after preprocessing."}
    
    # Make prediction
    prediction = model.predict([processed_text])[0]
    
    # Get prediction probabilities
    probabilities = model.predict_proba([processed_text])[0]
    classes = model.classes_
    
    # Create probability dictionary
    prob_dict = dict(zip(classes, probabilities))
    
    return {
        "prediction": prediction,
        "probabilities": prob_dict
    }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """API endpoint for text classification"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({"error": "Please enter some text to classify."}), 400
        
        if len(text) < 50:
            return jsonify({"error": "Please enter at least 50 characters."}), 400
        
        # Make prediction
        result = predict_category(text, model)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    model = load_model()
    
    if model is None:
        print("Failed to load model. Exiting.")
        exit(1)
    
    print("Model loaded successfully!")
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)