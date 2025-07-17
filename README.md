# 📰 News Classification using Machine Learning

This project builds a machine learning model to classify news articles into predefined categories such as Politics, Sports, Technology, Business, and more. It uses natural language processing (NLP) techniques combined with supervised learning to automate the classification of text-based news data.

## 🚀 Features

- Text preprocessing (tokenization, stopword removal, stemming)
- Feature extraction using TF-IDF / CountVectorizer
- ML models: Logistic Regression, Naive Bayes, Support Vector Machine
- Evaluation using Accuracy, Precision, Recall, F1-Score
- Streamlit web app (optional)
- Visualizations: Confusion Matrix, Word Clouds, Category Distribution

## 🧠 Technologies Used

Languages & Libraries:
- Python
- Scikit-learn
- NLTK / spaCy
- pandas, NumPy
- matplotlib, seaborn, wordcloud

Optional Web App:
- Flask / Streamlit

Dataset:
- Kaggle News Dataset or custom CSV

## 📁 Project Structure

news-classification/
│
├── data/                 # Dataset files
│   └── news.csv
├── models/               # Trained ML models
│   └── news_model.pkl
├── notebook.ipynb        # Main Jupyter Notebook
├── app.py                # Streamlit/Flask app (optional)
├── utils.py              # Helper functions
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

## ⚙️ Installation & Setup

1. Clone the Repository
   git clone https://github.com/your-username/news-classification.git
   cd news-classification

2. Install Dependencies
   pip install -r requirements.txt

3. Run Jupyter Notebook
   jupyter notebook notebook.ipynb

4. (Optional) Run the Web App
   For Streamlit:
   streamlit run app.py

   For Flask:
   python app.py

## 📊 Sample Model Results

| Model               | Accuracy | F1-Score |
|--------------------|----------|----------|
| Logistic Regression| 89%      | 0.88     |
| Naive Bayes        | 85%      | 0.84     |
| SVM (Linear)       | 91%      | 0.90     |

## 📝 Example Prediction

Input:
"The prime minister addressed the rising inflation and economic reforms in parliament today."

Predicted Category:
Politics

## ✅ To Do

- [x] Text preprocessing
- [x] Model building & evaluation
- [x] Create web UI (Streamlit/Flask)
- [ ] Add deep learning model (LSTM/BERT)
- [ ] Multilingual support

## 🤝 Contributing

Contributions are welcome!  
Steps:
1. Fork the repo  
2. Create a branch (`feature-xyz`)  
3. Commit changes  
4. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- Scikit-learn
- NLTK
- Kaggle
- Streamlit
