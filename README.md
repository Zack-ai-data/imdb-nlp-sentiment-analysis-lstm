# IMDB Sentiment Analysis using NLP and LSTM

This project performs end-to-end sentiment analysis on the IMDB 50K Movie Reviews dataset using both traditional machine learning and deep learning approaches. The goal is to compare classical NLP techniques with sequence-based neural models for binary sentiment classification.

---

## 📌 Project Overview

- Dataset: IMDB 50,000 Movie Reviews (balanced positive & negative)
- Task: Binary sentiment classification
- Approaches compared:
  - TF-IDF + Multinomial Naive Bayes
  - Tokenization + Padding + Bidirectional LSTM (TensorFlow)

---

## 🔍 Key Features

- Text cleaning (HTML removal, punctuation, stopwords, lemmatization)
- Exploratory Data Analysis (EDA)
  - Word clouds
  - Word and sentence length distributions
- Feature extraction using TF-IDF
- Deep learning pipeline using:
  - Tokenizer
  - Padding
  - Embedding layer
  - Bidirectional LSTM
- Model evaluation using accuracy, loss curves, and classification report
- Early stopping to mitigate overfitting

---

## 🧠 Models Implemented

### 1. Traditional ML Model
- **TF-IDF Vectorization**
- **Multinomial Naive Bayes**
- Accuracy: ~85%

### 2. Deep Learning Model
- **Embedding Layer**
- **Bidirectional LSTM**
- **Dropout Regularization**
- Accuracy: ~86%

---

## 📊 Results Summary

| Model | Accuracy |
|------|----------|
| TF-IDF + Naive Bayes | ~85% |
| Bidirectional LSTM | ~86% |

The LSTM model shows slightly better performance by capturing sequential dependencies in text.

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn, WordCloud

---

## 📁 Dataset

The dataset is sourced from Kaggle:
**IMDB Dataset of 50K Movie Reviews**

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/imdb-sentiment-analysis-lstm.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the notebook
```bash
jupyter notebook
```

4. Open `Movie_Reviews_Vectorization_NLP.ipynb` and run all cells

## 📌 Future Improvements

- Hyperparameter tuning  
- Pretrained embeddings (GloVe / Word2Vec)  
- Transformer-based models (BERT)  
- Model deployment via REST API  

---

## 👤 Author

**Zack Chong Zhao Cheng**  
AI / Machine Learning | Data Scientist

🔗 LinkedIn: https://linkedin.com/in/chong-z-38b102131  
💻 GitHub: https://github.com/Zack-ai-data
