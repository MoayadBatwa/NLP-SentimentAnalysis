# NLP Sentiment Analysis for Mental Health

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MoayadBatwa/NLP-SentimentAnalysis/blob/main/SentimentAnalysis_NLP.ipynb)

A machine learning project that performs sentiment analysis on mental health-related text data using Natural Language Processing (NLP) techniques. This project classifies user statements into different mental health status categories using TF-IDF feature extraction and Logistic Regression.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Workflow](#project-workflow)
- [Model Performance](#model-performance)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project analyzes text statements to identify potential mental health conditions. It uses advanced NLP preprocessing techniques including lemmatization, stopword removal, and TF-IDF vectorization with bigrams to extract meaningful features from text data.

The classifier can categorize statements into 7 different mental health status classes, making it useful for:
- Mental health screening tools
- Social media sentiment monitoring
- Healthcare chatbot systems
- Research on mental health discourse

## ✨ Features

- **Advanced Text Preprocessing**: Removes special characters, stopwords, and applies WordNet lemmatization
- **TF-IDF with Bigrams**: Captures context between adjacent words for better feature representation
- **Balanced Classification**: Handles class imbalance using weighted Logistic Regression
- **Comprehensive Evaluation**: Includes accuracy metrics, classification reports, and confusion matrices
- **Batch Inference**: Custom function to classify multiple new statements at once
- **Visualization**: Distribution plots and confusion matrix for model performance analysis

## 📊 Dataset

The project uses the **Sentiment Analysis for Mental Health** dataset from Kaggle, which contains:
- User statements related to mental health
- Labeled mental health status categories
- Multiple classes representing different mental health conditions

Dataset is automatically downloaded using the KaggleHub API.

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **NLTK** - Natural language processing toolkit
- **Scikit-learn** - Machine learning algorithms and metrics
- **Matplotlib & Seaborn** - Data visualization
- **KaggleHub** - Dataset management

## 📥 Installation

1. Clone the repository:
```bash
git clone https://github.com/MoayadBatwa/NLP-SentimentAnalysis.git
cd NLP-SentimentAnalysis
```

2. Install required packages:
```bash
pip install pandas nltk matplotlib seaborn kagglehub scikit-learn
```

3. Download NLTK resources (handled automatically in the notebook):
```python
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')
```

## 🚀 Usage

### Running the Notebook

1. Open the Jupyter notebook:
```bash
jupyter notebook SentimentAnalysis_NLP.ipynb
```

2. Run all cells sequentially to:
   - Install dependencies
   - Load and preprocess the dataset
   - Train the model
   - Evaluate performance
   - Make predictions on new data

### Using the Model for Predictions

```python
# Example of batch prediction
new_statements = [
    "I feel so overwhelmed and anxious all the time",
    "Having a great day, feeling positive!",
    "Can't sleep, constant worrying keeps me up"
]

# The notebook includes a batch prediction function
# predictions = predict_batch(new_statements)
```

## 🔄 Project Workflow

1. **Setup & Installation**: Install libraries and download NLTK resources
2. **Data Loading**: Fetch dataset using KaggleHub API
3. **Advanced Preprocessing**: 
   - Remove special characters and stopwords
   - Apply WordNet lemmatization
4. **Feature Extraction**: Convert text to TF-IDF vectors with bigrams (1,2 n-grams)
5. **Data Splitting**: 80/20 train-test split with fixed random seed
6. **Model Training**: Logistic Regression with balanced class weights
7. **Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix
8. **Visualization**: Plot distribution and confusion matrix
9. **Inference**: Batch prediction on new statements

## 📈 Model Performance

The model uses:
- **Algorithm**: Logistic Regression with L2 regularization
- **Feature Engineering**: TF-IDF with bigrams (1-2 word combinations)
- **Class Balancing**: Automatic weight adjustment for imbalanced classes
- **Evaluation Metrics**: 
  - Accuracy Score
  - Precision, Recall, F1-Score per class
  - Confusion Matrix

## 📊 Results

The model provides:
- Multi-class classification across 7 mental health categories
- Visual confusion matrix showing model predictions vs actual labels
- Detailed classification report with per-class metrics
- Distribution analysis of mental health status in the dataset

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Moayad Batwa**
- GitHub: [@MoayadBatwa](https://github.com/MoayadBatwa)

## 🙏 Acknowledgments

- Dataset provided by [Suchintika Sarkar](https://www.kaggle.com/suchintikasarkar) on Kaggle
- NLTK library for NLP tools
- Scikit-learn for machine learning algorithms

---

⭐ If you find this project useful, please consider giving it a star!