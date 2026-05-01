# Email Spam Detector ML Model

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A **comprehensive machine learning pipeline** for email spam detection using multiple classification algorithms with detailed performance analysis and interactive prediction functionality.

</div>

---


## 🎯 Overview

This project implements an **end-to-end email spam detection system** that classifies emails as either spam or legitimate (ham). The pipeline includes:

- **Data preprocessing** and text cleaning
- **Feature engineering** (TF-IDF vectorization + engineered numeric features)
- **Model training** across 5 different algorithms
- **Performance evaluation** with detailed metrics and visualizations
- **Live prediction function** for classifying new emails

The system automatically selects the best-performing model based on F1-score and provides a user-friendly function for real-time spam detection.

---

## ✨ Key Features

- 🔄 **Automated Data Pipeline**: Complete preprocessing and feature engineering workflow
- 📊 **Multi-Model Comparison**: Trains and evaluates 5 different classification algorithms
- 📈 **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC analysis
- 🎨 **Rich Visualizations**: Confusion matrices, ROC curves, feature importance, word clouds
- 🔍 **Feature Engineering**: Combines TF-IDF vectorization with custom numeric features
- 🚀 **Production-Ready**: Ready-to-use `detect_spam()` function for new email predictions
- 📝 **Detailed Analysis**: EDA with word clouds, correlation heatmaps, and distribution analysis

---

## 📦 Dataset

- **Name**: spam.csv
- **Format**: CSV (latin-1 encoding)
- **Size**: 5,572 emails (after deduplication)
- **Distribution**: 
  - Ham (Legitimate): ~86.6%
  - Spam: ~13.4%
- **Source**: UCI ML Repository (standard SMS/Email Spam dataset)
- **Columns**:
  - `v1`: Category (ham/spam)
  - `v2`: Email/message text

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or Anaconda

### Step 1: Clone the Repository
```bash
git clone https://github.com/SagarBhadra01/ML_Models.git
cd ML_Models/EmailSpamDetector_MLmodel
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas scikit-learn scipy nltk matplotlib seaborn wordcloud tabulate
```

### Step 3: Download NLTK Data
The notebook automatically downloads required NLTK data (stopwords) on first run.

---

## ⚡ Quick Start

### Running the Notebook
```bash
jupyter notebook SpamDetectModel.ipynb
```

### Basic Usage
```python
from SpamDetectModel import detect_spam

# Test with a sample email
email = "Congratulations! You have won $10,000! Click here to claim your prize now!"
result = detect_spam(email)
print(result)
# Output: Spam (confidence: 0.9876)
```

---

## 🧠 Model Architecture

### Data Processing Pipeline
```
Raw CSV
   ↓
Rename columns & remove duplicates
   ↓
Text Cleaning (lowercase, remove punctuation, remove stopwords)
   ↓
Feature Extraction
   ├── TF-IDF Vectorization (5000 features)
   └── Engineered Features:
       - Character count
       - Word count
       - Currency symbol presence
       - Number presence
       - Exclamation mark count
   ↓
Train/Test Split (80/20)
   ↓
Model Training & Evaluation
```

### Models Trained

| # | Model | Type | Best For |
|---|-------|------|----------|
| 1 | **ComplementNB** | Probabilistic | Fast, stable on text data |
| 2 | **Logistic Regression** | Linear | Interpretable, efficient |
| 3 | **LinearSVC (SVM)** | Margin-based | High-dimensional text |
| 4 | **Random Forest** | Ensemble | Feature importance, robustness |
| 5 | **Decision Tree** | Single Tree | Interpretability |

**Selection Criterion**: Best model is chosen by highest **F1-Score on test data** for balanced performance.

---

## 📊 Performance Results

All models are evaluated on:
- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / All positive predictions
- **Recall**: True positives / All actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

### Performance Metrics (Test Set)
*Last Updated: May 1, 2026*

```
Model                 Accuracy  Precision  Recall   F1-Score  ROC-AUC
──────────────────────────────────────────────────────────────────────
ComplementNB          0.9246    0.9200     0.9246   0.9200    0.9082
LogisticRegression    0.9816    0.9816     0.9816   0.9813    0.9911
LinearSVC             0.9478    0.9588     0.9478   0.9426    0.9940
RandomForest          0.9758    0.9774     0.9758   0.9759    0.9907
DecisionTree          0.9700    0.9695     0.9700   0.9693    0.9162
```

**Best Model**: Logistic Regression achieves the highest F1-Score (0.9813) on test data with excellent overall performance across all metrics.

---

## 📁 Project Structure

```
EmailSpamDetector_MLmodel/
├── README.md                 # This file
├── SpamDetectModel.ipynb    # Main Jupyter notebook
├── spam.csv                 # Datasetexplanations
├── anaconda_projects/       # Environment configuration (.gitignore)
│   └── db/                  # Database files (.gitignore)
└── .ipynb_checkpoints/      # Notebook checkpoints (.gitignore)
```

---

## 💻 Usage Examples

### Example 1: Detect Spam Email
```python
email1 = "Congratulations! You have won a luxury vacation package worth $10,000! Click here to claim your prize now!"
print(detect_spam(email1))
# Output: Spam (confidence: 0.9876)
```

### Example 2: Legitimate Email
```python
email2 = "Hey, are we still meeting for lunch tomorrow?"
print(detect_spam(email2))
# Output: Not Spam (confidence: 0.9654)
```

### Example 3: Suspicious Email
```python
email3 = "URGENT: Your account has been compromised. Send your password immediately to secure it."
print(detect_spam(email3))
# Output: Spam (confidence: 0.8934)
```

### Using Different Models
```python
# Use a different model instead of the best one
result = detect_spam(email_text, model=lr)  # Use Logistic Regression
result = detect_spam(email_text, model=svm)  # Use SVM
```

---

## 📚 Dependencies

### Core Libraries
- **numpy** (1.20+): Numerical computing
- **pandas** (1.2+): Data manipulation and analysis
- **scikit-learn** (0.24+): Machine learning algorithms and metrics
- **scipy** (1.6+): Scientific computing, sparse matrices

### NLP & Text Processing
- **nltk** (3.6+): Natural language processing, stopwords
- **re, string**: Python standard library text utilities

### Visualization
- **matplotlib** (3.3+): Plotting and visualization
- **seaborn** (0.11+): Statistical data visualization
- **wordcloud** (1.8+): Word cloud generation

### Utilities
- **tabulate** (0.8+): ASCII table formatting

### Install All at Once
```bash
pip install numpy pandas scikit-learn scipy nltk matplotlib seaborn wordcloud tabulate
```

---

## 🔬 Key Sections of the Notebook

### 1. **Import Libraries** (Cell 1)
All required packages for data processing, ML, and visualization

### 2. **Dataset Loading & Preprocessing** (Cells 2-7)
- Load and rename columns
- Remove duplicates
- Text cleaning: lowercase, remove punctuation, remove stopwords
- Feature engineering

### 3. **EDA & Visualization** (Cells 8-11)
- Distribution of spam vs ham
- Word clouds for both categories
- Statistics by category
- Correlation heatmap

### 4. **Feature Extraction & Train/Test Split** (Cells 12-13)
- TF-IDF vectorization
- Combine with engineered features
- 80/20 train/test split

### 5. **Model Training & Evaluation** (Cells 14-22)
- Define comprehensive `evaluate_model()` function
- Train and evaluate 5 models
- Generate detailed metrics and plots

### 6. **Model Comparison & Selection** (Cells 23-24)
- Compare all models across metrics
- Visualize performance
- Select best model by F1-Score

### 7. **Spam Detection Function** (Cells 25-26)
- Production-ready `detect_spam()` function
- Test with sample emails

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Complete ML pipeline from data to production
- ✅ Text preprocessing and NLP techniques
- ✅ Feature engineering and vectorization
- ✅ Multiple classification algorithms
- ✅ Model evaluation and comparison
- ✅ Visualization best practices
- ✅ Python best practices and code organization

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request


---

## 👤 Author

**Sagar Bhadra**
- GitHub: [@SagarBhadra01](https://github.com/SagarBhadra01)
- LinkedIn: [Sagar Bhadra](https://www.linkedin.com/in/sagarbhadra01)
- Email: sagarbhadra404@gmail.com
- Instagram: [@sagarbhadra01](https://www.instagram.com/sagarbhadra01)
- Repository: [ML_Models](https://github.com/SagarBhadra01/ML_Models)

---

## 📖 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [Spam Detection in NLP](https://en.wikipedia.org/wiki/Email_filtering)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)


---

<div align="center">

**If you found this project helpful, please consider giving it a ⭐ star on GitHub!**

</div>
