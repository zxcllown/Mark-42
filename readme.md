# Spam Email Classification Project

## Project Overview
This project develops and evaluates machine learning models for binary classification of emails into spam and non-spam (ham) categories. The core focus is comparing model effectiveness when using different feature types (textual, structural, and combined) and evaluating generalization capabilities across independent datasets.

## Objectives
1. Compare model performance using:
   - Text-only features (TF-IDF)
   - Structural-only features
   - Combined text and structural features
2. Assess model generalization on an independent dataset
3. Investigate the impact of classification threshold tuning

## Datasets
### Primary Dataset (Training)
- **Source**: Enron Spam Dataset (Kaggle)
- **Size**: 33,716 emails
- **Class Distribution**: 50.93% spam, 49.07% ham
- **Structure**:
  - `Subject` - email subject line
  - `Message` - email body content
  - `Spam/Ham` - class label
  - `Date` - sending date

### Independent Dataset (Testing)
- **Source**: Email Dataset (GitHub)
- **Size**: 19,528 emails in `.eml` format
- **Class Distribution**: 44.8% spam (8,752), 55.2% ham (10,776)

## Technical Stack
- **Python** 3.x
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Models**: LogisticRegression, LinearSVC
- **Vectorization**: TF-IDF (5,000 features)

## Data Preprocessing

### Main Stages
1. **Removing uninformative columns** (`Message ID`)
2. **Handling missing values**:
   - Replacing empty fields with markers: `NO_SUBJECT`, `NO_MESSAGE`
   - Creating binary features for empty fields
3. **Combining text fields**: `Subject` + `[BODY]` + `Message`
4. **Text normalization**: lowercase conversion, whitespace reduction
5. **Removing duplicates** at text level
6. **Target variable transformation**: `ham` → 0, `spam` → 1

### Feature Engineering
#### Structural Features
```python
features = [
    'is_empty_subject',    # Empty subject flag
    'is_empty_message',    # Empty message flag
    'subject_len',         # Subject length
    'message_len',         # Message length
    'subject_caps_ratio',  # Uppercase ratio in subject
    'message_caps_ratio',  # Uppercase ratio in message
    'subject_digit_ratio', # Digit ratio in subject
    'message_digit_ratio', # Digit ratio in message
    'subject_special_count', # Special character count in subject
    'message_special_count'  # Special character count in message
]
```

#### Textual Features
- TF-IDF vector representation (5,000 most frequent words)
- English stop words removal

## Exploratory Data Analysis (EDA)

### Key Insights
1. **Class Distribution**: Nearly balanced (51% spam, 49% ham)
2. **Empty Fields**:
   - Emails with empty subjects have 85% probability of being spam
   - Emails with empty bodies have 85% probability of being spam
3. **Text Length**:
   - Most emails are short (<2,000 characters)
   - Spam emails are slightly longer on average
4. **Lexical Differences**:
   - **Spam**: `http`, `online`, `click`, `save`, `money`, `software`
   - **Ham**: `enron`, `subject`, `body`, `thanks`, `attached`

### Correlation Analysis
Structural features show weak linear correlation with the target variable but demonstrate non-linear patterns useful for classification.

## Modeling

### Experiment Architecture
1. **TF-IDF only** (textual features)
2. **Structural features only**
3. **Combined TF-IDF + structural features**

### Models
1. **LogisticRegression**:
   - `max_iter=1000`
   - `class_weight='balanced'` (handles class imbalance)
   - Provides probability estimates
2. **LinearSVC**:
   - `class_weight='balanced'`
   - More robust to outliers due to hinge loss

### Results on Primary Dataset

#### LogisticRegression
| Features | Precision (spam) | Recall (spam) | F1-Score (spam) | Accuracy |
|----------|------------------|---------------|-----------------|----------|
| TF-IDF | 0.98 | 1.00 | 0.99 | 0.99 |
| Structural | 0.64 | 0.67 | 0.65 | 0.66 |
| Combined | 0.98 | 1.00 | 0.99 | 0.99 |

#### LinearSVC (TF-IDF + structural)
- **Precision (spam)**: 0.99
- **Recall (spam)**: 0.99  
- **F1-Score (spam)**: 0.99
- **Accuracy**: 0.99

## Cross-Dataset Validation

### Experiment 1: Train on Enron, Test on Email Dataset
| Model | Features | Precision | Recall | F1-Score | Accuracy |
|--------|----------|-----------|--------|----------|----------|
| LogisticRegression | TF-IDF | 0.44 | 0.75 | 0.56 | 0.46 |
| LinearSVC | TF-IDF | 0.44 | 0.70 | 0.54 | 0.45 |
| LogisticRegression | TF-IDF + structural | 0.46 | 0.76 | 0.57 | 0.48 |
| LinearSVC | TF-IDF + structural | 0.57 | 0.74 | 0.57 | 0.49 |

### Experiment 2: Train on Email Dataset, Test on Enron
| Model | Features | Precision | Recall | F1-Score | Accuracy |
|--------|----------|-----------|--------|----------|----------|
| LogisticRegression | TF-IDF | 0.44 | 0.43 | 0.43 | - |
| LinearSVC | TF-IDF | 0.46 | 0.46 | 0.46 | - |
| LogisticRegression | TF-IDF + structural | 0.45 | 0.43 | 0.44 | - |
| LinearSVC | TF-IDF + structural | 0.45 | 0.43 | 0.44 | - |

## Classification Threshold Tuning (LogisticRegression)

### Methodology
```python
# Get probability estimates
y_proba = model.predict_proba(X_test)[:, 1]

# Analyze precision-recall for different thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
```

### Results
| Threshold | False Positives | Effect |
|-------|-----------------|--------|
| 0.35 | 120 | Maximum recall, many FP |
| 0.50 | 63 | Default option |
| 0.70 | 23 | Minimum FP, reduced recall |

**Recommendation**: Threshold 0.7 for minimizing false positives.

## Project Structure
```
project/
├── data/                    # Datasets
│   ├── enron_spam_data.csv
│   └── email-dataset-main/
├── src/                     # Source code
│   ├── preprocessing.py     # Data preprocessing
│   ├── feature_engineering.py # Feature engineering
│   ├── models.py           # Model implementations
│   ├── evaluation.py       # Model evaluation
│   └── utils.py            # Utility functions
├── notebooks/              # Jupyter notebooks for analysis
├── outputs/                # Results and reports
│   ├── models/             # Saved models
│   ├── reports/            # Metric reports
│   └── visualizations/     # Plots and diagrams
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd spam-classification

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Train and evaluate models
python run_training.py

# Run data analysis
python eda.py
```

## Key Findings

### 1. Feature Effectiveness
- **Structural features alone** are insufficient for quality classification (F1=0.66)
- **TF-IDF alone** shows excellent results (F1=0.99)
- **Feature combination** slightly improves results on primary dataset but significantly increases robustness to domain shift

### 2. Model Comparison
- **LogisticRegression** and **LinearSVC** show similar effectiveness
- **LogisticRegression** provides probability estimates for threshold tuning
- **LinearSVC** is more robust to outliers

### 3. Generalization Capability
- Models show metric degradation on independent dataset
- Domain shift accounts for ~40% F1-score reduction
- **Structural features improve generalization** (F1 increase of 1-4%)

### 4. Practical Recommendations
1. **For production**: Use combined TF-IDF and structural features
2. **For minimizing false positives**: Set classification threshold to 0.7
3. **For maximizing spam detection**: Set threshold to 0.35

## Future Work

### Short-Term Improvements
1. **Feature expansion**:
   - Word embeddings (Word2Vec, FastText)
   - Contextual embeddings (BERT)
   - Syntactic features (POS tags, sentence structure)
2. **Model experiments**:
   - Ensemble methods (Random Forest, XGBoost)
   - Neural networks (LSTM, CNN)
3. **Special case handling**:
   - Image-based emails
   - HTML markup processing
   - Multilingual emails

### Long-Term Directions
1. **Online learning** for adapting to new spam types
2. **Personalized filtering** based on user preferences
3. **Integration with email systems** (IMAP, Exchange)
4. **New threat detection** through anomaly analysis

## Authors
- **Dmitriy Litvinov** - modeling and evaluation
- **Arunur Bassarov** - exploratory data analysis (EDA)
- **Pavel Plekhanov** - data preprocessing and feature engineering

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## References
1. [Enron Spam Dataset](https://www.kaggle.com/datasets/wanderfj/enron-spam) - primary dataset
2. [Email Dataset](https://github.com/...) - independent dataset
3. [Scikit-learn documentation](https://scikit-learn.org/) - machine learning library

---
made for final work by Bassarov Arnur , Plekhanov Pavel , Litvinov Dmitriy