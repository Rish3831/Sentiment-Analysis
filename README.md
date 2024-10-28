# Sentiment Analysis of IMDB Movie Reviews

This project utilizes natural language processing (NLP) to perform sentiment analysis on IMDB movie reviews, aiming to classify each review as either positive or negative. The project focuses on data preprocessing, tokenization, feature extraction, and evaluation using various machine learning models.

## Project Structure

- **Data Preprocessing**: Includes data cleaning, lowercasing, emoji handling, removing special characters and stopwords, and text lemmatization.
- **Exploratory Data Analysis**: Visualizations showing the distribution of positive and negative reviews and word clouds for sentiment-based keywords.
- **Machine Learning Models**: Trained and evaluated using Logistic Regression, Support Vector Machine (SVM), and Random Forest, with hyperparameter tuning for optimized performance.
  
## Project Files

- **`data/IMDB.csv`**: Dataset containing 50,000 IMDB movie reviews labeled as either positive or negative.
- **`notebooks/`**: Contains Jupyter notebooks for each phase of the project including preprocessing, model training, and analysis.
- **`README.md`**: Project documentation (this file).

## Data Processing Steps

1. **Cleaning**: Removal of HTML tags, special characters, frequent words, and emojis.
2. **Lowercasing**: Ensures uniformity in text data.
3. **Tokenization**: Splits reviews into individual words (tokens).
4. **Removing Stopwords**: Eliminates common words without semantic significance.
5. **Lemmatization**: Converts words to their base forms.
6. **Sequencing and Padding**: Prepares text data for model input by converting text to sequences and padding to standardize input lengths.

## Exploratory Data Analysis

Visualizations include:
- **Sentiment Distribution**: Histogram and pie chart displaying positive and negative reviews.
- **Word Clouds**: Highlights prominent words in positive and negative reviews.

## Machine Learning Models and Results

1. **Logistic Regression**:
   - Baseline accuracy: ~50%.
   - Tuned accuracy after Grid Search: ~49%.
   
2. **Support Vector Machine (SVM)**:
   - Feature extraction using TF-IDF vectorizer.
   - Achieved accuracy: ~84% (after tuning).
   
3. **Random Forest**:
   - Accuracy: ~54% with default parameters.
   - Optimized accuracy: ~53% after hyperparameter tuning.

## Dependencies

- `pandas`
- `numpy`
- `nltk`
- `tensorflow`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `WordCloud`

Install the dependencies with:
```bash
pip install -r requirements.txt
