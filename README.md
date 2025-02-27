# IMDB Dataset of 50K Movie Reviews

**Author**: SamiUllah568

## About Dataset

IMDB dataset having 50K movie reviews for natural language processing or Text analytics.

## Context

The IMDB dataset is widely used for sentiment analysis, particularly in the field of Natural Language Processing (NLP). It contains 50,000 movie reviews labeled as positive or negative. The dataset offers a substantial amount of data, allowing for more accurate and robust sentiment classification models compared to smaller datasets used in earlier benchmarks. It is typically used for binary classification tasks, where the objective is to classify each review as either positive or negative.

## Aim

The primary aim of this project is to predict the sentiment of movie reviews using classification algorithms.

Specifically, the goal is to:
- Preprocess the text data (such as tokenization, removing stop words, stemming, etc.).
- Train a Bidirectional LSTM model to classify movie reviews as positive or negative.
- Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1-score.

## Importing Required Libraries

```python
# !pip install wordcloud
# !pip install nltk
# !pip install spacy
# nltk.download('stopwords')
import os
import json
import re
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from wordcloud import WordCloud
# NLP (Natural Language Processing) libraries
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

# Deep Learning libraries (for building models)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional

import warnings
warnings.filterwarnings('ignore')
```

## Load DataSet

Load Data directly through using Kaggle API because the dataset is a little huge which takes a long time on Google Colab.

```python
# Load Kaggle API Key
from google.colab import files
uploaded = files.upload()
# Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
!mkdir -p ~/.kaggle  # Create Kaggle directory
!mv kaggle.json ~/.kaggle/  # Copy your Kaggle API key (Upload kaggle.json first)
!chmod 600 ~/.kaggle/kaggle.json # Set file permissions
!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
```

Dataset URL: [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Unzip Dataset

```python
from zipfile import ZipFile
with ZipFile("/content/imdb-dataset-of-50k-movie-reviews.zip", 'r') as zip_ref:
    zip_ref.extractall()
    print("Dataset extracted successfully!")
```

## Data Analysis

### Show Top 5 Rows

```python
movies = pd.read_csv("/content/IMDB Dataset.csv")
movies.head(5)
```

### Check Missing Values

```python
# Missing Values Check
movies.isnull().sum()
```

### Handling Duplicated Values

```python
print("Before Drop duplicated values Size of Data Set -->> ", movies.shape)
print("Duplicated values -->> ", movies.duplicated().sum())
movies.drop_duplicates(inplace=True)
print("Drop Duplicates Successfully")
print("After Drop duplicated values Size of Data Set -->> ", movies.shape)
print("Duplicated Values --->> ", movies.duplicated().sum())
```

### Sentiment Distribution Analysis

```python
# Creating the subplots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Countplot for sentiment
sns.countplot(data=movies, x="sentiment", ax=ax[0])
ax[0].set_title("Sentiment Distribution in Movies")
ax[0].set_xlabel("Sentiment")
ax[0].set_ylabel("Count")
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')

# Pie chart for sentiment value counts
sentiment_counts = movies["sentiment"].value_counts()
ax[1].pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', shadow=True, startangle=90)
ax[1].set_title("Sentiment Proportions")

# Display the plot
plt.tight_layout()
plt.show()
```

## Text Preprocessing

### Steps:
- Removing HTML Tags
- Removing Punctuation
- Converting to Lower Case
- Tokenizing (Spacy)
- Removing Stopwords
- Stemming

```python
# Load default stopwords
stop_words = set(stopwords.words("english"))

# Remove sentiment-related words from the stopword list
sentiment_words = {"good", "bad", "great", "awesome", "terrible", "horrible", "fantastic", "amazing", "awful", "love", "hate"}
stop_words = stop_words - sentiment_words  # Remove these words from stopwords

# Initialize stemmer and stopwords
stemming = PorterStemmer()
# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[.,!?]', '', text)  # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabets and spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.lower().strip()  # Convert to lowercase and strip spaces

    # Process the text with spaCy
    doc = nlp(text)

    # Tokenize, remove stopwords, and perform Stemming
    cleaned_text = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    return ' '.join(cleaned_text)  # Join words back into a string

# Apply function on Review Column
movies["clean_text"] = movies["review"].apply(clean_text)
```

### Displaying Text Before and After Preprocessing

```python
movies.reset_index(inplace=True)
print("============Before Preprocessing============\n")
print(movies["review"][0])
print("\n============After Preprocessing============\n")
print(movies["clean_text"][0])
```

## Visualizing Sentiment-Based Word Clouds

### Generating Word Clouds for Positive and Negative Reviews

```python
# Separate positive and negative reviews
pos_reviews = movies[movies["sentiment"] == 'positive']["clean_text"]
neg_reviews = movies[movies["sentiment"] == 'negative']["clean_text"]

# Combine all text for each sentiment
pos_text = ' '.join(pos_reviews)
neg_text = ' '.join(neg_reviews)

# Generate word clouds
wordcloud_pos = WordCloud(
    height=800, width=800, background_color='white', max_words=200
).generate(pos_text)

wordcloud_neg = WordCloud(
    height=800, width=800, background_color='black', max_words=200
).generate(neg_text)

# Plot both word clouds
plt.figure(figsize=(16, 8))

# Positive Sentiment
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.title("Positive Sentiment Word Cloud", fontsize=16)
plt.axis("off")

# Negative Sentiment
plt.subplot(1, 2, 2)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title("Negative Sentiment Word Cloud", fontsize=16)
plt.axis("off")

plt.tight_layout()
plt.show()
```

## Defining Features and Target for Sentiment Analysis

```python
X = movies["clean_text"]
y = movies["sentiment"].to_numpy()
print(f"Unique Labels: {np.unique(y)}")
```

## Text Preprocessing and Embedding Preparation for Sentiment Analysis

### Text Tokenization and Sequence Preparation for NLP Model

```python
# Initialize the tokenizer with OOV token and vocabulary size limit
tokenizer = Tokenizer(oov_token="<OOV>", num_words=120000)

# Fit tokenizer on the text data
tokenizer.fit_on_texts(X)

# Define vocabulary size
voc_size = len(tokenizer.word_index) + 1
print(f"Vocabulary Size: {voc_size}")

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(X)

print(f"\nMaximum sequence length: {max(len(seq) for seq in sequences)}")
print(f"Minimum sequence length: {min(len(seq) for seq in sequences)}")

# Print the first sequence for reference
print(f"First sequence length: {len(sequences[0])}")
print(f"First sequence: {sequences[0]}")
```

### Padding Sequences for Uniform Input Length

```python
# Define maximum sequence length
max_length = 1307

# Apply padding to sequences
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

# Print the shape of the padded sequences
print(f"Padded Sequence Shape: {padded_sequences.shape}")

# Print the first padded sequence
print(f"First Padded Sequence:\n{padded_sequences[0]}")
```

### Label Encoding and Dataset Splitting for Model Training

```python
# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

# Print shapes
print(f"Training Data Shape: {x_train.shape}")
print(f"Testing Data Shape: {x_test.shape}")
print(f"Training Labels Shape: {y_train.shape}")
print(f"Testing Labels Shape: {y_test.shape}")

# Encode labels
encode = LabelEncoder()
y_train = encode.fit_transform(y_train)
y_test = encode.transform(y_test)
```

### Word2Vec

#### Training a Word2Vec Model for Word Embeddings

```python
sentences = [sentence.split() for sentence in X]
w2v_model = Word2Vec(sentences, window=7, vector_size=200, min_count=1, sg=0)
print(len(w2v_model.wv[0]))

embedding_matrix = np.zeros((voc_size, 200))
for word, i in tokenizer.word_index.items():
    if i < voc_size:
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
```

## Model Training

### Building and Compiling a Bidirectional LSTM Model for Sentiment Analysis

```python
embedding = Embedding(input_dim=voc_size , output_dim=200,weights=[embedding_matrix],input_length=max_length,trainable=True)

# Define your model
model = Sequential()

model.add(embedding)

# LSTM layers with dropout
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.50))

model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.40))

# Dense layers with dropout
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.30))

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.20))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Explicitly build the model
model.build(input_shape=(None, max_length))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=["accuracy"])

# Show the model summary
model.summary()
```

### Model Training with Early Stopping for Optimization

```python
earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   # Monitor validation loss
    min_delta=0.001,          # Minimum change to qualify as an improvement
    patience=5,           # Stop training after 5 epochs without improvement
    verbose=1,
    mode='auto',
    restore_best_weights=True
)

history = model.fit(x_train, y_train, validation_split=0.25, epochs=50, batch_size=25, callbacks=[earlystopping])
```

### Plotting Model Accuracy Over Epochs

```python
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.show()
```

### Plotting Model Loss Over Epochs

```python
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Test"])
plt.show()
```

## Model Evaluation and Performance Metrics

```python
# Get model predictions (probabilities)
pred_probs_train = model.predict(x_train)
pred_probs_test = model.predict(x_test)

# Convert probabilities to class labels (for binary classification)
pred_train = (pred_probs_train > 0.55).astype(int)
pred_test = (pred_probs_test > 0.55).astype(int)

# Evaluate performance on training data
print("Training Accuracy:", accuracy_score(y_train, pred_train))
print("Training Classification Report:\n", classification_report(y_train, pred_train))
print("Training Confusion Matrix:\n", confusion_matrix(y_train, pred_train))

# Evaluate performance on test data
print("Test Accuracy:", accuracy_score(y_test, pred_test))
print("Test Classification Report:\n", classification_report(y_test, pred_test))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, pred_test))
```

## Model Saving

```python
import pickle
from gensim.models import Word2Vec
from tensorflow import keras  # Make sure you're using TensorFlow ≥2.16

# Save label encoder
with open("encode.pkl", 'wb') as file1:
    pickle.dump(encode, file1)

# Save tokenizer
with open("tokenizer.pkl", 'wb') as file2:
    pickle.dump(tokenizer, file2)

# Save Word2Vec model
w2v_model.save("Word2vModel.model")

# Save model
model.save("bilstm_model.keras")
```

## Model Loading and Prediction

```python
# Load encoder
with open("encode.pkl", 'rb') as f:
    encode = pickle.load(f)

# Load tokenizer
with open("tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)

# Load Word2Vec model
w2v_model = Word2Vec.load("Word2vModel.model")

# Load Keras model
model = keras.models.load_model("bilstm_model.keras")

def prediction(review):
    try:
        review = clean_text(review)
        if not review:
            return "Neutral"

        seq = tokenizer.texts_to_sequences([review])
        print(f"Tokenized Sequence: {seq}")  # Debugging

        if not seq or not seq[0]:
            return "Neutral"

        padded = pad_sequences(seq, maxlen=max_length, padding='post')
        # print(f"Padded Sequence: {padded}")  # Debugging

        prob = model.predict(padded)
        print(f"Prediction Probability: {prob}")  # Debugging

        return "Positive" if prob > 0.55 else "Negative"

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Neutral"

text = "This is a very good movie and very interesting movie. I like it."
print(prediction(text))
Prediction:
      'Positive'

text = "Despite the visually stunning cinematography and a few decent performances, the movie ultimately falls flat due to its predictable plot, poorly developed characters, and an over-reliance on clichés, making it a tedious and uninspired experience that fails to leave any lasting impact."
print(prediction(text))
Prediction:
   'Negative'

text = "This is a bad movie and very bad movie. I like it."
print(prediction(text))
Prediction:
   'Negative'
```

