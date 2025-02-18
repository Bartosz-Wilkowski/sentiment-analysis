"""
LSTM Model Training and Evaluation Module

This module provides functions for preprocessing, training, evaluating, and saving an LSTM model for sentiment analysis.

Functions:
    - _preprocess_data(df, max_length=128, vocab_size=10000, stop_words=False):
      Tokenizes and pads text data for model training.

    - _tokenize_with_existing_tokenizer(df, tokenizer, max_length, stop_words=False):
      Tokenizes and pads text using a pre-trained tokenizer.

    - train_lstm_model_and_tokenizer(train_df, val_df, model, learning_rate=0.001, epochs=10, batch_size=32, max_length=128, vocab_size=10000, stop_words=False):
      Trains an LSTM model and returns the trained model, history, and tokenizer.

    - evaluate_model(model, tokenizer, test_df, max_length=128, stop_words=False):
      Evaluates the trained model and prints a classification report.

    - save_classification_report(y_test, y_pred, output_path, name):
      Saves the classification report to a file.

Dependencies:
    - numpy, tensorflow, nltk, sklearn, pandas, os
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os


def _preprocess_data(df, max_length=128, vocab_size=10000, stop_words=False):
    """
    Preprocesses text data for training a LSTM model.

    This function performs the following preprocessing steps:
    - Removes missing or empty values from the 'Review' column.
    - Applies lemmatization to normalize words.
    - Tokenizes the text and converts it into sequences of integers.
    - Pads the sequences to ensure uniform input length.

    Args:
        df (pd.DataFrame): DataFrame containing at least two columns: 'Review' (text data) and 'Polarity' (labels).
        max_length (int, optional): Maximum length of tokenized sequences. Defaults to 128.
        vocab_size (int, optional): Maximum number of words to keep in the tokenizer vocabulary. Defaults to 10000.
        stop_words (bool or set, optional): If True, use NLTK English stopwords. If a set is provided, use it instead.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Padded sequences representing tokenized text data.
            - np.ndarray: Labels corresponding to the text data.
            - Tokenizer: The trained Keras Tokenizer instance for later use.
    """
    # initialize tokenizer and lemmatizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    lemmatizer = WordNetLemmatizer()

    # remove missing or empty values
    df = df.dropna(subset=['Review']).copy()
    df = df[df['Review'].str.strip() != '']

    # extract texts and labels
    texts = df['Review'].astype(str).tolist()
    labels = df['Polarity'].values

    if stop_words:
        if isinstance(stop_words, bool) and stop_words:
            stop_words = set(stopwords.words('english'))
        texts = [" ".join([lemmatizer.lemmatize(word) for word in text.split() if word.lower() not in stop_words]) for text in texts]

        # remove empty entries after stopwords removal
        filtered_data = [(text, label) for text, label in zip(texts, labels) if text.strip()]
        texts, labels = zip(*filtered_data) if filtered_data else ([], [])
    else:
        texts = [" ".join([lemmatizer.lemmatize(word) for word in text.split()]) for text in texts]

    # tokenize and pad sequences
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    return padded_sequences, np.array(labels), tokenizer


def _tokenize_with_existing_tokenizer(df, tokenizer, max_length, stop_words=False):
    """
    Tokenizes and pads text data using an existing tokenizer.

    This function takes a DataFrame containing text data and uses a pre-trained Keras Tokenizer
    to convert the text into numerical sequences. The sequences are then padded to a fixed length
    to ensure uniform input size.

    Args:
        df (pd.DataFrame): DataFrame containing at least two columns:
            - 'Review' (str): The text data to be tokenized.
            - 'Polarity' (int): Corresponding labels for the text data.
        tokenizer (Tokenizer): A pre-trained Keras Tokenizer instance.
        max_length (int): Maximum length of sequences after padding.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Padded sequences representing tokenized text data.
            - np.ndarray: Labels corresponding to the text data.
    """
    texts = df['Review'].astype(str).tolist()
    labels = df['Polarity'].values

    if stop_words:
        if isinstance(stop_words, bool) and stop_words:
            stop_words = set(stopwords.words('english'))
        texts = [" ".join([word for word in text.split() if word.lower() not in stop_words]) for text in texts]

        filtered_data = [(text, label) for text, label in zip(texts, labels) if text.strip()]
        texts, labels = zip(*filtered_data) if filtered_data else ([], [])

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    return padded_sequences, np.array(labels)


def train_lstm_model_and_tokenizer(train_df, val_df, model, learning_rate=0.001, epochs=10, batch_size=32, max_length=128, vocab_size=10000, stop_words=False):
    """
    Trains an LSTM model on a given dataset and returns the trained model, training history, and tokenizer.

    This function performs the following steps:
    - Preprocesses the training and validation datasets.
    - Tokenizes text data and converts it into numerical sequences.
    - Compiles and trains the provided LSTM model.
    - Uses early stopping to prevent overfitting.
    - Evaluates the trained model on the validation dataset and prints a classification report.

    Args:
        train_df (pd.DataFrame): Training dataset containing at least:
            - 'Review' (str): The text data.
            - 'Polarity' (int): The corresponding sentiment labels.
        val_df (pd.DataFrame): Validation dataset with the same format as `train_df`.
        model (tf.keras.Model): A Keras LSTM model to be trained.
        learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 0.001.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        batch_size (int, optional): Number of samples per training batch. Defaults to 32.
        max_length (int, optional): Maximum sequence length for padding. Defaults to 128.
        vocab_size (int, optional): Maximum number of words in the tokenizer vocabulary. Defaults to 10000.

    Returns:
        tuple: A tuple containing:
            - tf.keras.Model: The trained LSTM model.
            - tf.keras.callbacks.History: Training history containing loss and accuracy metrics.
            - Tokenizer: The trained Keras Tokenizer instance.
    """
    # preprocess training data
    X_train, y_train, tokenizer = _preprocess_data(train_df, max_length=max_length, vocab_size=vocab_size)

    # preprocess validation data
    X_val, y_val = _tokenize_with_existing_tokenizer(val_df, tokenizer=tokenizer, max_length=max_length, stop_words=stop_words)

    # build the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    # train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    # evaluate the model on the validation set
    print("\nClassification Report for Validation Set:\n")
    y_val_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
    print(classification_report(y_val, y_val_pred))

    return model, history, tokenizer


def evaluate_model(model, tokenizer, test_df, max_length=128, stop_words=False):
    """
    Evaluates a trained LSTM model on a test dataset and prints a classification report.

    This function tokenizes and pads the text data from the test dataset using a pre-trained tokenizer,
    makes predictions using the trained LSTM model, and generates a classification report showing the
    model's performance.

    Args:
        model (tf.keras.Model): The trained LSTM model to be evaluated.
        tokenizer (Tokenizer): A pre-trained Keras Tokenizer instance used for text tokenization.
        test_df (pd.DataFrame): Test dataset containing at least:
            - 'Review' (str): The text data.
            - 'Polarity' (int): The corresponding sentiment labels.

    Returns:
        None: The function prints the classification report but does not return any values.
    """
    X_test, y_test = _tokenize_with_existing_tokenizer(test_df, tokenizer, max_length, stop_words)
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

    print("\nClassification Report for Test Set:\n")
    print(classification_report(y_test, y_pred))


def save_classification_report(y_test, y_pred, output_path, name):
    """
    Saves the classification report to a file.

    Args:
        y_test: True test labels.
        y_pred: Predicted labels.
        output_path: Path to the directory where the file will be saved.
        name: Prefix to use for the saved file.

    Returns:
        None
    """
    report = classification_report(y_test, y_pred)
    os.makedirs(output_path, exist_ok=True)
    report_file = os.path.join(output_path, f'{name}_lstm_classification_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_file}")
