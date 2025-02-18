"""
BERT Model Fine-Tuning and Evaluation Module

This module provides functions for tokenizing, fine-tuning, and evaluating a BERT-based model for sentiment classification.

Functions:
    - _tokenize(df, model_name, max_length=128, batch_size=16):
      Tokenizes and prepares text data for BERT model training or evaluation.

    - fine_tune_bert(train_df, val_df, model_name='bert-base-uncased', epochs=4, max_length=128, batch_size=16, learning_rate=2e-5):
      Fine-tunes a BERT model on given datasets and returns the trained model and history.

    - evaluate_bert(model, test_dataset):
      Evaluates the trained BERT model on a test dataset and prints a classification report.

Dependencies:
    - numpy, tensorflow, transformers, sklearn
"""

import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping


def evaluate_bert(model, test_dataset):
    """
    Evaluates the performance of an BERT model on the test set and prints the results.

    Args:
        model (TFDistilBertForSequenceClassification): The trained BERT model.
        test_dataset (tf.data.Dataset): Tokenized test dataset.
    """
    print("\nEvaluating on Test Set...")

    # predict logits from the model
    test_predictions = model.predict(test_dataset)
    predicted_labels = np.argmax(test_predictions.logits, axis=1)

    # extract true labels directly from the tokenized test dataset
    true_labels = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)

    # generate and print the classification report
    print(classification_report(true_labels, predicted_labels))


def _tokenize(df, model_name, max_length=128, batch_size=16):
    """
    Tokenizes text data from a DataFrame for use with a BERT model.

    This function preprocesses text data by removing empty or NaN rows,
    tokenizing the text using a pre-trained BERT tokenizer, and converting
    the tokenized data into a TensorFlow dataset for training or evaluation.

    Args:
        df (pd.DataFrame): The input DataFrame containing text and labels.
                           Expects two columns: 'Review' (text) and 'Polarity' (labels).
        model_name (str): Name of the pre-trained BERT model to use for tokenization (e.g., 'distilbert-base-uncased').
        tokenizer (BertTokenizer): A pre-trained BERT tokenizer object.
        max_length (int): Maximum length of tokenized sequences (default is 128).
        batch_size (int): Batch size for the TensorFlow dataset (default is 16).

    Returns:
        tf.data.Dataset: A TensorFlow dataset containing tokenized input tensors
                         ('input_ids' and 'attention_mask') and labels (polarity).
    """
    # remove rows where 'Review' is NaN or empty
    df = df.dropna(subset=['Review']).copy()
    df = df[df['Review'].str.strip() != '']

    # initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # convert polarity to integers if it's not already
    if not np.issubdtype(df['Polarity'].dtype, np.integer):
        try:
            df['Polarity'] = df['Polarity'].astype(int)
        except ValueError:
            raise ValueError("'Polarity' column must contain integer values representing sentiment.")

    # tokenize reviews
    inputs = tokenizer(
        df['Review'].tolist(),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='tf'
    )

    # convert polarity labels to TensorFlow tensor
    labels = tf.convert_to_tensor(df['Polarity'].values, dtype=tf.int32)

    return tf.data.Dataset.from_tensor_slices(({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }, labels)).batch(batch_size)


def fine_tune_bert(train_df, val_df, model_name='bert-base-uncased', epochs=4, max_length=128, batch_size=16, learning_rate=2e-5):
    """
    Fine-tunes a BERT model on a given training and validation dataset for a binary classification task.

    Workflow:
        1. Preprocess the training and validation datasets using the `_tokenize` function.
        2. Initialize the tokenizer and BERT model for sequence classification.
        3. Compile the model with an Adam optimizer, `learning_rate`, and accuracy as the evaluation metric.
        4. Define the `EarlyStopping` callback to stop training when `val_loss` stops improving.
        5. Train the model using the `fit()` method on the training dataset and validate on the validation dataset.
        6. Evaluate the model on the validation dataset and display a classification report.

    Args:
        train_df (pd.DataFrame): The training dataset containing text and labels.
                                 Assumes two columns: 'Review' (text) and 'Polarity' (labels).
        val_df (pd.DataFrame): The validation dataset containing text and labels.
                               Assumes two columns: 'Review' (text) and 'Polarity' (labels).
        model_name (str): Name of the pre-trained BERT model to use (default is 'bert-base-uncased').
        epochs (int): Number of training epochs (default is 4).
        max_length (int): Maximum token length for text sequences (default is 128).
        batch_size (int): Batch size for training and validation (default is 16).
        learning_rate (float): Learning rate for the Adam optimizer (default is 2e-5).

    Returns:
        tuple: A tuple containing:
            - model (TFDistilBertForSequenceClassification): The fine-tuned BERT model.
            - history (History): The training history object containing metrics for each epoch.
    """

    # preprocess datasets
    train_dataset = _tokenize(
        df=train_df,
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size
    )

    val_dataset = _tokenize(
        df=val_df,
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size
    )

    # initialize tokenizer and model
    model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    # train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # evaluate the model on the validation set
    print("\nValidation Results:")
    val_predictions = model.predict(val_dataset)
    predicted_labels = np.argmax(val_predictions.logits, axis=1)
    true_labels = np.concatenate([y.numpy() for _, y in val_dataset], axis=0)
    print(classification_report(true_labels, predicted_labels))

    return model, history
