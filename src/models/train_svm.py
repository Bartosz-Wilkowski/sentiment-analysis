from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
from joblib import dump
import os


def check_svm_model_cv(train_df, test_df, kernel='linear', C=1.0, max_features=5000, ngram_range=(1, 2)):
    """
    Prepare an SVM model and display cross-validation metrics.

    Args:
        train_df (pd.DataFrame): Preprocessed training data containing the 'Review' and 'Polarity' columns.
        test_df (pd.DataFrame): Preprocessed test data containing the 'Review' and 'Polarity' columns.
        kernel (str): Kernel type for the SVM ('linear', 'rbf', 'poly').
        C (float): Regularization parameter for the SVM.
        max_features (int): Maximum number of features for TF-IDF vectorizer.
        ngram_range (tuple): N-gram range for TF-IDF (e.g., (1, 1), (1, 2)).

    Returns:
        None
    """

    # define TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    # transform text into TF-IDF vectors
    X_train = vectorizer.fit_transform(train_df['Review'])
    y_train = train_df['Polarity']

    # define SVM model
    svm_model = SVC(kernel=kernel, C=C, random_state=42)

    # perform 5-fold cross-validation for accuracy
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(svm_model, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1)

    # display cross-validation results
    print("\nCross-Validation Accuracy (5-Fold):")
    print(f"Mean Accuracy: {np.mean(cv_results):.4f}")
    print(f"Standard Deviation: {np.std(cv_results):.4f}")


def evaluate_svm(model, X_test, y_test):
    """
    Evaluates the performance of an SVM model on the test set and prints the results.

    Args:
        model: Trained SVM model.
        X_test: Features of the test set.
        y_test: True labels of the test set.

    Returns:
        None
    """
    # predict on the test set
    y_test_pred = model.predict(X_test)

    # calculate metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # print metrics
    print("\nTest Set Metrics:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")

    # print classificaion report
    print("\nClassification Report for Test Set:\n")
    print(classification_report(y_test, y_test_pred))


def save_svm_model_and_vectorizer(model, vectorizer, output_path, name):
    """
    Saves the given model and vectorizer to the specified output path with a given name prefix.

    Args:
        model: Trained machine learning model to save.
        vectorizer: Trained vectorizer to save.
        output_path: Path to the directory where the files will be saved.
        name: Prefix to use for the saved files.

    Returns:
        None
    """
    os.makedirs(output_path, exist_ok=True)

    model_file = os.path.join(output_path, f'{name}_svm_model.joblib')
    dump(model, model_file)
    print(f"Model saved to {model_file}")

    vectorizer_file = os.path.join(output_path, f'{name}_svm_vectorizer.joblib')
    dump(vectorizer, vectorizer_file)
    print(f"Vectorizer saved to {vectorizer_file}")


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
    report_file = os.path.join(output_path, f'{name}_svm_classification_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_file}")
