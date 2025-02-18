# Sentiment Analysis with NLP Models

## Project Overview
This project focuses on **Sentiment Analysis** using various **Natural Language Processing (NLP)** models. It compares the effectiveness and efficiency of different machine learning approaches, including **Support Vector Machine (SVM), Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN), and Bidirectional Encoder Representations from Transformers (BERT)**. The study evaluates these models based on multiple performance metrics, data preprocessing techniques, and the impact of **Transfer Learning**.

## Project Structure
```
SENTIMENT-ANALYSIS/
│── data/
│   ├── embeddings/        # Word embedding files 
│   ├── preprocessed/      # Preprocessed datasets
│   ├── raw/               # Raw datasets (Amazon, Sentiment140, Yelp)
│
│── notebooks/
│   ├── 01_data_preprocessing.ipynb    # Data preprocessing steps
│   ├── 02_data_augm.ipynb             # Data augmentation techniques
│   ├── 03_model_svm.ipynb             # SVM model training and evaluation
│   ├── 04_model_lstm.ipynb            # LSTM model training and evaluation
│   ├── 05_model_cnn.ipynb             # CNN model training and evaluation
│   ├── 06_model_bert.ipynb            # BERT model fine-tuning and evaluation
│
│── outputs/
│   ├── logs/                          # Training logs
│   ├── models/                        # Saved models
│
│── src/
│   ├── models/
│   │   ├── train_bert.py              # BERT fine-tuning
│   │   ├── train_cnn.py               # CNN training
│   │   ├── train_lstm.py              # LSTM training
│   │   ├── train_svm.py               # SVM training
│   ├── preprocessing/
│   │   ├── data_utils.py              # Data preprocessing and analysis utilities
│
│── .gitignore                         # Ignore unnecessary files
│── poetry.lock                         # Dependency lock file
│── pyproject.toml                      # Project dependencies and settings
│── README.md                           # Project documentation
```

## Installation
To set up the project, ensure you have **Python 3.8+** installed, then follow these steps:

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd SENTIMENT-ANALYSIS
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```
   If Poetry is not installed, install it first:
   ```bash
   pip install poetry
   ```

## Datasets
This project utilizes three benchmark datasets for sentiment analysis:

- **[Amazon Books Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)**: A dataset containing reviews and descriptions of books from the Amazon.
- **[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)**: A dataset consisting of 1,600,000 tweets collected via the Twitter API, labeled as negative or positive.
- **[Yelp Reviews](https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset)**: A dataset containing user reviews from the Yelp platform, with sentiment classification based on star ratings.


## Models Studied
### 1. **Support Vector Machine (SVM)**
- Utilizes **TF-IDF vectorization** for text feature extraction.
- Investigates the impact of **Linear, Polynomial, and RBF kernels** on sentiment classification.
- Evaluates model performance using **cross-validation, accuracy, precision, recall, and F1-score**.
- Compares **different hyperparameter settings**, including regularization and kernel selection.
- Tests the model on **original and augmented datasets**.

### 2. **Long Short-Term Memory (LSTM)**
- Compares **LSTM, BiLSTM, and BiLSTM with Attention Mechanism** for sentiment classification.
- Uses **FastText word embeddings** for better semantic representation of words.
- Tests **dropout regularization** and **learning rate tuning** to prevent overfitting.
- Analyzes **model robustness on noisy and augmented datasets**.

### 3. **Convolutional Neural Network (CNN)**
- Compares **CNN, BiCNN, and Temporal Convolutional Network (TCN)** for sentiment classification.
- Uses **FastText word embeddings** for word representation.
- Investigates the impact of **1D convolutions and dropout** on generalization.
- Evaluates the **effectiveness of CNN-based models compared to sequential models** (LSTM).

### 4. **Transformer-Based Models (DistilBERT)**
- Fine-tunes **DistilBERT** for sentiment classification.
- Utilizes **pre-trained transformer embeddings** from Hugging Face.
- Compares performance on **original and augmented datasets**.
- Assesses **training time, memory consumption, and effectiveness** in handling sentiment classification tasks.

Each model is tested on **Amazon Books Reviews, Sentiment140, and Yelp Reviews**, with experiments conducted on **both original and augmented datasets** to assess **generalization and robustness**.


## Results and Findings
The project evaluates model performance based on:
- **Accuracy**: Overall correctness of predictions.
- **Precision**: Correctness of positive classifications.
- **Recall**: Ability to capture all positive instances.
- **F1-score**: Balance between precision and recall.
- **Training Time**: Computational efficiency.
- **Robustness to Noise**: Performance on augmented datasets.

## Citation
If you use this project for research purposes, please cite:

```
Bartosz Wilkowski, "Comparison of the Effectiveness and Efficiency of Selected NLP Models in Sentiment Analysis Using Transfer Learning Techniques ", Warsaw, 2025.
```

## License
This project is released under the **MIT License**.

