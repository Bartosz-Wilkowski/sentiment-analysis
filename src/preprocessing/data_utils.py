"""
Module: data_utils
This module provides a class for preprocessing, analyzing and visualizing data.
"""

from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split


class DataUtils:
    """
    A class to handle and analyze text data.

    This class provides static methods for handling datasets.
    """

    @staticmethod
    def preprocess_data(df):
        """
        Preprocess text data in a DataFrame.

        This method preprocesses the text in the 'Review' column of the given DataFrame.
        The preprocessing steps include:
        1. Removing special characters, URLs, mentions, and hashtags.
        2. Converting all text to lowercase.
        3. Tokenizing the text into individual words.
        4. Removing stop words (e.g., "the", "and") for cleaner text.
        5. Lemmatizing tokens to their base forms (e.g., "running" -> "run").
        6. Removing rows with empty or meaningless reviews after preprocessing.

        Parameters:
            df (pd.DataFrame): DataFrame containing column:
                            - 'Review': Text data to preprocess.

        Returns:
            pd.DataFrame: A DataFrame where the 'Review' column
                        contains preprocessed text.
        """
        # ensure required columns are present
        if 'Review' not in df.columns:
            raise ValueError("Input DataFrame must contain 'Review' column.")

        # initialize NLP tools
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        processed_reviews = []
        for text in df['Review']:
            # remove special characters and URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
            text = re.sub(r'\@\w+|\#', '', text)
            text = re.sub(r'[^A-Za-z\s]', '', text)
            text = text.lower()

            # tokenize
            tokens = word_tokenize(text)

            # remove stop words and apply lemmatization
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
            processed_reviews.append(" ".join(tokens))

        # remove rows where 'Review' is empty after processing
        df = df[df['Review'].str.strip().fillna('') != '']

        # replace the 'Review' column with processed text
        df['Review'] = processed_reviews

        return df

    @staticmethod
    def count_tokens(text):
        """
        Counts the number of tokens in a given text.

        Parameters:
        ----------
        text : str
            The text to tokenize and count tokens for.

        Returns:
        -------
        int
            The number of tokens in the text.
        """
        tokens = word_tokenize(str(text))
        return len(tokens)

    @staticmethod
    def display_top_tokens(dataset, column_name, num_tokens=25):
        """
        Displays the most frequently occurring tokens in the specified column of the dataset.

        This method tokenizes all text in the specified column of the dataset,
        counts the frequency of each token, and displays the top `num_tokens` tokens
        along with their counts in a tabular format.

        Parameters:
        ----------
        dataset : pd.DataFrame
            The dataset containing text data.
        column_name : str
            The name of the column with text to be analyzed.
        num_tokens : int, optional
            The number of most frequent tokens to display (default is 25).

        Returns:
        -------
        None
        """
        # tokenize each text entry and aggregate all tokens into a single list
        all_tokens = []
        for text in dataset[column_name]:
            tokens = word_tokenize(str(text))
            all_tokens.extend(tokens)

        # count the frequency of each token
        token_counts = Counter(all_tokens)
        most_common_tokens = token_counts.most_common(num_tokens)

        # convert the results to a DataFrame for better visualization
        df_tokens = pd.DataFrame(most_common_tokens, columns=["Token", "Count"])

        print(df_tokens)

    @staticmethod
    def _process_column_temp(dataset, column_name):
        """
        Processes the specified column in a copy of the dataset and adds a 'Token_Count' column.

        Parameters:
        ----------
        dataset : pd.DataFrame
            The dataset containing text data.
        column_name : str
            The name of the column containing text data to process.

        Returns:
        -------
        pd.DataFrame
            A new DataFrame with the added 'Token_Count' column.
        """
        # create a copy of the dataset to avoid modifying the original
        dataset_copy = dataset.copy()

        # convert the column to string
        dataset_copy[column_name] = dataset_copy[column_name].astype(str)

        # fill NaN values with empty strings
        dataset_copy[column_name] = dataset_copy[column_name].fillna('')

        # calculate the number of tokens and add a new column
        dataset_copy['Token_Count'] = dataset_copy[column_name].apply(DataUtils.count_tokens)

        return dataset_copy

    @staticmethod
    def plot_token_distribution(dataset, dataset_name, column_name='Review', bins=30):
        """
        Plots the distribution of token counts in the dataset.

        Parameters:
        ----------
        dataset : pd.DataFrame
            The dataset containing the text data.
        dataset_name : str
            The name of the dataset for labeling the plot.
        column_name : str, optional
            The name of the column to process (default is 'Review').
        bins : int, optional
            The number of bins for the histogram (default is 30).

        Returns:
        -------
        None
        """
        # process the column to add 'Token_Count' temporarily
        dataset_copy = DataUtils._process_column_temp(dataset, column_name)

        # plot the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(dataset_copy['Token_Count'], bins=bins)
        plt.title(f'Distribution of Token Counts in {dataset_name}')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Number of Reviews')
        plt.show()

    @staticmethod
    def plot_polarity_distribution(dataset, dataset_name):
        """
        Plots the distribution of polarity in the dataset.

        This method groups data by polarity, counts the number of records in each group,
        and visualizes the results as a bar chart.

        Parameters:
        ----------
        dataset : pd.DataFrame
            The dataset containing polarity data.
        dataset_name : str
            The name of the dataset for labeling the plot.

        Returns:
        -------
        None
        """
        # group data by polarity and count the number of records
        grouped_data = dataset.groupby('Polarity').size().reset_index(name='Number of Reviews')

        # ceate the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(grouped_data['Polarity'], grouped_data['Number of Reviews'])
        plt.title(f'Polarity Distribution in the {dataset_name} Dataset')
        plt.xlabel('Polarity')
        plt.ylabel('Number of Reviews')
        plt.show()

    @staticmethod
    def plot_polarity_distribution_with_labels(dataset, dataset_name):
        """
        Plots a bar chart showing the distribution of polarity in the dataset,
        with the number of records labeled above each bar.

        Parameters:
        ----------
        dataset : pd.DataFrame
            The dataset containing polarity data.
        dataset_name : str
            The name of the dataset for labeling the chart.

        Returns:
        -------
        None
        """
        # group data by polarity and count the number of records
        grouped_data = dataset.groupby('Polarity').size().reset_index(name='Number of Reviews')

        # create the bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(grouped_data['Polarity'], grouped_data['Number of Reviews'])

        # add the number of records above each bar
        for bar, count in zip(bars, grouped_data['Number of Reviews']):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom', fontsize=12)

        # configure the chart
        plt.title(f'Polarity Distribution in the {dataset_name} Dataset')
        plt.xlabel('Polarity')
        plt.ylabel('Number of Reviews')
        plt.show()

    @staticmethod
    def split_and_save_dataset(output_dir, dataset, file_prefix):
        """
        Split a dataset into training, validation, and test sets, and save them as CSV files.

        This method splits the input dataset into three subsets:
        1. Training set (70%).
        2. Validation set (15%).
        3. Test set (15%).

        The subsets are saved as separate CSV files in the specified output directory.

        Parameters:
        ----------
            output_dir (str): Directory where the resulting CSV files will be saved.
                            The directory must exist prior to calling this method.
            dataset (pd.DataFrame): The dataset to be split, provided as a pandas DataFrame.
            file_prefix (str): Prefix for the output file names. The method appends '_train.csv',
                            '_val.csv', and '_test.csv' to this prefix for the respective splits.

        Returns:
        ----------
            None: The method saves the split datasets as CSV files and does not return anything.
        """

        # split into train (70%) and temp (30%)
        train, temp = train_test_split(dataset, test_size=0.3, random_state=42)
        # split temp into validation (15%) and test (15%)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        # save the train set to a CSV file
        train.to_csv(os.path.join(output_dir, f"{file_prefix}_train.csv"), index=False, encoding='utf-8')
        # save the validation set to a CSV file
        val.to_csv(os.path.join(output_dir, f"{file_prefix}_val.csv"), index=False, encoding='utf-8')
        # save the test set to a CSV file
        test.to_csv(os.path.join(output_dir, f"{file_prefix}_test.csv"), index=False, encoding='utf-8')
