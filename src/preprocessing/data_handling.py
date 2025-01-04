"""
Module: data_handling
This module provides a class for handling, analyzing and visualizing data.
"""

from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize


class DataHandler:
    """
    A class to handle and analyze text data.

    This class provides static methods for handling datasets with text columns.
    """

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
        dataset_copy['Token_Count'] = dataset_copy[column_name].apply(DataHandler.count_tokens)

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
        dataset_copy = DataHandler._process_column_temp(dataset, column_name)

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
