import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("ticks")
sns.color_palette("Paired")

def clean_header(header):
    # Create uniform table header
    header = header.lower()
    header = re.sub(r"\W+", "_", header)
    header = header.rstrip("_")
    return header

def check_unique_id(df: pd.DataFrame, id_column: str):
    """
    Check if the ID column is unique.
    """
    assert df[id_column].is_unique, f"The '{id_column}' column is not unique."

def check_missing_values(df, output_file):
    """
    Check for missing values in all columns of the DataFrame.
    """
    missing_summary = df.isnull().sum().reset_index()
    missing_summary.columns = ['Column', 'Missing Values']
    missing_summary['Percentage Missing'] = (missing_summary['Missing Values'] / len(df)) * 100

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Column', y='Percentage Missing', data=missing_summary)
    plt.xlabel('Column')
    plt.xticks(rotation=90)
    plt.ylabel('Missing Values (%)')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

