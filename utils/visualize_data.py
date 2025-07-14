import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import numpy as np

def load_dataset():
    """Load the cleaned dataset."""
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cleaned_dataset.csv')
    return pd.read_csv(dataset_path)

def preprocess_data(df):
    """Add analytical columns to the dataframe."""
    # Add length-based columns
    df['prompt_length'] = df['prompt'].str.len()
    df['response_length'] = df['response'].str.len()
    
    # Add word count columns
    df['prompt_words'] = df['prompt'].str.split().str.len()
    df['response_words'] = df['response'].str.split().str.len()
    
    return df

def create_length_distribution(data, column, title, bins=50):
    """Create a distribution plot of text lengths."""
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x=column, bins=bins)
    plt.title(title)
    plt.xlabel("Length (characters)")
    plt.ylabel("Count")
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'{title.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def create_word_count_distribution(data, column, title, bins=50):
    """Create a distribution plot of word counts."""
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x=column, bins=bins)
    plt.title(title)
    plt.xlabel("Number of Words")
    plt.ylabel("Count")
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'{title.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def create_length_comparison(data):
    """Create a scatter plot comparing prompt and response lengths."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x='prompt_length', y='response_length', alpha=0.5)
    plt.title("Prompt Length vs Response Length")
    plt.xlabel("Prompt Length (characters)")
    plt.ylabel("Response Length (characters)")
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'prompt_vs_response_length.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def main():
    try:
        # Load and preprocess the dataset
        print("Loading and preprocessing dataset...")
        df = load_dataset()
        df = preprocess_data(df)
        
        # Display basic information about the dataset
        print("\nDataset Info:")
        print(df.info())
        
        print("\nFirst few rows of the dataset:")
        print(df.head())
        
        print("\nGenerating visualizations...")
        
        # Create various plots
        plots = []
        plots.append(create_length_distribution(df, 'prompt_length', 'Prompt Length Distribution'))
        plots.append(create_length_distribution(df, 'response_length', 'Response Length Distribution'))
        plots.append(create_word_count_distribution(df, 'prompt_words', 'Prompt Word Count Distribution'))
        plots.append(create_word_count_distribution(df, 'response_words', 'Response Word Count Distribution'))
        plots.append(create_length_comparison(df))
        
        # Display summary statistics
        print("\nSummary Statistics:")
        print("\nPrompt Statistics:")
        print(f"Average prompt length: {df['prompt_length'].mean():.2f} characters")
        print(f"Average prompt word count: {df['prompt_words'].mean():.2f} words")
        print(f"Shortest prompt: {df['prompt_length'].min()} characters")
        print(f"Longest prompt: {df['prompt_length'].max()} characters")
        
        print("\nResponse Statistics:")
        print(f"Average response length: {df['response_length'].mean():.2f} characters")
        print(f"Average response word count: {df['response_words'].mean():.2f} words")
        print(f"Shortest response: {df['response_length'].min()} characters")
        print(f"Longest response: {df['response_length'].max()} characters")
        
        print("\nPlots have been saved in the data/plots directory:")
        for plot in plots:
            print(f"- {os.path.basename(plot)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 