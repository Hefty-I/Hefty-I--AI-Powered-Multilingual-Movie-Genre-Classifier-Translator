import streamlit as st
import base64
import io
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def display_metrics(metrics):
    """
    Display model evaluation metrics in a formatted manner.
    
    Args:
        metrics (dict): Dictionary of evaluation metrics
    """
    # Create columns for metrics display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        st.metric("Precision (micro)", f"{metrics['precision_micro']:.4f}")
    
    with col2:
        st.metric("Recall (micro)", f"{metrics['recall_micro']:.4f}")
        st.metric("F1 Score (micro)", f"{metrics['f1_micro']:.4f}")
    
    with col3:
        st.metric("Precision (macro)", f"{metrics['precision_macro']:.4f}")
        st.metric("Recall (macro)", f"{metrics['recall_macro']:.4f}")
        st.metric("F1 Score (macro)", f"{metrics['f1_macro']:.4f}")

def create_download_link(data, filename, text):
    """
    Create a download link for data.
    
    Args:
        data (bytes): Data to download
        filename (str): Filename for the downloaded file
        text (str): Text to display for the download link
        
    Returns:
        str: HTML for the download link
    """
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

def plot_genre_distribution(genres):
    """
    Plot the distribution of genres.
    
    Args:
        genres (list): List of all genres
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Count genre occurrences
    genre_counts = pd.Series(genres).value_counts()
    
    # Plot distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax)
    ax.set_title("Genre Distribution")
    ax.set_xlabel("Count")
    ax.set_ylabel("Genre")
    
    return fig

def plot_confusion_matrix(conf_matrix):
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix (numpy.ndarray): Confusion matrix
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    return fig
