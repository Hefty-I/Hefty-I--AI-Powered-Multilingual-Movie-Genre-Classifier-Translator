import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import aiohttp
import asyncio
import io

# Download NLTK resources
try:
    nltk.data.find('corpus/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpus/wordnet')
    nltk.data.find('corpus/omw-1.4')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# async def fetch_data(url):
#     """
#     Asynchronously fetch data from a URL.
    
#     Args:
#         url (str): URL to fetch data from
        
#     Returns:
#         str: Text content of the response
#     """
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             return await response.text()  # Properly await the text content
def preprocess_text(text):
    """
    Preprocess text by performing the following steps:
    1. Convert to lowercase
    2. Remove special characters and numbers
    3. Tokenize
    4. Remove stopwords
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    
    # Simple tokenization instead of using word_tokenize
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def load_and_preprocess_data(plot_summaries_path, metadata_path):
    """
    Load and preprocess movie summaries and metadata files.
    
    Args:
        plot_summaries_path: Path to the file containing movie summaries
        metadata_path: Path to the file containing movie metadata
        
    Returns:
        tuple: (
            df_processed: Processed dataframe with movie ID, summary, and genres
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            vectorizer: Fitted TF-IDF vectorizer
            mlb: Fitted MultiLabelBinarizer
        )
    """
    # Read plot summaries (tab-separated)
    with open(plot_summaries_path, 'r', encoding='utf-8') as f:
        plot_summaries_content = f.read()
    
    plot_summaries_data = []
    
    for line in plot_summaries_content.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) >= 2:
            movie_id = parts[0]
            summary = parts[1]
            plot_summaries_data.append([movie_id, summary])
    
    df_summaries = pd.DataFrame(plot_summaries_data, columns=['movie_id', 'summary'])
    
    # Read metadata (tab-separated)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_content = f.read()
    
    metadata_data = []
    
    for line in metadata_content.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) >= 9:  # Ensure the genres column exists
            movie_id = parts[0]
            genres = parts[8].split()  # Space-separated genres
            # Remove curly braces if present
            genres = [g.strip('{}') for g in genres]
            metadata_data.append([movie_id, genres])
    
    df_metadata = pd.DataFrame(metadata_data, columns=['movie_id', 'genres'])
    
    # Merge dataframes
    df = pd.merge(df_summaries, df_metadata, on='movie_id')
    
    # Preprocess summaries
    df['processed_summary'] = df['summary'].apply(preprocess_text)
    
    # Filter out rows with empty summaries or genres
    df = df[df['processed_summary'].str.strip() != '']
    df = df[df['genres'].apply(len) > 0]
    
    # Ensure we have genre diversity by counting unique genres
    unique_genres = set()
    for genre_list in df['genres']:
        unique_genres.update(genre_list)
    print(f"Found {len(unique_genres)} unique genres in the dataset")
    
    # Get movies with diverse genres (at least one movie per genre)
    selected_movies = []
    genre_count = {}
    
    # Initialize genre counter
    for genre in unique_genres:
        genre_count[genre] = 0
    
    # Select movies to ensure genre diversity
    for idx, row in df.iterrows():
        movie_id = row['movie_id']
        genres = row['genres']
        
        # Check if any genre in this movie needs more representation
        needs_more = False
        for genre in genres:
            if genre_count[genre] < 5:  # Aim for at least 5 movies per genre
                needs_more = True
                break
        
        if needs_more:
            selected_movies.append(movie_id)
            for genre in genres:
                genre_count[genre] += 1
    
    # Add more movies until we have enough
    if len(selected_movies) < 500:
        remaining = df[~df['movie_id'].isin(selected_movies)]
        additional = min(500 - len(selected_movies), len(remaining))
        additional_ids = remaining.head(additional)['movie_id'].tolist()
        selected_movies.extend(additional_ids)
    
    # Filter to selected movies
    print(f"Selected {len(selected_movies)} movies for processing")
    df = df[df['movie_id'].isin(selected_movies)]
    
    # Create final processed dataframe
    df_processed = df[['movie_id', 'summary', 'processed_summary', 'genres']]
    
    # Prepare for modeling
    X = df_processed['processed_summary']
    
    # Convert genres to multi-label format
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df_processed['genres'])
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    return df_processed, X_train, X_test, y_train, y_test, vectorizer, mlb
