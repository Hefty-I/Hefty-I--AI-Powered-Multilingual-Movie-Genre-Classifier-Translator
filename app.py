import streamlit as st
import pandas as pd
import os
import time
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from preprocessing import preprocess_text, load_and_preprocess_data
from translation import translate_text
from audio_conversion import text_to_speech
from genre_prediction import (
    train_genre_model, 
    predict_genre, 
    evaluate_model,
    get_top_features
)
from utils import display_metrics, create_download_link

# Set page config
st.set_page_config(
    page_title="Filmception - AI-powered Movie Summary Processor",
    page_icon="🎬",
    layout="wide",
)

# Initialize session states
if "preprocessed_data" not in st.session_state:
    st.session_state["preprocessed_data"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "vectorizer" not in st.session_state:
    st.session_state["vectorizer"] = None
if "multilabel_binarizer" not in st.session_state:
    st.session_state["multilabel_binarizer"] = None
if "evaluation_metrics" not in st.session_state:
    st.session_state["evaluation_metrics"] = None
if "confusion_matrix" not in st.session_state:
    st.session_state["confusion_matrix"] = None
if "translated_summaries" not in st.session_state:
    st.session_state["translated_summaries"] = {}
if "audio_files" not in st.session_state:
    st.session_state["audio_files"] = {}
if "top_genres" not in st.session_state:
    st.session_state["top_genres"] = None
if "has_trained" not in st.session_state:
    st.session_state["has_trained"] = False
if "auto_process" not in st.session_state:
    st.session_state["auto_process"] = False
    
# Define paths to data files
plot_summaries_path = "attached_assets/plot_summaries.txt"
metadata_path = "attached_assets/movie.metadata.tsv"

# Auto-process all files if they exist
if os.path.exists(plot_summaries_path) and os.path.exists(metadata_path) and not st.session_state["has_trained"] and not st.session_state["auto_process"]:
    with st.spinner("Automatically processing all data files... This may take a few minutes"):
        # Set flag to prevent repeated processing
        st.session_state["auto_process"] = True
        
        # Load and preprocess data
        df_processed, X_train, X_test, y_train, y_test, vectorizer, mlb = load_and_preprocess_data(
            plot_summaries_path, metadata_path
        )
        
        # Store in session state
        st.session_state["preprocessed_data"] = {
            "df_processed": df_processed,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        st.session_state["vectorizer"] = vectorizer
        st.session_state["multilabel_binarizer"] = mlb
        
        # Train model with default parameters (Logistic Regression)
        model, evaluation_metrics, conf_matrix = train_genre_model(
            X_train, X_test, y_train, y_test, "Logistic Regression", {"C": 1.0, "max_iter": 100, "solver": "liblinear"}
        )
        
        # Store model and metrics in session state
        st.session_state["model"] = model
        st.session_state["evaluation_metrics"] = evaluation_metrics
        st.session_state["confusion_matrix"] = conf_matrix
        st.session_state["has_trained"] = True
        
        # Get top genres for feature analysis
        top_genres = st.session_state["multilabel_binarizer"].classes_
        st.session_state["top_genres"] = top_genres

# Main title
st.title("🎬 Filmception")
st.subheader("AI-powered Multilingual movie summary translator and genre classifier")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Welcome", "Model Training", "Explore Features", "Use the App", "Project Report"]
)

# Welcome page
if app_mode == "Welcome":
    st.write("""
    ## Overview
    Welcome to Filmception - an AI-powered system for processing movie summaries, predicting movie genres, 
    and converting movie summaries into audio formats in multiple languages.
    
    ### Features:
    - Automatically preprocess and clean movie summaries from the CMU Movie Summary dataset
    - Translate movie summaries into multiple languages (Arabic, Urdu, Korean)
    - Convert translated summaries to audio
    - Predict movie genres based on summaries using machine learning
    
    ### How to use:
    1. **Model Training**: Train a machine learning model to predict movie genres
    2. **Explore Features**: Analyze the model features and performance
    3. **Use the App**: Input your own movie summary and explore the features
    4. **Project Report**: View detailed information about the project
    
    The data is automatically processed when you start the application!
    """)
    
    st.info("📝 Note: This application automatically processes the CMU Movie Summary dataset from the attached_assets folder when it starts.\n\nYou can proceed directly to 'Model Training' in the sidebar to train a model using the preprocessed data.")
    
    # Show progress indicator when data is being processed
    if st.session_state["preprocessed_data"] is not None:
        # Calculate and display dataset statistics
        if "df_processed" in st.session_state["preprocessed_data"]:
            df = st.session_state["preprocessed_data"]["df_processed"]
            
            # Display dataset statistics
            st.subheader("Dataset Statistics")
            
            # Sum all genres
            all_genres = []
            for genres in df['genres']:
                all_genres.extend(genres)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Movies", len(df))
            with col2:
                st.metric("Unique Genres", len(set(all_genres)))
            with col3:
                st.metric("Avg. Genres per Movie", round(len(all_genres) / len(df), 2))
                
            # Display training status
            if st.session_state["has_trained"]:
                st.success("✅ Data processing and model training completed successfully! You can now explore all features of the application.")
            else:
                st.info("Data processing completed. Head to the 'Model Training' section to train a genre prediction model.")
    else:
        st.warning("Data processing is in progress. Please wait until it completes...")

# Model Training page
elif app_mode == "Model Training":
    st.title("Model Training")
    
    if st.session_state["preprocessed_data"] is None:
        st.warning("Please complete the data preprocessing step before training the model.")
    else:
        st.write("### Model Type: Logistic Regression")
        
        # Hyperparameters for Logistic Regression
        c_value = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0, 0.1)
        max_iter = st.slider("Maximum Iterations", 100, 1000, 100, 50)
        solver = st.selectbox("Solver", ["liblinear", "saga"])
        
        hyperparams = {
            "C": c_value,
            "max_iter": max_iter,
            "solver": solver
        }
        
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a few minutes"):
                # Access data from session state
                X_train = st.session_state["preprocessed_data"]["X_train"]
                X_test = st.session_state["preprocessed_data"]["X_test"]
                y_train = st.session_state["preprocessed_data"]["y_train"]
                y_test = st.session_state["preprocessed_data"]["y_test"]
                
                # Train model
                model, evaluation_metrics, conf_matrix = train_genre_model(
                    X_train, X_test, y_train, y_test, model_type="Logistic Regression", hyperparams=hyperparams
                )
                
                # Store model and metrics in session state
                st.session_state["model"] = model
                st.session_state["evaluation_metrics"] = evaluation_metrics
                st.session_state["confusion_matrix"] = conf_matrix
                st.session_state["has_trained"] = True
                
                # Get top genres for feature analysis
                top_genres = st.session_state["multilabel_binarizer"].classes_
                st.session_state["top_genres"] = top_genres
                
                st.success("Model training completed!")
        
        # Display evaluation metrics and confusion matrix
        if st.session_state["has_trained"]:
            st.subheader("Model Evaluation")
            metrics = st.session_state["evaluation_metrics"]
            display_metrics(metrics)
            
            st.subheader("Confusion Matrix")
            conf_matrix = st.session_state["confusion_matrix"]
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Confusion Matrix (Multi-label)")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

# Explore Features page
elif app_mode == "Explore Features":
    st.header("Explore Model Features")
    
    if not st.session_state["has_trained"]:
        st.warning("Please train a model first to explore its features.")
    else:
        # Get top features for each genre
        model_name = st.session_state["model"].__class__.__name__
        if model_name == "MultiOutputClassifier":
            st.subheader("Top Words for Each Genre")
            
            vectorizer = st.session_state["vectorizer"]
            model = st.session_state["model"]
            top_genres = st.session_state["top_genres"]
            
            if len(top_genres) == 0:
                st.warning("No genres were found in the dataset. Feature exploration is not available.")
            else:
                # Allow user to select genre
                selected_genre = st.selectbox("Select a genre to view top words", top_genres)
                
                genre_idx = np.where(top_genres == selected_genre)[0][0]
                
                # Get top words for selected genre
                top_words = get_top_features(vectorizer, model, genre_idx, n=20)
                
                # Check if we have meaningful features to display
                if len(top_words) == 1 and top_words[0][0].startswith("No feature"):
                    st.info(top_words[0][0])
                else:
                    # Display as bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Sort by coefficients
                    top_words_sorted = sorted(top_words, key=lambda x: x[1])
                    words = [word for word, coef in top_words_sorted]
                    coefs = [coef for word, coef in top_words_sorted]
                    
                    sns.barplot(x=coefs, y=words, palette="viridis", ax=ax)
                    ax.set_title(f"Top 20 Words Associated with {selected_genre}")
                    ax.set_xlabel("Coefficient Value")
                    st.pyplot(fig)
                    
                    # Write explanation
                    st.write(f"""
                    **Interpretation**: These words have the strongest association with the genre "{selected_genre}" 
                    according to our model. Words with higher coefficient values are more predictive of this genre.
                    """)
        else:
            st.info("Feature importance visualization is currently only available for classification models.")
        
        # Sample predictions on test data
        st.subheader("Sample Predictions")
        
        if "df_processed" in st.session_state["preprocessed_data"]:
            df = st.session_state["preprocessed_data"]["df_processed"]
            
            # Select a random sample
            sample_idx = np.random.randint(0, len(df))
            sample = df.iloc[sample_idx]
            
            st.write("**Movie Summary:**")
            st.write(sample['summary'])
            
            st.write("**Actual Genres:**")
            st.write(", ".join(sample['genres']))
            
            # Predict genre
            preprocessed_text = preprocess_text(sample['summary'])
            predicted_genres = predict_genre(
                preprocessed_text,
                st.session_state["model"],
                st.session_state["vectorizer"],
                st.session_state["multilabel_binarizer"]
            )
            
            st.write("**Predicted Genres:**")
            st.write(", ".join(predicted_genres))

# Use the App page
elif app_mode == "Use the App":
    st.header("Filmception - Interactive Movie Summary Processor")
    
    if not st.session_state["has_trained"]:
        st.warning("Please train a model first to use all app features.")
    
    # Input section
    st.subheader("Enter a Movie Summary")
    user_input = st.text_area("Type or paste a movie summary here:", height=200)
    
    if user_input:
        # Process the input
        preprocessed_input = preprocess_text(user_input)
        
        # User can select what to do with the summary
        st.subheader("Choose an Action")
        
        # Create tabs for different actions
        tab1, tab2 = st.tabs(["Genre Prediction", "Translation & Audio"])
        
        with tab1:
            if st.session_state["has_trained"]:
                if st.button("Predict Genre", key="predict_genre"):
                    with st.spinner("Predicting genre..."):
                        # Predict genre
                        predicted_genres = predict_genre(
                            preprocessed_input,
                            st.session_state["model"],
                            st.session_state["vectorizer"],
                            st.session_state["multilabel_binarizer"]
                        )
                        
                        # Display results
                        st.subheader("Predicted Genres")
                        st.write(", ".join(predicted_genres))
                        
                        # Create a simple visualization of the genres
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.barh(range(len(predicted_genres)), [1] * len(predicted_genres), color='skyblue')
                        ax.set_yticks(range(len(predicted_genres)))
                        ax.set_yticklabels(predicted_genres)
                        ax.set_title("Predicted Movie Genres")
                        ax.set_xlabel("Confidence")
                        st.pyplot(fig)
            else:
                st.info("Train a model first to enable genre prediction.")
        
        with tab2:
            # Translation options
            st.subheader("Translation Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_language = st.selectbox(
                    "Select target language", 
                    ["Arabic", "Urdu", "Korean"]
                )
                
                language_codes = {
                    "Arabic": "ar",
                    "Urdu": "ur",
                    "Korean": "ko"
                }
                
                language_code = language_codes[target_language]
            
            # Translation action
            if st.button("Translate", key="translate"):
                with st.spinner(f"Translating to {target_language}..."):
                    # Check if already translated
                    if user_input not in st.session_state["translated_summaries"] or \
                       language_code not in st.session_state["translated_summaries"][user_input]:
                        # Translate text
                        translated_text = translate_text(user_input, language_code)
                        
                        # Store in session state
                        if user_input not in st.session_state["translated_summaries"]:
                            st.session_state["translated_summaries"][user_input] = {}
                        
                        st.session_state["translated_summaries"][user_input][language_code] = translated_text
                    else:
                        translated_text = st.session_state["translated_summaries"][user_input][language_code]
                    
                    # Display translation
                    st.subheader(f"Translation ({target_language})")
                    st.write(translated_text)
                    
                    # Check if audio already exists
                    key = f"{user_input}_{language_code}"
                    
                    # Create audio
                    with st.spinner("Converting to audio..."):
                        if key not in st.session_state["audio_files"]:
                            audio_bytes = text_to_speech(translated_text, language_code)
                            st.session_state["audio_files"][key] = audio_bytes
                        else:
                            audio_bytes = st.session_state["audio_files"][key]
                        
                        # Display audio player
                        st.subheader("Audio Playback")
                        st.audio(audio_bytes, format="audio/mp3")
                        
                        # Download link
                        st.download_button(
                            label="Download Audio",
                            data=audio_bytes,
                            file_name=f"movie_summary_{target_language}.mp3",
                            mime="audio/mp3"
                        )

# Project Report page
elif app_mode == "Project Report":
    st.header("Filmception Project Report")
    
    st.write("""
    ## Project Overview
    
    Filmception is an AI-powered tool that combines natural language processing, machine learning, and audio generation 
    to provide a comprehensive solution for movie summary analysis. It automatically processes movie data from the 
    CMU Movie Summary Corpus, extracts patterns and insights from movie plots, and offers multilingual support 
    for translations and audio generation.
    
    ### Core Components
    
    1. **Data Processing Pipeline**
       - Automatic loading of CMU Movie Summary dataset files
       - Text preprocessing with efficient tokenization and stop-word removal
       - Multi-label genre encoding for machine learning compatibility
       
    2. **Machine Learning Model**
       - Genre prediction using multi-label classification
       - Support for multiple model types (Logistic Regression, Random Forest, SVM)
       - Feature importance analysis to understand genre-specific keywords
       
    3. **Multilingual Support**
       - Translation to multiple languages (Arabic, Urdu, Korean)
       - Text-to-speech conversion for accessibility
       - Audio file generation and download
       
    4. **User Interface**
       - Interactive web application built with Streamlit
       - Dynamic visualization of model metrics and features
       - Seamless user experience with automatic data processing
    """)
    
    # Technical Implementation Details
    st.subheader("Technical Implementation Details")
    
    # Show tabs for different aspects of the implementation
    tech_tab1, tech_tab2, tech_tab3, tech_tab4 = st.tabs([
        "Data Processing", "Machine Learning", "Translation & Audio", "Error Handling"
    ])
    
    with tech_tab1:
        st.write("""
        ### Data Processing Implementation
        
        The data processing pipeline handles the CMU Movie Summary dataset, which consists of movie summaries and metadata. 
        Key challenges addressed include:
        
        - **Efficient Text Preprocessing**: The pipeline converts raw text to a clean, tokenized format suitable for machine learning
        - **Genre Diversity Sampling**: The application selects movies to ensure all genres have sufficient representation
        - **Multi-label Classification**: Each movie can belong to multiple genres, requiring specialized encoding techniques
        
        The preprocessing steps include:
        1. Loading and merging movie metadata and summary files
        2. Cleaning and tokenizing text data
        3. Removing stopwords for improved feature extraction
        4. Creating TF-IDF features to represent text documents
        5. Encoding multi-label genre targets using MultiLabelBinarizer
        """)
        
        # Show dataset statistics if available
        if "preprocessed_data" in st.session_state and st.session_state["preprocessed_data"] is not None and "df_processed" in st.session_state["preprocessed_data"]:
            df = st.session_state["preprocessed_data"]["df_processed"]
            
            # Display sample data
            st.subheader("Sample of Processed Data")
            st.dataframe(df.head())
            
            # Sum all genres
            all_genres = []
            for genres in df['genres']:
                all_genres.extend(genres)
            
            # Display genre distribution
            st.subheader("Genre Distribution")
            genre_counts = pd.Series(all_genres).value_counts().head(15)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="viridis", ax=ax)
            ax.set_title("Top 15 Movie Genres")
            ax.set_xlabel("Count")
            ax.set_ylabel("Genre")
            st.pyplot(fig)
            
            # Display dataset statistics
            st.subheader("Dataset Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Movies", len(df))
            with col2:
                st.metric("Unique Genres", len(set(all_genres)))
            with col3:
                st.metric("Avg. Genres per Movie", round(len(all_genres) / len(df), 2))
    
    with tech_tab2:
        st.write("""
        ### Machine Learning Implementation
        
        The genre prediction system uses multi-label classification to predict multiple genres for each movie summary. 
        The machine learning pipeline includes:
        
        - **Feature Engineering**: TF-IDF vectorization transforms text into numerical features
        - **Model Architecture**: MultiOutputClassifier wraps base classifiers to handle multi-label targets
        - **Robust Error Handling**: Fallback mechanisms detect and handle edge cases like insufficient class diversity
        - **Model Evaluation**: Custom metrics account for multi-label performance (accuracy, precision, recall, F1 score)
        
        Supported model types:
        - Logistic Regression (default)  
        - Random Forest  
        - Support Vector Machine  
        """)
        
        # Show model evaluation if available
        if st.session_state["has_trained"]:
            st.subheader("Current Model Evaluation")
            metrics = st.session_state["evaluation_metrics"]
            display_metrics(metrics)
            
            # Confusion matrix visualization
            st.subheader("Confusion Matrix")
            conf_matrix = st.session_state["confusion_matrix"]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Confusion Matrix (Multi-label)")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
    
    with tech_tab3:
        st.write("""
        ### Translation & Audio Generation
        
        The application provides multilingual support for movie summaries through:
        
        - **Google Translate Integration**: Translate text to Arabic, Urdu, and Korean
        - **Google Text-to-Speech**: Convert translated text to audio in the target language
        - **Audio File Management**: Efficient storage and retrieval of generated audio files
        
        The translation and audio components are optimized to handle longer texts common in movie summaries, 
        with proper error handling for network issues or service limitations.
        """)
        
        # Show sample translation interface
        st.subheader("Supported Languages")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Arabic (ar)**")
            st.write("Right-to-left script with connected letters")
        with col2:
            st.markdown("**Urdu (ur)**")
            st.write("Perso-Arabic script with Nastaliq style")
        with col3:
            st.markdown("**Korean (ko)**")
            st.write("Hangul alphabet with syllabic blocks")
    
    with tech_tab4:
        st.write("""
        ### Error Handling & Robustness
        
        The application includes comprehensive error handling mechanisms:
        
        - **Data Processing Fallbacks**: Handles missing or corrupted dataset files
        - **Machine Learning Safeguards**: Detects and resolves issues with class imbalance or insufficient data
        - **Service Integration Resilience**: Graceful handling of network errors in translation and audio services
        - **User Interface Feedback**: Clear error messages and progress indicators
        
        Key robustness features:
        
        1. Automatic detection of class diversity issues
        2. Fallback to dummy classifiers when necessary
        3. Exception handling for model training and prediction
        4. Safe handling of feature importance extraction
        """)
    
    # Project Architecture
    st.subheader("Project Architecture")
    
    st.write("""
    The application follows a modular architecture with clear separation of concerns:
    
    ```
    ├── app.py                 # Main Streamlit application
    ├── preprocessing.py       # Data loading and text processing
    ├── genre_prediction.py    # Machine learning models and evaluation
    ├── translation.py         # Text translation functionality
    ├── audio_conversion.py    # Text-to-speech conversion
    └── utils.py               # Helper functions and visualization
    ```
    
    Each module handles a specific aspect of the application, making the codebase maintainable and extensible.
    Session state management ensures efficient data flow between components.
    """)
    
    # Future Enhancements
    st.subheader("Future Enhancement Opportunities")
    
    st.write("""
    Potential areas for future development:
    
    1. **Expanded Language Support**: Add more languages for translation and audio
    2. **Advanced NLP Models**: Integrate transformer-based models for improved genre prediction
    3. **User Account System**: Save user preferences and history
    4. **API Endpoint**: Expose functionality via REST API for integration with other systems
    5. **Custom Audio Voices**: Allow users to select different voices for audio generation
    6. **Cloud Deployment**: Optimize for cloud deployment with distributed processing
    """)
    
    # Academic References
    st.subheader("Academic References")
    
    st.write("""
    This project builds upon established research in the fields of natural language processing, 
    multi-label classification, and machine translation:
    
    1. **CMU Movie Summary Corpus**: The primary dataset used for movie summaries and metadata
       - Bamman, D., O'Connor, B., & Smith, N. (2013). Learning Latent Personas of Film Characters.
    
    2. **Text Classification Techniques**:  
       - Joachims, T. (1998). Text categorization with Support Vector Machines: Learning with many relevant features.  
       - Wang, S., & Manning, C. D. (2012). Baselines and bigrams: Simple, good sentiment and topic classification.  
    
    3. **Multi-label Classification**:  
       - Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification.  
       - Zhang, M. L., & Zhou, Z. H. (2014). A review on multi-label learning algorithms.  
    """)
    
    # Project Statistics
    st.subheader("Project Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Core Python Modules", "5")
        st.metric("ML Dependencies", "scikit-learn, numpy, pandas")
        st.metric("Visualization Libraries", "matplotlib, seaborn")
    with col2:
        st.metric("External APIs", "Google Translate, Google TTS")
        st.metric("UI Framework", "Streamlit")
        st.metric("Dataset Source", "CMU Movie Summary Corpus")
        
    # Download Project Report
    st.subheader("Download Complete Report")
    
    st.markdown("""
    For a complete PDF report on the project, you can download the original project document:
    """)
    
    if os.path.exists("attached_assets/Semester Project - Filmception.pdf"):
        with open("attached_assets/Semester Project - Filmception.pdf", "rb") as file:
            btn = st.download_button(
                label="Download Complete PDF Report",
                data=file,
                file_name="Filmception_Project_Report.pdf",
                mime="application/pdf"
            )
    else:
        st.info("Project PDF report not available for download.")

