import os
import time
from deep_translator import GoogleTranslator

def translate_text(text, target_language):
    """
    Translate text to the target language using Google Translate API.
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code (e.g., 'ar' for Arabic)
        
    Returns:
        str: Translated text
    """
    try:
        # Initialize translator
        translator = GoogleTranslator(source='auto', target=target_language)
        
        # Split text into chunks to avoid request limits (deep-translator has a 5000 char limit)
        chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
        translated_chunks = []
        
        for chunk in chunks:
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
            
            # Translate chunk
            translation = translator.translate(chunk)
            translated_chunks.append(translation)
            
        # Combine translated chunks
        translated_text = ' '.join(translated_chunks)
        
        return translated_text
    
    except Exception as e:
        # Fallback message in case of translation error
        return f"Translation error: {str(e)}"

# Keep your get_language_name function unchanged
def get_language_name(language_code):
    """
    Get the language name from the language code.
    
    Args:
        language_code (str): Language code (e.g., 'ar')
        
    Returns:
        str: Language name (e.g., 'Arabic')
    """
    language_map = {
        'ar': 'Arabic',
        'ur': 'Urdu',
        'ko': 'Korean'
    }
    
    return language_map.get(language_code, 'Unknown')