import os
import io
import tempfile
from gtts import gTTS

def text_to_speech(text, language_code):
    """
    Convert text to speech using Google Text-to-Speech.
    
    Args:
        text (str): Text to convert to speech
        language_code (str): Language code (e.g., 'ar' for Arabic)
        
    Returns:
        bytes: Audio data as bytes
    """
    try:
        # Language code mapping for gTTS
        language_map = {
            'ar': 'ar',   # Arabic
            'ur': 'ur',   # Urdu
            'ko': 'ko'    # Korean
        }
        
        # Use the mapped language code or default to English
        tts_lang = language_map.get(language_code, 'en')
        
        # Create a bytes buffer to store the audio
        audio_bytes = io.BytesIO()
        
        # Create TTS object
        tts = gTTS(text=text, lang=tts_lang)
        
        # Save to bytes buffer
        tts.write_to_fp(audio_bytes)
        
        # Reset buffer position to beginning
        audio_bytes.seek(0)
        
        return audio_bytes.getvalue()
    
    except Exception as e:
        # Create an error audio message
        error_audio = io.BytesIO()
        error_message = f"Error generating audio: {str(e)}"
        error_tts = gTTS(text=error_message, lang='en')
        error_tts.write_to_fp(error_audio)
        error_audio.seek(0)
        
        return error_audio.getvalue()
