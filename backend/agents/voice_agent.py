"""
Voice Agent

Handles audio transcription for voice-based queries.
Uses SpeechRecognition library to convert speech to text.
"""

import logging
import os
from typing import Optional, Tuple
import speech_recognition as sr
from pydub import AudioSegment
import io

logger = logging.getLogger(__name__)

class VoiceAgent:
    """
    Voice Agent for handling audio input.
    """
    
    def __init__(self):
        """Initialize Voice Agent"""
        self.recognizer = sr.Recognizer()
        
    def transcribe(self, audio_file: bytes, filename: str = "audio.wav") -> Tuple[str, float]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_file: Audio file bytes
            filename: Original filename (for extension detection)
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        try:
            # Create a file-like object
            audio_io = io.BytesIO(audio_file)
            
            # Convert to wav if necessary (using pydub if installed)
            # For now, we'll assume the input is compatible or handle basic conversions
            # In a full production env, we'd use ffmpeg/pydub to normalize to WAV
            
            # Load audio file
            with sr.AudioFile(audio_io) as source:
                # Record the audio data
                audio_data = self.recognizer.record(source)
                
                # Recognize speech using Google Speech Recognition (free tier)
                # In production, this should be replaced with a robust model like Whisper
                text = self.recognizer.recognize_google(audio_data)
                
                # Google API doesn't return confidence in the simple response, 
                # but we can assume a high baseline if it succeeded and returned text.
                confidence = 0.9 
                
                logger.info(f"Transcribed audio: {text}")
                return text, confidence
                
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            raise ValueError("Could not understand audio")
        except sr.RequestError as e:
            logger.error(f"Could not request results from speech recognition service; {e}")
            raise RuntimeError(f"Speech service error: {e}")
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {str(e)}")

# Global instance
voice_agent = VoiceAgent()
