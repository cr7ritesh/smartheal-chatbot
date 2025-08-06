import os
import wave
import pyaudio
import whisper
import ffmpeg
import tempfile
import logging
from typing import Dict, List
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import time

load_dotenv()

class SpeechHandler:
    """Handles speech-to-text transcription and speaker diarization"""
    
    def __init__(self):
        self.whisper_model = None
        self.diarization_pipeline = None
        self.supported_languages = ['en', 'hi', 'bn']
        self._load_models()
    
    def _load_models(self):
        """Load Whisper and diarization models"""
        try:
            # Load Whisper model
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            print("Whisper model loaded successfully!")
            
            # Load diarization pipeline if token available
            hf_token = os.environ.get("HUGGINGFACE_TOKEN")
            if hf_token:
                print("Loading speaker diarization model...")
                try:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token
                    )
                    print("Diarization model loaded successfully!")
                except Exception as e:
                    print(f"Warning: Could not load diarization model: {e}")
                    self.diarization_pipeline = None
            else:
                print("Warning: No HuggingFace token found. Diarization disabled.")
                
        except Exception as e:
            logging.error(f"Error loading speech models: {e}")
            raise
    
    def _create_temp_file(self, suffix: str = '.wav') -> str:
        """Create a temporary file and return its path"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_filename = temp_file.name
        temp_file.close()
        return temp_filename
    
    def _cleanup_file(self, file_path: str, max_retries: int = 3) -> bool:
        """Safely cleanup file with retries"""
        if not file_path or not os.path.exists(file_path):
            return True
            
        for attempt in range(max_retries):
            try:
                os.unlink(file_path)
                print(f"Cleaned up file: {file_path}")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Brief delay before retry
                else:
                    logging.warning(f"Could not cleanup file {file_path}: {e}")
                    return False
        return False
    
    def _detect_audio_format(self, file_path: str) -> str:
        """Detect audio file format from header"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(12)
                
            if header.startswith(b'RIFF') and b'WAVE' in header:
                return 'wav'
            elif header.startswith(b'\x1a\x45\xdf\xa3'):
                return 'webm'
            else:
                return 'unknown'
        except Exception:
            return 'unknown'
    
    def _convert_to_wav(self, input_file: str) -> str:
        """Convert audio file to WAV format using ffmpeg"""
        output_file = self._create_temp_file('.wav')
        
        try:
            print(f"Converting to WAV format...")
            (
                ffmpeg
                .input(input_file)
                .output(output_file, format='wav', acodec='pcm_s16le', ar=16000, ac=1)
                .overwrite_output()
                .run(quiet=True)
            )
            print(f"Conversion completed")
            return output_file
        except Exception as e:
            self._cleanup_file(output_file)
            raise Exception(f"Audio conversion failed: {e}")
    
    def record_audio(self, duration: int = 10, sample_rate: int = 16000) -> str:
        """Record audio from microphone"""
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        
        p = pyaudio.PyAudio()
        
        try:
            print(f"Recording for {duration} seconds...")
            
            stream = p.open(
                format=format,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk
            )
            
            frames = []
            for _ in range(0, int(sample_rate / chunk * duration)):
                data = stream.read(chunk)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            # Save to temporary file
            temp_filename = self._create_temp_file('.wav')
            
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            print(f"Recording completed!")
            return temp_filename
            
        finally:
            p.terminate()
    
    def save_uploaded_audio(self, uploaded_file) -> str:
        """Save uploaded audio and convert to WAV if needed"""
        # Get file extension
        original_filename = uploaded_file.filename or "audio.wav"
        extension = os.path.splitext(original_filename)[1] or ".wav"
        
        # Save to temporary file
        temp_filename = self._create_temp_file(extension)
        uploaded_file.save(temp_filename)
        
        file_size = os.path.getsize(temp_filename)
        print(f"Audio file saved. Size: {file_size} bytes")
        
        # Check format and convert if needed
        audio_format = self._detect_audio_format(temp_filename)
        print(f"Detected format: {audio_format}")
        
        if audio_format == 'wav':
            return temp_filename
        else:
            # Convert to WAV
            converted_file = self._convert_to_wav(temp_filename)
            self._cleanup_file(temp_filename)  # Clean up original
            return converted_file
    
    def transcribe_audio(self, audio_file: str, transliterate_to_english: bool = True) -> Dict:
        """Transcribe audio with transliteration to English for non-English languages"""
        if not self.whisper_model:
            raise Exception("Whisper model not loaded")
        
        if not os.path.exists(audio_file):
            raise Exception(f"Audio file not found: {audio_file}")
        
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            raise Exception("Audio file is empty")
        
        print(f"Transcribing audio... (Size: {file_size} bytes)")
        
        try:
            # First, detect the language
            result = self.whisper_model.transcribe(audio_file, fp16=False)
            detected_language = result.get('language', 'unknown')
            print(f"Detected language: {detected_language}")
            
            # If it's Hindi or Bengali, use translate task to get English text
            if transliterate_to_english and detected_language in ['hi', 'bn']:
                print(f"Converting {detected_language} to English text...")
                result = self.whisper_model.transcribe(
                    audio_file, 
                    task="translate",  # This converts to English
                    fp16=False
                )
                print(f"Translation to English completed")
            else:
                print(f"Keeping original language transcription")
            
            return {
                'text': result['text'].strip(),
                'language': detected_language,  # Keep original detected language
                'segments': result.get('segments', [])
            }
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            return {'text': '', 'language': 'unknown', 'segments': []}
    
    def perform_diarization(self, audio_file: str) -> List[Dict]:
        """Perform speaker diarization if available"""
        if not self.diarization_pipeline:
            return [{'speaker': 'Speaker_0', 'start': 0, 'end': 0}]
        
        try:
            print(f"Performing speaker diarization...")
            diarization = self.diarization_pipeline(audio_file)
            
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'speaker': speaker,
                    'start': turn.start,
                    'end': turn.end
                })
            
            print(f"Diarization completed. Found {len(segments)} segments")
            return segments
            
        except Exception as e:
            logging.error(f"Error in diarization: {e}")
            return [{'speaker': 'Speaker_0', 'start': 0, 'end': 0}]
    
    def _combine_transcription_diarization(self, transcription: Dict, diarization: List[Dict]) -> List[Dict]:
        """Combine transcription segments with speaker information"""
        if not diarization or not transcription.get('segments'):
            return [{
                'speaker': 'Speaker_0',
                'text': transcription.get('text', ''),
                'start': 0,
                'end': 0,
                'language': transcription.get('language', 'unknown')
            }]
        
        combined = []
        for segment in transcription['segments']:
            # Find matching speaker
            matching_speaker = 'Speaker_0'
            for diar_seg in diarization:
                if (diar_seg['start'] <= segment['start'] <= diar_seg['end'] or
                    diar_seg['start'] <= segment['end'] <= diar_seg['end']):
                    matching_speaker = diar_seg['speaker']
                    break
            
            combined.append({
                'speaker': matching_speaker,
                'text': segment['text'].strip(),
                'start': segment['start'],
                'end': segment['end'],
                'language': transcription.get('language', 'unknown')
            })
        
        return combined
    
    def process_audio_file(self, audio_file: str) -> Dict:
        """Main method to process audio file with automatic cleanup"""
        try:
            print(f"Processing audio file: {audio_file}")
            
            # Transcribe
            transcription = self.transcribe_audio(audio_file)
            if not transcription['text']:
                raise Exception("No text transcribed from audio")
            
            # Diarize
            diarization = self.perform_diarization(audio_file)
            
            # Combine results
            combined_result = self._combine_transcription_diarization(transcription, diarization)
            
            return {
                'success': True,
                'full_text': transcription['text'],
                'language': transcription['language'],
                'segments': combined_result,
                'speaker_count': len(set(seg['speaker'] for seg in combined_result))
            }
            
        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            return {
                'success': False,
                'error': str(e),
                'full_text': '',
                'language': 'unknown',
                'segments': [],
                'speaker_count': 0
            }
        finally:
            # Always cleanup
            self._cleanup_file(audio_file)

# Initialize global speech handler
speech_handler = SpeechHandler()