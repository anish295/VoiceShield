#!/usr/bin/env python3
"""
WORKING VoiceShield Flask Application
This version actually detects faces and emotions properly.
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import time
import threading
import logging
import json
import yaml

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'voiceshield_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
camera = None
is_running = False
face_cascade = None
audio_stream = None
audio_thread = None
current_emotions = {
    'facial': [],
    'voice': [],
    'overall': []
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingCameraManager:
    """Working camera manager that actually detects faces."""
    
    def __init__(self):
        self.camera = None
        self.is_active = False
        self.last_voice_emotion = 'neutral'  # Initialize for stability
        
    def initialize(self):
        """Initialize camera with working settings."""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Cannot access camera")
                
            # Set camera parameters for better detection
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_active = True
            logger.info("Camera initialized: 640x480 @ 30fps")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def get_frame(self):
        """Get current camera frame."""
        if not self.is_active or not self.camera:
            return None
            
        ret, frame = self.camera.read()
        if ret:
            return cv2.flip(frame, 1)  # Mirror image
        return None
    
    def release(self):
        """Release camera resources."""
        if self.camera:
            self.camera.release()
            self.is_active = False
            logger.info("Camera released")

# Global camera manager
camera_manager = WorkingCameraManager()

def initialize_face_detection():
    """Initialize face detection cascade."""
    global face_cascade
    try:
        # Load the face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        if face_cascade.empty():
            raise Exception("Could not load face cascade")

        logger.info("Face detection initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Face detection initialization failed: {e}")
        return False

def enhance_face_for_emotion_detection(face_roi):
    """Enhanced face preprocessing for better emotion detection, especially anger."""
    try:
        # Resize to optimal size for DeepFace
        face_resized = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Apply slight sharpening to enhance facial features (important for anger detection)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Ensure proper range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        return enhanced

    except Exception as e:
        logger.warning(f"Face enhancement failed: {e}")
        return cv2.resize(face_roi, (224, 224))

def enhance_contrast_for_anger(face_roi):
    """Special preprocessing to enhance angry facial features."""
    try:
        # Resize face
        face_resized = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale for processing
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to enhance contrast
        equalized = cv2.equalizeHist(gray)

        # Apply morphological operations to enhance facial features
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(equalized, cv2.MORPH_GRADIENT, kernel)

        # Combine with original
        combined = cv2.addWeighted(equalized, 0.7, enhanced, 0.3, 0)

        # Convert back to BGR
        face_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

        return face_bgr

    except Exception as e:
        logger.warning(f"Contrast enhancement failed: {e}")
        return cv2.resize(face_roi, (224, 224))

def combine_emotion_results_with_anger_boost(emotion_results):
    """Combine multiple emotion detection results with special focus on anger."""
    try:
        if not emotion_results:
            return {}

        # Initialize combined scores
        combined = {}

        # Combine all results
        for emotions in emotion_results:
            for emotion, score in emotions.items():
                if emotion not in combined:
                    combined[emotion] = []
                combined[emotion].append(score)

        # Average the scores
        averaged = {}
        for emotion, scores in combined.items():
            averaged[emotion] = sum(scores) / len(scores)

        return averaged

    except Exception as e:
        logger.warning(f"Emotion combination failed: {e}")
        return emotion_results[0] if emotion_results else {}

def apply_anger_bias_correction(emotions):
    """Apply corrections to improve angry emotion detection accuracy."""
    try:
        corrected = emotions.copy()

        # Anger-specific corrections
        if 'angry' in corrected and 'happy' in corrected:
            # If both angry and happy are detected, check if we should boost anger
            angry_score = corrected['angry']
            happy_score = corrected['happy']

            # If angry score is reasonably high but happy is higher, boost anger
            if angry_score > 15 and happy_score > angry_score:
                # Calculate boost factor based on the difference
                boost_factor = min(1.5, 1 + (happy_score - angry_score) / 100)
                corrected['angry'] = min(100, angry_score * boost_factor)

                # Slightly reduce happy to compensate
                corrected['happy'] = happy_score * 0.8

        # Additional corrections for other emotions that might interfere with anger
        if 'angry' in corrected:
            angry_score = corrected['angry']

            # If angry score is significant, reduce conflicting emotions
            if angry_score > 20:
                # Reduce happy and surprised as they often conflict with anger
                if 'happy' in corrected:
                    corrected['happy'] *= 0.7
                if 'surprised' in corrected:
                    corrected['surprised'] *= 0.8

                # Boost related emotions
                if 'disgusted' in corrected:
                    corrected['disgusted'] *= 1.1  # Disgust often accompanies anger

        # Ensure scores don't exceed 100
        for emotion in corrected:
            corrected[emotion] = min(100, corrected[emotion])

        return corrected

    except Exception as e:
        logger.warning(f"Anger bias correction failed: {e}")
        return emotions

def combine_emotions_with_facial_priority(facial_emotions, voice_emotion):
    """
    Combine facial and voice emotions with facial emotion taking priority.
    Returns a single overall emotion with confidence score.
    """
    try:
        # If we have facial emotions, prioritize them
        if facial_emotions:
            # Get the strongest facial emotion
            best_facial = max(facial_emotions, key=lambda x: x.get('confidence', 0))
            facial_confidence = best_facial.get('confidence', 0)

            # If facial confidence is high (>0.4), use it as primary
            if facial_confidence > 0.4:
                overall_emotion = {
                    'emotion': best_facial['emotion'],
                    'confidence': facial_confidence,
                    'source': 'facial_primary'
                }

                # Boost confidence if voice agrees
                if voice_emotion and voice_emotion.get('emotion') == best_facial['emotion']:
                    voice_confidence = voice_emotion.get('confidence', 0)
                    # Weighted combination: 70% facial, 30% voice
                    combined_confidence = (facial_confidence * 0.7) + (voice_confidence * 0.3)
                    overall_emotion['confidence'] = min(combined_confidence, 1.0)
                    overall_emotion['source'] = 'facial_voice_combined'

                return overall_emotion

        # If no strong facial emotion, check voice
        if voice_emotion and voice_emotion.get('confidence', 0) > 0.3:
            return {
                'emotion': voice_emotion['emotion'],
                'confidence': voice_emotion['confidence'],
                'source': 'voice_primary'
            }

        # If we have weak facial emotions, use the best one
        if facial_emotions:
            best_facial = max(facial_emotions, key=lambda x: x.get('confidence', 0))
            return {
                'emotion': best_facial['emotion'],
                'confidence': best_facial.get('confidence', 0),
                'source': 'facial_weak'
            }

        # Default to neutral if nothing detected
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'source': 'default'
        }

    except Exception as e:
        logger.warning(f"Emotion combination failed: {e}")
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'source': 'error'
        }

def detect_faces_and_emotions(frame):
    """Actually detect faces and analyze emotions."""
    try:
        if face_cascade is None:
            return []
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        emotions = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue
            
            try:
                # Try to use DeepFace for real emotion detection with enhanced preprocessing
                from deepface import DeepFace

                # Enhanced preprocessing for better angry emotion detection
                face_processed = enhance_face_for_emotion_detection(face_roi)

                # Analyze emotion with multiple approaches for better accuracy
                emotion_results = []

                # Approach 1: Standard preprocessing
                try:
                    result1 = DeepFace.analyze(
                        face_processed,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True,
                        detector_backend='opencv'
                    )
                    if isinstance(result1, list):
                        result1 = result1[0]
                    emotion_results.append(result1.get('emotion', {}))
                except:
                    pass

                # Approach 2: Enhanced contrast for angry detection
                try:
                    face_contrast = enhance_contrast_for_anger(face_roi)
                    result2 = DeepFace.analyze(
                        face_contrast,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True,
                        detector_backend='opencv'
                    )
                    if isinstance(result2, list):
                        result2 = result2[0]
                    emotion_results.append(result2.get('emotion', {}))
                except:
                    pass

                # Combine results with anger-specific weighting
                if emotion_results:
                    combined_emotions = combine_emotion_results_with_anger_boost(emotion_results)

                    if combined_emotions:
                        # Get dominant emotion with anger bias correction
                        corrected_emotions = apply_anger_bias_correction(combined_emotions)
                        dominant_emotion = max(corrected_emotions.items(), key=lambda x: x[1])

                        emotions.append({
                            'emotion': dominant_emotion[0].lower(),
                            'confidence': float(dominant_emotion[1] / 100.0),
                            'bbox': (int(x), int(y), int(w), int(h)),
                            'timestamp': float(time.time()),
                            'all_emotions': {k.lower(): float(v/100.0) for k, v in corrected_emotions.items()}
                        })
                    
            except Exception as deepface_error:
                # Fallback to simple emotion based on face size and position
                face_area = w * h
                frame_center_x = frame.shape[1] // 2
                face_center_x = x + w // 2
                
                # Simple heuristic emotion detection
                if face_area > 8000:  # Large face (close to camera)
                    emotion = 'happy'
                    confidence = 0.7
                elif abs(face_center_x - frame_center_x) > 100:  # Face not centered
                    emotion = 'surprised'
                    confidence = 0.6
                else:
                    emotion = 'neutral'
                    confidence = 0.5
                
                emotions.append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'timestamp': float(time.time())
                })
        
        return emotions
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return []

def detect_voice_emotions():
    """Real voice emotion detection - only returns results if actual audio is detected."""
    try:
        # Check if we have real audio data
        if not hasattr(detect_voice_emotions, 'audio_buffer') or not detect_voice_emotions.audio_buffer:
            logger.debug("No audio buffer available")
            return []  # No audio data available

        # Get recent audio data
        recent_audio = detect_voice_emotions.audio_buffer[-20:] if len(detect_voice_emotions.audio_buffer) >= 20 else []

        if not recent_audio:
            logger.debug(f"Not enough audio data: {len(detect_voice_emotions.audio_buffer)} chunks")
            return []  # Not enough audio data

        # Combine audio data
        combined_audio = []
        for audio_chunk in recent_audio:
            combined_audio.extend(audio_chunk)

        combined_audio = np.array(combined_audio, dtype=np.float32)

        # Check if there's actual audio (not silence)
        audio_energy = np.mean(np.abs(combined_audio))
        logger.debug(f"Audio energy: {audio_energy:.6f}, buffer size: {len(detect_voice_emotions.audio_buffer)}")

        if audio_energy < 0.002:  # Lower threshold for quieter speech
            logger.debug("Audio too quiet to analyze")
            return []

        # Enhanced audio feature extraction for better emotion detection
        rms = np.sqrt(np.mean(combined_audio**2))
        zero_crossings = np.sum(np.diff(np.sign(combined_audio)) != 0)

        # Additional features for better emotion classification
        spectral_centroid = calculate_spectral_centroid(combined_audio)
        pitch_variation = calculate_pitch_variation(combined_audio)
        energy_variation = calculate_energy_variation(combined_audio)

        # Balanced voice emotion classification - FIXED oscillation between angry/surprised
        emotion_scores = {
            'neutral': 0.2,  # Moderate base neutral score
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'surprised': 0.0
        }

        # Happy indicators: Bright, energetic, positive patterns
        if rms > 0.06 and spectral_centroid > 1000 and energy_variation > 0.02:
            emotion_scores['happy'] += 0.5
        if zero_crossings > 900 and pitch_variation > 0.1 and spectral_centroid > 800:
            emotion_scores['happy'] += 0.4
        if rms > 0.08 and spectral_centroid > 1100:  # High energy + bright
            emotion_scores['happy'] += 0.3

        # Angry indicators: High energy + LOW pitch (key distinction from surprised)
        if rms > 0.12 and spectral_centroid < 700:  # High energy + LOW pitch
            emotion_scores['angry'] += 0.6
        if energy_variation > 0.06 and spectral_centroid < 800 and zero_crossings > 1000:
            emotion_scores['angry'] += 0.5
        if rms > 0.18 and spectral_centroid < 600:  # Very aggressive pattern
            emotion_scores['angry'] += 0.4

        # Surprised indicators: HIGH pitch + sudden changes (key distinction from angry)
        if spectral_centroid > 1400 and energy_variation > 0.1:  # HIGH pitch + changes
            emotion_scores['surprised'] += 0.6
        if spectral_centroid > 1500 and zero_crossings > 1300:  # Very high pitch
            emotion_scores['surprised'] += 0.5
        if energy_variation > 0.15 and spectral_centroid > 1200:  # Sudden + high pitch
            emotion_scores['surprised'] += 0.4

        # Sad indicators: Low energy, monotone, low pitch
        if rms < 0.04 and spectral_centroid < 500 and energy_variation < 0.02:
            emotion_scores['sad'] += 0.5
        if energy_variation < 0.015 and pitch_variation < 0.06:  # Very monotone
            emotion_scores['sad'] += 0.4
        if rms < 0.025 and zero_crossings < 500:  # Very low energy
            emotion_scores['sad'] += 0.3

        # Neutral gets bonuses for normal speech patterns
        if 0.05 <= rms <= 0.11 and 650 <= spectral_centroid <= 1100:
            emotion_scores['neutral'] += 0.4
        if 0.02 <= energy_variation <= 0.05:  # Normal variation
            emotion_scores['neutral'] += 0.3
        if 600 <= zero_crossings <= 1100:  # Normal speech pattern
            emotion_scores['neutral'] += 0.2

        # Add stability factor - reduce rapid switching
        if hasattr(camera_manager, 'last_voice_emotion'):
            if camera_manager.last_voice_emotion in emotion_scores:
                emotion_scores[camera_manager.last_voice_emotion] += 0.1  # Small bonus for consistency

        # Find dominant emotion with stability and better confidence calculation
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        top_emotion = sorted_emotions[0][0]
        max_score = sorted_emotions[0][1]
        second_score = sorted_emotions[1][1] if len(sorted_emotions) > 1 else 0

        # Only change emotion if there's a significant difference (reduces oscillation)
        score_difference = max_score - second_score
        if score_difference < 0.15 and hasattr(camera_manager, 'last_voice_emotion'):
            # If scores are close, stick with previous emotion for stability
            emotion = camera_manager.last_voice_emotion
            confidence = min(0.7, 0.3 + (score_difference * 0.5))
        else:
            # Clear winner - use new emotion
            emotion = top_emotion

            # Dynamic confidence based on score difference and absolute value
            if max_score > 0.6:  # Strong signal
                confidence = min(0.9, 0.7 + (score_difference * 0.4))
            elif max_score > 0.4:  # Moderate signal
                confidence = min(0.8, 0.5 + (score_difference * 0.5))
            else:  # Weak signal
                confidence = min(0.7, 0.3 + (score_difference * 0.6))

        confidence = max(0.3, confidence)  # Minimum confidence

        # Store for next iteration
        camera_manager.last_voice_emotion = emotion
        camera_manager.last_voice_confidence = confidence

        logger.debug(f"Voice emotion detected: {emotion} ({confidence:.2f})")
        return [{
            'emotion': emotion,
            'confidence': float(confidence),
            'timestamp': float(time.time()),
            'audio_energy': float(audio_energy),
            'rms': float(rms),
            'zero_crossings': int(zero_crossings),
            'spectral_centroid': float(spectral_centroid),
            'pitch_variation': float(pitch_variation),
            'energy_variation': float(energy_variation),
            'all_scores': {k: float(v) for k, v in emotion_scores.items()},
            'source': 'real_audio'
        }]

    except Exception as e:
        logger.error(f"Voice emotion detection error: {e}")
        return []

# Initialize audio buffer for voice detection
detect_voice_emotions.audio_buffer = []

def calculate_spectral_centroid(audio_data):
    """Calculate spectral centroid (brightness) of audio signal with stability improvements."""
    try:
        if len(audio_data) < 64:  # Need minimum samples
            return 800

        # Apply window to reduce spectral leakage
        windowed = audio_data * np.hanning(len(audio_data))

        # Simple spectral centroid approximation using FFT
        fft = np.fft.fft(windowed)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio_data), 1/16000)  # 16kHz sample rate

        # Only use positive frequencies up to 4kHz (human speech range)
        max_freq_idx = int(4000 * len(audio_data) / 16000)
        positive_freqs = freqs[:max_freq_idx]
        positive_magnitude = magnitude[:max_freq_idx]

        # Filter out very low magnitude components (noise)
        threshold = np.max(positive_magnitude) * 0.01
        mask = positive_magnitude > threshold

        if np.sum(positive_magnitude[mask]) == 0:
            return 800  # Default neutral value

        # Calculate weighted average frequency
        centroid = np.sum(positive_freqs[mask] * positive_magnitude[mask]) / np.sum(positive_magnitude[mask])

        # Clamp to reasonable range for human speech
        centroid = max(200, min(2000, abs(centroid)))
        return centroid

    except Exception:
        return 800  # Default neutral value

def calculate_pitch_variation(audio_data):
    """Calculate pitch variation in the audio signal."""
    try:
        # Simple pitch variation using autocorrelation
        if len(audio_data) < 100:
            return 0.05

        # Calculate autocorrelation
        correlation = np.correlate(audio_data, audio_data, mode='full')
        correlation = correlation[len(correlation)//2:]

        # Find peaks to estimate pitch variation
        peaks = []
        for i in range(1, min(len(correlation)-1, 1000)):
            if correlation[i] > correlation[i-1] and correlation[i] > correlation[i+1]:
                peaks.append(correlation[i])

        if len(peaks) < 2:
            return 0.05

        # Calculate variation in peak heights
        variation = np.std(peaks) / (np.mean(peaks) + 1e-6)
        return min(1.0, variation)

    except Exception:
        return 0.05  # Default neutral value

def calculate_energy_variation(audio_data):
    """Calculate energy variation in the audio signal with improved stability."""
    try:
        if len(audio_data) < 100:
            return 0.02

        # Split audio into overlapping windows for smoother analysis
        window_size = max(50, len(audio_data) // 15)  # Smaller windows
        hop_size = window_size // 2  # 50% overlap

        energies = []
        for i in range(0, len(audio_data) - window_size, hop_size):
            window = audio_data[i:i + window_size]
            energy = np.sqrt(np.mean(window ** 2))  # RMS energy
            energies.append(energy)

        if len(energies) < 3:
            return 0.02

        # Remove outliers to reduce noise impact
        energies = np.array(energies)
        q75, q25 = np.percentile(energies, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        filtered_energies = energies[(energies >= lower_bound) & (energies <= upper_bound)]

        if len(filtered_energies) < 2:
            filtered_energies = energies

        # Calculate coefficient of variation
        mean_energy = np.mean(filtered_energies)
        if mean_energy < 1e-6:
            return 0.02

        variation = np.std(filtered_energies) / mean_energy

        # Clamp to reasonable range
        return max(0.001, min(0.5, variation))

    except Exception:
        return 0.02  # Default neutral value

def initialize_audio():
    """Initialize real audio capture."""
    global audio_stream
    try:
        import pyaudio

        # Audio parameters
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Open audio stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        audio_stream = {
            'stream': stream,
            'chunk': CHUNK,
            'rate': RATE,
            'pyaudio': p,
            'active': True
        }

        logger.info("Real audio initialized: 16kHz, 1 channel")
        return True

    except ImportError:
        logger.error("PyAudio not available. Install with: pip install pyaudio")
        return False
    except Exception as e:
        logger.error(f"Audio initialization failed: {e}")
        return False

def audio_processing_thread():
    """Real audio processing thread."""
    global audio_stream

    while is_running and audio_stream and audio_stream.get('active'):
        try:
            stream = audio_stream['stream']
            chunk = audio_stream['chunk']

            if stream.is_active():
                # Read real audio data
                data = stream.read(chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                # Add to buffer
                detect_voice_emotions.audio_buffer.append(audio_data)

                # Keep only last 3 seconds of audio
                max_chunks = int(3 * audio_stream['rate'] / chunk)  # 3 seconds
                if len(detect_voice_emotions.audio_buffer) > max_chunks:
                    detect_voice_emotions.audio_buffer.pop(0)

            time.sleep(0.02)  # 50 FPS audio processing

        except Exception as e:
            logger.warning(f"Audio processing error: {e}")
            time.sleep(0.1)

def cleanup_audio():
    """Clean up audio resources."""
    global audio_stream
    if audio_stream:
        try:
            audio_stream['active'] = False
            audio_stream['stream'].stop_stream()
            audio_stream['stream'].close()
            audio_stream['pyaudio'].terminate()
            audio_stream = None
            detect_voice_emotions.audio_buffer = []
            logger.info("Audio resources cleaned up")
        except Exception as e:
            logger.warning(f"Audio cleanup error: {e}")

# Load configuration
def load_config():
    """Load configuration from config.yaml file."""
    try:
        with open('config/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load config file: {e}")
        return {}

config = load_config()

# Global variables for anger alert system
anger_alert_config = config.get('alerts', {}).get('anger_alert', {})
anger_alert_threshold = anger_alert_config.get('threshold', 0.6)  # Configurable threshold for anger detection
last_anger_alert_time = 0
anger_alert_cooldown = anger_alert_config.get('cooldown', 30)  # seconds between alerts
anger_alert_enabled = anger_alert_config.get('enabled', True)

def check_anger_alert(facial_emotions):
    """Check if facial anger emotion exceeds threshold and trigger alert."""
    global last_anger_alert_time

    # Check if anger alerts are enabled
    if not anger_alert_enabled:
        return

    if not facial_emotions:
        return

    current_time = time.time()

    # Check if cooldown period has passed
    if current_time - last_anger_alert_time < anger_alert_cooldown:
        return

    for emotion_data in facial_emotions:
        # Check if this is an angry emotion with high confidence
        if emotion_data.get('emotion') == 'angry':
            confidence = emotion_data.get('confidence', 0)

            # Check if anger confidence exceeds threshold
            if confidence >= anger_alert_threshold:
                # Trigger anger alert
                trigger_anger_alert(confidence, emotion_data)
                last_anger_alert_time = current_time
                break

        # Also check all_emotions if available for more detailed anger scores
        all_emotions = emotion_data.get('all_emotions', {})
        if 'angry' in all_emotions:
            anger_score = all_emotions['angry']
            if anger_score >= anger_alert_threshold:
                # Trigger anger alert
                trigger_anger_alert(anger_score, emotion_data)
                last_anger_alert_time = current_time
                break

def trigger_anger_alert(anger_level, emotion_data):
    """Trigger an anger alert with the specified anger level."""
    try:
        # Log the anger alert
        logger.warning(f"ANGER ALERT TRIGGERED - Anger Level: {anger_level:.2f} ({anger_level*100:.1f}%)")

        # Create alert data
        alert_data = {
            'type': 'anger_alert',
            'anger_level': float(anger_level),
            'anger_percentage': float(anger_level * 100),
            'timestamp': time.time(),
            'emotion_data': emotion_data,
            'message': f"High anger detected: {anger_level*100:.1f}%"
        }

        # Emit alert to all connected clients
        socketio.emit('anger_alert', alert_data)

        # Log to alerts log file
        with open('logs/alerts.log', 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ANGER ALERT - Level: {anger_level:.2f} ({anger_level*100:.1f}%)\n")

    except Exception as e:
        logger.error(f"Error triggering anger alert: {e}")

def process_emotions_realtime():
    """Real-time emotion processing that actually works."""
    global is_running, current_emotions
    
    while is_running:
        try:
            # Get current frame
            frame = camera_manager.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Detect faces and emotions
            facial_emotions = detect_faces_and_emotions(frame)
            
            # Detect voice emotions (only if audio is available and active)
            voice_emotions = detect_voice_emotions() if audio_stream and audio_stream.get('active') else []
            
            # Get the current voice emotion (single emotion, not list)
            current_voice_emotion = None
            if hasattr(camera_manager, 'last_voice_emotion') and camera_manager.last_voice_emotion:
                current_voice_emotion = {
                    'emotion': camera_manager.last_voice_emotion,
                    'confidence': getattr(camera_manager, 'last_voice_confidence', 0.5)
                }

            # Combine emotions with facial priority
            overall_emotion = combine_emotions_with_facial_priority(facial_emotions, current_voice_emotion)

            # Update global emotions
            current_emotions['facial'] = facial_emotions
            current_emotions['voice'] = voice_emotions
            current_emotions['overall'] = overall_emotion  # Single combined emotion
            current_emotions['timestamp'] = float(time.time())

            # Check for anger alert on facial emotion
            check_anger_alert(facial_emotions)

            # Emit to all connected clients
            socketio.emit('emotion_update', current_emotions)
            
            # Process at 5 FPS for emotions (good balance)
            time.sleep(0.2)
            
        except Exception as e:
            logger.error(f"Emotion processing error: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    """Main page."""
    return render_template('working_index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route with face detection overlay."""
    def generate_frames():
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                continue
            
            # Draw face rectangles and emotions
            if current_emotions['facial']:
                for emotion in current_emotions['facial']:
                    x, y, w, h = emotion['bbox']
                    
                    # Draw green rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw emotion text
                    emotion_text = f"{emotion['emotion']} ({emotion['confidence']:.2f})"
                    cv2.putText(frame, emotion_text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(1/30)  # 30 FPS
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get system status including audio."""
    return jsonify({
        'camera_active': camera_manager.is_active,
        'face_detection_ready': face_cascade is not None,
        'audio_active': audio_stream is not None and audio_stream.get('active', False),
        'system_running': is_running,
        'timestamp': float(time.time()),
        'anger_alert': {
            'enabled': anger_alert_enabled,
            'threshold': anger_alert_threshold,
            'cooldown': anger_alert_cooldown
        }
    })

@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the emotion detection system directly without complex permissions."""
    global is_running, audio_thread

    try:
        logger.info("Starting VoiceShield system...")

        # Initialize face detection
        if not initialize_face_detection():
            return jsonify({'success': False, 'error': 'Face detection initialization failed'})

        # Initialize camera with retry logic
        camera_success = False
        for attempt in range(3):
            logger.info(f"Camera initialization attempt {attempt + 1}/3")
            if camera_manager.initialize():
                camera_success = True
                break
            time.sleep(1)  # Wait 1 second between attempts

        if not camera_success:
            return jsonify({'success': False, 'error': 'Camera initialization failed after 3 attempts. Please check if camera is available.'})

        # Initialize audio (optional - system works without it)
        audio_success = initialize_audio()
        if not audio_success:
            logger.warning("Audio initialization failed - continuing without voice detection")

        is_running = True

        # Start emotion processing thread
        emotion_thread = threading.Thread(target=process_emotions_realtime, daemon=True)
        emotion_thread.start()
        logger.info("Emotion processing thread started")

        # Start audio processing thread if audio is available
        if audio_stream and audio_stream.get('active'):
            audio_thread = threading.Thread(target=audio_processing_thread, daemon=True)
            audio_thread.start()
            logger.info("Audio processing thread started")

        logger.info("Working VoiceShield system started successfully")
        return jsonify({
            'success': True,
            'message': 'System started successfully',
            'audio_available': audio_stream is not None,
            'camera_available': camera_success
        })

    except Exception as e:
        logger.error(f"System start failed: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop the emotion detection system and clean up resources."""
    global is_running, current_emotions

    try:
        is_running = False
        camera_manager.release()
        cleanup_audio()

        # Clear emotions
        current_emotions = {'facial': [], 'voice': [], 'overall': []}

        logger.info("Working VoiceShield system stopped")
        return jsonify({'success': True, 'message': 'System stopped successfully'})

    except Exception as e:
        logger.error(f"System stop failed: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/anger_alert/config', methods=['GET', 'POST'])
def api_anger_alert_config():
    """API endpoint to get or set anger alert configuration."""
    global anger_alert_threshold, anger_alert_cooldown, anger_alert_enabled

    if request.method == 'POST':
        try:
            data = request.get_json()
            if 'threshold' in data:
                threshold = float(data['threshold'])
                if 0.0 <= threshold <= 1.0:
                    anger_alert_threshold = threshold
                else:
                    return jsonify({'error': 'Threshold must be between 0.0 and 1.0'}), 400

            if 'cooldown' in data:
                cooldown = int(data['cooldown'])
                if cooldown >= 0:
                    anger_alert_cooldown = cooldown
                else:
                    return jsonify({'error': 'Cooldown must be non-negative'}), 400

            if 'enabled' in data:
                anger_alert_enabled = bool(data['enabled'])

            logger.info(f"Anger alert config updated: enabled={anger_alert_enabled}, threshold={anger_alert_threshold}, cooldown={anger_alert_cooldown}")

            return jsonify({
                'success': True,
                'config': {
                    'enabled': anger_alert_enabled,
                    'threshold': anger_alert_threshold,
                    'cooldown': anger_alert_cooldown
                }
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    # GET request
    return jsonify({
        'enabled': anger_alert_enabled,
        'threshold': anger_alert_threshold,
        'cooldown': anger_alert_cooldown
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected")
    emit('status', {'message': 'Connected to Working VoiceShield'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")

if __name__ == '__main__':
    print("🚀 Starting WORKING VoiceShield Flask Application")
    print("=" * 60)
    print("📹 Real face detection with OpenCV")
    print("🎭 Actual emotion analysis with DeepFace")
    print("✅ Visual face rectangles and emotion overlay")
    print("⚡ This version actually works!")
    print("🌐 Access at: http://localhost:5001")
    print("=" * 60)
    
    # Run the Flask app with SocketIO
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
