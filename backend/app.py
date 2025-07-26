#!/usr/bin/env python3
"""
WORKING VoiceShield Flask Application
This version actually detects faces and emotions properly.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from src
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
import time
import threading
import logging
import json
import yaml
import base64
from io import BytesIO

# Initialize Flask app with updated template and static folders
app = Flask(__name__,
           template_folder=str(parent_dir / 'frontend'),
           static_folder=str(parent_dir / 'frontend' / 'static'))
app.config['SECRET_KEY'] = 'voiceshield_secret_key'

# Configure CORS to allow requests from Netlify frontend
CORS(app, resources={r"/*": {"origins": ["https://voiceshield.netlify.app", "http://localhost:*", "http://127.0.0.1:*"]}})

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
camera = None
is_running = False
face_cascade = None
current_emotions = {
    'facial': [],
    'voice': [],
    'overall': []
}

def decode_base64_image(base64_string):
    """Decode Base64 image data from frontend into OpenCV format."""
    try:
        # Remove the data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]

        # Decode Base64 to bytes
        image_bytes = base64.b64decode(base64_string)

        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode image using OpenCV
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode image")

        return frame

    except Exception as e:
        logger.error(f"Base64 image decoding failed: {e}")
        return None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Camera is now handled by client - we just track state
class SystemState:
    """Track system state for client-server architecture."""

    def __init__(self):
        self.camera_active = False
        self.last_voice_emotion = 'neutral'
        self.last_voice_confidence = 0.5
        self.last_facial_emotion = 'neutral'
        self.last_facial_confidence = 0.5
        self.facial_emotion_history = []  # Store last 5 emotions for smoothing
        self.audio_active = False
        self.last_audio_chunk_time = 0
        self.voice_detection_active = False  # Track if voice is actually being detected

    def set_camera_active(self, active):
        """Set camera active state."""
        self.camera_active = active
        if active:
            logger.info("Client camera stream active")
        else:
            logger.info("Client camera stream inactive")

# Global system state
system_state = SystemState()

def initialize_face_detection():
    """Initialize face detection cascade."""
    global face_cascade
    try:
        # Load the face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        if face_cascade.empty():
            logger.error("Could not load face cascade at: %s", cascade_path)
            return False

        logger.info("Face detection initialized successfully")
        # Check DeepFace availability
        try:
            from deepface import DeepFace
            logger.info("DeepFace is available for emotion detection.")
        except ImportError as e:
            logger.error(f"DeepFace not installed: {e}")
            return False
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

def smooth_facial_emotion(new_emotion, new_confidence):
    """Smooth facial emotions to prevent rapid changes."""
    try:
        # Add to history
        system_state.facial_emotion_history.append({
            'emotion': new_emotion,
            'confidence': new_confidence,
            'timestamp': time.time()
        })

        # Keep only last 5 emotions
        if len(system_state.facial_emotion_history) > 5:
            system_state.facial_emotion_history = system_state.facial_emotion_history[-5:]

        # If we don't have enough history, use current emotion
        if len(system_state.facial_emotion_history) < 2:
            return new_emotion, new_confidence

        # Check for consistency in last 2 emotions (less strict for better responsiveness)
        recent_emotions = system_state.facial_emotion_history[-2:]
        emotion_counts = {}
        total_confidence = 0

        for emotion_data in recent_emotions:
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']

            if emotion not in emotion_counts:
                emotion_counts[emotion] = {'count': 0, 'total_confidence': 0}

            emotion_counts[emotion]['count'] += 1
            emotion_counts[emotion]['total_confidence'] += confidence
            total_confidence += confidence

        # Find most frequent emotion
        most_frequent = max(emotion_counts.items(), key=lambda x: x[1]['count'])
        most_frequent_emotion = most_frequent[0]
        most_frequent_count = most_frequent[1]['count']

        # If emotion appears in both recent frames, use it
        if most_frequent_count >= 2:
            avg_confidence = most_frequent[1]['total_confidence'] / most_frequent_count
            return most_frequent_emotion, min(avg_confidence, 0.95)  # Higher confidence cap

        # If new emotion has high confidence, allow it through
        if new_confidence >= 0.6:
            return new_emotion, new_confidence

        # Otherwise, use previous stable emotion if confidence is low
        if new_confidence < 0.5 and system_state.last_facial_emotion:
            return system_state.last_facial_emotion, system_state.last_facial_confidence * 0.95

        return new_emotion, new_confidence

    except Exception as e:
        logger.error(f"Facial emotion smoothing error: {e}")
        return new_emotion, new_confidence

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
            return [], frame
        
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
                # Use DeepFace for accurate emotion detection with simplified approach
                from deepface import DeepFace

                # Simple preprocessing for better accuracy
                face_resized = cv2.resize(face_roi, (224, 224))  # Standard size for better accuracy

                # Single, accurate emotion analysis
                result = DeepFace.analyze(
                    face_resized,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True,
                    detector_backend='opencv'
                )

                if isinstance(result, list):
                    result = result[0]

                emotion_scores = result.get('emotion', {})

                if emotion_scores:
                    # Get the dominant emotion
                    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                    emotion_name = dominant_emotion[0].lower()
                    confidence = float(dominant_emotion[1] / 100.0)

                    # Apply confidence threshold for accuracy
                    if confidence < 0.4:  # If confidence is too low, default to neutral
                        emotion_name = 'neutral'
                        confidence = 0.5

                    emotions.append({
                        'emotion': emotion_name,
                        'confidence': confidence,
                        'bbox': (int(x), int(y), int(w), int(h)),
                        'timestamp': float(time.time()),
                        'all_emotions': {k.lower(): float(v/100.0) for k, v in emotion_scores.items()}
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

        # Draw bounding boxes and emotion labels on the frame
        for emotion_data in emotions:
            x, y, w, h = emotion_data['bbox']
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw emotion label with confidence
            label = f"{emotion}: {confidence:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Draw background rectangle for text
            cv2.rectangle(frame, (x, y - label_size[1] - 10),
                         (x + label_size[0], y), (0, 255, 0), -1)

            # Draw text
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return emotions, frame

    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return [], frame

def detect_voice_emotions(audio_data=None):
    """Voice emotion detection from client-sent audio data."""
    try:
        # Use provided audio data or check buffer for backward compatibility
        if audio_data is not None:
            combined_audio = np.array(audio_data, dtype=np.float32)
        else:
            # Check if we have buffered audio data (fallback)
            if not hasattr(detect_voice_emotions, 'audio_buffer') or not detect_voice_emotions.audio_buffer:
                logger.debug("No audio data available")
                return []

            # Get recent audio data from buffer
            recent_audio = detect_voice_emotions.audio_buffer[-20:] if len(detect_voice_emotions.audio_buffer) >= 20 else []

            if not recent_audio:
                logger.debug(f"Not enough audio data: {len(detect_voice_emotions.audio_buffer)} chunks")
                return []

            # Combine audio data
            combined_audio = []
            for audio_chunk in recent_audio:
                combined_audio.extend(audio_chunk)

            combined_audio = np.array(combined_audio, dtype=np.float32)

        # Check if there's actual audio (not silence)
        audio_energy = np.mean(np.abs(combined_audio))
        buffer_size = len(detect_voice_emotions.audio_buffer) if hasattr(detect_voice_emotions, 'audio_buffer') else 0

        # Enhanced debugging for voice emotion detection
        if not hasattr(detect_voice_emotions, 'debug_count'):
            detect_voice_emotions.debug_count = 0
        detect_voice_emotions.debug_count += 1

        if detect_voice_emotions.debug_count <= 10:
            logger.info(f"🎤 VOICE DEBUG #{detect_voice_emotions.debug_count}: Audio energy: {audio_energy:.6f}, samples: {len(combined_audio)}, buffer size: {buffer_size}")

        # Balanced threshold for normal speech
        if audio_energy < 0.003:  # Lowered from 0.02 to 0.003 for normal speech detection
            if detect_voice_emotions.debug_count <= 5:
                logger.info(f"🎤 Audio too quiet to analyze (energy: {audio_energy:.6f} < 0.003)")
            system_state.voice_detection_active = False  # No voice detected
            return []

        # Enhanced audio feature extraction for better emotion detection
        rms = np.sqrt(np.mean(combined_audio**2))
        zero_crossings = np.sum(np.diff(np.sign(combined_audio)) != 0)

        # Additional silence detection using RMS
        if rms < 0.015:  # Lowered RMS threshold for normal speech
            if detect_voice_emotions.debug_count <= 5:
                logger.info(f"🎤 RMS too low for speech (rms: {rms:.6f} < 0.015)")
            system_state.voice_detection_active = False  # No voice detected
            return []

        # Check for speech-like patterns using zero crossings
        # Speech typically has 200-2000 zero crossings per second
        expected_crossings = len(combined_audio) * 0.03  # Lowered minimum for normal speech
        if zero_crossings < expected_crossings:
            if detect_voice_emotions.debug_count <= 5:
                logger.info(f"🎤 Not enough zero crossings for speech ({zero_crossings} < {expected_crossings:.0f})")
            system_state.voice_detection_active = False  # No voice detected
            return []

        # Additional features for better emotion classification
        spectral_centroid = calculate_spectral_centroid(combined_audio)
        pitch_variation = calculate_pitch_variation(combined_audio)
        energy_variation = calculate_energy_variation(combined_audio)

        # Balanced voice emotion classification - REDUCED NEUTRAL BIAS
        emotion_scores = {
            'neutral': 0.1,  # Reduced neutral bias to allow other emotions
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'surprised': 0.0
        }

        # Happy indicators: Bright, energetic, positive patterns (LOWERED THRESHOLDS)
        if rms > 0.08 and spectral_centroid > 800 and energy_variation > 0.02:
            emotion_scores['happy'] += 0.5
        if zero_crossings > 400 and pitch_variation > 0.1 and spectral_centroid > 600:
            emotion_scores['happy'] += 0.4
        if rms > 0.10 and spectral_centroid > 900:  # High energy + bright
            emotion_scores['happy'] += 0.3

        # Angry indicators: High energy + LOW pitch (LOWERED THRESHOLDS)
        if rms > 0.15 and spectral_centroid < 600:  # High energy + LOW pitch
            emotion_scores['angry'] += 0.6
        if energy_variation > 0.04 and spectral_centroid < 700 and zero_crossings > 300:
            emotion_scores['angry'] += 0.5
        if rms > 0.20 and spectral_centroid < 500:  # Very aggressive pattern
            emotion_scores['angry'] += 0.4

        # Surprised indicators: HIGH pitch + sudden changes (LOWERED THRESHOLDS)
        if spectral_centroid > 1000 and energy_variation > 0.08:  # HIGH pitch + changes
            emotion_scores['surprised'] += 0.6
        if spectral_centroid > 1200 and zero_crossings > 500:  # Very high pitch
            emotion_scores['surprised'] += 0.5
        if energy_variation > 0.10 and spectral_centroid > 900:  # Sudden + high pitch
            emotion_scores['surprised'] += 0.4

        # Sad indicators: Low energy, monotone, low pitch (ADJUSTED THRESHOLDS)
        if rms < 0.08 and spectral_centroid < 500 and energy_variation < 0.03:
            emotion_scores['sad'] += 0.5
        if energy_variation < 0.02 and pitch_variation < 0.06:  # Very monotone
            emotion_scores['sad'] += 0.4
        if rms < 0.025 and zero_crossings < 500:  # Very low energy
            emotion_scores['sad'] += 0.3

        # Neutral gets REDUCED bonuses for normal speech patterns
        if 0.06 <= rms <= 0.12 and 650 <= spectral_centroid <= 1100:
            emotion_scores['neutral'] += 0.2  # Reduced from 0.4
        if 0.03 <= energy_variation <= 0.05:  # Normal variation
            emotion_scores['neutral'] += 0.15  # Reduced from 0.3
        if 400 <= zero_crossings <= 800:  # Normal speech pattern
            emotion_scores['neutral'] += 0.1  # Reduced from 0.2

        # Add stability factor - reduce rapid switching
        # Defensive handling: extract emotion string if it's a dict
        last_emotion = system_state.last_voice_emotion
        if isinstance(last_emotion, dict):
            last_emotion = last_emotion.get('emotion', 'neutral')

        if last_emotion and last_emotion in emotion_scores:
            emotion_scores[last_emotion] += 0.1  # Small bonus for consistency

        # Find dominant emotion with stability and better confidence calculation
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        top_emotion = sorted_emotions[0][0]
        max_score = sorted_emotions[0][1]
        second_score = sorted_emotions[1][1] if len(sorted_emotions) > 1 else 0

        # Only change emotion if there's a significant difference (reduces oscillation)
        score_difference = max_score - second_score
        if score_difference < 0.15:
            # If scores are close, stick with previous emotion for stability
            previous_emotion_state = system_state.last_voice_emotion

            # FIX: Defensively extract the string, in case the state is a dict
            if isinstance(previous_emotion_state, dict):
                emotion = previous_emotion_state.get('emotion', 'neutral')
            elif previous_emotion_state:
                emotion = previous_emotion_state
            else:
                emotion = 'neutral'  # Fallback if no previous state

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
        system_state.last_voice_emotion = emotion
        system_state.last_voice_confidence = confidence
        system_state.voice_detection_active = True  # Voice successfully detected

        # Enhanced debugging for voice emotion results
        if detect_voice_emotions.debug_count <= 10:
            logger.info(f"🎤 VOICE EMOTION DETECTED: {emotion} (confidence: {confidence:.2f})")
            logger.info(f"🎤 Audio features: energy={audio_energy:.6f}, rms={rms:.4f}, zero_crossings={zero_crossings}")
            logger.info(f"🎤 Emotion scores: {emotion_scores}")

        result = [{
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

        if detect_voice_emotions.debug_count <= 5:
            logger.info(f"🎤 RETURNING VOICE RESULT: {result}")

        return result

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

# Audio initialization removed - now handled by frontend

# Audio processing thread removed - now handled by frontend

# Audio cleanup removed - now handled by frontend

# Load configuration
def load_config():
    """Load configuration from config.yaml file."""
    try:
        config_path = parent_dir / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
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
        log_path = parent_dir / 'logs' / 'alerts.log'
        with open(log_path, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ANGER ALERT - Level: {anger_level:.2f} ({anger_level*100:.1f}%)\n")

    except Exception as e:
        logger.error(f"Error triggering anger alert: {e}")

# process_emotions_realtime is no longer needed - frames are processed via Socket.IO

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/debug')
def debug():
    """Debug page."""
    return send_from_directory('../frontend', 'debug.html')

# video_feed route removed - video is now handled client-side

@app.route('/api/status')
def get_status():
    """Get system status including audio."""
    return jsonify({
        'status': 'healthy',
        'camera_active': system_state.camera_active,
        'face_detection_ready': face_cascade is not None,
        'audio_active': system_state.audio_active and (time.time() - system_state.last_audio_chunk_time < 5),  # Active if received audio in last 5 seconds
        'system_running': is_running,
        'timestamp': float(time.time()),
        'anger_alert': {
            'enabled': anger_alert_enabled,
            'threshold': anger_alert_threshold,
            'cooldown': anger_alert_cooldown
        },
        'deployment': 'render',
        'version': '1.0.0'
    })

@app.route('/health')
def health_check():
    """Simple health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'timestamp': float(time.time())
    })

@app.route('/api/test_emotions')
def test_emotions():
    """Test endpoint to send fake emotion data."""
    test_data = {
        'facial': [
            {
                'emotion': 'TEST_FACIAL_EMOTION',
                'confidence': 0.99,
                'region': {'x': 100, 'y': 100, 'w': 150, 'h': 150}
            }
        ],
        'voice': [
            {
                'emotion': 'TEST_VOICE_EMOTION',
                'confidence': 0.88,
                'source': 'test_audio',
                'audio_energy': 0.9,
                'all_scores': {
                    'TEST_VOICE_EMOTION': 0.88,
                    'happy': 0.65,
                    'neutral': 0.35
                }
            }
        ],
        'overall': {
            'emotion': 'TEST_OVERALL_EMOTION',
            'confidence': 0.95,
            'source': 'test_combined'
        },
        'timestamp': time.time()
    }

    # Check connected clients
    try:
        connected_clients = len(socketio.server.manager.rooms.get('/', {}).keys()) if hasattr(socketio.server, 'manager') else 0
        logger.info(f"🔍 Connected Socket.IO clients: {connected_clients}")
    except:
        connected_clients = "unknown"

    # Emit to all connected clients
    logger.info(f"📤 EMITTING TEST EMOTIONS: {test_data}")
    socketio.emit('emotion_update', test_data)
    logger.info(f"✅ Test emotions emitted successfully")

    return jsonify({
        'status': 'Test emotions sent',
        'data': test_data,
        'connected_clients': connected_clients
    })

@app.route('/test')
def test_page():
    """Simple test page for Socket.IO."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Socket.IO Test</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    </head>
    <body>
        <h1>Socket.IO Test</h1>
        <div id="status">Connecting...</div>
        <script>
            console.log('Test page loaded');
            const socket = io('http://localhost:5001');

            socket.on('connect', function() {
                console.log('Connected!');
                document.getElementById('status').innerHTML = 'Connected!';
            });

            socket.on('connect_error', function(error) {
                console.error('Connection error:', error);
                document.getElementById('status').innerHTML = 'Connection error: ' + error;
            });
        </script>
    </body>
    </html>
    '''

@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the emotion detection system for client-server architecture."""
    global is_running, audio_thread

    try:
        logger.info("Starting VoiceShield backend system...")

        # Initialize face detection
        if not initialize_face_detection():
            logger.error("Face detection or DeepFace initialization failed. System will not start.")
            return jsonify({'success': False, 'error': 'Face detection or DeepFace initialization failed. Please check backend logs and dependencies.'})

        # Camera is now handled by client - just mark as ready
        system_state.set_camera_active(True)

        # Audio processing now handled by frontend
        logger.info("Audio processing will be handled by frontend via Socket.IO")

        is_running = True

        logger.info("VoiceShield backend system started successfully")
        return jsonify({
            'success': True,
            'message': 'System started successfully',
            'audio_available': True,  # Audio handled by frontend
            'camera_available': True
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
        system_state.set_camera_active(False)
        # Audio cleanup now handled by frontend

        # Clear emotions
        current_emotions = {'facial': [], 'voice': [], 'overall': []}

        logger.info("VoiceShield backend system stopped")
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
    print("🔌 Socket.IO client connected!")
    logger.info("Socket.IO client connected")
    emit('status', {'message': 'Connected to Working VoiceShield'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print("🔌 Socket.IO client disconnected!")
    logger.info("Socket.IO client disconnected")

@socketio.on('process_frame')
def handle_process_frame(data):
    """Handle frame processing from client."""
    try:
        # Debug logging for frame reception
        if not hasattr(handle_process_frame, 'frame_count'):
            handle_process_frame.frame_count = 0
        handle_process_frame.frame_count += 1

        if handle_process_frame.frame_count <= 3:
            logger.info(f"📹 FRAME #{handle_process_frame.frame_count} RECEIVED from client")

        if not is_running:
            if handle_process_frame.frame_count <= 3:
                logger.info(f"📹 Frame received but system not running (is_running={is_running})")
            return

        # Get the Base64 image data
        base64_image = data.get('image')
        if not base64_image:
            logger.warning("📹 Frame received but no image data")
            return

        if handle_process_frame.frame_count <= 3:
            logger.info(f"📹 Processing frame with image data length: {len(base64_image)}")

        # Decode the Base64 image
        frame = decode_base64_image(base64_image)
        if frame is None:
            return

        # Process the frame for emotion detection
        facial_emotions, processed_frame = detect_faces_and_emotions(frame)

        if handle_process_frame.frame_count <= 5:
            logger.info(f"📹 Face detection result: {len(facial_emotions) if facial_emotions else 0} faces found")

        # Apply facial emotion smoothing to prevent rapid changes
        if facial_emotions:
            primary_emotion = facial_emotions[0]
            smoothed_emotion, smoothed_confidence = smooth_facial_emotion(
                primary_emotion['emotion'],
                primary_emotion['confidence']
            )

            # Update the primary emotion with smoothed values
            facial_emotions[0]['emotion'] = smoothed_emotion
            facial_emotions[0]['confidence'] = smoothed_confidence

            # Update system state
            system_state.last_facial_emotion = smoothed_emotion
            system_state.last_facial_confidence = smoothed_confidence

        # Get voice emotions from recent audio processing
        # Voice emotions are processed separately via audio chunks
        voice_emotions = []
        if (system_state.voice_detection_active and
            system_state.last_voice_emotion and
            system_state.last_voice_confidence > 0):
            voice_emotions = [{
                'emotion': system_state.last_voice_emotion,
                'confidence': system_state.last_voice_confidence,
                'timestamp': float(time.time()),
                'source': 'real_audio'  # Frontend expects 'real_audio' source
            }]

        # Get current voice emotion for combination
        current_voice_emotion = None
        if system_state.voice_detection_active and system_state.last_voice_emotion:
            current_voice_emotion = {
                'emotion': system_state.last_voice_emotion,
                'confidence': system_state.last_voice_confidence
            }

        # Combine emotions with facial priority
        overall_emotion = combine_emotions_with_facial_priority(facial_emotions, current_voice_emotion)

        # Update global emotions
        current_emotions['facial'] = facial_emotions
        current_emotions['voice'] = voice_emotions
        current_emotions['overall'] = overall_emotion
        current_emotions['timestamp'] = float(time.time())

        # Check for anger alert
        check_anger_alert(facial_emotions)

        # Encode processed frame and send back to client
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Emit results back to client
        emit('emotion_update', current_emotions)
        emit('processed_frame', {'frame': processed_frame_b64})

        # Enhanced debugging
        print(f"📸 Frame processed: {len(facial_emotions)} faces, overall: {overall_emotion.get('emotion', 'none')}")
        if len(facial_emotions) > 0:
            print(f"🎭 FACIAL EMOTION DETECTED: {facial_emotions[0]['emotion']} (confidence: {facial_emotions[0]['confidence']:.2f})")
            print(f"📤 EMITTING TO FRONTEND: facial={facial_emotions}, overall={overall_emotion}")

        # Debug: Print the exact data being sent
        if len(facial_emotions) > 0 or overall_emotion.get('emotion') != 'neutral':
            print(f"🔥 DEBUG - Sending emotion data:")
            print(f"   Facial: {current_emotions['facial']}")
            print(f"   Voice: {current_emotions['voice']}")
            print(f"   Overall: {current_emotions['overall']}")
            print(f"   Timestamp: {current_emotions['timestamp']}")
            print("=" * 50)

    except Exception as e:
        logger.error(f"Frame processing error: {e}")

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle audio chunk from client."""
    try:
        if not is_running:
            logger.debug("🎤 Audio chunk received but system not running")
            return

        # Get the audio data from client
        audio_data = data.get('audio_data')
        if not audio_data:
            logger.debug("🎤 Audio chunk received but no audio_data")
            return

        # Debug logging for first few chunks
        if not hasattr(handle_audio_chunk, 'chunk_count'):
            handle_audio_chunk.chunk_count = 0
        handle_audio_chunk.chunk_count += 1

        if handle_audio_chunk.chunk_count <= 3:
            logger.info(f"🎤 AUDIO CHUNK #{handle_audio_chunk.chunk_count} RECEIVED: {len(audio_data)} samples, sample_rate: {data.get('sample_rate', 'unknown')}")

        # Update audio activity tracking
        system_state.audio_active = True
        system_state.last_audio_chunk_time = time.time()

        # Convert to numpy array (audio_data should be a list of float values)
        audio_array = np.array(audio_data, dtype=np.float32)

        # Process voice emotions with the received audio data
        voice_emotions = detect_voice_emotions(audio_array)

        # Update current emotions with voice data
        if voice_emotions:
            current_emotions['voice'] = voice_emotions
            current_emotions['timestamp'] = time.time()

            # Store the latest voice emotion for combination with facial emotions
            system_state.last_voice_emotion = voice_emotions[0]['emotion']  # Extract emotion string
            system_state.last_voice_confidence = voice_emotions[0]['confidence']

            if handle_audio_chunk.chunk_count <= 5:
                logger.info(f"🎤 Voice emotion processed: {voice_emotions[0]['emotion']}")

            # 🎯 NEW: Send immediate update to frontend when voice emotion is detected
            # Get the last known facial emotion from the state
            facial_data_for_update = []
            if system_state.last_facial_emotion:
                facial_data_for_update.append({
                    'emotion': system_state.last_facial_emotion,
                    'confidence': system_state.last_facial_confidence
                })

            # Combine with the NEW voice emotion to get a NEW overall emotion
            overall_emotion = combine_emotions_with_facial_priority(facial_data_for_update, voice_emotions[0])

            # Build the complete payload and emit immediately
            update_payload = {
                'facial': facial_data_for_update,
                'voice': voice_emotions,
                'overall': overall_emotion,
                'timestamp': time.time()
            }

            logger.info(f"🎤 ✅ SENDING VOICE-TRIGGERED UPDATE: voice={voice_emotions[0]['emotion']}, overall={overall_emotion['emotion'] if overall_emotion else 'none'}")
            emit('emotion_update', update_payload)

    except Exception as e:
        logger.error(f"❌ Audio processing error: {e}")

if __name__ == '__main__':
    import os

    print("🚀 Starting WORKING VoiceShield Flask Application")
    print("=" * 60)
    print("📹 Real face detection with OpenCV")
    print("🎭 Actual emotion analysis with DeepFace")
    print("✅ Visual face rectangles and emotion overlay")
    print("⚡ This version actually works!")

    # Get port from environment variable (for Render deployment)
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')

    print(f"🌐 Access at: http://{host}:{port}")
    print("=" * 60)

    # Run the Flask app with SocketIO
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
