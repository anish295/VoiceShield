# VoiceShield Backend

This is the backend server for the VoiceShield emotion detection system.

## Features

- **Real-time Face Detection**: Uses OpenCV for face detection
- **Emotion Analysis**: DeepFace for facial emotion recognition
- **Voice Emotion Detection**: Real-time audio processing for voice emotions
- **WebSocket Communication**: Real-time updates to frontend
- **Anger Alert System**: Configurable alerts for anger detection
- **Multi-modal Fusion**: Combines facial and voice emotions

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask application:
```bash
python app.py
```

The server will start on `http://localhost:5001`

## API Endpoints

- `GET /` - Main application page
- `GET /video_feed` - Video stream with face detection overlay
- `GET /api/status` - System status
- `POST /api/start` - Start emotion detection system
- `POST /api/stop` - Stop emotion detection system
- `GET/POST /api/anger_alert/config` - Anger alert configuration

## WebSocket Events

- `emotion_update` - Real-time emotion data
- `anger_alert` - Anger detection alerts

## Configuration

The system uses `../config/config.yaml` for configuration settings.

## Logging

Logs are written to `../logs/` directory:
- `alerts.log` - Anger alert logs
- `emotions.log` - Emotion detection logs
- `voiceshield.log` - General system logs
