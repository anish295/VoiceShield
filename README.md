# 🛡️ VoiceShield - Real-Time Emotion Detection & Anger Alert System

VoiceShield is a **modern Flask-based** real-time emotion detection system that combines facial expression analysis and voice emotion recognition with intelligent anger alert capabilities. Built with a clean, responsive web interface, it provides instant emotion detection and proactive anger monitoring.

## ✨ Key Features

- **🎭 Dual-Modal Emotion Detection**: Real-time facial + voice emotion analysis
- **⚠️ Anger Alert System**: Configurable alerts when anger levels exceed thresholds
- **⚡ Lightning Fast**: Instant startup, smooth 30 FPS video processing
- **🎨 Modern Web Interface**: Beautiful Flask-based UI with real-time updates
- **🔒 Privacy-First**: All processing done locally, no cloud dependencies
- **🎯 High Accuracy**: DeepFace for facial emotions, advanced audio analysis for voice
- **📱 Responsive Design**: Works on desktop and mobile browsers
- **🔄 Real-Time Updates**: WebSocket-based live emotion streaming
- **⚙️ Configurable Alerts**: Customizable anger thresholds and cooldown periods

## 🎭 Supported Emotions & Alerts

### Facial Emotions (DeepFace)
- 😠 **Angry** - Aggressive facial expressions ⚠️ *Triggers anger alerts*
- 😊 **Happy** - Smiles, positive expressions
- 😢 **Sad** - Downturned expressions, frowns
- 😨 **Fear** - Surprised, worried expressions
- 😐 **Neutral** - Calm, baseline expressions
- 😲 **Surprise** - Wide eyes, raised eyebrows
- 🤢 **Disgust** - Negative, repulsed expressions

### Voice Emotions (Advanced Audio Analysis)
- 😠 **Angry** - High energy + Low pitch patterns
- 😊 **Happy** - Bright, energetic speech patterns
- 😢 **Sad** - Low energy, monotone delivery
- 😲 **Surprised** - High pitch + Sudden changes
- 😐 **Neutral** - Normal conversational patterns

### ⚠️ Anger Alert System
- **Real-time monitoring** of facial anger levels
- **Configurable threshold** (10% - 100% anger confidence)
- **Cooldown protection** to prevent alert spam
- **Visual popup alerts** with anger level details
- **Alert logging** to `logs/alerts.log` file

## 🏗️ Project Structure

VoiceShield is now organized with a clear separation between backend and frontend:

```
VoiceShield/
├── backend/                    # Flask application & ML processing
│   ├── app.py                 # Main Flask server
│   ├── src/                   # Backend modules
│   └── requirements.txt       # Python dependencies
├── frontend/                  # Web interface
│   ├── templates/index.html   # Main web interface
│   └── static/               # Images and assets
├── config/                    # Configuration files
├── logs/                      # System logs
└── main.py                    # Entry point
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Camera Feed   │    │  Microphone     │
│   (OpenCV)      │    │   (PyAudio)     │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│   DeepFace      │    │ Audio Feature   │
│ Face Detection  │    │  Extraction     │
│ + Emotion AI    │    │                 │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     ▼
          ┌─────────────────┐
          │ Emotion Fusion  │
          │ + Anger Alert   │ ⚠️
          └─────────┬───────┘
                    ▼
          ┌─────────────────┐
          │  Flask Server   │
          │  + WebSocket    │
          └─────────┬───────┘
                    ▼
          ┌─────────────────┐
          │ Modern Web UI   │
          │ + Alert Popups  │ ⚠️
          └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Recommended: Python 3.9-3.11)
- **Webcam** (Built-in or USB camera)
- **Microphone** (Built-in or external mic)
- **4GB RAM minimum** (8GB recommended)
- **Windows 10/11, macOS 10.15+, or Linux**

### ⚡ Installation

1. **Clone and Setup**
   ```bash
   git clone https://github.com/your-username/voiceshield.git
   cd voiceshield
   python -m venv venv

   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run VoiceShield**
   ```bash
   python working_flask_app.py
   ```

4. **Open Browser & Configure**
   ```
   🌐 Navigate to: http://localhost:5001
   ⚙️ Configure anger alerts in the web interface
   🚀 Click "Start WORKING System"
   ```

## 🎮 Usage

### 🚀 Running VoiceShield

```bash
python working_flask_app.py
# Opens at: http://localhost:5001
```

### 🎯 Using the Web Interface

1. **Configure Anger Alerts**
   - Adjust **Anger Threshold** slider (10% - 100%)
   - Set **Alert Cooldown** period (5-300 seconds)
   - Enable/disable alerts with checkbox
   - Click **"Save Configuration"**

2. **Start the System**
   - Click **"Start WORKING System"**
   - Grant camera/microphone permissions when prompted
   - System starts instantly with real-time monitoring

3. **Monitor Emotions & Alerts**
   - **📹 Live Video Feed** - Face detection with emotion overlay
   - **🎭 Facial Emotions** - Real-time emotion analysis
   - **🎤 Voice Emotions** - Live voice emotion detection
   - **⚠️ Anger Alerts** - Popup alerts when anger exceeds threshold
   - **📊 Confidence Scores** - Accuracy indicators for each emotion

### ⚠️ Anger Alert Features

- **Real-time Detection** - Monitors facial anger continuously
- **Visual Alerts** - Red popup notifications with anger level
- **Alert Logging** - Records all alerts to `logs/alerts.log`
- **Configurable Thresholds** - Customize sensitivity (default: 60%)
- **Cooldown Protection** - Prevents alert spam (default: 30 seconds)
- **Auto-dismiss** - Alerts disappear after 10 seconds

## 🔧 Technical Details

### 🎭 Core Technologies

**Facial Emotion Detection**
- **DeepFace**: Pre-trained deep learning models for emotion classification
- **OpenCV**: Real-time face detection and video processing
- **Processing**: 30 FPS with multi-face support

**Voice Emotion Analysis**
- **Audio Features**: RMS energy, spectral centroid, pitch variation
- **Real-time Processing**: Live audio buffer analysis
- **Anti-oscillation**: Prevents rapid emotion switching

**Anger Alert System**
- **Threshold Detection**: Configurable anger confidence levels
- **Alert Management**: Cooldown periods and popup notifications
- **Logging**: Persistent alert records in `logs/alerts.log`

### 🏗️ Architecture

**Backend (Flask + SocketIO)**
- Real-time emotion processing
- WebSocket communication
- API endpoints for configuration

**Frontend (Modern Web UI)**
- Live video feed with emotion overlay
- Configurable anger alert settings
- Real-time popup notifications

## ⚙️ Configuration

### 🔧 Anger Alert Configuration

**Via Web Interface (Recommended)**
- Access configuration panel at http://localhost:5001
- Adjust anger threshold slider (10% - 100%)
- Set cooldown period (5-300 seconds)
- Enable/disable alerts with checkbox

**Via Config File** (`config/config.yaml`)
```yaml
alerts:
  anger_alert:
    enabled: true
    threshold: 0.6  # 60% anger confidence
    cooldown: 30    # seconds between alerts
    popup_duration: 10  # seconds to show popup
```

**Via API** (for integration)
```bash
# Get current configuration
curl http://localhost:5001/api/anger_alert/config

# Update configuration
curl -X POST http://localhost:5001/api/anger_alert/config \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.7, "cooldown": 20, "enabled": true}'
```

## 🔍 Troubleshooting

### Common Issues & Solutions

**1. Camera Not Working**
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Try different camera index in working_flask_app.py:
camera = cv2.VideoCapture(1)  # Try 1, 2, 3, etc.
```

**2. Audio/Microphone Issues**
```bash
# List audio devices
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]; p.terminate()"

# Make sure microphone permissions are granted in browser
```

**3. Slow Performance**
- **Close other applications** using camera/microphone
- **Check system resources** - ensure adequate RAM/CPU
- **Update graphics drivers** for better OpenCV performance
- **Use Chrome/Edge** for best WebSocket performance

**4. Dependencies Issues**
```bash
# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# For TensorFlow issues:
pip install tensorflow==2.13.0 --force-reinstall
```

## 📁 Project Structure

```
VoiceShield/
├── 📄 working_flask_app.py      # Main Flask application with anger alerts
├── 📄 test_anger_alert.py       # Test script for anger alert system
├── 📄 requirements.txt          # Dependencies
├── 📄 README.md                 # Documentation
├── 📁 config/
│   └── 📄 config.yaml          # Configuration including anger alerts
├── 📁 templates/
│   └── 📄 working_index.html   # Web interface with alert controls
├── 📁 logs/
│   └── � alerts.log           # Anger alert log file
└── �📁 venv/                    # Virtual environment
```

### 🎯 Key Files

**working_flask_app.py** - Core application
- Real-time emotion detection (facial + voice)
- Anger alert system with configurable thresholds
- Flask server with WebSocket support
- API endpoints for configuration

**templates/working_index.html** - Web interface
- Live video feed with emotion display
- Anger alert configuration controls
- Real-time popup notifications
- Responsive design

**config/config.yaml** - Configuration
- Anger alert settings (threshold, cooldown, etc.)
- System configuration options

**logs/alerts.log** - Alert logging
- Records all triggered anger alerts
- Includes timestamp and anger levels

## 🔌 API Endpoints

### Anger Alert Configuration
```bash
# Get current configuration
GET /api/anger_alert/config

# Update configuration
POST /api/anger_alert/config
Content-Type: application/json
{
  "threshold": 0.7,      # 0.0 - 1.0 (anger confidence level)
  "cooldown": 30,        # seconds between alerts
  "enabled": true        # enable/disable alerts
}
```

### System Status
```bash
# Get system status including anger alert settings
GET /api/status
```

### System Control
```bash
# Start emotion detection system
POST /api/start

# Stop emotion detection system
POST /api/stop
```

## � System Requirements

**Minimum Requirements**
- Python 3.8+
- 4GB RAM
- Webcam + Microphone
- Modern browser (Chrome/Edge recommended)

**Performance**
- 30 FPS video processing
- Real-time emotion detection
- <100ms latency for alerts

## � Use Cases

**Personal & Home**
- Family wellness monitoring with anger alerts
- Personal mood tracking and emotional awareness
- Smart home integration based on detected emotions

**Professional & Educational**
- Customer service training with emotion feedback
- Research in emotion recognition and psychology
- Classroom engagement monitoring
- Therapy and counseling assistance

**Development & Integration**
- API integration for custom applications
- AI/ML research and development
- Building emotion-aware systems

## 🔒 Privacy & Security

- **100% Local Processing** - No data leaves your device
- **No Cloud Dependencies** - All AI models run locally
- **No Data Storage** - Emotions processed in real-time, not saved
- **Open Source** - Transparent, auditable code

## 🤝 Contributing

1. Fork the repository on GitHub
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

##  Acknowledgments

**Core Technologies**
- **[DeepFace](https://github.com/serengil/deepface)** - Facial emotion recognition
- **[OpenCV](https://opencv.org/)** - Computer vision and camera handling
- **[Flask](https://flask.palletsprojects.com/)** - Web framework and server

---

## 🎯 Quick Summary

**VoiceShield** is a modern emotion detection system with intelligent anger alerts:

✅ **Real-time emotion detection** - Face + voice analysis
✅ **Configurable anger alerts** - Customizable thresholds and notifications
✅ **Privacy-first** - 100% local processing, no cloud dependencies
✅ **Modern web interface** - Beautiful, responsive design
✅ **Easy setup** - Works out of the box

**🚀 Get started:**
```bash
git clone [repo] && cd voiceshield && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt && python working_flask_app.py
```

**🛡️ VoiceShield - Intelligent Emotion Detection with Anger Alerts** ⚠️🎭✨
