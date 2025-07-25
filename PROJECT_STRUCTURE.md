# VoiceShield Project Structure

This document describes the organized structure of the VoiceShield project after separating backend and frontend components.

## Directory Structure

```
VoiceShield/
├── backend/                    # Backend Flask application
│   ├── __init__.py            # Package initialization
│   ├── app.py                 # Main Flask application (formerly working_flask_app.py)
│   ├── requirements.txt       # Python dependencies
│   ├── README.md             # Backend documentation
│   └── src/                  # Backend source modules
│       ├── alerts/           # Alert system components
│       ├── emotion_detection/# Emotion detection modules
│       ├── fusion/           # Multi-modal fusion logic
│       ├── ui/               # UI-related backend code
│       └── utils/            # Utility functions
│
├── frontend/                  # Frontend web interface
│   ├── templates/            # HTML templates
│   │   └── index.html        # Main web interface (formerly working_index.html)
│   ├── static/               # Static assets
│   │   ├── Logo.jpg          # VoiceShield logo
│   │   └── Generals_logo.jpg # The Generals logo
│   ├── package.json          # Frontend package configuration
│   └── README.md             # Frontend documentation
│
├── config/                    # Configuration files
│   └── config.yaml           # System configuration
│
├── logs/                      # Log files
│   ├── alerts.log            # Anger alert logs
│   ├── emotions.log          # Emotion detection logs
│   └── voiceshield.log       # General system logs
│
├── data/                      # Data storage
├── models/                    # ML models
├── tests/                     # Test files
├── assets/                    # Additional assets
├── venv/                      # Virtual environment
├── main.py                    # Main entry point
├── requirements.txt           # Root Python dependencies
├── README.md                  # Project documentation
└── PROJECT_STRUCTURE.md       # This file
```

## Key Changes Made

### 1. Backend Organization
- **Moved**: `working_flask_app.py` → `backend/app.py`
- **Updated**: Template and static folder paths to point to frontend
- **Added**: Backend-specific documentation and requirements
- **Moved**: `src/` directory to `backend/src/`

### 2. Frontend Organization
- **Moved**: `templates/working_index.html` → `frontend/templates/index.html`
- **Moved**: `static/*.jpg` → `frontend/static/`
- **Added**: Frontend-specific documentation and package.json
- **Maintained**: All CSS, JavaScript, and HTML functionality

### 3. Path Updates
- **Flask app**: Updated template_folder and static_folder paths
- **Config loading**: Updated to use relative paths from backend
- **Log files**: Updated to use relative paths from backend
- **Main entry**: Updated to run backend/app.py

## Running the Application

### Option 1: Using main.py (Recommended)
```bash
python main.py
```

### Option 2: Direct backend execution
```bash
cd backend
python app.py
```

### Option 3: Using Flask CLI
```bash
cd backend
flask --app app run --host=0.0.0.0 --port=5001
```

## Benefits of This Structure

1. **Clear Separation**: Backend and frontend are clearly separated
2. **Maintainability**: Easier to maintain and update each component
3. **Scalability**: Can easily add more backend services or frontend components
4. **Development**: Different teams can work on backend and frontend independently
5. **Deployment**: Can deploy backend and frontend separately if needed

## File Relationships

- `backend/app.py` serves files from `frontend/templates/` and `frontend/static/`
- Configuration files remain in the root `config/` directory
- Logs are written to the root `logs/` directory
- The main entry point (`main.py`) launches the backend application

## Future Enhancements

This structure supports:
- Adding API versioning in the backend
- Implementing a separate frontend build process
- Adding more static assets and resources
- Implementing microservices architecture
- Adding database models to the backend
- Creating separate frontend applications (React, Vue, etc.)
