# VoiceShield Frontend

This is the frontend interface for the VoiceShield emotion detection system.

## Structure

```
frontend/
├── index.html             # Main web interface
├── static/
│   ├── Logo.jpg           # VoiceShield logo
│   └── Generals_logo.jpg  # The Generals logo
├── package.json           # Frontend package configuration
└── README.md              # This documentation
```

## Features

- **Modern Web Interface**: Responsive design with dark theme
- **Real-time Video Feed**: Live camera feed with face detection overlay
- **Emotion Display**: Separate displays for facial, voice, and overall emotions
- **System Status**: Real-time status indicators
- **Anger Alert Configuration**: Configurable anger detection settings
- **Alert Popups**: Visual and audio alerts for anger detection

## Technologies Used

- **HTML5**: Modern semantic markup
- **CSS3**: Advanced styling with gradients, animations, and responsive design
- **JavaScript**: Real-time WebSocket communication
- **Socket.IO**: Real-time bidirectional communication

## Interface Components

### Main Sections
1. **Video Section**: Live camera feed and controls
2. **Emotions Section**: Real-time emotion displays
3. **Status Section**: System component status
4. **Configuration**: Anger alert settings

### Responsive Design
- Desktop: Two-column layout
- Tablet: Single column layout
- Mobile: Optimized for small screens

## Customization

The interface can be customized by modifying:
- CSS variables for colors and themes
- Layout grid in the container section
- Animation timings and effects
- Alert popup styling and behavior

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+
