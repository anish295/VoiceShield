#!/bin/bash

# VoiceShield Render Deployment Script

echo "🚀 Preparing VoiceShield for Render deployment..."

# Add all changes
git add .

# Commit changes
git commit -m "🚀 Add Render deployment configuration

- Added render.yaml for service configuration
- Created cloud-specific requirements without PyAudio
- Added health check endpoints
- Updated app.py for production deployment
- Added comprehensive deployment guide"

# Push to GitHub
echo "📤 Pushing to GitHub..."
git push origin main

echo "✅ Deployment files ready!"
echo ""
echo "Next steps:"
echo "1. Go to https://dashboard.render.com"
echo "2. Create new Web Service"
echo "3. Connect your GitHub repository"
echo "4. Use the settings from RENDER_DEPLOYMENT_GUIDE.md"
echo ""
echo "🌐 Your app will be available at: https://voiceshield-backend.onrender.com"
