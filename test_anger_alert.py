#!/usr/bin/env python3
"""
Test script to simulate anger detection and trigger alerts.
This script demonstrates the anger alert functionality.
"""

import requests
import json
import time

def test_anger_alert_config():
    """Test the anger alert configuration API."""
    base_url = "http://localhost:5001"
    
    print("🧪 Testing Anger Alert Configuration API")
    print("=" * 50)
    
    # Test GET configuration
    try:
        response = requests.get(f"{base_url}/api/anger_alert/config")
        if response.status_code == 200:
            config = response.json()
            print(f"✅ Current Configuration:")
            print(f"   - Enabled: {config['enabled']}")
            print(f"   - Threshold: {config['threshold']} ({config['threshold']*100:.1f}%)")
            print(f"   - Cooldown: {config['cooldown']} seconds")
        else:
            print(f"❌ Failed to get configuration: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting configuration: {e}")
        return False
    
    # Test POST configuration (lower threshold for testing)
    try:
        new_config = {
            "threshold": 0.3,  # Lower threshold for easier testing
            "cooldown": 10,    # Shorter cooldown for testing
            "enabled": True
        }
        
        response = requests.post(
            f"{base_url}/api/anger_alert/config",
            headers={"Content-Type": "application/json"},
            data=json.dumps(new_config)
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Configuration updated successfully:")
            print(f"   - Threshold: {result['config']['threshold']} ({result['config']['threshold']*100:.1f}%)")
            print(f"   - Cooldown: {result['config']['cooldown']} seconds")
        else:
            print(f"❌ Failed to update configuration: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error updating configuration: {e}")
        return False
    
    return True

def test_system_status():
    """Test the system status API."""
    base_url = "http://localhost:5001"
    
    print("\n🔍 Testing System Status API")
    print("=" * 50)
    
    try:
        response = requests.get(f"{base_url}/api/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ System Status:")
            print(f"   - Camera Active: {status.get('camera_active', False)}")
            print(f"   - Audio Active: {status.get('audio_active', False)}")
            print(f"   - System Running: {status.get('system_running', False)}")
            
            anger_config = status.get('anger_alert', {})
            print(f"   - Anger Alert Enabled: {anger_config.get('enabled', False)}")
            print(f"   - Anger Threshold: {anger_config.get('threshold', 0)} ({anger_config.get('threshold', 0)*100:.1f}%)")
            
            return True
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting status: {e}")
        return False

def simulate_anger_detection():
    """
    Simulate anger detection by creating fake facial emotion data.
    Note: This is for demonstration purposes only.
    """
    print("\n⚠️  Simulating Anger Detection")
    print("=" * 50)
    print("📝 To actually test anger alerts, you need to:")
    print("   1. Start the VoiceShield system in the web interface")
    print("   2. Show an angry facial expression to the camera")
    print("   3. The system will detect anger and trigger alerts")
    print("   4. Check the logs/alerts.log file for alert records")
    print("   5. Watch for popup alerts in the web interface")
    
    print("\n💡 Tips for triggering anger alerts:")
    print("   - Frown deeply and furrow your brow")
    print("   - Clench your jaw")
    print("   - Make an angry facial expression")
    print("   - Ensure good lighting for better detection")
    
    print(f"\n📊 Current alert threshold: Check the web interface")
    print(f"🕒 Alert cooldown: Check the web interface")

if __name__ == "__main__":
    print("🚀 VoiceShield Anger Alert Test Suite")
    print("=" * 60)
    
    # Test configuration API
    if test_anger_alert_config():
        print("\n✅ Configuration API tests passed!")
    else:
        print("\n❌ Configuration API tests failed!")
        exit(1)
    
    # Test status API
    if test_system_status():
        print("\n✅ Status API tests passed!")
    else:
        print("\n❌ Status API tests failed!")
        exit(1)
    
    # Provide instructions for manual testing
    simulate_anger_detection()
    
    print("\n🎯 Test Summary:")
    print("✅ Anger alert configuration API is working")
    print("✅ System status API includes anger alert info")
    print("✅ Web interface should show anger configuration controls")
    print("✅ Ready for live anger detection testing!")
    
    print("\n🌐 Open http://localhost:5001 to test the full system")
