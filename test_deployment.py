#!/usr/bin/env python3
"""
Test script to verify VoiceShield deployment
"""

import requests
import json

def test_deployment(base_url):
    """Test the deployed VoiceShield backend."""
    
    print(f"🧪 Testing VoiceShield deployment at: {base_url}")
    print("=" * 60)
    
    tests = [
        {
            'name': 'Health Check',
            'endpoint': '/health',
            'method': 'GET'
        },
        {
            'name': 'System Status',
            'endpoint': '/api/status',
            'method': 'GET'
        },
        {
            'name': 'Main Page',
            'endpoint': '/',
            'method': 'GET'
        }
    ]
    
    results = []
    
    for test in tests:
        try:
            print(f"Testing {test['name']}...")
            
            url = f"{base_url}{test['endpoint']}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {test['name']}: PASSED")
                if 'json' in response.headers.get('content-type', ''):
                    data = response.json()
                    print(f"   Response: {json.dumps(data, indent=2)}")
                results.append(True)
            else:
                print(f"❌ {test['name']}: FAILED (Status: {response.status_code})")
                results.append(False)
                
        except Exception as e:
            print(f"❌ {test['name']}: ERROR - {e}")
            results.append(False)
        
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Deployment is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the deployment configuration.")
    
    return passed == total

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter your Render app URL (e.g., https://voiceshield-backend.onrender.com): ")
    
    # Remove trailing slash
    url = url.rstrip('/')
    
    success = test_deployment(url)
    sys.exit(0 if success else 1)
