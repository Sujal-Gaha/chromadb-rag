import requests
import os
from dotenv import load_dotenv

load_dotenv()

ollama_url = os.getenv("OLLAMA_SERVER_URL", "http://localhost:11434")

try:
    # Test basic connectivity
    response = requests.get(f"{ollama_url}/api/tags", timeout=5)
    if response.status_code == 200:
        print(f"✅ Connected to Ollama at {ollama_url}")
        print(f"Available models: {response.json().get('models', [])}")
    else:
        print(f"❌ Ollama responded with status {response.status_code}")

except requests.exceptions.ConnectionError:
    print(f"❌ Cannot connect to Ollama at {ollama_url}")
    print("Check if:")
    print("1. Ollama is running on the remote device")
    print("2. Tailscale is connected on both devices")
    print("3. Firewall allows port 11434")
except Exception as e:
    print(f"❌ Error: {e}")
