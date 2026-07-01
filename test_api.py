import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CRICKET_API_KEY")

url = "https://api.cricapi.com/v1/currentMatches"
params = {"apikey": API_KEY, "offset": 0}

response = requests.get(url, params=params)
data = response.json()

print("Status:", data.get("status"))
print("Number of matches:", len(data.get("data", [])))
print("Sample match:", data.get("data", [])[0] if data.get("data") else "No matches right now")