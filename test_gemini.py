import requests, json

with open('.streamlit/secrets.toml', 'r') as f:
    content = f.read()
key = content.split('"')[1]
print("Key found:", key[:10], "...")

r = requests.post(
    f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}',
    headers={'Content-Type': 'application/json'},
    json={'contents': [{'parts': [{'text': 'Say hello in one word'}]}]},
    timeout=15
)
print('Status:', r.status_code)
print('Response:', json.dumps(r.json(), indent=2))
