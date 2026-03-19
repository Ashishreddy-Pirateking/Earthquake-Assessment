import requests, json
key = open('.streamlit/secrets.toml').read().split(chr(34))[1]
r = requests.post(
    'https://api.groq.com/openai/v1/chat/completions',
    headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
    json={'model': 'llama-3.1-8b-instant', 'messages': [{'role': 'user', 'content': 'Say hello in one word'}], 'max_tokens': 10},
    timeout=15
)
print('Status:', r.status_code)
print(json.dumps(r.json(), indent=2))
