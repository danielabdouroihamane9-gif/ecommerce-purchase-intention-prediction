from __future__ import annotations

import os

import httpx

API_URL = os.getenv('API_URL', 'http://localhost:8000')

payload = {
    'Administrative': 0,
    'Administrative_Duration': 0.0,
    'Informational': 0,
    'Informational_Duration': 0.0,
    'ProductRelated': 2,
    'ProductRelated_Duration': 64.0,
    'BounceRates': 0.0,
    'ExitRates': 0.1,
    'SpecialDay': 0.0,
    'Month': 'Feb',
    'OperatingSystems': 2,
    'Browser': 2,
    'Region': 1,
    'TrafficType': 2,
    'VisitorType': 'Returning_Visitor',
    'Weekend': False,
}

response = httpx.post(f'{API_URL}/predict', json=payload, timeout=30.0)
print(f'Status code: {response.status_code}')
print(response.text)

if response.status_code != 200:
    raise SystemExit(1)

body = response.json()
required_fields = {'prediction', 'probability', 'predicted_class'}
missing = required_fields - body.keys()
if missing:
    raise SystemExit(f'Missing response fields: {sorted(missing)}')

prediction = body['prediction']
probability = float(body['probability'])
if not isinstance(prediction, bool):
    raise SystemExit(f'Prediction must be boolean, got {type(prediction).__name__}')
if not 0.0 <= probability <= 1.0:
    raise SystemExit(f'Probability must be in [0,1], got {probability}')

print('Smoke test passed.')
