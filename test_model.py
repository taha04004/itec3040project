import joblib

try:
    bundle = joblib.load('mushroom_model.joblib')
    print('Model loaded successfully!')
    print(f'Features: {len(bundle["features"])} features')
    print(f'Strategy: {bundle["strategy"]}')
    print(f'First 5 features: {bundle["features"][:5]}')
except Exception as e:
    print(f'Error loading model: {e}')

