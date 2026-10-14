import sys
sys.stdout.reconfigure(encoding='utf-8')
from app import app

client = app.test_client()

test_keywords = [
    ("diabetes", "diabetes"),
    ("sugar", "diabetes"),
    ("fever", "fever"),
    ("jvara", "fever"),
    ("headache", "headache"),
    ("migraine", "headache"),
    ("cough", "cough"),
    ("asthma", "asthma"),
    ("allergy", "allergy"),
    ("vaccine", "vaccination"),
    ("vaccination", "vaccination"),
    ("bp", "hypertension"),
    ("blood pressure", "hypertension"),
    ("cold", "cold"),
    ("burn", "first_aid"),
    ("chest pain", "emergency"),
    ("depression", "mental_wellness"),
    ("stress", "mental_wellness"),
    ("nutrition", "nutrition"),
    ("diet", "nutrition"),
    ("stomach ache", "digestive_health"),
    ("acidity", "digestive_health"),
    ("soap", "hygiene"),
    ("first aid", "first_aid")
]

print(f"{'KEYWORD':15} | {'EXPECTED':18} | {'PREDICTED':18} | {'CONF':6} | {'FALLBACK':8} | {'STATUS'}")
print("-" * 85)

all_passed = True
for kw, expected in test_keywords:
    res = client.post('/api/chat', json={"message": kw, "model": "dl"}).get_json()
    predicted = res.get('intent')
    conf = res.get('confidence', 0.0)
    fb = res.get('is_fallback', False)
    is_ok = (predicted == expected and not fb)
    if not is_ok:
        all_passed = False
    status = "PASS" if is_ok else "FAIL"
    print(f"{kw:15} | {expected:18} | {predicted:18} | {conf:<6.2f} | {str(fb):8} | {status}")

print("-" * 85)
if all_passed:
    print("SUCCESS: Every single keyword from the dataset was accurately understood and suggested!")
else:
    print("FAILED: Some keywords did not pass.")
