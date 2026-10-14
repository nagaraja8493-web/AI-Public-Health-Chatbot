"""
Health Knowledge Base and Response Generation Module.
Provides verified medical awareness responses, emergency detection,
confidence fallback handling, and ethical medical disclaimers.
"""

import re
from typing import Tuple, Dict, Any

# Critical Emergency Keywords triggering immediate emergency safety layer
EMERGENCY_KEYWORDS = [
    r"\bchest pain\b",
    r"\bheart attack\b",
    r"\bheart arrest\b",
    r"\bcardiac arrest\b",
    r"\bcannot breathe\b",
    r"\bcan\'t breathe\b",
    r"\bsevere difficulty breathing\b",
    r"\bunconscious\b",
    r"\blost consciousness\b",
    r"\bunresponsive\b",
    r"\bheavy bleeding\b",
    r"\bsevere bleeding\b",
    r"\bbleeding won\'t stop\b",
    r"\bstroke\b",
    r"\bface drooping\b",
    r"\bslurred speech\b",
    r"\bsudden paralysis\b",
    r"\bpoisoning\b",
    r"\bpoison\b",
    r"\bswallowed poison\b",
    r"\bchoking\b",
    r"\bblue lips\b",
    r"\bsuicide\b",
    r"\bsuicidal\b",
    r"\bkilling myself\b",
    r"\banaphylaxis\b",
    r"\bthroat closing\b"
]

# Emergency Protocol Response
EMERGENCY_RESPONSE = (
    "🚨 **CRITICAL MEDICAL EMERGENCY DETECTED**\n\n"
    "Based on the critical symptoms described, **do not wait for an online assessment**.\n\n"
    "**Immediate Actions:**\n"
    "1. **Call Emergency Services immediately:**\n"
    "   - 📞 **Dial 112** (India National Unified Emergency Helpline)\n"
    "   - 🚑 **Dial 102 / 108** (Ambulance Services)\n"
    "2. If the person is unconscious and not breathing, initiate bystander CPR immediately.\n"
    "3. Keep the patient in a safe, resting position and loosen tight clothing.\n"
    "4. Proceed immediately to the nearest hospital Emergency Department (Casualty/ER)."
)

FALLBACK_RESPONSE = (
    "I'm not completely confident I understood your health query (Confidence is below the clinical safety threshold). "
    "Please try rephrasing your question with specific symptoms or health topics (e.g., 'What are symptoms of fever?', "
    "'How to manage high blood pressure?', 'What is a balanced diet?'). "
    "For specific individual health conditions, always consult a certified medical practitioner."
)

MEDICAL_DISCLAIMER = (
    "\n\n*⚠️ Medical Disclaimer: This system provides general public-health awareness information only "
    "and does not constitute medical diagnosis, treatment, or clinical prescription. "
    "Always consult a qualified healthcare professional for personalized medical advice.*"
)

# Verified Knowledge Base Responses
KNOWLEDGE_BASE = {
    "greeting": (
        "Hello! I am your AI-powered Public Health Awareness Assistant. "
        "I provide evidence-based general health information, preventive tips, disease awareness, and wellness guidance. "
        "How can I assist you with your health questions today?"
    ),
    "goodbye": (
        "Thank you for consulting the Public Health Awareness Chatbot! "
        "Remember to stay hydrated, maintain good hygiene, and seek professional healthcare when needed. "
        "Have a healthy and safe day ahead!"
    ),
    "fever": (
        "**Fever Awareness & Care:**\n"
        "A fever is a temporary elevation in body temperature (above 38°C / 100.4°F), typically signaling an immune response to a viral or bacterial infection.\n\n"
        "• **Home Management:** Rest in a cool, well-ventilated room, stay well hydrated (water, oral rehydration salts, soups), and apply lukewarm sponge compresses.\n"
        "• **When to Seek Medical Care:** Consult a doctor if the fever exceeds 39.4°C (103°F), persists beyond 3 days, or is accompanied by a stiff neck, persistent vomiting, or difficulty breathing."
    ),
    "cold": (
        "**Common Cold Awareness & Care:**\n"
        "The common cold is a viral upper respiratory infection causing nasal congestion, sneezing, a runny nose, mild sore throat, and fatigue.\n\n"
        "• **Self-Care:** Most colds resolve naturally within 7–10 days. Get ample rest, drink warm liquids (herbal tea with honey), use steam inhalation, and use saline nasal drops.\n"
        "• **Note:** Antibiotics do not cure viral colds. Seek a physician if symptoms persist longer than 10 days or if high fever and severe sinus pain develop."
    ),
    "cough": (
        "**Cough Awareness & Care:**\n"
        "A cough is a vital defensive reflex that clears irritants and mucus from your respiratory airways.\n\n"
        "• **Dry vs Wet Cough:** Dry coughs are often caused by viral recovery, smoke, or allergies; wet coughs expel phlegm.\n"
        "• **Relief:** Stay hydrated, sip warm fluids with honey and ginger, avoid dust and cigarette smoke, and use a humidifier.\n"
        "• **When to Seek Care:** Consult a healthcare provider if coughing lasts over 3 weeks, produces rust-colored or bloody sputum, or causes severe shortness of breath."
    ),
    "diabetes": (
        "**Diabetes Mellitus Awareness:**\n"
        "Diabetes is a metabolic condition marked by elevated blood glucose (blood sugar) due to insufficient insulin production (Type 1) or cellular insulin resistance (Type 2).\n\n"
        "• **Common Symptoms:** Excessive thirst (polydipsia), frequent urination especially at night (polyuria), unexplained weight loss, blurry vision, and slow-healing wounds.\n"
        "• **Prevention & Management:** Maintain a balanced low-glycemic diet rich in fiber, engage in 150+ minutes of weekly aerobic exercise, maintain healthy body weight, and monitor fasting blood glucose and HbA1c periodically."
    ),
    "hypertension": (
        "**Hypertension (High Blood Pressure) Awareness:**\n"
        "Hypertension occurs when blood exerts persistent high pressure against artery walls (consistently 130/80 mmHg or higher). Known as the 'silent killer', it often produces no symptoms until complications arise.\n\n"
        "• **Prevention:** Limit sodium intake to under 2,000 mg/day (1 teaspoon of salt), consume potassium-rich vegetables, exercise regularly, limit alcohol, avoid tobacco, and manage stress.\n"
        "• **Emergency Warning:** Extremely high BP (>180/120 mmHg) with severe headache, chest pain, or visual changes requires immediate emergency room care."
    ),
    "nutrition": (
        "**Nutrition & Balanced Diet Guidance:**\n"
        "Proper nutrition fuels daily metabolism, supports immune defense, and prevents chronic cardiovascular and metabolic diseases.\n\n"
        "• **Key Guidelines:** Fill half your plate with colorful vegetables and fruits, one quarter with whole grains (brown rice, oats, whole wheat), and one quarter with lean proteins (legumes, lentils, eggs, lean poultry).\n"
        "• **Hydration:** Drink 2 to 3 liters of clean water daily. Minimize ultra-processed foods, refined sugars, deep-fried snacks, and trans fats."
    ),
    "vaccination": (
        "**Vaccination & Immunization Awareness:**\n"
        "Vaccines safely stimulate your body's immune system to create antibodies against specific pathogens without causing the actual illness.\n\n"
        "• **Benefits:** Immunization protects against life-threatening diseases such as measles, hepatitis, influenza, tetanus, polio, and HPV, fostering community herd immunity.\n"
        "• **Adult Boosters:** Adults should ensure up-to-date tetanus boosters (every 10 years), annual influenza vaccines, and specific vaccines recommended for chronic conditions or travel."
    ),
    "hygiene": (
        "**Hygiene & Infection Prevention:**\n"
        "Consistent hygiene practices break the chain of infection for bacterial, viral, and parasitic diseases.\n\n"
        "• **Handwashing:** Wash hands thoroughly with soap and water for at least 20 seconds before meals, after using restrooms, and after coughing or sneezing.\n"
        "• **Food & Water Hygiene:** Drink clean, filtered or boiled water, cook food thoroughly, keep raw meats separate from produce, and sanitize frequently touched surfaces."
    ),
    "mental_wellness": (
        "**Mental Wellness & Stress Management:**\n"
        "Mental wellness is essential for overall health, emotional resilience, and productive living.\n\n"
        "• **Stress Reduction:** Practice diaphragmatic breathing (4-7-8 breathing), engage in daily physical exercise, maintain 7–8 hours of consistent sleep, and stay connected with supportive family and friends.\n"
        "• **Professional Support:** If you experience persistent sadness, severe anxiety, panic attacks, or feelings of hopelessness lasting more than 2 weeks, consult a certified psychologist or psychiatrist."
    ),
    "asthma": (
        "**Asthma Awareness & Management:**\n"
        "Asthma is a chronic respiratory condition characterized by airway inflammation, hyper-reactivity, and bronchospasm.\n\n"
        "• **Symptoms:** Wheezing (whistling sound when breathing), chest tightness, breathlessness, and nocturnal coughing.\n"
        "• **Management:** Identify and avoid known triggers (dust mites, pollen, pet dander, cold air, smoke). Always keep prescribed rescue inhalers accessible.\n"
        "• **Emergency:** Seek emergency care immediately if breathing is severely labored, lips or nailbeds turn bluish, or the rescue inhaler fails to provide prompt relief."
    ),
    "headache": (
        "**Headache & Migraine Awareness:**\n"
        "Headaches range from common tension headaches (dull, band-like tightness) to vascular migraines (throbbing, usually on one side, with sensitivity to light and sound).\n\n"
        "• **Relief:** Stay hydrated, rest in a quiet and dark room, apply cold or warm compresses, avoid skipping meals, and minimize digital screen glare.\n"
        "• **Red Flags:** Seek emergency medical evaluation for a sudden explosive ('thunderclap') headache, headache following head injury, or headache with fever and neck stiffness."
    ),
    "heart_health": (
        "**Cardiovascular & Heart Health Awareness:**\n"
        "Maintaining heart health reduces the risk of coronary artery disease, heart failure, and stroke.\n\n"
        "• **Recommendations:** Engage in 150 minutes of moderate aerobic exercise weekly (brisk walking, cycling), adopt a heart-healthy Mediterranean-style diet, keep blood cholesterol within target ranges, and avoid smoking completely.\n"
        "• **Emergency Signs:** Crushing chest pain, pain radiating to the left arm, neck, or jaw, accompanied by sweating and nausea, warrants calling 112 immediately."
    ),
    "allergy": (
        "**Allergy & Hypersensitivity Awareness:**\n"
        "Allergies occur when the immune system overreacts to ordinarily harmless environmental substances such as pollen, mold, dust mites, insect venom, or specific foods.\n\n"
        "• **Management:** Avoid known allergen triggers, use hypoallergenic bedding, keep windows closed during high pollen counts, and use doctor-recommended antihistamines for mild reactions.\n"
        "• **Anaphylaxis Warning:** Sudden swelling of the lips, tongue, or throat, dizziness, or breathing distress is anaphylaxis—a life-threatening emergency requiring immediate emergency services (112)."
    ),
    "digestive_health": (
        "**Digestive Health & Acidity Care:**\n"
        "A healthy gastrointestinal system ensures efficient nutrient absorption and supports the gut microbiome.\n\n"
        "• **Acidity & Indigestion:** Avoid lying down within 2–3 hours after eating, eat smaller meals, reduce fried or overly spicy foods, and avoid excessive caffeine.\n"
        "• **Bowel Regularity:** Increase dietary fiber gradually with legumes, whole grains, and vegetables, and drink plenty of fluids.\n"
        "• **Medical Evaluation:** Consult a gastroenterologist if you experience persistent severe abdominal pain, difficulty swallowing, or unexplained weight loss."
    ),
    "first_aid": (
        "**Essential First Aid Awareness:**\n"
        "Prompt, appropriate first aid can stabilize acute injuries before professional clinical treatment is available.\n\n"
        "• **Minor Burns:** Cool immediately under cool running tap water for 10–15 minutes. Never apply ice, butter, or toothpaste.\n"
        "• **Bleeding:** Apply firm, continuous direct pressure with a clean cloth or bandage.\n"
        "• **Sprains (RICE):** Rest the injured limb, Ice for 15 minutes at a time, Compress gently with an elastic bandage, and Elevate above heart level.\n"
        "• **Choking:** Perform the Heimlich maneuver (abdominal thrusts) if an adult is conscious but unable to breathe."
    ),
    "emergency": EMERGENCY_RESPONSE
}

# Comprehensive Dataset Keyword & Health Problem Mapping
# Maps any clinical keyword, symptom, Kannada term, or health problem from the dataset to its intent
DATASET_KEYWORDS_MAP = {
    "emergency": [
        r"\bemergency\b", r"\bchest[\s\-_]*pain\b", r"\bheart[\s\-_]*attack\b", r"\bcardiac[\s\-_]*arrest\b",
        r"\bstroke\b", r"\bunconscious\b", r"\blost[\s\-_]*consciousness\b", r"\bcollapse(d)?\b",
        r"\bcannot[\s\-_]*breathe\b", r"\bcan\'?t[\s\-_]*breathe\b", r"\bdifficulty[\s\-_]*breathing\b",
        r"\bsevere[\s\-_]*bleeding\b", r"\bheavy[\s\-_]*bleeding\b", r"\bambulance\b", r"\b112\b",
        r"\b108\b", r"\bdying\b", r"\bchoking\b", r"\bsuicid(e|al)\b", r"\bpoison(ing)?\b"
    ],
    "fever": [
        r"\bfevers?\b", r"\bhigh[\s\-_]*fever\b", r"\btemperature\b", r"\bhigh[\s\-_]*temp\b",
        r"\bchills\b", r"\bshivering\b", r"\bjvara\b", r"\bjwara\b", r"\bfeverish\b",
        r"\bbody[\s\-_]*heat\b", r"\bpyrexia\b"
    ],
    "cold": [
        r"\bcold\b", r"\bcommon[\s\-_]*cold\b", r"\brunny[\s\-_]*nose\b", r"\brunning[\s\-_]*nose\b",
        r"\bsneez(e|ing)\b", r"\bblocked[\s\-_]*nose\b", r"\bnasal[\s\-_]*congestion\b",
        r"\bcongestion\b", r"\bsore[\s\-_]*throat\b", r"\bflu\b", r"\binfluenza\b",
        r"\bseetha\b", r"\bstuffy[\s\-_]*nose\b"
    ],
    "cough": [
        r"\bcough(ing)?\b", r"\bdry[\s\-_]*cough\b", r"\bwet[\s\-_]*cough\b", r"\bkemmu\b",
        r"\bphlegm\b", r"\bmucus\b", r"\bsputum\b", r"\bhacking[\s\-_]*cough\b",
        r"\bticklish[\s\-_]*throat\b", r"\bthroat[\s\-_]*irritation\b"
    ],
    "diabetes": [
        r"\bdiabet(es|ic)\b", r"\bsugar\b", r"\bblood[\s\-_]*sugar\b", r"\bhigh[\s\-_]*sugar\b",
        r"\bglucose\b", r"\binsulin\b", r"\bhba1c\b", r"\bhypoglycemia\b",
        r"\bhyperglycemia\b", r"\bsakkare[\s\-_]*kayile\b", r"\bprediabetes\b", r"\bglycemic\b"
    ],
    "hypertension": [
        r"\bhypertension\b", r"\bhigh[\s\-_]*bp\b", r"\bblood[\s\-_]*pressure\b",
        r"\bhigh[\s\-_]*blood[\s\-_]*pressure\b", r"\bbp\b", r"\bsystolic\b", r"\bdiastolic\b",
        r"\brakta[\s\-_]*ottada\b", r"\bhypertensive\b"
    ],
    "asthma": [
        r"\basthma\b", r"\bwheez(e|ing)\b", r"\binhalers?\b", r"\bbronchial[\s\-_]*asthma\b",
        r"\basthma[\s\-_]*attacks?\b", r"\bshortness[\s\-_]*of[\s\-_]*breath\b", r"\bdammu\b",
        r"\busirata[\s\-_]*samasye\b", r"\bairway[\s\-_]*tightness\b"
    ],
    "headache": [
        r"\bheadaches?\b", r"\bmigraines?\b", r"\bhead[\s\-_]*pain\b", r"\bhead[\s\-_]*ache\b",
        r"\bthrobbing[\s\-_]*head\b", r"\btale[\s\-_]*novu\b", r"\bcluster[\s\-_]*headaches?\b",
        r"\btension[\s\-_]*headaches?\b", r"\bsinus[\s\-_]*headaches?\b"
    ],
    "heart_health": [
        r"\bheart[\s\-_]*health\b", r"\bcardiovascular\b", r"\bcholesterol\b",
        r"\bldl\b", r"\bhdl\b", r"\bheart[\s\-_]*disease\b", r"\barteries\b",
        r"\bhrudaya\b", r"\bangina\b", r"\batherosclerosis\b", r"\bcardiac[\s\-_]*health\b"
    ],
    "allergy": [
        r"\ballerg(y|ies|ic)\b", r"\bhives\b", r"\bantihistamines?\b", r"\bskin[\s\-_]*rash(es)?\b",
        r"\brash(es)?\b", r"\bitching\b", r"\bitchy\b", r"\bpollen\b",
        r"\bfood[\s\-_]*allergy\b", r"\bdust[\s\-_]*allergy\b", r"\bdust[\s\-_]*mites?\b",
        r"\banaphylaxis\b", r"\brhinitis\b"
    ],
    "digestive_health": [
        r"\bdigest(ive|ion)\b", r"\bacidity\b", r"\bacid[\s\-_]*reflux\b", r"\bgerd\b",
        r"\bheartburn\b", r"\bconstipation\b", r"\bbloat(ing)?\b", r"\bgas\b",
        r"\bgastritis\b", r"\bstomach[\s\-_]*ache\b", r"\bstomach[\s\-_]*pain\b",
        r"\btummy[\s\-_]*ache\b", r"\bhotte[\s\-_]*novu\b", r"\bindigestion\b",
        r"\bgut[\s\-_]*health\b", r"\bdiarrhea\b", r"\bloose[\s\-_]*motions?\b",
        r"\bcramps?[\s\-_]*stomach\b", r"\bnausea\b", r"\bvomit(ing)?\b"
    ],
    "nutrition": [
        r"\bnutrition\b", r"\bdiet(ary)?\b", r"\bbalanced[\s\-_]*diet\b", r"\bvitamins?\b",
        r"\bproteins?\b", r"\bcarbohydrates?\b", r"\bhealthy[\s\-_]*food\b",
        r"\baahara[\s\-_]*krama\b", r"\bminerals?\b", r"\bcaloric\b", r"\bfiber\b",
        r"\bintermittent[\s\-_]*fasting\b", r"\bhealthy[\s\-_]*eating\b", r"\bmeal[\s\-_]*plan(ning)?\b"
    ],
    "vaccination": [
        r"\bvaccin(e|es|ation)\b", r"\bimmuniz(e|ation)\b", r"\blasike\b",
        r"\bbooster[\s\-_]*doses?\b", r"\bflu[\s\-_]*shots?\b", r"\btetanus\b", r"\bjabs?\b",
        r"\bcovid[\s\-_]*vaccine\b", r"\bpolio\b", r"\bhepatitis[\s\-_]*vaccine\b", r"\bmrna[\s\-_]*vaccine\b"
    ],
    "hygiene": [
        r"\bhygiene\b", r"\bhand[\s\-_]*wash(ing)?\b", r"\bsoaps?\b", r"\bswachate\b",
        r"\bsanitiz(e|er|ing)\b", r"\bdisinfect(ant)?\b", r"\bclean[\s\-_]*hands\b",
        r"\bwash[\s\-_]*hands\b", r"\binfection[\s\-_]*prevention\b", r"\bpurify[\s\-_]*water\b"
    ],
    "mental_wellness": [
        r"\bmental[\s\-_]*(health|wellness)\b", r"\bstress(ed)?\b", r"\banxiety\b",
        r"\bdepress(ion|ed)?\b", r"\bmanasika[\s\-_]*ottada\b", r"\bburnout\b",
        r"\bpanic[\s\-_]*attacks?\b", r"\bmindfulness\b", r"\bmeditation\b",
        r"\boverthinking\b", r"\binsomnia\b", r"\bsleeplessness\b", r"\btherapy\b", r"\bcounseling\b"
    ],
    "first_aid": [
        r"\bfirst[\s\-_]*aid\b", r"\bburns?\b", r"\bminor[\s\-_]*burns?\b", r"\bcuts?\b",
        r"\bwounds?\b", r"\bbleeding\b", r"\bprathamachikitse\b", r"\bsprains?\b",
        r"\bbandage\b", r"\brice[\s\-_]*protocol\b", r"\bnosebleeds?\b",
        r"\binsect[\s\-_]*stings?\b", r"\bbee[\s\-_]*stings?\b", r"\bfractures?\b",
        r"\bscrapes?\b", r"\babrasions?\b"
    ],
    "greeting": [
        r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bnamaskara\b", r"\bnamaste\b",
        r"\bgood[\s\-_]*morning\b", r"\bgood[\s\-_]*evening\b", r"\bgood[\s\-_]*afternoon\b", r"\bgreetings\b"
    ],
    "goodbye": [
        r"\bbye\b", r"\bgoodbye\b", r"\bthanks\b", r"\bthank[\s\-_]*you\b",
        r"\bdhanyavada\b", r"\bsee[\s\-_]*you\b", r"\bsigning[\s\-_]*off\b"
    ]
}

def detect_dataset_keyword(text: str) -> str:
    """
    Scans user query for any canonical health keyword, disease, or problem from the dataset.
    Returns matching intent string, or None if no direct keyword matches.
    Prioritizes emergency, then clinical health conditions, then greetings/goodbye.
    """
    clean_text = text.lower().strip()
    
    # Priority order ensures acute/specific conditions take precedence over broad terms
    priority_order = [
        "emergency", "first_aid", "diabetes", "hypertension", "asthma",
        "headache", "fever", "cold", "cough", "heart_health", "allergy",
        "digestive_health", "nutrition", "vaccination", "hygiene",
        "mental_wellness", "greeting", "goodbye"
    ]
    
    for intent in priority_order:
        patterns = DATASET_KEYWORDS_MAP.get(intent, [])
        for pattern in patterns:
            if re.search(pattern, clean_text):
                return intent
                
    return None

def check_emergency(text: str) -> bool:
    """Checks if the user text contains immediate critical emergency keywords."""
    clean = text.lower()
    for pattern in EMERGENCY_KEYWORDS:
        if re.search(pattern, clean):
            return True
    return False

def get_response_for_intent(intent: str, confidence: float, threshold: float = 0.60) -> Tuple[str, bool]:
    """
    Retrieves the response from the knowledge base based on intent and confidence.
    Returns (response_text, is_fallback).
    """
    if confidence < threshold:
        return FALLBACK_RESPONSE + MEDICAL_DISCLAIMER, True
    
    response = KNOWLEDGE_BASE.get(intent)
    if not response:
        return FALLBACK_RESPONSE + MEDICAL_DISCLAIMER, True
    
    # Emergency responses don't need additional general disclaimer
    if intent == "emergency":
        return response, False
    
    return response + MEDICAL_DISCLAIMER, False
