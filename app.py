import os
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
import requests  # Uncomment this if you are using an external API
import openai

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
DATASET_PATH = r"C:\Users\sanga\Downloads\try\Dataset"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Disease information dictionary
disease_info = {
    "Wound": {
        "suggestions": [
            "Clean the wound thoroughly with mild soap and water.",
            "Apply an antibiotic ointment to prevent infection.",
            "Cover th e wound with a sterile bandage or dressing.",
            "Use an antibiotic ointment such as POLYSPORIN for children under 5 years.",
            "Recommend oral antibiotics like Amoxicillin-clavulanate for ages above 15."
        ],
        "precautions": [
            "Avoid touching the wound with dirty hands.",
            "Change the bandage regularly to keep the wound clean and dry.",
            "Seek medical attention if the wound shows signs of infection."
        ]
    },
    "joint pain": {
        "suggestions": [
            "Apply ice packs to reduce inflammation and pain.",
            "Take over-the-counter pain relievers, such as ibuprofen or acetaminophen.",
            "Rest the affected joint and avoid strenuous activities.",
            "For children under 10, acetaminophen or ibuprofen is recommended.",
            "For older individuals, consult a doctor for proper medication."
        ],
        "precautions": [
            "Avoid activities that worsen the pain.",
            "Use proper posture to prevent further strain.",
            "Consult a healthcare professional for a diagnosis and treatment plan."
        ]
    },
    "Pimples": {
        "suggestions": [
            "Wash your face twice daily with a gentle cleanser.",
            "Use non-comedogenic products to avoid clogging pores.",
            "Apply treatments containing benzoyl peroxide or salicylic acid.",
            "For children under 10, tetracycline or macrolides are recommended.",
            "For adults over 20, clindamycin, erythromycin, and Vitamin C serum are helpful."
        ],
        "precautions": [
            "Avoid picking or squeezing pimples.",
            "Keep hair and hands away from your face.",
            "Maintain a healthy diet and drink water to support clear skin."
        ]
    },
    "teeth": {
        "suggestions": [
            "Brush your teeth twice a day with fluoride toothpaste.",
            "Floss daily to remove plaque and food particles.",
            "Limit sugary and acidic foods to prevent tooth decay.",
            "Consider using Pepsodent toothpaste."
        ],
        "precautions": [
            "Visit your dentist regularly for check-ups.",
            "Avoid biting hard objects with your teeth.",
            "If you experience tooth pain, consult a dentist for evaluation."
        ]
    }
}

# Function to get suggestions and precautions for the disease
def get_disease_info(disease_name):
    return disease_info.get(disease_name, {"suggestions": [], "precautions": []})

# Function to calculate similarity between two images
def calculate_similarity(image1, image2):
    if image1 is None or image2 is None:
        return 0.0
    image1_resized = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    gray_image1 = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return ssim(gray_image1, gray_image2)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Load the uploaded image and convert it to an array
            uploaded_image = Image.open(file_path).convert('RGB')
            uploaded_image = np.array(uploaded_image)

            similarity_scores = []
            # Iterate over each disease folder in the dataset
            for disease_folder in os.listdir(DATASET_PATH):
                disease_folder_path = os.path.join(DATASET_PATH, disease_folder)
                for image_file in os.listdir(disease_folder_path):
                    image_path = os.path.join(disease_folder_path, image_file)
                    try:
                        # Load each sample image for comparison
                        dataset_image = np.array(Image.open(image_path).convert('RGB'))
                        similarity_score = calculate_similarity(uploaded_image, dataset_image)
                        similarity_scores.append((disease_folder, similarity_score))
                    except FileNotFoundError:
                        print(f"Sample image for {disease_folder} not found at {image_path}.")

            # Check if we found any matching images
            if similarity_scores:
                similarity_scores.sort(key=lambda x: x[1], reverse=True)
                top_match = similarity_scores[0]
                disease_name = top_match[0] if top_match[1] >= 0.5 else "Unknown"
                similarity_score = top_match[1]
                
                # Get disease-specific suggestions and precautions
                disease_data = get_disease_info(disease_name)
                suggestions = "\n".join(disease_data["suggestions"])
                precautions = "\n".join(disease_data["precautions"])

                # Retrieve suggested medicines based on the disease name (adjust as needed)
                medicine_info = {
                    "Wound": ["Polysporin", "Amoxicillin-clavulanate"],
                    "joint pain": ["Ibuprofen", "Acetaminophen"],
                    "Pimples": ["Benzoyl peroxide", "Clindamycin"],
                    "teeth": ["Fluoride toothpaste", "Pepsodent toothpaste"]
                }
                medicines = ", ".join(medicine_info.get(disease_name, ["No specific medicines found"]))

                return render_template(
                    'result.html',
                    disease_name=disease_name,
                    similarity_score=similarity_score,
                    suggestions=suggestions,
                    precautions=precautions,
                    medicines=medicines
                )
            else:
                # If no matches found, display a message to the user
                return render_template(
                    'result.html',
                    disease_name="Unknown",
                    similarity_score="N/A",
                    suggestions="No suggestions available.",
                    precautions="No precautions available.",
                    medicines="No specific medicines found"
                )

    return render_template('upload.html')



from fuzzywuzzy import fuzz

# Example function to fetch disease data for symptoms-based predictions
def get_disease_info_symptoms(disease_name):
    disease_info = {
        "Flu": {
            "suggestions": ["Rest", "Stay hydrated", "Take over-the-counter medication for fever"],
            "precautions": ["Avoid close contact with people", "Wash hands frequently"],
            "suggested_medicines": ["Paracetamol", "Ibuprofen"]
        },
        "Allergic Reaction": {
            "suggestions": ["Use antihistamines", "Avoid allergens", "Apply soothing lotion to affected areas"],
            "precautions": ["Identify allergens", "Wear protective clothing if necessary"],
            "suggested_medicines": ["Cetirizine", "Loratadine"]
        },
        
        
    "Migraine": {
        "suggestions": ["Rest in a dark and quiet room", "Apply a cold compress", "Avoid triggers such as certain foods or stress"],
        "precautions": ["Maintain a headache diary", "Stay hydrated", "Follow a regular sleep schedule"],
        "suggested_medicines": ["Aspirin", "Sumatriptan", "Ibuprofen"]
    },
    "Heart Disease": {
        "suggestions": ["Consult a cardiologist immediately", "Take prescribed heart medications", "Reduce stress and anxiety"],
        "precautions": ["Avoid fatty foods", "Exercise regularly under medical guidance", "Quit smoking"],
        "suggested_medicines": ["Aspirin", "Beta blockers", "Statins"]
    },
    "Arthritis": {
        "suggestions": ["Use warm or cold compresses", "Engage in low-impact exercises", "Consider physiotherapy"],
        "precautions": ["Maintain a healthy weight", "Avoid repetitive motions", "Follow ergonomic practices"],
        "suggested_medicines": ["Ibuprofen", "Naproxen", "Methotrexate"]
    },
    "Strep Throat": {
        "suggestions": ["Stay hydrated", "Use throat lozenges", "Gargle with warm salt water"],
        "precautions": ["Avoid sharing personal items", "Wash hands frequently", "Stay home until fever-free"],
        "suggested_medicines": ["Amoxicillin", "Penicillin", "Acetaminophen"]
    },
    "Viral Infection": {
        "suggestions": ["Get plenty of rest", "Drink warm fluids", "Use over-the-counter pain relievers for symptoms"],
        "precautions": ["Avoid crowded places", "Boost immunity with healthy foods", "Practice good hygiene"],
        "suggested_medicines": ["Paracetamol", "Ibuprofen", "Antiviral medications (if prescribed)"]
    },
    "Food Poisoning": {
        "suggestions": ["Drink oral rehydration solutions", "Avoid solid foods until recovery", "Eat bland foods like rice and toast"],
        "precautions": ["Avoid contaminated food and water", "Wash hands before meals", "Ensure food is cooked thoroughly"],
        "suggested_medicines": ["Oral rehydration salts", "Antiemetics", "Probiotics"]
    },
    "Diabetes": {
        "suggestions": ["Monitor blood sugar levels regularly", "Maintain a balanced diet", "Engage in regular exercise"],
        "precautions": ["Avoid sugary foods", "Take medications on time", "Stay hydrated"],
        "suggested_medicines": ["Metformin", "Insulin", "Glipizide"]
    },
    "Malaria": {
        "suggestions": ["Seek immediate medical attention", "Stay hydrated", "Take antimalarial medications"],
        "precautions": ["Use mosquito repellents", "Sleep under mosquito nets", "Avoid mosquito-prone areas"],
        "suggested_medicines": ["Chloroquine", "Artemisinin-based combination therapies (ACTs)", "Primaquine"]
    },
    "Irritable Bowel Syndrome (IBS)": {
        "suggestions": ["Follow a low-FODMAP diet", "Manage stress levels", "Take probiotics"],
        "precautions": ["Avoid trigger foods like caffeine and spicy foods", "Eat smaller meals", "Stay hydrated"],
        "suggested_medicines": ["Loperamide", "Alosetron", "Lubiprostone"]
    },
    "Chickenpox": {
        "suggestions": ["Apply calamine lotion to soothe itching", "Keep nails trimmed", "Take lukewarm baths with baking soda"],
        "precautions": ["Avoid scratching lesions", "Stay isolated to prevent spread", "Get vaccinated"],
        "suggested_medicines": ["Paracetamol", "Antihistamines", "Acyclovir (for severe cases)"]
    },
    "Lupus": {
        "suggestions": ["Avoid prolonged sun exposure", "Maintain a healthy diet", "Rest during flare-ups"],
        "precautions": ["Get regular check-ups", "Avoid stress", "Exercise moderately"],
        "suggested_medicines": ["Hydroxychloroquine", "NSAIDs", "Corticosteroids"]
    },
    "Common Cold": {
        "suggestions": ["Stay hydrated", "Use steam inhalation", "Take vitamin C supplements"],
        "precautions": ["Avoid cold beverages", "Wash hands frequently", "Cover mouth when sneezing"],
        "suggested_medicines": ["Paracetamol", "Decongestants", "Antihistamines"]
    },
    "Tuberculosis": {
        "suggestions": ["Follow the full course of antibiotics", "Rest in a well-ventilated room", "Eat nutritious meals"],
        "precautions": ["Avoid close contact with others", "Cover mouth when coughing", "Get vaccinated"],
        "suggested_medicines": ["Isoniazid", "Rifampin", "Pyrazinamide"]
    },
    "Kidney Disease": {
        "suggestions": ["Control blood pressure and diabetes", "Reduce salt intake", "Stay hydrated but avoid overhydration"],
        "precautions": ["Avoid NSAIDs and nephrotoxic drugs", "Follow a kidney-friendly diet", "Monitor fluid intake"],
        "suggested_medicines": ["ACE inhibitors", "Diuretics", "Erythropoiesis-stimulating agents"]
    },
    "Fibromyalgia": {
        "suggestions": ["Engage in light physical activities", "Practice relaxation techniques like yoga", "Use heat therapy for muscle pain"],
        "precautions": ["Avoid overexertion", "Maintain a consistent sleep schedule", "Follow a balanced diet"],
        "suggested_medicines": ["Pregabalin", "Duloxetine", "Amitriptyline"]
    }


        # Add more diseases with suggestions, precautions, and suggested medicines as needed
    }
    return disease_info.get(disease_name, {})


# Route to predict disease based on symptoms
@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms():
    if request.method == 'POST':
        # Get selected symptoms from checkboxes and text input
        selected_symptoms = request.form.getlist('symptoms')
        custom_symptoms = request.form.get('custom_symptoms', '').split(',')

        # Combine and sort symptoms for consistent key matching
        all_symptoms = sorted(selected_symptoms + [s.strip() for s in custom_symptoms if s.strip()])

        # Disease mappings
        disease_mapping = {
            "cough": "Flu",
            "fever": "Flu",
            "fever,itching,rash": "Allergic Reaction",
            "headache,nausea": "Migraine",
            "chest_pain,shortness_of_breath": "Heart Disease",
            "fatigue,joint_pain": "Arthritis",
            "sore_throat,fever": "Strep Throat",
            "body_aches,fever": "Viral Infection",
            "diarrhea,nausea,vomiting": "Food Poisoning",
            "fatigue,weight_loss": "Diabetes",
            "chills,fever,sweating": "Malaria",
            "abdominal_pain,bloating": "Irritable Bowel Syndrome",
            "blurred_vision,thirst": "Diabetes",
            "fever,skin_lesions": "Chickenpox",
            "joint_pain,rash": "Lupus",
            "runny_nose,sneezing": "Common Cold",
            "fever,night_sweats": "Tuberculosis",
            "frequent_urination,thirst": "Kidney Disease",
            "muscle_pain,tiredness": "Fibromyalgia",
            # Add more mappings as needed
        }

        # Search for partial matches using fuzzy matching
        best_match = None
        highest_score = 0
        for symptoms_key, disease in disease_mapping.items():
            score = fuzz.partial_ratio(",".join(all_symptoms), symptoms_key)
            if score > highest_score:
                highest_score = score
                best_match = disease

        # Default to Unknown if no match is found
        predicted_disease = best_match if highest_score > 70 else "Unknown"

        # Fetch disease information if available
        disease_data = get_disease_info_symptoms(predicted_disease) if predicted_disease != "Unknown" else {}
        suggestions = "\n".join(disease_data.get("suggestions", ["No specific suggestions available."]))
        precautions = "\n".join(disease_data.get("precautions", ["No specific precautions available."]))
        suggested_medicines = "\n".join(disease_data.get("suggested_medicines", ["No specific medicines available."]))

        return render_template(
            'syresult.html',
            disease_name=predicted_disease,
            suggestions=suggestions,
            precautions=precautions,
            suggested_medicines=suggested_medicines
        )
    return render_template('symptoms.html')



MEDICINE_API_KEY = os.getenv("MEDICINE_API_KEY")  # Replace with the actual environment variable name
MEDICINE_API_URL = "https://api.medicine.example/v1/lookup"  # Replace with actual API endpoint


# Route to search for medicine informationfrom rapidfuzz import process, fuzz


@app.route('/medicine', methods=['GET', 'POST'])
def medicine():
    if request.method == 'POST':
        tablet_name = request.form.get("tablet_name")

        # First, attempt to fetch data from the external API
        if MEDICINE_API_KEY and MEDICINE_API_URL:
            headers = {
                "Authorization": f"Bearer {MEDICINE_API_KEY}",
                "Content-Type": "application/json"
            }

            try:
                # Make an API request to get information about the specified medicine
                response = requests.get(f"{MEDICINE_API_URL}/{tablet_name}", headers=headers)
                response.raise_for_status()  # Check for HTTP errors
                medicine_info = response.json()

                # Return API response if successful
                return jsonify(medicine_info)

            except requests.exceptions.RequestException as e:
                print(f"API request error: {e}")

        # If API call fails or no API key, use predefined dictionary as fallback
        medicine_info = {
            "Paracetamol": {
                "uses": "Pain relief and fever reduction",
                "side_effects": "Nausea, liver issues with prolonged use"
            },
            "Ibuprofen": {
                "uses": "Anti-inflammatory, pain relief",
                "side_effects": "Stomach upset, kidney issues with prolonged use"
            },
            "Amoxicillin": {
                "uses": "Treatment of bacterial infections",
                "side_effects": "Nausea, rash, diarrhea"
            },
            "Aspirin": {
                "uses": "Pain relief, anti-inflammatory, blood thinning",
                "side_effects": "Stomach upset, bleeding, allergic reactions"
            },
            "Metformin": {
                "uses": "Management of type 2 diabetes",
                "side_effects": "Nausea, vomiting, risk of lactic acidosis"
            },
            "Lisinopril": {
                "uses": "Treatment of high blood pressure and heart failure",
                "side_effects": "Cough, dizziness, elevated blood potassium levels"
            },
            "Omeprazole": {
                "uses": "Acid reflux and stomach ulcer treatment",
                "side_effects": "Headache, abdominal pain, long-term risk of kidney issues"
            },
            "Cetirizine": {
                "uses": "Relief from allergy symptoms",
                "side_effects": "Drowsiness, dry mouth"
            },
            "Prednisone": {
                "uses": "Anti-inflammatory, immune system suppression",
                "side_effects": "Weight gain, mood changes, high blood pressure"
            },
            "Amlodipine": {
                "uses": "Management of high blood pressure and angina",
                "side_effects": "Swelling, dizziness, fatigue"
            }
        }
        
        # Return information from local data if not found in API
        return jsonify(medicine_info.get(tablet_name, {"error": "No information available"}))

    return render_template('medicine.html')


# Load environment variables (if using .env, otherwise set directly in environment)
#CHATBOT_API_KEY = os.getenv("CHATBOT_API_KEY")  # Replace with the actual environment variable name
#CHATBOT_API_URL = "https://api.chatbot.example/v1/message"  # Replace with actual chatbot API endpoint

# Route for chatbot interaction
# OpenAI API Key
openai.api_key = "YOUR_OPENAI_API_KEYsk-proj-J9SO-wfwRlaZM7uqRDZmo0D_4DCDLJ2iJ25k8MFaUDX7qLtRyGFCbjq8i9bxygjb0nGjFBDMq8T3BlbkFJofugYQPaNhvcceGInkTywLTCh9-Yqh3dpW7pKpejU8-BprBUU5mt1O_Ia2B0hDsal9_ORvXTIA"

# Chatbot Route
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.form.get("message")
        
        if not user_message:
            return jsonify({"reply": "Please enter a message."})
        
        try:
            # Using OpenAI API to generate a chatbot response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant for a medical project. Help users with medical queries."},
                    {"role": "user", "content": user_message}
                ]
            )
            bot_reply = response.choices[0].message["content"].strip()
        except Exception as e:
            print(f"Error connecting to OpenAI API: {e}")
            bot_reply = "There was an issue connecting to the chatbot service. Please try again later."
        
        return jsonify({"reply": bot_reply})
    
    return render_template('chatbot.html')




if __name__ == '__main__':
    app.run(debug=True)
