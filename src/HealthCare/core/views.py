"""
Views for the core Django application.

This file contains the logic that handles requests and returns responses.
It includes views for the home page, user authentication (login, logout, signup),
and the user profile page.
"""
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from urllib.parse import urlencode
from .forms import SymptomForm, SignUpForm, LoginForm
import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import matplotlib.pyplot as plt
from django.conf import settings
from django.utils.safestring import mark_safe
import json

# --- Mock Data for Demonstration (Expanded for better results) ---
# This data is used as a fallback if the .pkl model files are not found.
diseases_symptoms_map = {
    'Fungal infection': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches'],
    'Allergy': ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes', 'sneezing', 'itching', 'runny_nose', 'redness_of_eyes'],
    'GERD': ['stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting', 'cough', 'chest_pain'],
    'Chronic cholestasis': ['itching', 'vomiting', 'yellowish_skin', 'nausea', 'loss_of_appetite'],
    'Drug Reaction': ['itching', 'skin_rash', 'stomach_pain', 'burning_micturition', 'spotting_urination'],
    'Peptic ulcer disease': ['vomiting', 'indigestion', 'stomach_pain', 'loss_of_appetite'],
    'AIDS': ['muscle_wasting', 'patches_in_throat', 'high_fever', 'extra_marital_contacts'],
    'Diabetes': ['fatigue', 'weight_loss', 'restlessness', 'lethargy', 'irregular_sugar_level'],
    'Gastroenteritis': ['vomiting', 'diarrhoea', 'dehydration', 'sunken_eyes'],
    'Bronchial Asthma': ['fatigue', 'cough', 'high_fever', 'breathlessness', 'chest_pain'],
    'Hypertension': ['headache', 'chest_pain', 'dizziness', 'loss_of_balance'],
    'Migraine': ['acidity', 'indigestion', 'headache', 'blurred_and_distorted_vision', 'vomiting', 'nausea', 'light_sensitivity'],
    'Cervical spondylosis': ['back_pain', 'weakness_in_limbs', 'neck_pain', 'dizziness'],
    'Psoriasis': ['skin_rash', 'joint_pain', 'skin_peeling', 'silver_like_dusting'],
    'Chicken pox': ['skin_rash', 'itching', 'high_fever', 'red_spots', 'loss_of_appetite'],
    'Osteoarthritis': ['joint_pain', 'neck_pain', 'knee_pain', 'swelling_joints'],
    'Typhoid': ['chills', 'high_fever', 'nausea', 'stomach_pain', 'diarrhoea'],
    'Common Cold': ['cough', 'runny_nose', 'sneezing', 'sore_throat'],
    'Influenza': ['high_fever', 'cough', 'fatigue', 'headache'],
    'Stomach_Flu': ['stomach_pain', 'vomiting', 'diarrhoea', 'nausea'],
}

severity_data = {
    'high_fever': 3,
    'vomiting': 4,
    'headache': 2,
    'fatigue': 2,
    'diarrhoea': 3,
    'cough': 1,
    'runny_nose': 1,
    'stomach_pain': 3,
    'joint_pain': 2,
    'weight_loss': 3,
    'chest_pain': 4,
    'dizziness': 2,
    'breathlessness': 4,
    'itching': 1,
    'skin_rash': 2,
    'chills': 2,
    'nausea': 2,
    'loss_of_appetite': 2,
    'acidity': 1,
    'ulcers_on_tongue': 1,
    'dehydration': 3,
    'sunken_eyes': 2,
}

# -----------------
# Load models and data safely
# -----------------
# Get the absolute path to the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the Models directory
MODELS_DIR = os.path.join(BASE_DIR, "Models")

fpg_model = None
precautions_vec = None
precautions_data_list = None

# Load the FP-growth model (association rules)
try:
    fpg_path = os.path.join(MODELS_DIR, "FPgrowth.pkl")
    fpg_model = joblib.load(fpg_path)
    if not isinstance(fpg_model, pd.DataFrame) or fpg_model.empty:
        raise ValueError("FPgrowth.pkl loaded but is not a valid or non-empty DataFrame.")
    print("Disease prediction model (FPgrowth.pkl) loaded successfully!")
except (FileNotFoundError, ValueError) as e:
    print(f"Error loading FPgrowth.pkl: {e}. Using mock data.")

# Load the precautions vectorizer and data
try:
    precautions_vec_path = os.path.join(MODELS_DIR, "precautionsVEC.pkl")
    precautions_data_path = os.path.join(MODELS_DIR, "precautions_data.pkl")
    
    precautions_vec = joblib.load(precautions_vec_path)
    if not hasattr(precautions_vec, 'transform'):
        raise TypeError("precautionsVEC.pkl is not a valid vectorizer.")
    
    precautions_data_list = joblib.load(precautions_data_path)
    if not isinstance(precautions_data_list, list):
        raise TypeError("precautions_data.pkl is not a valid list.")
        
    print("Precautions model and data loaded successfully!")
except (FileNotFoundError, TypeError) as e:
    print(f"Error loading precautions models: {e}. Using mock data.")
    precautions_data_list = [
        "consult a physician", "drink plenty of fluids", "get plenty of rest",
        "eat light and healthy food", "use an antihistamine", "avoid sun exposure",
        "wear a face mask", "maintain hygiene", "avoid cold food", "take antibiotics",
        "take vaccination", "exercise regularly", "meditation and yoga", "avoid stress"
    ]
    precautions_vec = TfidfVectorizer()
    precautions_vec.fit_transform(precautions_data_list)


# A predefined list of symptoms for the dropdowns
SYMPTOM_LIST = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
    'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain',
    'anxiety', 'cold_hands_and_feet', 'mood_swings', 'weight_loss', 'restlessness',
    'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever',
    'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
    'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
    'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
    'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
    'light_sensitivity', 'sore_throat', 'sneezing',
]

def calculate_jaccard_similarity(set1, set2):
    """Calculates the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# -----------------
# Helper function for prediction
# -----------------
def safe_predict(symptoms):
    """
    symptoms: list of input symptoms (strings)
    returns: dict with predictions, precautions, severity, and other info.
    """
    results = {
        "predictions": [],
        "precautions": [],
        "severity": "Unknown",
        "possible_diseases": {}
    }

    # FP-Growth prediction (Primary)
    if fpg_model is not None:
        try:
            symptom_set = frozenset(symptoms)
            # Find a direct match based on the `antecedents` of the association rules
            direct_match = fpg_model[fpg_model['antecedents'] == symptom_set]
            
            if not direct_match.empty:
                best_match = direct_match.sort_values(by='confidence', ascending=False).iloc[0]
                predicted_disease = list(best_match['consequents'])[0] if isinstance(best_match['consequents'], frozenset) else best_match['consequents']
                results["predictions"] = [predicted_disease]
            else:
                # If no direct match, use Jaccard similarity for other possibilities
                symptom_set = set(symptoms)
                
                # FIX: Corrected variable name to use symptoms from diseases_symptoms_map
                jaccard_scores = {disease: calculate_jaccard_similarity(symptom_set, set(disease_symptoms))
                                  for disease, disease_symptoms in diseases_symptoms_map.items()}
                
                # Sort and filter diseases with a Jaccard score above a threshold (e.g., 0.3)
                possible_diseases = sorted(
                    [(disease, score) for disease, score in jaccard_scores.items() if score > 0.0],
                    key=lambda item: item[1], reverse=True
                )
                
                if possible_diseases:
                    # Select the top 5 possible diseases
                    top_possible_diseases = possible_diseases[:5]
                    results["predictions"] = ["No direct match found."]
                    results["possible_diseases"] = {d: round(s * 100, 2) for d, s in top_possible_diseases}
                else:
                    results["predictions"] = ["No close match found."]

        except Exception as e:
            results["predictions"] = [f"Error in FP-Growth prediction: {e}"]
    else:
        # Fallback to pure Jaccard similarity if no model is loaded
        symptom_set = set(symptoms)
        jaccard_scores = {disease: calculate_jaccard_similarity(symptom_set, set(disease_symptoms))
                          for disease, disease_symptoms in diseases_symptoms_map.items()}
        
        possible_diseases = sorted(
            [(disease, score) for disease, score in jaccard_scores.items() if score > 0.0],
            key=lambda item: item[1], reverse=True
        )
        
        if possible_diseases:
            top_possible_diseases = possible_diseases[:5]
            results["predictions"] = ["No direct match found."]
            results["possible_diseases"] = {d: round(s * 100, 2) for d, s in top_possible_diseases}
        else:
            results["predictions"] = ["No close match found."]

    # Precautions recommendation (cosine similarity)
    if precautions_vec is not None and precautions_data_list:
        try:
            input_text = " ".join(symptoms)
            input_vec = precautions_vec.transform([input_text])
            precautions_vec_matrix = precautions_vec.transform(precautions_data_list)
            sims = cosine_similarity(input_vec, precautions_vec_matrix)

            # Get top indices with better filtering
            top_indices = np.argsort(sims[0])[-10:][::-1]
            top_precautions = [precautions_data_list[i] for i in top_indices]
            
            # Remove duplicates while preserving order
            unique_precautions = []
            seen = set()
            for p in top_precautions:
                if p not in seen:
                    unique_precautions.append(p)
                    seen.add(p)

            # If we don't have enough unique precautions, add random ones
            if len(unique_precautions) < 5:
                all_unique_precautions = list(set(precautions_data_list))
                random.shuffle(all_unique_precautions)
                for p in all_unique_precautions:
                    if p not in seen and len(unique_precautions) < 5:
                        unique_precautions.append(p)
                        seen.add(p)
            
            results["precautions"] = unique_precautions

        except Exception as e:
            results["precautions"] = [f"Error in precaution matching: {e}"]
    else:
        results["precautions"] = ["Precautions model missing."]

    # Severity Calculation
    if severity_data:
        try:
            total_severity = sum(severity_data.get(symptom, 0) for symptom in symptoms)
            num_symptoms = len([s for s in symptoms if s in severity_data])
            if num_symptoms > 0:
                avg_severity = total_severity / num_symptoms
                if avg_severity > 3:
                    results["severity"] = "High"
                elif avg_severity > 1:
                    results["severity"] = "Medium"
                else:
                    results["severity"] = "Low"
            else:
                results["severity"] = "Low"
        except Exception as e:
            results["severity"] = "Error in severity calculation."

    return results

# -----------------
# Django Views
# -----------------

def home(request):
    """
    Handles the home page logic with the symptom input form.
    Upon successful submission, it redirects to the results page.
    """
    form = SymptomForm(request.POST or None, symptoms=SYMPTOM_LIST)

    if request.method == 'POST':
        if form.is_valid():
            symptoms = [form.cleaned_data[f'symptom_{i}'] for i in range(1, 5) if form.cleaned_data.get(f'symptom_{i}')]
            text_symptom = form.cleaned_data.get('symptom_text')

            if text_symptom:
                processed_text_symptom = text_symptom.lower().replace(' ', '_')
                if processed_text_symptom in SYMPTOM_LIST:
                    symptoms.append(processed_text_symptom)

            if not symptoms:
                return render(request, "core/home.html", {'form': form, "error": "Please select or enter at least one symptom."})
            else:
                results = safe_predict(symptoms)

                # Generate and save the pie chart
                if results.get("possible_diseases"):
                    labels = list(results["possible_diseases"].keys())
                    values = list(results["possible_diseases"].values())
                    fig, ax = plt.subplots()
                    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
                    ax.axis("equal")
                    if not os.path.exists(settings.MEDIA_ROOT):
                        os.makedirs(settings.MEDIA_ROOT)
                    chart_path = os.path.join(settings.MEDIA_ROOT, "disease_pie.png")
                    plt.savefig(chart_path)
                    plt.close()
                    results["chart_url"] = settings.MEDIA_URL + "disease_pie.png"

                # Pass the data to the results view via URL parameters
                query_string = urlencode({
                    'predictions': ','.join(results["predictions"]),
                    'precautions': ','.join(results["precautions"]),
                    'severity': results["severity"],
                    'chart_url': results.get("chart_url", ""),
                    'input_symptoms': ','.join(symptoms),
                    'possible_diseases': str(results.get("possible_diseases", {}))
                })
                return redirect(f'{reverse("results")}?{query_string}')

    return render(request, "core/home.html", {'form': form})

def results_view(request):
    """
    Displays the analysis results. Data is passed via URL parameters.
    Saves the result to the user's session for the profile page.
    """
    predictions = request.GET.get('predictions', '').split(',')
    precautions = request.GET.get('precautions', '').split(',')
    severity = request.GET.get('severity', '')
    chart_url = request.GET.get('chart_url', '')
    input_symptoms = request.GET.get('input_symptoms', '').split(',')
    possible_diseases_str = request.GET.get('possible_diseases', '{}')
    
    # Safely convert the string back to a dictionary
    try:
        possible_diseases = eval(possible_diseases_str)
    except (SyntaxError, NameError):
        possible_diseases = {}

    # Store the result in the session for the profile page
    health_history = request.session.get('health_history', [])
    health_history.append({
        'symptoms': input_symptoms,
        'predictions': predictions,
        'possible_diseases': possible_diseases
    })
    request.session['health_history'] = health_history

    context = {
        "predictions": predictions,
        "precautions": precautions,
        "severity": severity,
        "chart_url": chart_url,
        "input_symptoms": input_symptoms,
        "possible_diseases": possible_diseases,
    }

    return render(request, 'core/results.html', context)


def login_view(request):
    """
    Handles user login.
    """
    if request.user.is_authenticated:
        return redirect('home')
    form = LoginForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                form.add_error(None, "Invalid username or password.")
    return render(request, 'core/login.html', {'form': form})


def signup_view(request):
    """
    Handles new user registration.
    """
    if request.user.is_authenticated:
        return redirect('home')
    form = SignUpForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    return render(request, 'core/signup.html', {'form': form})


def logout_view(request):
    """
    Logs the user out and redirects to the home page.
    """
    logout(request)
    return redirect('home')


@login_required
def profile(request):
    """
    Displays the user's profile page with charts of their past inputs.
    """
    health_history = request.session.get('health_history', [])

    symptom_counts = defaultdict(int)
    disease_counts = defaultdict(int)

    for entry in health_history:
        # Count symptoms
        for symptom in entry.get('symptoms', []):
            if symptom and symptom != "No close match found.":
                symptom_counts[symptom] += 1
        
        # Count predicted diseases
        predictions = entry.get('predictions', [])
        if "No direct match found." in predictions:
            # If no direct match, count the possible diseases
            for disease, score in entry.get('possible_diseases', {}).items():
                disease_counts[disease] += 1
        else:
            for disease in predictions:
                if disease and disease != "No close match found.":
                    disease_counts[disease] += 1
    
    symptom_data = {
        'labels': list(symptom_counts.keys()),
        'data': list(symptom_counts.values()),
    }
    
    disease_data = {
        'labels': list(disease_counts.keys()),
        'data': list(disease_counts.values()),
    }
    
    context = {
        'user': request.user,
        'symptom_data': mark_safe(json.dumps(symptom_data)),
        'disease_data': mark_safe(json.dumps(disease_data)),
    }
    
    return render(request, 'core/profile.html', context)
