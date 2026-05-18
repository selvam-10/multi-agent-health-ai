"""
Generate random live data and make predictions using the trained model.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime


def generate_random_patient_data(num_samples: int = 1, feature_names: list = None):
    """
    Generate random patient data based on realistic ranges from the dataset.
    """
    
    # Default feature ranges based on typical medical values
    ranges = {
        'age': (29, 76),  # reasonable age range
        'sex': (0, 1),  # binary: 0=female, 1=male
        'chest_pain_type': (0, 3),  # 4 types (0-3)
        'resting_bp_s': (90, 200),  # systolic BP range
        'cholesterol': (0, 600),  # cholesterol range (0 means not recorded)
        'fasting_blood_sugar': (0, 1),  # binary
        'resting_ecg': (0, 2),  # 3 types
        'max_heart_rate': (60, 220),  # max heart rate
        'exercise_angina': (0, 1),  # binary
        'oldpeak': (0, 6.2),  # ST depression
        'st_slope': (0, 2),  # 3 types
    }
    
    random_data = []
    for _ in range(num_samples):
        sample = {}
        for feature in feature_names:
            if feature in ranges:
                low, high = ranges[feature]
                if feature in ['sex', 'fasting_blood_sugar', 'exercise_angina', 'chest_pain_type', 'resting_ecg', 'st_slope']:
                    # Integer features
                    sample[feature] = float(np.random.randint(int(low), int(high) + 1))
                else:
                    # Continuous features
                    sample[feature] = float(np.random.uniform(low, high))
        random_data.append(sample)
    
    return pd.DataFrame(random_data)


def predict_heart_disease(num_samples: int = 5):
    """
    Generate random patient data and predict heart disease risk.
    """
    
    # Load model, scaler, and feature names
    try:
        model = joblib.load("model.joblib")
        scaler = joblib.load("scaler.joblib")
        feature_names = joblib.load("feature_names.joblib")
        print("✅ Model loaded successfully!\n")
    except FileNotFoundError:
        print("❌ Model files not found. Please run train_model.py first.")
        return
    
    # Generate random patient data
    print(f"🔄 Generating {num_samples} random patient profiles...\n")
    patient_data = generate_random_patient_data(num_samples, feature_names)
    
    # Make predictions
    X_scaled = scaler.transform(patient_data)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Display results
    print("=" * 100)
    print(f"HEART DISEASE PREDICTION RESULTS | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print()
    
    for idx, (row_data, pred, prob) in enumerate(zip(patient_data.to_dict('records'), predictions, probabilities)):
        risk_percentage = prob[1] * 100  # probability of heart disease
        prediction_label = "🔴 HIGH RISK - Heart Disease Likely" if pred == 1 else "🟢 LOW RISK - No Heart Disease"
        
        print(f"PATIENT {idx + 1}:")
        print(f"  {prediction_label}")
        print(f"  Heart Disease Probability: {risk_percentage:.2f}%")
        print()
        
        # Display patient vitals
        print(f"  Patient Vitals:")
        print(f"    Age: {row_data['age']:.1f} years")
        print(f"    Sex: {'Male' if row_data['sex'] == 1 else 'Female'}")
        print(f"    Resting BP (Systolic): {row_data['resting_bp_s']:.1f} mmHg")
        print(f"    Cholesterol: {row_data['cholesterol']:.1f} mg/dL")
        print(f"    Max Heart Rate: {row_data['max_heart_rate']:.1f} bpm")
        print(f"    Chest Pain Type: {int(row_data['chest_pain_type'])}")
        print(f"    ST Depression (oldpeak): {row_data['oldpeak']:.2f}")
        print(f"    Exercise Angina: {'Yes' if row_data['exercise_angina'] == 1 else 'No'}")
        print()
        print("-" * 100)
        print()
    
    # Save predictions to CSV
    patient_data['prediction'] = predictions
    patient_data['heart_disease_probability_%'] = (probabilities[:, 1] * 100).round(2)
    patient_data['risk_level'] = patient_data['heart_disease_probability_%'].apply(
        lambda x: 'HIGH RISK' if x >= 70 else ('MEDIUM RISK' if x >= 40 else 'LOW RISK')
    )
    
    output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    patient_data.to_csv(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}\n")
    
    return patient_data


if __name__ == "__main__":
    import sys
    
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    predict_heart_disease(num_samples)
