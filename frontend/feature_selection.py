import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

def feature_selection_page():

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .prediction-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .pest {
            # background-color: #ffebee;
            border: 1px solid #ffcdd2;
        }
        .no-pest {
            # background-color: #e8f5e9;
            border: 1px solid #c8e6c9;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("🌾 Cotton Pest Prediction Dashboard")
    st.markdown("Enter crop conditions to predict the likelihood of pest infestation using machine learning models.")

    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv("../data/processed_crop.csv")
        original_df = pd.read_csv("../data/Crop.csv")
        return df, original_df

    try:
        df, original_df = load_data()
    except FileNotFoundError:
        st.error("Required data files not found. Please ensure 'processed_crop.csv' and 'Crop.csv' are in the correct directory.")
        st.stop()

    # Get the feature names from training data
    feature_columns = df.drop(columns=["pest"]).columns.tolist()

    # Setup encoders for dropdowns
    label_encoder = LabelEncoder()
    district_encoder = LabelEncoder()

    # Recreate label mappings
    label_encoder.fit(original_df["label"])
    district_encoder.fit(original_df["district"])

    # Prepare data
    X = df.drop(columns=["pest"])
    y = df["pest"]

    

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train models
    @st.cache_resource
    def train_models():
        print(y)
        print(X_scaled)
        nb_model = GaussianNB()
        nb_model.fit(X_scaled, y)

        adb_model = AdaBoostClassifier(n_estimators=100, random_state=42)
        adb_model.fit(X_scaled, y)
        
        return nb_model, adb_model

    nb_model, adb_model = train_models()

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Environmental Conditions")
        nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0, max_value=1000.0, value=50.0)
        phosphorus = st.number_input("Phosphorus (kg/ha)", min_value=0.0, max_value=1000.0, value=50.0)
        potassium = st.number_input("Potassium (kg/ha)", min_value=0.0, max_value=1000.0, value=50.0)
        temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
        hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=100.0)
        gdd = st.number_input("Growing Degree Days (°C)", min_value=0.0, max_value=1000.0, value=100.0)

    with col2:
        st.subheader("Location & Crop")
        label = st.selectbox("Crop Label", options=label_encoder.classes_.tolist())
        district = st.selectbox("District", options=district_encoder.classes_.tolist())

    # Prediction function
    def predict_pest(temp, hum, ph, rainfall, label, district, nitrogen, phosphorus, potassium, gdd):
        # Calculate GDD
        GDD = max(0, temp - 10)

        # Encode categorical features
        label_encoded = label_encoder.transform([label])[0]
        district_encoded = district_encoder.transform([district])[0]

        # Create base features dictionary with correct feature names
        input_dict = {
            'N': nitrogen,
            'P': phosphorus,
            'K': potassium,
            'temperature': temp,
            'humidity': hum,
            'ph': ph,
            'rainfall': rainfall,
            'GDD': GDD,  # Note: Using uppercase GDD to match training data
            'label_encoded': label_encoded,
            'district_encoded': district_encoded
        }

        # Create a DataFrame with all feature columns (initialize to 0)
        input_df = pd.DataFrame(columns=feature_columns)
        input_df = pd.concat([input_df, pd.DataFrame([input_dict])], ignore_index=True).fillna(0)

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Get predictions
        nb_pred = nb_model.predict(input_scaled)[0]
        nb_prob = nb_model.predict_proba(input_scaled)[0][1]

        adb_pred = adb_model.predict(input_scaled)[0]
        adb_prob = adb_model.predict_proba(input_scaled)[0][1]

        return {
            "nb_pred": nb_pred,
            "nb_prob": nb_prob,
            "adb_pred": adb_pred,
            "adb_prob": adb_prob,
            "gdd": GDD
        }

    # Prediction button
    if st.button("Predict Pest Infestation", type="primary"):
        results = predict_pest(temp, hum, ph, rainfall, label, district, nitrogen, phosphorus, potassium, gdd)
        
        # Display results in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Naive Bayes Model")
            nb_class = "pest" if results["nb_pred"] == 1 else "no-pest"
            st.markdown(f"""
                <div class="prediction-box {nb_class}">
                    <h4>Prediction: {'🌿 Pest Detected' if results['nb_pred'] == 1 else '✅ No Pest'}</h4>
                    <p>Confidence: {results['nb_prob']:.1%}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### AdaBoost Model")
            adb_class = "pest" if results["adb_pred"] == 1 else "no-pest"
            st.markdown(f"""
                <div class="prediction-box {adb_class}">
                    <h4>Prediction: {'🌿 Pest Detected' if results['adb_pred'] == 1 else '✅ No Pest'}</h4>
                    <p>Confidence: {results['adb_prob']:.1%}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Display GDD
        st.markdown(f"### Growing Degree Days (GDD): {results['gdd']:.1f}°C")

