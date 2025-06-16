import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
            border: 1px solid #ffcdd2;
        }
        .no-pest {
            border: 1px solid #c8e6c9;
        }
        .debug-info {
            font-family: monospace;
            padding: 1rem;
            border-radius: 0.5rem;
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
        
        # Calculate GDD consistently
        df['GDD'] = df['temperature'].apply(lambda x: max(0, x - 10))
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
    label_encoder.fit(original_df["label"])
    district_encoder.fit(original_df["district"])

    # Prepare data
    X = df.drop(columns=["pest"])
    y = df["pest"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    @st.cache_resource
    def train_models():
        # Train models
        nb_model = GaussianNB()
        adb_model = AdaBoostClassifier(n_estimators=100, random_state=42)
        
        nb_model.fit(X_train_scaled, y_train)
        adb_model.fit(X_train_scaled, y_train)
        
        # Get predictions
        nb_train_pred = nb_model.predict(X_train_scaled)
        nb_test_pred = nb_model.predict(X_test_scaled)
        adb_train_pred = adb_model.predict(X_train_scaled)
        adb_test_pred = adb_model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'nb_train_acc': accuracy_score(y_train, nb_train_pred),
            'nb_test_acc': accuracy_score(y_test, nb_test_pred),
            'adb_train_acc': accuracy_score(y_train, adb_train_pred),
            'adb_test_acc': accuracy_score(y_test, adb_test_pred),
            'nb_test_report': classification_report(y_test, nb_test_pred),
            'adb_test_report': classification_report(y_test, adb_test_pred),
            'nb_conf_matrix': confusion_matrix(y_test, nb_test_pred),
            'adb_conf_matrix': confusion_matrix(y_test, adb_test_pred),
            'train_results': pd.DataFrame({
                'Actual': y_train,
                'NB_Prediction': nb_train_pred,
                'NB_Probability': nb_model.predict_proba(X_train_scaled)[:, 1],
                'AdaBoost_Prediction': adb_train_pred,
                'AdaBoost_Probability': adb_model.predict_proba(X_train_scaled)[:, 1]
            }),
            'test_results': pd.DataFrame({
                'Actual': y_test,
                'NB_Prediction': nb_test_pred,
                'NB_Probability': nb_model.predict_proba(X_test_scaled)[:, 1],
                'AdaBoost_Prediction': adb_test_pred,
                'AdaBoost_Probability': adb_model.predict_proba(X_test_scaled)[:, 1]
            })
        }
        
        return nb_model, adb_model, metrics

    nb_model, adb_model, metrics = train_models()

    # Display model performance
    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Naive Bayes")
        st.write(f"Training Accuracy: {metrics['nb_train_acc']:.2%}")
        st.write(f"Testing Accuracy: {metrics['nb_test_acc']:.2%}")
        fig, ax = plt.subplots()
        sns.heatmap(metrics['nb_conf_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### AdaBoost")
        st.write(f"Training Accuracy: {metrics['adb_train_acc']:.2%}")
        st.write(f"Testing Accuracy: {metrics['adb_test_acc']:.2%}")
        fig, ax = plt.subplots()
        sns.heatmap(metrics['adb_conf_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

    # Input form
    st.subheader("Make a Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0, value=50.0)
            phosphorus = st.number_input("Phosphorus (kg/ha)", min_value=0.0, value=50.0)
            potassium = st.number_input("Potassium (kg/ha)", min_value=0.0, value=50.0)
            temp = st.number_input("Temperature (°C)", min_value=-50.0, value=25.0)
            hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
            
        with col2:
            ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)
            label = st.selectbox("Crop Label", label_encoder.classes_)
            district = st.selectbox("District", district_encoder.classes_)
        
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Calculate GDD consistently with training data
        GDD = max(0, temp - 10)
        
        # Prepare input data with correct feature order
        input_data = [
            nitrogen, phosphorus, potassium, temp, hum, ph, rainfall, GDD,
            label_encoder.transform([label])[0],
            district_encoder.transform([district])[0]
        ]
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data], columns=feature_columns)
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Get predictions
        nb_pred = nb_model.predict(input_scaled)[0]
        nb_prob = nb_model.predict_proba(input_scaled)[0][1]
        adb_pred = adb_model.predict(input_scaled)[0]
        adb_prob = adb_model.predict_proba(input_scaled)[0][1]
        
        # Display results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
                <div class="prediction-box {'pest' if nb_pred == 1 else 'no-pest'}">
                    <h3>Naive Bayes</h3>
                    <p>Prediction: {'🌿 Pest Detected' if nb_pred == 1 else '✅ No Pest'}</p>
                    <p>Confidence: {nb_prob:.1%}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="prediction-box {'pest' if adb_pred == 1 else 'no-pest'}">
                    <h3>AdaBoost</h3>
                    <p>Prediction: {'🌿 Pest Detected' if adb_pred == 1 else '✅ No Pest'}</p>
                    <p>Confidence: {adb_prob:.1%}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Debug information
        with st.expander("Debug Information"):
            st.markdown("""
                <div class="debug-info">
                    <h4>Input Features</h4>
                    <pre>{}</pre>
                    <h4>Scaled Features</h4>
                    <pre>{}</pre>
                    <h4>Feature Columns Order</h4>
                    <pre>{}</pre>
                </div>
            """.format(
                input_df.to_string(),
                pd.DataFrame(input_scaled, columns=feature_columns).to_string(),
                "\n".join(feature_columns)
            ), unsafe_allow_html=True)

        # Add user input to results
        user_input_row = pd.DataFrame({
            'Actual': [-1],
            'NB_Prediction': [nb_pred],
            'NB_Probability': [nb_prob],
            'AdaBoost_Prediction': [adb_pred],
            'AdaBoost_Probability': [adb_prob],
            'User_Input': [True]
        })
        
        # Display results with user input
        st.subheader("Training Set Predictions with Your Input")
        train_results_with_input = pd.concat([metrics['train_results'], user_input_row])
        st.dataframe(train_results_with_input.tail(10))
        
        # Plot distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Naive Bayes plot
        sns.histplot(data=train_results_with_input[train_results_with_input['Actual'] != -1], 
                    x='NB_Probability', hue='Actual', bins=20, ax=ax1)
        ax1.axvline(x=nb_prob, color='red', linestyle='--', label='Your Input')
        ax1.set_title('Naive Bayes Probability Distribution')
        ax1.legend()
        
        # AdaBoost plot
        sns.histplot(data=train_results_with_input[train_results_with_input['Actual'] != -1], 
                    x='AdaBoost_Probability', hue='Actual', bins=20, ax=ax2)
        ax2.axvline(x=adb_prob, color='red', linestyle='--', label='Your Input')
        ax2.set_title('AdaBoost Probability Distribution')
        ax2.legend()
        
        st.pyplot(fig)

if __name__ == "__main__":
    feature_selection_page()