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

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    @st.cache_resource
    def train_models():
        # Train Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(X_train_scaled, y_train)
        nb_train_pred = nb_model.predict(X_train_scaled)
        nb_test_pred = nb_model.predict(X_test_scaled)
        nb_train_prob = nb_model.predict_proba(X_train_scaled)[:, 1]
        nb_test_prob = nb_model.predict_proba(X_test_scaled)[:, 1]
        
        # Train AdaBoost
        adb_model = AdaBoostClassifier(n_estimators=100, random_state=42)
        adb_model.fit(X_train_scaled, y_train)
        adb_train_pred = adb_model.predict(X_train_scaled)
        adb_test_pred = adb_model.predict(X_test_scaled)
        adb_train_prob = adb_model.predict_proba(X_train_scaled)[:, 1]
        adb_test_prob = adb_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        nb_train_acc = accuracy_score(y_train, nb_train_pred)
        nb_test_acc = accuracy_score(y_test, nb_test_pred)
        adb_train_acc = accuracy_score(y_train, adb_train_pred)
        adb_test_acc = accuracy_score(y_test, adb_test_pred)
        
        # Create DataFrames for predictions
        train_results = pd.DataFrame({
            'Actual': y_train,
            'NB_Prediction': nb_train_pred,
            'NB_Probability': nb_train_prob,
            'AdaBoost_Prediction': adb_train_pred,
            'AdaBoost_Probability': adb_train_prob
        })
        
        test_results = pd.DataFrame({
            'Actual': y_test,
            'NB_Prediction': nb_test_pred,
            'NB_Probability': nb_test_prob,
            'AdaBoost_Prediction': adb_test_pred,
            'AdaBoost_Probability': adb_test_prob
        })
        
        return nb_model, adb_model, {
            'nb_train_acc': nb_train_acc,
            'nb_test_acc': nb_test_acc,
            'adb_train_acc': adb_train_acc,
            'adb_test_acc': adb_test_acc,
            'nb_test_report': classification_report(y_test, nb_test_pred),
            'adb_test_report': classification_report(y_test, adb_test_pred),
            'nb_conf_matrix': confusion_matrix(y_test, nb_test_pred),
            'adb_conf_matrix': confusion_matrix(y_test, adb_test_pred),
            'train_results': train_results,
            'test_results': test_results
        }

    nb_model, adb_model, metrics = train_models()

    # Display model performance metrics
    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Naive Bayes Model")
        st.write(f"Training Accuracy: {metrics['nb_train_acc']:.2%}")
        st.write(f"Testing Accuracy: {metrics['nb_test_acc']:.2%}")
        st.write("Classification Report:")
        st.text(metrics['nb_test_report'])
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(metrics['nb_conf_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Naive Bayes Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### AdaBoost Model")
        st.write(f"Training Accuracy: {metrics['adb_train_acc']:.2%}")
        st.write(f"Testing Accuracy: {metrics['adb_test_acc']:.2%}")
        st.write("Classification Report:")
        st.text(metrics['adb_test_report'])
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(metrics['adb_conf_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('AdaBoost Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig)

    # Add a new section for train-test predictions
    st.subheader("Train-Test Predictions")
    
    # Create tabs for train and test predictions
    train_tab, test_tab = st.tabs(["Training Set Predictions", "Test Set Predictions"])
    
    with train_tab:
        st.markdown("### Training Set Predictions")
        # Display sample of training predictions
        st.write("Sample of Training Set Predictions (First 10 rows):")
        st.dataframe(metrics['train_results'].head(10))
        
        # Plot training predictions distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Naive Bayes training predictions
        sns.histplot(data=metrics['train_results'], x='NB_Probability', hue='Actual', 
                    bins=20, ax=ax1)
        ax1.set_title('Naive Bayes Training Predictions Distribution')
        ax1.set_xlabel('Prediction Probability')
        ax1.set_ylabel('Count')
        
        # AdaBoost training predictions
        sns.histplot(data=metrics['train_results'], x='AdaBoost_Probability', hue='Actual', 
                    bins=20, ax=ax2)
        ax2.set_title('AdaBoost Training Predictions Distribution')
        ax2.set_xlabel('Prediction Probability')
        ax2.set_ylabel('Count')
        
        st.pyplot(fig)
    
    with test_tab:
        st.markdown("### Test Set Predictions")
        # Display sample of test predictions
        st.write("Sample of Test Set Predictions (First 10 rows):")
        st.dataframe(metrics['test_results'].head(10))
        
        # Plot test predictions distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Naive Bayes test predictions
        sns.histplot(data=metrics['test_results'], x='NB_Probability', hue='Actual', 
                    bins=20, ax=ax1)
        ax1.set_title('Naive Bayes Test Predictions Distribution')
        ax1.set_xlabel('Prediction Probability')
        ax1.set_ylabel('Count')
        
        # AdaBoost test predictions
        sns.histplot(data=metrics['test_results'], x='AdaBoost_Probability', hue='Actual', 
                    bins=20, ax=ax2)
        ax2.set_title('AdaBoost Test Predictions Distribution')
        ax2.set_xlabel('Prediction Probability')
        ax2.set_ylabel('Count')
        
        st.pyplot(fig)

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
            'GDD': GDD,
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
        
        # Create a new row for user input
        user_input_row = pd.DataFrame({
            'Actual': [-1],  # Use -1 instead of None to represent user input
            'NB_Prediction': [results["nb_pred"]],
            'NB_Probability': [results["nb_prob"]],
            'AdaBoost_Prediction': [results["adb_pred"]],
            'AdaBoost_Probability': [results["adb_prob"]],
            'User_Input': [True]  # Flag to identify user input
        })
        
        # Append user input to both train and test results
        train_results_with_input = pd.concat([metrics['train_results'], user_input_row], ignore_index=True)
        test_results_with_input = pd.concat([metrics['test_results'], user_input_row], ignore_index=True)
        
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
        
        # Update the train-test predictions section with user input
        st.subheader("Train-Test Predictions with User Input")
        
        # Create tabs for train and test predictions
        train_tab, test_tab = st.tabs(["Training Set Predictions", "Test Set Predictions"])
        
        with train_tab:
            st.markdown("### Training Set Predictions")
            # Display sample of training predictions with user input
            st.write("Sample of Training Set Predictions (Last 10 rows including user input):")
            st.dataframe(train_results_with_input.tail(10).style.applymap(
                lambda x: 'background-color: #ffebee' if x == True else '', 
                subset=['User_Input']
            ))
            
            # Plot training predictions distribution with user input
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Naive Bayes training predictions
            sns.histplot(data=train_results_with_input[train_results_with_input['Actual'] != -1], 
                        x='NB_Probability', hue='Actual', bins=20, ax=ax1)
            # Add user input point
            ax1.axvline(x=results["nb_prob"], color='red', linestyle='--', 
                       label='User Input')
            ax1.set_title('Naive Bayes Training Predictions Distribution')
            ax1.set_xlabel('Prediction Probability')
            ax1.set_ylabel('Count')
            ax1.legend()
            
            # AdaBoost training predictions
            sns.histplot(data=train_results_with_input[train_results_with_input['Actual'] != -1], 
                        x='AdaBoost_Probability', hue='Actual', bins=20, ax=ax2)
            # Add user input point
            ax2.axvline(x=results["adb_prob"], color='red', linestyle='--', 
                       label='User Input')
            ax2.set_title('AdaBoost Training Predictions Distribution')
            ax2.set_xlabel('Prediction Probability')
            ax2.set_ylabel('Count')
            ax2.legend()
            
            st.pyplot(fig)
        
        with test_tab:
            st.markdown("### Test Set Predictions")
            # Display sample of test predictions with user input
            st.write("Sample of Test Set Predictions (Last 10 rows including user input):")
            st.dataframe(test_results_with_input.tail(10).style.applymap(
                lambda x: 'background-color: #ffebee' if x == True else '', 
                subset=['User_Input']
            ))
            
            # Plot test predictions distribution with user input
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Naive Bayes test predictions
            sns.histplot(data=test_results_with_input[test_results_with_input['Actual'] != -1], 
                        x='NB_Probability', hue='Actual', bins=20, ax=ax1)
            # Add user input point
            ax1.axvline(x=results["nb_prob"], color='red', linestyle='--', 
                       label='User Input')
            ax1.set_title('Naive Bayes Test Predictions Distribution')
            ax1.set_xlabel('Prediction Probability')
            ax1.set_ylabel('Count')
            ax1.legend()
            
            # AdaBoost test predictions
            sns.histplot(data=test_results_with_input[test_results_with_input['Actual'] != -1], 
                        x='AdaBoost_Probability', hue='Actual', bins=20, ax=ax2)
            # Add user input point
            ax2.axvline(x=results["adb_prob"], color='red', linestyle='--', 
                       label='User Input')
            ax2.set_title('AdaBoost Test Predictions Distribution')
            ax2.set_xlabel('Prediction Probability')
            ax2.set_ylabel('Count')
            ax2.legend()
            
            st.pyplot(fig)

