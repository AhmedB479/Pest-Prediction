import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import lime
import lime.lime_tabular
import shap

def model_comparison_page():
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .model-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🌾 Model Comparison and Explanation Dashboard")
    st.markdown("Compare different machine learning models and understand their predictions using LIME and SHAP explanations.")

    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv("../data/processed_crop.csv")
        return df

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Required data file not found. Please ensure 'processed_crop.csv' is in the correct directory.")
        st.stop()

    # Prepare data
    X = df.drop(columns=["pest"])
    y = df["pest"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    @st.cache_resource
    def train_models():
        # Decision Tree
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        dt_preds = dt.predict(X_test)
        dt_probs = dt.predict_proba(X_test)

        # Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_probs = rf.predict_proba(X_test)

        # SVM
        svm = make_pipeline(StandardScaler(), SVC(probability=True))
        svm.fit(X_train, y_train)
        svm_preds = svm.predict(X_test)
        svm_probs = svm.predict_proba(X_test)

        return {
            'dt': dt,
            'rf': rf,
            'svm': svm,
            'dt_preds': dt_preds,
            'rf_preds': rf_preds,
            'svm_preds': svm_preds,
            'dt_probs': dt_probs,
            'rf_probs': rf_probs,
            'svm_probs': svm_probs
        }

    models = train_models()

    # Display model performance
    st.subheader("Model Performance Comparison")
    
    # Create tabs for different models
    dt_tab, rf_tab, svm_tab = st.tabs(["Decision Tree", "Random Forest", "SVM"])
    
    with dt_tab:
        st.markdown("### Decision Tree Performance")
        st.write("Classification Report:")
        st.text(classification_report(y_test, models['dt_preds']))
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, models['dt_preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Decision Tree Confusion Matrix')
        st.pyplot(fig)

    with rf_tab:
        st.markdown("### Random Forest Performance")
        st.write("Classification Report:")
        st.text(classification_report(y_test, models['rf_preds']))
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, models['rf_preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Random Forest Confusion Matrix')
        st.pyplot(fig)

    with svm_tab:
        st.markdown("### SVM Performance")
        st.write("Classification Report:")
        st.text(classification_report(y_test, models['svm_preds']))
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, models['svm_preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('SVM Confusion Matrix')
        st.pyplot(fig)

    # Model Explanation Section
    st.subheader("Model Explanations")
    
    # Select model for explanation
    model_choice = st.selectbox(
        "Select Model for Explanation",
        ["Random Forest", "Decision Tree", "SVM"]
    )
    
    # Select explanation method
    explanation_method = st.selectbox(
        "Select Explanation Method",
        ["LIME", "SHAP"]
    )
    
    # Select test instance
    instance_idx = st.slider("Select Test Instance", 0, len(X_test)-1, 0)
    test_instance = X_test.iloc[instance_idx:instance_idx+1]
    
    if explanation_method == "LIME":
        st.markdown("### LIME Explanation")
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X.columns.tolist(),
            class_names=[str(c) for c in sorted(y.unique())],
            mode='classification'
        )
        
        # Get model for explanation
        model = models['rf'] if model_choice == "Random Forest" else \
                models['dt'] if model_choice == "Decision Tree" else \
                models['svm']
        
        # Generate LIME explanation
        lime_exp = explainer.explain_instance(
            test_instance.values[0],
            model.predict_proba,
            num_features=6
        )
        
        # Display LIME explanation
        fig = lime_exp.as_pyplot_figure()
        st.pyplot(fig)
        
    else:  # SHAP
        st.markdown("### SHAP Explanation")
        
        if model_choice == "Random Forest":
            # Create SHAP explainer
            explainer = shap.TreeExplainer(models['rf'])
            shap_values = explainer.shap_values(test_instance)
            
            # Get predicted class
            predicted_class = models['rf'].predict(test_instance)[0]
            class_idx = list(models['rf'].classes_).index(predicted_class)
            
            # Generate SHAP decision plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.decision_plot(
                explainer.expected_value[class_idx],
                shap_values[class_idx][0],
                test_instance.values[0],
                feature_names=X_test.columns.tolist(),
                show=False
            )
            st.pyplot(fig)
            
            # Show SHAP summary plot
            st.markdown("### SHAP Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                explainer.shap_values(X_train),
                X_train,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig)
        else:
            st.warning("SHAP explanations are currently only available for Random Forest model.")

    # Add debug information
    with st.expander("Debug Information"):
        st.write("Test Instance Features:")
        st.write(test_instance)
        st.write("\nPredicted Class:", predicted_class if explanation_method == "SHAP" else model.predict(test_instance)[0])
        st.write("\nFeature Names:", X.columns.tolist())

if __name__ == "__main__":
    model_comparison_page()