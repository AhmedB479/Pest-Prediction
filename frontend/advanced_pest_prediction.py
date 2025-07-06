import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import shap
from lime.lime_tabular import LimeTabularExplainer
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def advanced_pest_prediction_page():
    st.title("🔬 Advanced Pest Prediction & Model Interpretation")
    st.markdown("Comprehensive pest prediction system using trained StackingRegressor models with detailed analysis")
    
    # Load data and models
    @st.cache_data
    def load_data():
        try:
            df = pd.read_excel("../data/final_data.xlsx")
            df.fillna(0, inplace=True)
            df = df.drop(columns=['TOTAL AREA VISITED', 'TOTAL SPOTS VISITED'], errors='ignore')
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    @st.cache_resource
    def load_models():
        model_dir = "../best_models"
        models = {}
        
        target_cols = ['W. FLY(ABOVE ETL)', 'JASSID(ABOVE ETL)', 'THRIPS(ABOVE ETL)',
                       'M.BUG(ABOVE ETL)', 'MITES(ABOVE ETL)', 'APHIDS(ABOVE ETL)',
                       'DUSKY COTTON BUG(ABOVE ETL)', 'W. FLY(BELOW ETL)', 'JASSID(BELOW ETL)',
                       'THRIPS(BELOW ETL)', 'MITES(BELOW ETL)', 'APHIDS(BELOW ETL)',
                       'DUSKY COTTON BUG(BELOW ETL)', 'SBW(ABOVE ETL)', 'PBW(ABOVE ETL)',
                       'ABW(ABOVE ETL)', 'Army Worm(ABOVE ETL)', 'SBW(BELOW ETL)',
                       'PBW(BELOW ETL)', 'ABW(BELOW ETL)', 'CLCV(%SPOT)', 'CLCV(%AREA)',
                       'WILT(%SPOT)', 'WILT(%AREA)']
        
        for target in target_cols:
            model_path = os.path.join(model_dir, f"best_model_StackingRegressor_{target}.joblib")
            if os.path.exists(model_path):
                try:
                    models[target] = joblib.load(model_path)
                except Exception as e:
                    st.warning(f"Could not load model for {target}: {e}")
        
        return models, target_cols
    
    # Load data and models
    df = load_data()
    if df is None:
        return
        
    models, target_cols = load_models()
    
    if not models:
        st.error("No models found in best_models directory!")
        return
    
    # Prepare data
    X = df.drop(columns=target_cols, errors='ignore')
    y = df[target_cols]
    
    # Sidebar for navigation
    st.sidebar.title("📊 Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["📈 Train/Test Performance", "🔮 Future Predictions", "🎯 Model Interpretation", "📊 Feature Analysis"]
    )
    
    if analysis_type == "📈 Train/Test Performance":
        show_train_test_performance(X, y, models, target_cols)
    elif analysis_type == "🔮 Future Predictions":
        show_future_predictions(X, y, models, target_cols)
    elif analysis_type == "🎯 Model Interpretation":
        show_model_interpretation(X, y, models, target_cols)
    elif analysis_type == "📊 Feature Analysis":
        show_feature_analysis(X, y, models, target_cols)

def show_train_test_performance(X, y, models, target_cols):
    st.header("📈 Train/Test Split Performance Analysis")
    
    # Train/test split parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
    with col2:
        random_state = st.selectbox("Random State", [0, 21, 42, 77, 101])
    with col3:
        selected_targets = st.multiselect("Select Targets", target_cols, default=target_cols[:5])
    
    if st.button("🚀 Run Performance Analysis"):
        with st.spinner("Running performance analysis..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Calculate performance metrics
            results = []
            for target in selected_targets:
                if target in models:
                    model = models[target]
                    
                    # Train predictions
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    train_r2 = r2_score(y_train[target], y_train_pred)
                    test_r2 = r2_score(y_test[target], y_test_pred)
                    train_mae = mean_absolute_error(y_train[target], y_train_pred)
                    test_mae = mean_absolute_error(y_test[target], y_test_pred)
                    train_rmse = np.sqrt(mean_squared_error(y_train[target], y_train_pred))
                    test_rmse = np.sqrt(mean_squared_error(y_test[target], y_test_pred))
                    
                    results.append({
                        'Target': target,
                        'Train R²': train_r2,
                        'Test R²': test_r2,
                        'Train MAE': train_mae,
                        'Test MAE': test_mae,
                        'Train RMSE': train_rmse,
                        'Test RMSE': test_rmse,
                        'Overfitting': train_r2 - test_r2
                    })
            
            # Display results
            results_df = pd.DataFrame(results)
            st.subheader("📊 Performance Metrics")
            st.dataframe(results_df.round(4))
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("R² Score Comparison")
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(results_df))
                width = 0.35
                ax.bar(x - width/2, results_df['Train R²'], width, label='Train R²', alpha=0.8)
                ax.bar(x + width/2, results_df['Test R²'], width, label='Test R²', alpha=0.8)
                ax.set_xlabel('Targets')
                ax.set_ylabel('R² Score')
                ax.set_title('Train vs Test R² Scores')
                ax.set_xticks(x)
                ax.set_xticklabels(results_df['Target'], rotation=45, ha='right')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.subheader("Overfitting Analysis")
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(results_df['Target'], results_df['Overfitting'])
                ax.set_ylabel('Overfitting (Train R² - Test R²)')
                ax.set_title('Model Overfitting Analysis')
                ax.tick_params(axis='x', rotation=45)
                
                # Color bars based on overfitting level
                for i, bar in enumerate(bars):
                    overfitting = results_df.iloc[i]['Overfitting']
                    if overfitting > 0.1:
                        bar.set_color('red')
                    elif overfitting > 0.05:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
                
                plt.tight_layout()
                st.pyplot(fig)

def show_future_predictions(X, y, models, target_cols):
    st.header("🔮 Future Pest Prediction")
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Create input form
    st.subheader("📝 Enter Field Conditions")
    
    # Organize inputs in columns
    input_data = {}
    
    # Split features into categories for better organization
    weather_features = [f for f in feature_names if any(word in f.lower() for word in ['temp', 'humid', 'rain', 'wind'])]
    soil_features = [f for f in feature_names if any(word in f.lower() for word in ['soil', 'ph', 'moist'])]
    crop_features = [f for f in feature_names if any(word in f.lower() for word in ['crop', 'variety', 'stage', 'age'])]
    location_features = [f for f in feature_names if any(word in f.lower() for word in ['district', 'region', 'location'])]
    other_features = [f for f in feature_names if f not in weather_features + soil_features + crop_features + location_features]
    
    # Create tabs for different feature categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌤️ Weather", "🌱 Soil", "🌾 Crop", "📍 Location", "📊 Other"])
    
    with tab1:
        st.subheader("Weather Conditions")
        col1, col2 = st.columns(2)
        for i, feature in enumerate(weather_features):
            with col1 if i % 2 == 0 else col2:
                if X[feature].dtype in ['int64', 'float64']:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    input_data[feature] = st.number_input(f"{feature}", 
                                                        min_value=min_val, 
                                                        max_value=max_val, 
                                                        value=mean_val,
                                                        step=(max_val-min_val)/100)
                else:
                    unique_vals = X[feature].unique()
                    input_data[feature] = st.selectbox(f"{feature}", unique_vals)
    
    with tab2:
        st.subheader("Soil Conditions")
        col1, col2 = st.columns(2)
        for i, feature in enumerate(soil_features):
            with col1 if i % 2 == 0 else col2:
                if X[feature].dtype in ['int64', 'float64']:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    input_data[feature] = st.number_input(f"{feature}", 
                                                        min_value=min_val, 
                                                        max_value=max_val, 
                                                        value=mean_val,
                                                        step=(max_val-min_val)/100)
                else:
                    unique_vals = X[feature].unique()
                    input_data[feature] = st.selectbox(f"{feature}", unique_vals)
    
    with tab3:
        st.subheader("Crop Information")
        col1, col2 = st.columns(2)
        for i, feature in enumerate(crop_features):
            with col1 if i % 2 == 0 else col2:
                if X[feature].dtype in ['int64', 'float64']:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    input_data[feature] = st.number_input(f"{feature}", 
                                                        min_value=min_val, 
                                                        max_value=max_val, 
                                                        value=mean_val,
                                                        step=(max_val-min_val)/100)
                else:
                    unique_vals = X[feature].unique()
                    input_data[feature] = st.selectbox(f"{feature}", unique_vals)
    
    with tab4:
        st.subheader("Location Information")
        col1, col2 = st.columns(2)
        for i, feature in enumerate(location_features):
            with col1 if i % 2 == 0 else col2:
                if X[feature].dtype in ['int64', 'float64']:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    input_data[feature] = st.number_input(f"{feature}", 
                                                        min_value=min_val, 
                                                        max_value=max_val, 
                                                        value=mean_val,
                                                        step=(max_val-min_val)/100)
                else:
                    unique_vals = X[feature].unique()
                    input_data[feature] = st.selectbox(f"{feature}", unique_vals)
    
    with tab5:
        st.subheader("Other Parameters")
        col1, col2 = st.columns(2)
        for i, feature in enumerate(other_features):
            with col1 if i % 2 == 0 else col2:
                if X[feature].dtype in ['int64', 'float64']:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    input_data[feature] = st.number_input(f"{feature}", 
                                                        min_value=min_val, 
                                                        max_value=max_val, 
                                                        value=mean_val,
                                                        step=(max_val-min_val)/100)
                else:
                    unique_vals = X[feature].unique()
                    input_data[feature] = st.selectbox(f"{feature}", unique_vals)
    
    # Prediction button
    if st.button("🔮 Predict Pest Levels"):
        with st.spinner("Generating predictions..."):
            # Create input dataframe
            input_df = pd.DataFrame([input_data])
            
            # Make predictions
            predictions = {}
            for target in target_cols:
                if target in models:
                    pred = models[target].predict(input_df)[0]
                    predictions[target] = max(0, pred)  # Ensure non-negative
            
            # Display predictions
            st.subheader("🎯 Prediction Results")
            
            # Create categories for better visualization
            above_etl = [k for k in predictions.keys() if 'ABOVE ETL' in k]
            below_etl = [k for k in predictions.keys() if 'BELOW ETL' in k]
            diseases = [k for k in predictions.keys() if '%' in k]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🔴 Above ETL (Critical)")
                for pest in above_etl:
                    value = predictions[pest]
                    color = "🔴" if value > 5 else "🟡" if value > 2 else "🟢"
                    st.metric(f"{color} {pest.replace('(ABOVE ETL)', '')}", f"{value:.2f}")
            
            with col2:
                st.subheader("🟡 Below ETL (Monitoring)")
                for pest in below_etl:
                    value = predictions[pest]
                    color = "🟡" if value > 3 else "🟢"
                    st.metric(f"{color} {pest.replace('(BELOW ETL)', '')}", f"{value:.2f}")
            
            with col3:
                st.subheader("🦠 Disease Risk (%)")
                for disease in diseases:
                    value = predictions[disease]
                    color = "🔴" if value > 20 else "🟡" if value > 10 else "🟢"
                    st.metric(f"{color} {disease}", f"{value:.1f}%")
            
            # Risk assessment
            st.subheader("⚠️ Risk Assessment")
            high_risk = [k for k, v in predictions.items() if ('ABOVE ETL' in k and v > 5) or ('%' in k and v > 20)]
            medium_risk = [k for k, v in predictions.items() if ('ABOVE ETL' in k and 2 < v <= 5) or ('%' in k and 10 < v <= 20)]
            
            if high_risk:
                st.error(f"🔴 HIGH RISK: {', '.join([k.split('(')[0] for k in high_risk])}")
            elif medium_risk:
                st.warning(f"🟡 MEDIUM RISK: {', '.join([k.split('(')[0] for k in medium_risk])}")
            else:
                st.success("🟢 LOW RISK: All pest levels within acceptable limits")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(15, 8))
            pest_names = [k.replace('(ABOVE ETL)', '').replace('(BELOW ETL)', '') for k in predictions.keys()]
            values = list(predictions.values())
            colors = ['red' if 'ABOVE ETL' in k else 'orange' if 'BELOW ETL' in k else 'blue' for k in predictions.keys()]
            
            bars = ax.bar(range(len(predictions)), values, color=colors, alpha=0.7)
            ax.set_xlabel('Pest/Disease Types')
            ax.set_ylabel('Predicted Levels')
            ax.set_title('Pest and Disease Prediction Results')
            ax.set_xticks(range(len(predictions)))
            ax.set_xticklabels(pest_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)

def show_model_interpretation(X, y, models, target_cols):
    st.header("🎯 Model Interpretation")
    
    # Select target for interpretation
    selected_target = st.selectbox("Select Target for Interpretation", target_cols)
    
    if selected_target not in models:
        st.error(f"Model not found for {selected_target}")
        return
    
    model = models[selected_target]
    
    # Interpretation method selection
    interpretation_method = st.selectbox(
        "Select Interpretation Method",
        ["🎯 Permutation Feature Importance", "🔍 SHAP Analysis", "🔬 LIME Explanation"]
    )
    
    if interpretation_method == "🎯 Permutation Feature Importance":
        show_pfi_analysis(X, y, model, selected_target)
    elif interpretation_method == "🔍 SHAP Analysis":
        show_shap_analysis(X, y, model, selected_target)
    elif interpretation_method == "🔬 LIME Explanation":
        show_lime_analysis(X, y, model, selected_target)

def show_pfi_analysis(X, y, model, target):
    st.subheader(f"🎯 Permutation Feature Importance - {target}")
    
    if st.button("Calculate PFI"):
        with st.spinner("Calculating permutation importance..."):
            pfi_result = permutation_importance(
                model, X, y[target], n_repeats=10, random_state=42
            )
            
            # Create results dataframe
            pfi_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': pfi_result.importances_mean,
                'Std': pfi_result.importances_std
            }).sort_values('Importance', ascending=False)
            
            # Display top features
            st.subheader("📊 Top Important Features")
            st.dataframe(pfi_df.head(15))
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            top_features = pfi_df.head(15)
            bars = ax.barh(range(len(top_features)), top_features['Importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.set_xlabel('Permutation Importance')
            ax.set_title(f'Top 15 Features - {target}')
            ax.invert_yaxis()
            
            # Add error bars
            ax.errorbar(top_features['Importance'], range(len(top_features)),
                       xerr=top_features['Std'], fmt='none', color='black', alpha=0.7)
            
            plt.tight_layout()
            st.pyplot(fig)

def show_shap_analysis(X, y, model, target):
    st.subheader(f"🔍 SHAP Analysis - {target}")
    
    # Use XGBoost as surrogate for SHAP
    if st.button("Calculate SHAP Values"):
        with st.spinner("Training surrogate model and calculating SHAP values..."):
            # Train XGBoost surrogate
            xgb_surrogate = XGBRegressor(n_estimators=100, random_state=42)
            xgb_surrogate.fit(X, y[target])
            
            # Calculate SHAP values
            explainer = shap.Explainer(xgb_surrogate, X.sample(100))  # Sample for speed
            shap_values = explainer(X.sample(100))
            
            # Feature importance
            feature_importance = np.abs(shap_values.values).mean(0)
            
            # Create results dataframe
            shap_df = pd.DataFrame({
                'Feature': X.columns,
                'SHAP_Importance': feature_importance
            }).sort_values('SHAP_Importance', ascending=False)
            
            # Display results
            st.subheader("📊 SHAP Feature Importance")
            st.dataframe(shap_df.head(15))
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            top_features = shap_df.head(15)
            bars = ax.barh(range(len(top_features)), top_features['SHAP_Importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title(f'SHAP Feature Importance - {target}')
            ax.invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig)

def show_lime_analysis(X, y, model, target):
    st.subheader(f"🔬 LIME Explanation - {target}")
    
    # Select instance to explain
    instance_idx = st.number_input(
        "Select instance index to explain", 
        min_value=0, 
        max_value=len(X)-1, 
        value=0
    )
    
    if st.button("Generate LIME Explanation"):
        with st.spinner("Generating LIME explanation..."):
            # Create LIME explainer
            explainer = LimeTabularExplainer(
                X.values,
                feature_names=X.columns.tolist(),
                mode='regression',
                random_state=42
            )
            
            # Get explanation
            instance = X.iloc[instance_idx].values
            lime_exp = explainer.explain_instance(
                instance, model.predict, num_features=10
            )
            
            # Extract results
            lime_values = lime_exp.as_list()
            features = [item[0] for item in lime_values]
            importance = [item[1] for item in lime_values]
            
            # Display results
            st.subheader(f"📊 LIME Explanation for Instance {instance_idx}")
            
            lime_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            })
            st.dataframe(lime_df)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = ['red' if imp < 0 else 'blue' for imp in importance]
            bars = ax.barh(range(len(features)), importance, color=colors, alpha=0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('LIME Feature Importance')
            ax.set_title(f'LIME Explanation - {target} (Instance {instance_idx})')
            ax.invert_yaxis()
            
            # Add vertical line at 0
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show instance values
            st.subheader("📋 Instance Feature Values")
            instance_df = pd.DataFrame({
                'Feature': X.columns,
                'Value': X.iloc[instance_idx].values
            })
            st.dataframe(instance_df)

def show_feature_analysis(X, y, models, target_cols):
    st.header("📊 Feature Analysis")
    
    # Feature statistics
    st.subheader("📈 Feature Statistics")
    feature_stats = X.describe()
    st.dataframe(feature_stats)
    
    # Feature correlations
    st.subheader("🔗 Feature Correlations")
    
    # Select features for correlation analysis
    selected_features = st.multiselect(
        "Select features for correlation analysis",
        X.columns.tolist(),
        default=X.columns.tolist()[:10]
    )
    
    if selected_features:
        corr_matrix = X[selected_features].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Feature distributions
    st.subheader("📊 Feature Distributions")
    
    selected_feature = st.selectbox("Select feature to analyze", X.columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(X[selected_feature], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {selected_feature}')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Box plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(X[selected_feature])
        ax.set_ylabel(selected_feature)
        ax.set_title(f'Box Plot of {selected_feature}')
        plt.tight_layout()
        st.pyplot(fig) 