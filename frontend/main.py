import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from feature_selection import feature_selection_page
from model_comparison import model_comparison_page
from advanced_pest_prediction import advanced_pest_prediction_page

# Set page config
st.set_page_config(
    page_title="Cotton Pest Analysis",
    page_icon="🌿",
    layout="wide"
)


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["🔍 Feature Selection", "📊 Model Comparison", "📈 Data Overview", "🔬 Advanced Pest Prediction"]
)

# Load and process data
@st.cache_data
def load_data():
    df = pd.read_csv('../data/Crop.csv')
    # Encode categorical columns
    le_label = LabelEncoder()
    le_district = LabelEncoder()
    df['label_encoded'] = le_label.fit_transform(df['label'])
    df['district_encoded'] = le_district.fit_transform(df['district'])

    # Drop original label and district columns
    df.drop(columns=['label', 'district'], inplace=True)

    # Add synthetic 'pest' column based on humidity and temperature
    df['pest'] = ((df['humidity'] > 80) & (df['temperature'] > 25)).astype(int)

    # Add Growing Degree Day (GDD)
    T_base = 10  # base temperature
    df['GDD'] = (df['temperature'] - T_base).apply(lambda x: max(0, x))

    # Save to new CSV
    df.to_csv("../data/processed_crop.csv", index=False)
    return df

# Load the data
df = load_data()

if page == "🔍 Feature Selection":
    feature_selection_page() 

elif page == "📊 Model Comparison":
    model_comparison_page()
elif page == "📈 Data Overview":
    # Page content based on selection
    st.title("📈 Data Overview")
    st.markdown("### Raw Data Preview")
    st.dataframe(df)
    
    # Add some basic statistics
    st.markdown("### Basic Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Pest Infestation Rate", f"{(df['pest'].mean() * 100):.1f}%")
    with col3:
        st.metric("Average Temperature", f"{df['temperature'].mean():.1f}°C")
        
elif page == "🔬 Advanced Pest Prediction":
    advanced_pest_prediction_page()


# Add footer
st.markdown("---")
