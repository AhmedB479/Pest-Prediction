# Pest-Prediction

A machine learning web application for predicting pest infestations in agricultural fields using interactive Streamlit frontend and comprehensive data analysis through Jupyter notebooks.

## 🎯 Overview

This project provides an interactive web-based system for pest prediction in cotton crops. It combines machine learning models with an intuitive Streamlit interface, allowing users to input crop conditions and receive real-time pest infestation predictions. The system includes comprehensive data analysis capabilities through Jupyter notebooks.

## 📋 Features

- **Interactive Web Interface**: Streamlit-based frontend with three main pages
- **Real-time Prediction**: Instant pest infestation predictions based on crop conditions
- **Multiple ML Models**: Comparison of Decision Tree, Random Forest, SVM, Naive Bayes, and AdaBoost
- **Model Interpretability**: LIME and SHAP explanations for model predictions
- **Data Visualization**: Interactive charts and confusion matrices
- **Comprehensive Analysis**: Jupyter notebooks for in-depth data exploration

## 🛠️ Prerequisites

- Python 3.8+
- All required packages listed in `requirements.txt`

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Pest-Prediction
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Windows
   python -m venv pest_venv
   pest_venv\Scripts\activate

   # Linux/Mac
   python3 -m venv pest_venv
   source pest_venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Requirements

The project contains multiple datasets for different purposes:

### **Frontend Data (Streamlit App)**:
- **`data/Crop.csv`**: Original crop dataset with columns: nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, label, district
- **`data/processed_crop.csv`**: Processed dataset (automatically created by the app)

### **Notebook Analysis Data**:
- **`data/final_data.xlsx`**: Main dataset for comprehensive pest prediction analysis (used in Copy_of_FYP_2.ipynb)
- **`data/cotton_summary_compilation.xlsx`**: Compiled cotton field data for advanced analysis
- **Additional processed files**: Generated during notebook execution

**Note**: The Streamlit frontend and Jupyter notebooks use different datasets and approaches:
- **Frontend**: Uses synthetic pest classification based on environmental thresholds
- **Notebooks**: Use actual pest infestation data from agricultural surveys

## 🚀 How to Start the Streamlit Application

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Navigate to Frontend Directory
```bash
cd frontend
```

### Step 3: Launch the App
```bash
streamlit run main.py
```

### Step 4: Access the Application
- Open your web browser
- Go to: **`http://localhost:8501`**
- The app will automatically open with four pages:
  - 🌾 **Feature Selection** - Basic pest predictions using simple models
  - 📊 **Model Comparison** - Compare different ML models
  - 📈 **Data Overview** - View dataset statistics
  - 🔬 **Advanced Pest Prediction** - **NEW!** Comprehensive pest analysis using trained models

---

### Alternative: Run from Root Directory
```bash
# If you want to run from the project root
streamlit run frontend/main.py
```

### Troubleshooting Startup
- **Port busy?** Use: `streamlit run main.py --server.port 8502`
- **Missing modules?** Run: `pip install -r requirements.txt`
- **Data files missing?** Ensure `data/Crop.csv` and `data/final_data.xlsx` exist
- **Model loading errors?** Ensure all 24 model files exist in `best_models/` directory

---

## 🚀 Alternative: Jupyter Notebook Analysis

```bash
# Launch Jupyter
jupyter notebook

# Open either:
# - Copy_of_FYP_2.ipynb (Comprehensive analysis)
# - data_work.ipynb (Basic exploration)
```

## 🌐 Streamlit Frontend

### Overview
The Streamlit frontend provides three main pages for different aspects of pest prediction and analysis.

### Frontend Pages

#### 🌾 Feature Selection Page
**Purpose**: Interactive pest prediction based on crop conditions

**Features**:
- **Input Form**: Enter crop parameters (nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall, crop type, district)
- **Real-time Prediction**: Get instant pest infestation predictions
- **Model Performance**: View accuracy metrics for Naive Bayes and AdaBoost models
- **Confusion Matrices**: Visual representation of model performance
- **Probability Scores**: See prediction confidence levels

**Models Used**:
- Naive Bayes (GaussianNB)
- AdaBoost Classifier

**Usage**:
1. Fill in the crop condition parameters
2. Click "Predict" to get results
3. View predictions from both models
4. Check model performance metrics

#### 📊 Model Comparison Page
**Purpose**: Compare different machine learning models and understand their predictions

**Features**:
- **Model Performance**: Compare Decision Tree, Random Forest, and SVM
- **Classification Reports**: Detailed performance metrics
- **Confusion Matrices**: Visual performance comparison
- **Model Explanations**: LIME and SHAP explanations for interpretability
- **Interactive Instance Selection**: Choose specific test cases for explanation

**Models Used**:
- Decision Tree Classifier
- Random Forest Classifier  
- Support Vector Machine (SVM)

**Usage**:
1. Select a model for detailed analysis
2. Choose explanation method (LIME or SHAP)
3. Select a test instance to explain
4. View feature importance and model reasoning

#### 📈 Data Overview Page
**Purpose**: Explore dataset statistics and distributions

**Features**:
- **Data Preview**: View raw dataset
- **Basic Statistics**: Total records, pest infestation rate, average temperature
- **Data Insights**: Quick overview of dataset characteristics

### Data Processing Pipeline

The frontend automatically processes data through these steps:

1. **Load Original Data**: Reads `data/Crop.csv`
2. **Feature Engineering**: 
   - Encodes categorical variables (label, district)
   - Creates synthetic 'pest' column based on humidity > 80% and temperature > 25°C
   - Calculates Growing Degree Days (GDD) with base temperature of 10°C
3. **Save Processed Data**: Creates `data/processed_crop.csv`

## 📁 Project Structure

```
Pest-Prediction/
├── data/                              # Data directory
│   ├── Crop.csv                      # Frontend: Original crop dataset
│   ├── processed_crop.csv            # Frontend: Processed dataset (auto-generated)
│   ├── final_data.xlsx               # Notebooks: Main pest prediction dataset
│   └── cotton_summary_compilation.xlsx # Notebooks: Cotton field compilation data
├── frontend/                          # Streamlit web application
│   ├── main.py                       # Main Streamlit app entry point
│   ├── feature_selection.py          # Feature selection and prediction page
│   ├── model_comparison.py           # Model comparison and explanation page
│   ├── advanced_pest_prediction.py   # Advanced pest prediction with trained models
│   └── data_work.py                 # Data processing utilities
├── best_models/                       # Trained StackingRegressor models
│   ├── best_model_StackingRegressor_{pest_name}.joblib # 24 trained models
│   └── ... (24 model files total)
├── pest_venv/                         # Virtual environment
├── Copy_of_FYP_2.ipynb               # Comprehensive pest analysis notebook
├── data_work.ipynb                   # Basic data exploration notebook
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## 🔧 Technical Details

### Machine Learning Models

**Feature Selection Page**:
- **Naive Bayes**: GaussianNB with StandardScaler preprocessing
- **AdaBoost**: AdaBoostClassifier with 100 estimators

**Model Comparison Page**:
- **Decision Tree**: DecisionTreeClassifier with random_state=42
- **Random Forest**: RandomForestClassifier with random_state=42
- **SVM**: SVC with probability=True, wrapped in StandardScaler pipeline

### Model Interpretability

**LIME (Local Interpretable Model-agnostic Explanations)**:
- Explains individual predictions
- Shows feature contributions for specific samples
- Available for all models in comparison page

**SHAP (SHapley Additive exPlanations)**:
- Provides global and local feature importance
- Uses TreeExplainer for Random Forest
- Generates decision plots for model understanding

### Performance Metrics

The system evaluates models using:
- **Accuracy Score**: Overall classification accuracy
- **Classification Report**: Precision, recall, F1-score
- **Confusion Matrix**: Visual representation of predictions vs actual
- **ROC-AUC**: Area under the ROC curve (where applicable)

## 🎨 Visualization Features

- **Interactive Charts**: Real-time updates based on user input
- **Confusion Matrices**: Heatmap visualizations using seaborn
- **Feature Importance**: Bar charts showing model explanations
- **Model Comparison**: Side-by-side performance visualization
- **Data Distribution**: Statistical overview of dataset

#### 🔬 Advanced Pest Prediction Page
**Purpose**: Comprehensive pest prediction using trained StackingRegressor models for 24 pest targets

**Key Features**:

##### **📈 Train/Test Performance Analysis**
- **Interactive Split Configuration**: Adjust test size and random state
- **Multi-target Evaluation**: Analyze performance across all 24 pest/disease targets
- **Comprehensive Metrics**: R², MAE, RMSE for both training and testing
- **Overfitting Detection**: Visual analysis of model generalization
- **Comparative Visualization**: Side-by-side train vs test performance

##### **🔮 Future Predictions**
- **Comprehensive Input System**: Organized into 5 tabs:
  - 🌤️ **Weather**: Temperature, humidity, rainfall, wind conditions
  - 🌱 **Soil**: Soil type, pH, moisture, nutrients
  - 🌾 **Crop**: Variety, growth stage, age, density
  - 📍 **Location**: District, region, geographic factors
  - 📊 **Other**: Additional environmental and management factors

- **24 Simultaneous Predictions**: All pest types and disease risks including:
  - **Above ETL**: W.FLY, JASSID, THRIPS, M.BUG, MITES, APHIDS, DUSKY COTTON BUG, SBW, PBW, ABW, Army Worm
  - **Below ETL**: Same pests at monitoring levels
  - **Diseases**: CLCV (Cotton Leaf Curl Virus), WILT with spot and area percentages

- **Risk Assessment**: Automatic categorization:
  - 🔴 **High Risk**: Above ETL > 5 or Disease > 20%
  - 🟡 **Medium Risk**: Above ETL 2-5 or Disease 10-20%
  - 🟢 **Low Risk**: Below monitoring thresholds

- **Visual Results**: Color-coded predictions with actionable insights

##### **🎯 Model Interpretation**
- **Permutation Feature Importance (PFI)**:
  - Identifies which features drive predictions
  - Shows feature importance with confidence intervals
  - Ranked importance across all features

- **SHAP Analysis**:
  - Global and local explanations using XGBoost surrogate models
  - Feature contribution analysis
  - Model decision transparency

- **LIME Explanations**:
  - Instance-level interpretability
  - Local feature importance for specific predictions
  - Positive/negative feature contributions

##### **📊 Feature Analysis**
- **Statistical Summaries**: Comprehensive feature statistics
- **Correlation Analysis**: Interactive correlation matrix
- **Distribution Analysis**: Histograms and box plots
- **Feature Relationships**: Identify patterns and dependencies

**Models Used**:
- **24 StackingRegressor Models**: One for each pest/disease target
- **Ensemble Architecture**: Combines Random Forest, XGBoost, and AdaBoost
- **Surrogate Models**: XGBoost for SHAP analysis

**Data Requirements**:
- `data/final_data.xlsx`: Main dataset with pest infestation records
- `best_models/`: Directory containing all 24 trained models

## 📊 Data Analysis Notebooks

### Copy_of_FYP_2.ipynb
**Purpose**: Comprehensive pest prediction analysis using actual agricultural data

**Key Features**:
- **Real Pest Data**: Uses `final_data.xlsx` with actual pest infestation records
- **Multi-target Prediction**: Predicts multiple pest species and disease conditions
- **Advanced Models**: Implements ensemble methods and hyperparameter tuning
- **Comprehensive Analysis**: Feature engineering, model evaluation, and interpretation

**Data Source**: `data/final_data.xlsx` and `data/cotton_summary_compilation.xlsx`

### data_work.ipynb  
**Purpose**: Basic data exploration and simple classification

**Key Features**:
- **Exploratory Analysis**: Basic statistics and data visualization
- **Simple Classification**: Binary pest/no-pest classification
- **Data Processing**: Preprocessing steps and feature selection
- **Baseline Models**: Simple machine learning approaches

**Data Source**: Various datasets including processed versions

### Key Differences

| Aspect | Copy_of_FYP_2.ipynb | data_work.ipynb |
|--------|---------------------|-----------------|
| **Complexity** | Advanced, production-ready | Basic, exploratory |
| **Data** | Real agricultural pest data | Simplified datasets |
| **Models** | Ensemble methods, hyperparameter tuning | Basic classifiers |
| **Targets** | Multi-target pest prediction | Binary classification |
| **Purpose** | Comprehensive analysis | Learning and exploration |

**Recommendation**: 
- Use **Copy_of_FYP_2.ipynb** for serious pest prediction analysis
- Use **data_work.ipynb** for understanding basics and data exploration
- Use **Streamlit frontend** for interactive predictions and demonstrations

## 🐛 Troubleshooting

### Common Issues

1. **Missing Data File**:
   ```
   FileNotFoundError: data/Crop.csv
   ```
   **Solution**: Ensure `data/Crop.csv` exists with the correct column structure

2. **Streamlit Not Starting**:
   ```
   ModuleNotFoundError: No module named 'streamlit'
   ```
   **Solution**: Install requirements: `pip install -r requirements.txt`

3. **Port Already in Use**:
   ```
   Port 8501 is already in use
   ```
   **Solution**: Use custom port: `streamlit run main.py --server.port 8502`

4. **Data Processing Errors**:
   ```
   KeyError: 'label' or 'district'
   ```
   **Solution**: Verify CSV file has required columns: nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, label, district

5. **Notebook Data File Issues**:
   ```
   FileNotFoundError: final_data.xlsx
   ```
   **Solution**: Ensure Excel files (`final_data.xlsx`, `cotton_summary_compilation.xlsx`) exist in `data/` directory

6. **Excel File Dependencies**:
   ```
   ImportError: Missing optional dependency 'openpyxl'
   ```
   **Solution**: Install Excel support: `pip install openpyxl xlrd`

### Performance Optimization

- **Caching**: Streamlit uses `@st.cache_data` and `@st.cache_resource` for efficient data loading
- **Model Training**: Models are cached after first training
- **Data Processing**: Processed data is saved for reuse

## 📋 Requirements

The system requires these main packages:
- `streamlit>=1.45.1` - Web interface
- `pandas>=2.3.0` - Data manipulation
- `scikit-learn>=1.7.0` - Machine learning models
- `matplotlib>=3.10.3` - Plotting
- `seaborn>=0.13.2` - Statistical visualization
- `lime>=0.2.0.1` - Model interpretability
- `shap>=0.48.0` - Model explanations

## 🚀 Getting Started Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Prepare data**: Ensure `data/Crop.csv` exists
3. **Launch app**: `cd frontend && streamlit run main.py`
4. **Access interface**: Open `http://localhost:8501`
5. **Start predicting**: Use the Feature Selection page for predictions
6. **Explore models**: Use Model Comparison page for detailed analysis

## 🔬 For Data Scientists

To extend or modify the system:

1. **Add new models**: Edit `feature_selection.py` or `model_comparison.py`
2. **Modify features**: Update data processing in `main.py`
3. **Add visualizations**: Extend plotting functions in respective page files
4. **Enhance interpretability**: Add new explanation methods to comparison page

## 📄 License

This project is designed for educational and research purposes. Please validate predictions with domain experts before making agricultural decisions.

---

**Note**: This system creates synthetic pest labels based on temperature and humidity thresholds. For production use, replace with actual pest infestation data.

