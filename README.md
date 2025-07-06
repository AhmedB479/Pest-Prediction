# Pest-Prediction

A comprehensive machine learning system for predicting pest infestations in agricultural fields using ensemble methods and advanced model interpretation techniques.

## 🎯 Overview

This project implements a multi-target regression system to predict various pest populations and disease incidences in cotton fields. The system uses ensemble learning methods including Random Forest, XGBoost, AdaBoost, and advanced ensemble techniques like Voting, Bagging, and Stacking.

## 📋 Features

- **Multi-target Prediction**: Predicts 24 different pest and disease targets
- **Ensemble Learning**: Combines multiple algorithms for better performance
- **Hyperparameter Optimization**: Automated tuning using RandomizedSearchCV
- **Model Interpretation**: SHAP, LIME, and Permutation Feature Importance
- **Interactive Visualizations**: Comprehensive matplotlib-based analysis tools
- **Parallel Processing**: Efficient multi-core training and evaluation

## 🛠️ Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)

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

## 📊 Data Preparation

1. **Place your data file**:
   - Ensure `final_data.xlsx` is in the root directory
   - The file should contain features and target columns as specified in the code

2. **Data structure**:
   - Features: All columns except target variables
   - Targets: 24 pest and disease columns (see code for exact names)
   - Missing values are automatically filled with 0

## 🚀 Quick Start

### Complete Setup and Run (All-in-One)

1. **Setup Environment**:
   ```bash
   python -m venv pest_venv
   pest_venv\Scripts\activate  # Windows
   # OR
   source pest_venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

2. **Run Everything**:
   ```bash
   # Train models
   python train_models.py
   
   # Launch web app
   cd frontend
   streamlit run main.py
   ```

3. **Access Results**:
   - **Web App**: Open `http://localhost:8501` in your browser
   - **Model Results**: Check CSV files in root directory
   - **Interpretation**: Run `python model_interpretation.py` for detailed analysis

---

## 🚀 Running the System (Detailed)

Run the main training script to train all models and evaluate performance:

```bash
python train_models.py
```

**What this does**:
- Trains 9 different models (RandomForest, DecisionTree, AdaBoost, SVM, KNN, XGBoost, VotingRegressor, BaggingRegressor, StackingRegressor)
- Performs hyperparameter optimization with reduced search space
- Evaluates models across 5 different random states
- Saves results to CSV files
- Generates performance summary

**Output files**:
- `{ModelName}_results.csv` - Individual model results
- `all_model_results.csv` - Combined results
- `model_performance_summary.csv` - Performance summary

### Step 2: Model Interpretation

Run the interpretation script to analyze model behavior:

```bash
python model_interpretation.py
```

**What this does**:
- Calculates Permutation Feature Importance (PFI)
- Generates SHAP explanations using XGBoost surrogate
- Creates LIME explanations for sample instances
- Provides interactive matplotlib visualizations

**Interactive Menu Options**:
1. Individual PFI plots for specific targets
2. Individual SHAP plots for specific targets
3. Individual LIME plots for specific targets
4. PFI Heatmap across all targets
5. SHAP Heatmap across all targets
6. Top Features Summary (PFI vs SHAP comparison)
7. All visualizations at once

### Step 3: Streamlit Frontend (Interactive Web App)

Launch the interactive web application for real-time pest prediction and model analysis:

```bash
cd frontend
streamlit run main.py
```

**What this provides**:
- **Interactive Prediction Dashboard**: Real-time pest prediction based on input parameters
- **Model Comparison**: Compare different ML models (Decision Tree, Random Forest, SVM)
- **Feature Selection Analysis**: Understand which features drive predictions
- **Model Explanations**: LIME and SHAP explanations for model interpretability
- **Data Overview**: Interactive data exploration and statistics

**Frontend Features**:
- 🌾 **Feature Selection Page**: Input crop conditions and get pest predictions
- 📊 **Model Comparison Page**: Compare model performance with visualizations
- 📈 **Data Overview Page**: Explore dataset statistics and distributions

**Access the app**: Open your browser and go to `http://localhost:8501`

### Step 4: Jupyter Notebook Analysis (Optional)

For detailed analysis and experimentation:

```bash
jupyter notebook
```

Then open:
- `Copy_of_FYP_2.ipynb` - Main analysis notebook
- `data_work.ipynb` - Data exploration and preprocessing

## 📁 Project Structure

```
Pest-Prediction/
├── data/                          # Data directory
│   ├── Crop.csv                  # Original crop dataset
│   └── processed_crop.csv        # Processed dataset for frontend
├── frontend/                      # Streamlit web application
│   ├── main.py                   # Main Streamlit app entry point
│   ├── feature_selection.py      # Feature selection and prediction page
│   ├── model_comparison.py       # Model comparison and explanation page
│   └── data_work.py             # Data processing utilities
├── pest_venv/                     # Virtual environment
├── best_models/                   # Saved trained models
├── model_interpretation_outputs/  # Interpretation results
├── Copy_of_FYP_2.ipynb           # Main analysis notebook
├── data_work.ipynb               # Data exploration notebook
├── train_models.py               # Main training script
├── model_interpretation.py       # Model interpretation script
├── requirements.txt              # Python dependencies
├── final_data.xlsx              # Input data file
└── README.md                    # This file
```

## 🌐 Streamlit Frontend

### Overview
The Streamlit frontend provides an interactive web interface for pest prediction and model analysis. It offers three main pages with different functionalities.

### Setup and Installation

1. **Install Streamlit** (if not already installed):
   ```bash
   pip install streamlit
   ```

2. **Prepare Data**:
   - Ensure `data/Crop.csv` exists (original dataset)
   - The app will automatically create `data/processed_crop.csv`

3. **Run the Application**:
   ```bash
   cd frontend
   streamlit run main.py
   ```

### Frontend Pages

#### 🌾 Feature Selection Page
**Purpose**: Interactive pest prediction based on crop conditions

**Features**:
- **Input Form**: Enter crop parameters (nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall, crop type, district)
- **Real-time Prediction**: Get instant pest infestation predictions
- **Model Performance**: View accuracy metrics for Naive Bayes and AdaBoost models
- **Confusion Matrices**: Visual representation of model performance
- **Probability Scores**: See prediction confidence levels

**Usage**:
1. Fill in the crop condition parameters
2. Click "Predict" to get results
3. View predictions from both Naive Bayes and AdaBoost models
4. Check model performance metrics

#### 📊 Model Comparison Page
**Purpose**: Compare different machine learning models and understand their predictions

**Features**:
- **Model Performance**: Compare Decision Tree, Random Forest, and SVM
- **Classification Reports**: Detailed performance metrics
- **Confusion Matrices**: Visual performance comparison
- **Model Explanations**: LIME and SHAP explanations for interpretability
- **Interactive Instance Selection**: Choose specific test cases for explanation

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

### Frontend Configuration

#### Customization Options
- **Theme**: Modify CSS in each page for custom styling
- **Models**: Add new models in `model_comparison.py`
- **Features**: Extend input parameters in `feature_selection.py`
- **Visualizations**: Customize plots and charts

#### Data Requirements
The frontend expects:
- `data/Crop.csv`: Original dataset with columns: nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, label, district
- Automatic processing creates additional features (GDD, pest classification)

#### Performance Optimization
- **Caching**: Uses `@st.cache_data` and `@st.cache_resource` for efficient data loading
- **Parallel Processing**: Model training is cached for faster subsequent runs
- **Responsive Design**: Optimized for different screen sizes

### Troubleshooting Frontend Issues

#### Common Problems

1. **Data File Not Found**:
   ```
   FileNotFoundError: data/Crop.csv
   ```
   **Solution**: Ensure `data/Crop.csv` exists in the correct location

2. **Streamlit Not Installed**:
   ```
   ModuleNotFoundError: No module named 'streamlit'
   ```
   **Solution**: `pip install streamlit`

3. **Port Already in Use**:
   ```
   Port 8501 is already in use
   ```
   **Solution**: Use `streamlit run main.py --server.port 8502`

4. **Model Loading Issues**:
   ```
   Error loading models
   ```
   **Solution**: Ensure all required packages are installed and data is properly formatted

#### Advanced Configuration

**Custom Port**:
```bash
streamlit run main.py --server.port 8502
```

**Custom Theme**:
```bash
streamlit run main.py --theme.base light
```

**Enable Debug Mode**:
```bash
streamlit run main.py --logger.level debug
```

### Frontend Dependencies

The frontend requires these additional packages:
```
streamlit>=1.28.0
seaborn>=0.12.0
lime>=0.2.0.1
shap>=0.42.0
```

Add to your `requirements.txt`:
```txt
streamlit>=1.28.0
seaborn>=0.12.0
lime>=0.2.0.1
shap>=0.42.0
```

## 🔧 Configuration

### Model Parameters

The system uses optimized hyperparameter grids:

- **RandomForest**: n_estimators [100, 200], max_depth [10, None]
- **XGBoost**: n_estimators [100, 200], max_depth [3, 5], learning_rate [0.05, 0.1]
- **Ensemble Methods**: Pre-configured with best-performing base models

### Training Settings

- **Test Size**: 30% (0.3)
- **Cross-validation**: 3 folds
- **Random States**: [0, 21, 42, 77, 101] for robust evaluation
- **Hyperparameter Search**: 5 iterations (optimized for speed)

## 📈 Performance Metrics

The system evaluates models using:
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error

## 🎨 Visualization Features

### Model Interpretation Visualizations

1. **Permutation Feature Importance (PFI)**:
   - Shows feature importance by permuting values
   - Robust to feature correlations

2. **SHAP (SHapley Additive exPlanations)**:
   - Uses XGBoost surrogate model
   - Provides local and global feature importance

3. **LIME (Local Interpretable Model-agnostic Explanations)**:
   - Explains individual predictions
   - Shows feature contributions for specific samples

### Interactive Features

- **Heatmaps**: Compare feature importance across all targets
- **Bar Charts**: Top features for each method
- **Comparison Plots**: PFI vs SHAP side-by-side analysis
- **Customizable Views**: Control number of features displayed

## 🐛 Troubleshooting

### Common Issues

1. **Missing data file**:
   ```
   FileNotFoundError: final_data.xlsx
   ```
   **Solution**: Ensure `final_data.xlsx` is in the root directory

2. **Missing models**:
   ```
   FileNotFoundError: best_models/best_model_*.joblib
   ```
   **Solution**: Run `train_models.py` first to generate models

3. **Memory issues**:
   ```
   MemoryError during training
   ```
   **Solution**: Reduce `n_iter` in RandomizedSearchCV or use smaller dataset

4. **Package conflicts**:
   ```
   ImportError: No module named 'shap'
   ```
   **Solution**: Reinstall requirements: `pip install -r requirements.txt`

### Performance Optimization

- **Faster training**: Reduce `n_iter` in hyperparameter search
- **Memory efficient**: Use smaller subset of data for testing
- **Parallel processing**: Already enabled with `n_jobs=-1`

## 📊 Expected Outputs

### Training Results
- Individual model performance CSV files
- Combined results summary
- Model performance ranking

### Interpretation Results
- PFI plots for each target
- SHAP summary plots
- LIME HTML explanations
- Interactive matplotlib visualizations

