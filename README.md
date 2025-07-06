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

## 🚀 Running the System

### Step 1: Model Training and Evaluation

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

### Step 3: Jupyter Notebook Analysis (Optional)

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
├── frontend/                      # Frontend application (if any)
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

