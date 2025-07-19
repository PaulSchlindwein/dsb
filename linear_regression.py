# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from data_cleaning import df_cleaned

# Function to prepare data for linear regression
def data_cleaning_for_lr(df):
    # Create a copy to avoid modifying the original dataframe
    df_lr = df.copy()
    
    print(f"Initial shape: {df_lr.shape}")
    print(f"Initial columns: {df_lr.columns.tolist()}")
    
    # Drop non-useful columns for linear regression
    columns_to_drop = ['ID', 'name', 'category', 'deadline', 'launched']
    df_lr = df_lr.drop(columns=columns_to_drop)
    
    print(f"After dropping columns: {df_lr.shape}")
    print(f"Remaining columns: {df_lr.columns.tolist()}")
    
    # Convert state to binary variable (1 for successful, 0 for everything else)
    df_lr['state'] = (df_lr['state'] == 'successful').astype(int)
    
    # Create dummy variables for categorical columns
    categorical_columns = ['main_category', 'currency', 'country']
    for col in categorical_columns:
        if col in df_lr.columns:
            print(f"Creating dummies for {col}...")
            dummies = pd.get_dummies(df_lr[col], prefix=col, drop_first=True)
            print(f"Created {dummies.shape[1]} dummy variables for {col}")
            df_lr = pd.concat([df_lr, dummies], axis=1)
            df_lr = df_lr.drop(columns=[col])
    
    print(f"After creating dummies: {df_lr.shape}")
    print(f"Columns after dummies: {df_lr.columns.tolist()}")
    
    # Convert all columns to numeric (this will handle the dummy variables)
    for col in df_lr.columns:
        if col != 'state':  # Don't convert the target variable
            df_lr[col] = pd.to_numeric(df_lr[col], errors='coerce')
    
    # Ensure all remaining columns are numeric
    numeric_columns = df_lr.select_dtypes(include=[np.number]).columns
    df_lr = df_lr[numeric_columns]
    
    print(f"After selecting numeric columns: {df_lr.shape}")
    print(f"Numeric columns: {df_lr.columns.tolist()}")

    # Drop all rows with any NaN values
    before = df_lr.shape[0]
    df_lr = df_lr.dropna()
    after = df_lr.shape[0]
    print(f"Dropped {before - after} rows with NaN values. Remaining rows: {after}")
    print(f"Final shape: {df_lr.shape}")
    
    return df_lr

# Apply the data cleaning function
df_lr = data_cleaning_for_lr(df_cleaned)

# Separate features and target
X = df_lr.drop('state', axis=1)
y = df_lr['state']

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Ridge Classifier': RidgeClassifier(alpha=1.0, random_state=42)
}

# Initialize scaler
scaler = StandardScaler()

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary to store results for each model
results = {model_name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'mse': [], 'r2': []} 
           for model_name in models.keys()}

print("Starting 5-fold cross-validation with multiple regression models...")
print("=" * 80)

# Perform 5-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    print(f"\nFold {fold}/5:")
    print("-" * 40)
    
    # Split data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        if model_name == 'Linear Regression':
            # For linear regression, we need to threshold the predictions
            y_pred_proba = model.predict(X_test_scaled)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # For other models, use predict method
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC AUC (handle cases where all predictions are the same)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.5  # Default value when ROC AUC cannot be calculated
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[model_name]['accuracy'].append(accuracy)
        results[model_name]['precision'].append(precision)
        results[model_name]['recall'].append(recall)
        results[model_name]['f1'].append(f1)
        results[model_name]['roc_auc'].append(roc_auc)
        results[model_name]['mse'].append(mse)
        results[model_name]['r2'].append(r2)
        
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

# Calculate and display final results
print("\n" + "=" * 80)
print("FINAL RESULTS - 5-FOLD CROSS-VALIDATION")
print("=" * 80)

# Create results summary
summary_data = []
for model_name, metrics in results.items():
    row = {
        'Model': model_name,
        'Avg Accuracy': np.mean(metrics['accuracy']),
        'Std Accuracy': np.std(metrics['accuracy']),
        'Avg Precision': np.mean(metrics['precision']),
        'Avg Recall': np.mean(metrics['recall']),
        'Avg F1': np.mean(metrics['f1']),
        'Avg ROC AUC': np.mean(metrics['roc_auc']),
        'Avg MSE': np.mean(metrics['mse']),
        'Avg R²': np.mean(metrics['r2'])
    }
    summary_data.append(row)

# Create and display summary dataframe
summary_df = pd.DataFrame(summary_data)
print("\nModel Performance Summary:")
print(summary_df.round(4))

# Find best model for each metric
print("\n" + "=" * 80)
print("BEST PERFORMING MODELS BY METRIC")
print("=" * 80)

metrics_to_check = ['Avg Accuracy', 'Avg F1', 'Avg ROC AUC', 'Avg R²']
for metric in metrics_to_check:
    best_model = summary_df.loc[summary_df[metric].idxmax()]
    print(f"Best {metric}: {best_model['Model']} ({best_model[metric]:.4f})")

# Detailed fold-by-fold results
print("\n" + "=" * 80)
print("DETAILED FOLD-BY-FOLD RESULTS")
print("=" * 80)

for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy: {[f'{acc:.4f}' for acc in metrics['accuracy']]}")
    print(f"  F1 Score: {[f'{f1:.4f}' for f1 in metrics['f1']]}")
    print(f"  ROC AUC:  {[f'{roc:.4f}' for roc in metrics['roc_auc']]}")

print(f"\nData shape: {X.shape}")
print(f"Features: {X.shape[1]}")
print(f"Target distribution: {y.value_counts().to_dict()}")
