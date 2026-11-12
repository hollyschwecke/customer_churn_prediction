import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split

def identify_feature_types(X):
    '''
    Identify categorical and numerical features in the dataset.
    
    SeniorCitizen is treated as a categorical feature, even though it is represented as a numerical value (0 or 1).
    '''
    
    # identify numerical columns
    numerical_features = [
        col for col in X.columns 
        if pd.api.types.is_numeric_dtype(X[col]) and col != 'SeniorCitizen'
    ]
    
    # identify categorical columns
    categorical_features = [
        col for col in X.columns
        if col not in numerical_features
    ]
    
    return numerical_features, categorical_features
    
def build_preprocessor(numerical_features, categorical_features):
    '''
    Build a preprocessing pipeline for numerical and categorical features.
    
    Numerical features:
        - Impute missing values using the median.
        - Standardize using StandardScaler.
        
    Categorical features:
        - Impute missing values using the most frequent value.
        - One-hot encode using OneHotEncoder.
    '''
    
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    return ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])
    
def split_data(X, y, test_size=0.2, random_state=42):
    '''
    Split the dataset into training and testing sets.
    
    Stratification preserves the target class distribution across both datasets.
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test