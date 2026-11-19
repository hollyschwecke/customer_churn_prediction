from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.base import clone 

import time

def get_candidate_models():
    '''
    Return a collection of classification models 
    for customer churn prediction
    '''
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
        
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }
    
    return models

def build_model_pipelines(models, preprocessor):
    '''
    Build a separate preprocessing and classification 
    pipeline for each candidate model
    '''
    model_pipelines = {}
    
    for name, model in models.items():
        model_pipeline = Pipeline(
            steps = [
                ("preprocessor", clone(preprocessor)),
                ("classifier", model)
            ]
        )
        
        model_pipelines[name] = model_pipeline
        
    return model_pipelines

def train_models(model_pipelines, X_train, y_train):
    '''
    Train each model pipeline using the training dataset.
    
    Returns:
        trained_models: Dictionary containing fitted models
        training_times: Dictionary containing training durations
    '''
    
    trained_models = {}
    training_times = {}
    
    for name, pipeline in model_pipelines.items():
        
        # start training timer
        start_time = time.time()
        
        # fit the model pipeline
        pipeline.fit(X_train, y_train)
        
        # calculate training duration
        elapsed_time = time.time() - start_time
        
        # store trained model and training time
        trained_models[name] = pipeline
        training_times[name] = elapsed_time
        
        print(f"{name} training completed.")
        print(f"Training time: {elapsed_time:.2f} seconds\n")
    
    return trained_models, training_times
    