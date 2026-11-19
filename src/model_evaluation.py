import pandas as pd 

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix, 
    classification_report
)

def evaluate_models(trained_models, X_test, y_test):
    '''
    Evaluate trained classification models 
    using multiple classification metrics.
    '''
    
    results = []
    
    for name, model in trained_models.items():
        
        # generate class predictions
        y_pred = model.predict(X_test)
        
        # generate probability predictions
        y_prob = model.predict_proba(X_test)[:,1]
        
        # calculate evaluation metrics
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })
        
    return pd.DataFrame(results)

    