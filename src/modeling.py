"""
Modulo de modelagem para o projeto Wine Quality Classification.
"""
 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve
)
from sklearn.model_selection import cross_val_score
 
 
def get_models():
    """Retorna dicionario com os modelos a serem treinados."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, random_state=42, eval_metric='logloss',
            scale_pos_weight=6
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, random_state=42
        ),
    }
    return models
 
 
def train_model(model, X_train, y_train):
    """Treina um modelo."""
    model.fit(X_train, y_train)
    return model
 
 
def evaluate_model(model, X_test, y_test):
    """Avalia um modelo e retorna dicionario com metricas."""
    y_pred = model.predict(X_test)
 
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
 
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba),
    }
 
    return metrics, y_pred, y_proba
 
 
def cross_validate_model(model, X, y, cv=5):
    """Realiza validacao cruzada."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    return scores
 
 
def get_feature_importance(model, feature_names):
    """Extrai importancia das features."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
 
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
 
    return feat_imp