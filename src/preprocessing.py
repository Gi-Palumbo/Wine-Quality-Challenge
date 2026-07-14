"""
Modulo de pre-processamento de dados para o projeto Wine Quality Classification.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(filepath: str) -> pd.DataFrame:
    """Carrega o dataset de vinhos."""
    df = pd.read_csv(filepath)
    return df


def create_binary_target(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    """Transforma a variavel quality em classificacao binaria."""
    df = df.copy()
    df['quality_label'] = (df['quality'] >= threshold).astype(int)
    return df


def remove_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove a coluna Id se existir."""
    df = df.copy()
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    return df


def get_features_and_target(df: pd.DataFrame):
    """Separa features e variavel alvo."""
    feature_cols = [c for c in df.columns if c not in ['quality', 'quality_label', 'Id']]
    X = df[feature_cols]
    y = df['quality_label']
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Aplica StandardScaler nas features."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train, random_state=42):
    """Aplica SMOTE para balancear as classes."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def split_data(X, y, test_size=0.2, random_state=42):
    """Divide em treino e teste com estratificacao."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)