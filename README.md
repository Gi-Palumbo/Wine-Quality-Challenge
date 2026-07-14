
# Classificação da Qualidade de Vinhos com Machine Learning

## Tech Challenge - Fase 2 | POSTECH FIAP

Projeto de classificação binária para prever a qualidade de vinhos tintos (Alta vs Baixa/Média) a partir de características físico-químicas, utilizando técnicas de Machine Learning.

---

## Objetivo

Desenvolver um modelo de classificação capaz de prever a qualidade de um vinho com base em suas características físico-químicas. A variável de qualidade foi transformada em classificação binária:

- **Alta Qualidade**: nota >= 7
- **Baixa/Média Qualidade**: nota < 7

## Dataset

**Fonte:** [Wine Quality Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

- **1.143 amostras** de vinhos tintos
- **11 variáveis** físico-químicas + 1 variável alvo (quality)
- **Sem dados faltantes**
- **Classes desbalanceadas**: 86.1% Baixa/Média vs 13.9% Alta Qualidade

### Variáveis

| Variável | Descrição |
|---|---|
| fixed acidity | Acidez fixa |
| volatile acidity | Acidez volátil |
| citric acid | Ácido cítrico |
| residual sugar | Açúcar residual |
| chlorides | Cloretos |
| free sulfur dioxide | Dióxido de enxofre livre |
| total sulfur dioxide | Dióxido de enxofre total |
| density | Densidade |
| pH | pH |
| sulphates | Sulfatos |
| alcohol | Teor alcoólico |
| quality | Qualidade (variável alvo original, notas 3-8) |

## Estrutura do Projeto

```
wine-quality-classification/
|
|-- data/                    # Base de dados utilizada
|   |-- WineQT.csv
|
|-- notebooks/               # Notebook com a análise e modelagem
|   |-- wine_quality_classification.ipynb
|
|-- src/                     # Scripts auxiliares
|   |-- __init__.py
|   |-- preprocessing.py     # Pré-processamento de dados
|   |-- modeling.py          # Treinamento e avaliação de modelos
|
|-- results/                 # Gráficos e métricas dos modelos
|   |-- 01_distribuicao_quality.png
|   |-- 02_distribuicao_variaveis.png
|   |-- 03_boxplots_outliers.png
|   |-- 04_matriz_correlacao.png
|   |-- 05_correlacao_com_qualidade.png
|   |-- 06_distribuicao_por_classe.png
|   |-- 07_smote_balanceamento.png
|   |-- 08_comparacao_metricas.png
|   |-- 09_matrizes_confusao.png
|   |-- 10_curvas_roc.png
|   |-- 11_feature_importance.png
|   |-- 12_top_features_melhor_modelo.png
|   |-- metricas_modelos.csv
|
|-- relatorio_executivo.md   # Relatório executivo completo
|-- requirements.txt         # Bibliotecas utilizadas
|-- README.md                # Este arquivo
```

## Metodologia

### 1. Compreensão do Problema
- Interpretação do contexto da indústria vitivinícola
- Transformação da variável quality em classificação binária (>= 7 = Alta, < 7 = Baixa/Média)

### 2. Análise Exploratória de Dados (EDA)
- Distribuição de todas as 11 variáveis (histogramas + KDE)
- Detecção de outliers pelo método IQR
- Matriz de correlação com justificativas
- Teste estatístico de Mann-Whitney U entre classes
- Análise do desbalanceamento (86% vs 14%)

### 3. Pré-processamento
- Remoção da coluna Id
- Divisão treino/teste (80/20) com estratificação
- Normalização com StandardScaler
- Balanceamento de classes com SMOTE (Synthetic Minority Over-sampling Technique)

### 4. Modelos Treinados

Foram treinados 4 modelos de classificação:

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **Gradient Boosting** | **0.8908** | **0.5946** | **0.6875** | **0.6377** | 0.8614 |
| XGBoost | 0.8865 | 0.5789 | 0.6875 | 0.6286 | 0.8907 |
| Random Forest | 0.8777 | 0.5556 | 0.6250 | 0.5882 | **0.9067** |
| Logistic Regression | 0.7904 | 0.3621 | 0.6562 | 0.4667 | 0.8484 |

**Melhor modelo por F1-Score: Gradient Boosting (0.6377)**

### 5. Principais Resultados
- **Alcohol** é a variável com maior influência positiva na qualidade
- **Volatile acidity** é a variável com maior influência negativa
- **Sulphates** e **citric acid** também são preditores importantes
- O tratamento do desbalanceamento com SMOTE melhorou o recall da classe minoritária

## Como Executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar o notebook
jupyter notebook notebooks/wine_quality_classification.ipynb
```

## Tecnologias Utilizadas

- **Python 3.12**
- **pandas** - Manipulação de dados
- **numpy** - Computação numérica
- **matplotlib / seaborn** - Visualização de dados
- **scikit-learn** - Modelos de ML e métricas
- **XGBoost** - Gradient boosting otimizado
- **imbalanced-learn** - Tratamento de desbalanceamento (SMOTE)
- **scipy** - Testes estatísticos


