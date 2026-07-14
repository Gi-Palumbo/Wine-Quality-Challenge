
# Classificacao da Qualidade de Vinhos com Machine Learning

## Tech Challenge - Fase 2 | POSTECH FIAP

Projeto de classificacao binaria para prever a qualidade de vinhos tintos (Alta vs Baixa/Media) a partir de caracteristicas fisico-quimicas, utilizando tecnicas de Machine Learning.

---

## Objetivo

Desenvolver um modelo de classificacao capaz de prever a qualidade de um vinho com base em suas caracteristicas fisico-quimicas. A variavel de qualidade foi transformada em classificacao binaria:

- ## Alta Qualidade: nota >= 7
- ## Baixa/Media Qualidade: nota < 7
  
## Dataset
**Fonte:** [Wine Quality Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

- **1.143 amostras** de vinhos tintos
- **11 variaveis** fisico-quimicas + 1 variavel alvo (quality)
- **Sem dados faltantes**
- **Classes desbalanceadas**: 86.1% Baixa/Media vs 13.9% Alta Qualidade

### Variaveis

**Variavel**	       |   **Descricao**
fixed acidity	       |   Acidez fixa
volatile acidity 	   |   Acidez volatil
citric acid	         |   Acido citrico
residual sugar       |   Acucar residual
chlorides	           |   Cloretos
free sulfur dioxide	 |   Dioxido de enxofre livre
total sulfur dioxide |	  Dioxido de enxofre total
density	             |    Densidade
pH	                 |   pH
sulphates	           |   Sulfatos
alcohol	             |   Teor alcoolico
quality	             |   "Qualidade (variavel alvo original, notas 3-8)"

## Estrutura do Projeto

```
wine-quality-classification/
|-- data/                    # Base de dados utilizada
|   |-- WineQT.csv
|
|-- notebooks/               # Notebook com a analise e modelagem
|   |-- wine_quality_classification.ipynb
|
|-- src/                     # Scripts auxiliares
|   |-- __init__.py
|   |-- preprocessing.py     # Pre-processamento de dados
|   |-- modeling.py          # Treinamento e avaliacao de modelos
|
|-- results/                 # Graficos e metricas dos modelos
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
|-- relatorio_executivo.md   # Relatorio executivo completo
|-- requirements.txt         # Bibliotecas utilizadas
|-- README.md                # Este arquivo
```

## Metodologia
## 1. Compreensao do Problema
- Interpretacao do contexto da industria vitivinicola
- Transformacao da variavel quality em classificacao binaria (>= 7 = Alta, < 7 = Baixa/Media)

## 2. Analise Exploratoria de Dados (EDA)
- Distribuicao de todas as 11 variaveis (histogramas + KDE)
- Deteccao de outliers pelo metodo IQR
- Matriz de correlacao com justificativas
- Teste estatistico de Mann-Whitney U entre classes
- Analise do desbalanceamento (86% vs 14%)
  
## 3. Pre-processamento
- Remocao da coluna Id
- Divisao treino/teste (80/20) com estratificacao
- Normalizacao com StandardScaler
- Balanceamento de classes com SMOTE (Synthetic Minority Over-sampling Technique)

## 4. Modelos Treinados
Foram treinados 4 modelos de classificacao:
Modelo             |	Accuracy | Precision | Recall	| F1-Score |AUC-ROC
**Gradient Boosting**  | **0.8908**	 | **0.5946**	   | **0.6875**	| **0.6377**   |0.8614
XGBoost	           |  0.8865	 | 0.5789	   | 0.6875 | 0.6286   |0.8907
Random Forest      |	0.8777	 | 0.5556	   | 0.6250	| 0.5882   |0.9067
Logistic Regression|	0.7904	 | 0.3621	   | 0.6562 | 0.4667   |0.8484

**Melhor modelo por F1-Score: Gradient Boosting (0.6377)**

## 5. Principais Resultados
- **Alcohol** e a variavel com maior influencia positiva na qualidade
- **Volatile acidity** e a variavel com maior influencia negativa
- **Sulphates** e **citric acid** tambem sao preditores importantes
- O tratamento do desbalanceamento com SMOTE melhorou o recall da classe minoritaria
  
## Como Executar
``` bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Executar o notebook
jupyter notebook notebooks/wine_quality_classification.ipynb
```

## Tecnologias Utilizadas
- **Python 3.12**
- **pandas** - Manipulacao de dados
- **numpy** - Computacao numerica
- **matplotlib / seaborn** - Visualizacao de dados
- **scikit-learn** - Modelos de ML e metricas
- **XGBoost** - Gradient boosting otimizado
- **imbalanced-learn** - Tratamento de desbalanceamento (SMOTE)
- **scipy** - Testes estatisticos

