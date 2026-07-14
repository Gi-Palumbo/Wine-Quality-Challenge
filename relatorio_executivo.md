# Relatório Executivo
## Classificação da Qualidade de Vinhos com Machine Learning

### Tech Challenge - Fase 2 | POSTECH FIAP

---

## 1. Contexto e Motivação

A avaliação da qualidade de vinhos é tradicionalmente realizada por painéis de degustação compostos por especialistas (enólogos e sommeliers). Este processo, embora valioso, apresenta limitações significativas:

- **Subjetividade**: diferentes avaliadores podem atribuir notas distintas ao mesmo vinho
- **Custo elevado**: manter um painel de especialistas é caro e demanda tempo
- **Escalabilidade limitada**: não é viável avaliar sensorialmente grandes volumes de produção

Com o avanço das técnicas de ciência de dados, tornou-se possível utilizar **dados físico-químicos** — obtidos por análises laboratoriais rápidas e padronizadas — para auxiliar na previsão da qualidade final do produto.

**Proposta deste projeto:** Desenvolver um modelo de Machine Learning capaz de classificar vinhos como "Alta Qualidade" ou "Baixa/Média Qualidade" com base exclusivamente em suas propriedades físico-químicas.

---

## 2. Dados Utilizados

### 2.1 Fonte
O dataset **Wine Quality Dataset** foi obtido do Kaggle e contém **1.143 amostras de vinhos tintos** com 11 variáveis físico-químicas e uma nota de qualidade atribuída por especialistas (escala de 3 a 8).

### 2.2 Variáveis Analisadas

| Variável | O que mede | Unidade |
|---|---|---|
| Fixed Acidity | Ácidos não-voláteis (tartárico) | g/dm3 |
| Volatile Acidity | Ácidos voláteis (acético) | g/dm3 |
| Citric Acid | Ácido cítrico (frescor) | g/dm3 |
| Residual Sugar | Açúcar residual após fermentação | g/dm3 |
| Chlorides | Teor de cloretos (sal) | g/dm3 |
| Free Sulfur Dioxide | SO2 livre (conservante ativo) | mg/dm3 |
| Total Sulfur Dioxide | SO2 total | mg/dm3 |
| Density | Densidade do vinho | g/cm3 |
| pH | Nível de acidez/alcalinidade | - |
| Sulphates | Sulfatos (conservante) | g/dm3 |
| Alcohol | Teor alcoólico | % vol. |
| Quality | Nota dos especialistas (3-8) | - |

### 2.3 Qualidade dos Dados
- **Zero valores faltantes** em todas as variáveis
- Dados consistentes e sem erros de digitação aparentes
- Presença de outliers em algumas variáveis (mantidos na análise por representarem composições reais)

### 2.4 Transformação da Variável Alvo
Para simplificar o problema, a nota de qualidade (3 a 8) foi transformada em **classificação binária**:
- **Alta Qualidade**: nota >= 7 (159 amostras, 13.9%)
- **Baixa/Média Qualidade**: nota < 7 (984 amostras, 86.1%)

**Distribuição original das notas:**
| Nota | Quantidade | Percentual |
|------|-----------|------------|
| 3 | 6 | 0.5% |
| 4 | 33 | 2.9% |
| 5 | 483 | 42.3% |
| 6 | 462 | 40.4% |
| 7 | 143 | 12.5% |
| 8 | 16 | 1.4% |

> **Observação importante:** As classes estão fortemente desbalanceadas (86% vs 14%), o que foi tratado na etapa de pré-processamento com a técnica SMOTE.

---

## 3. Análise Exploratória - Principais Insights

### 3.1 Variáveis que Mais Diferenciam Vinhos de Alta Qualidade

A análise estatística (teste de Mann-Whitney U) revelou que **todas as 11 variáveis** apresentam diferença estatisticamente significativa (p < 0.05) entre vinhos de alta e baixa/média qualidade. Porém, as mais relevantes são:

**Indicadores de ALTA qualidade:**
1. **Teor Alcoólico (alcohol)** - Vinhos de alta qualidade têm, em média, maior teor alcoólico. A correlação com qualidade é a mais forte entre todas as variáveis (+0.48).
2. **Sulfatos (sulphates)** - Níveis moderadamente mais altos de sulfatos estão associados a melhor qualidade (+0.27).
3. **Ácido Cítrico (citric acid)** - Contribui para o frescor e complexidade do sabor (+0.20).

**Indicadores de BAIXA qualidade:**
1. **Acidez Volátil (volatile acidity)** - É o principal indicador negativo (-0.31). Níveis elevados indicam presença excessiva de ácido acético, que confere sabor de vinagre.
2. **Dióxido de Enxofre Total** - Excesso de SO2 prejudica o aroma e sabor (-0.14).
3. **Densidade** - Vinhos mais densos tendem a ter menor qualidade (-0.14), correlacionado com menor teor alcoólico.

### 3.2 Correlações Físico-Químicas Importantes

Foram identificadas correlações esperadas do ponto de vista químico:
- **Acidez fixa vs pH** (correlação negativa forte): quanto mais ácido, menor o pH
- **Dióxido de enxofre livre vs total** (correlação positiva forte): SO2 livre é subconjunto do total
- **Densidade vs alcohol** (correlação negativa): álcool é menos denso que água
- **Acidez fixa vs ácido cítrico** (correlação positiva): ambos são ácidos presentes no vinho

### 3.3 Detecção de Outliers

Foram detectados outliers pelo método IQR (Interquartile Range) em diversas variáveis. Os mais notáveis:
- **Residual sugar**: vinhos com açúcar residual muito acima da média
- **Chlorides**: algumas amostras com níveis atípicos de cloretos
- **Total sulfur dioxide**: amostras com SO2 total muito elevado

**Decisão:** Os outliers foram **mantidos** na análise, pois representam composições reais de vinhos e podem conter informações valiosas para a classificação.

---

## 4. Tratamento dos Dados

### 4.1 Divisão dos Dados
- **80% para treinamento** (914 amostras)
- **20% para teste** (229 amostras)
- Divisão estratificada para manter a proporção de classes em ambos os conjuntos

### 4.2 Normalização
Aplicado **StandardScaler** para padronizar todas as variáveis para média 0 e desvio padrão 1. Isso é importante pois:
- Variáveis estão em escalas diferentes (ex: pH vai de 2.7 a 4.0, enquanto total sulfur dioxide vai de 6 a 289)
- Modelos como Logistic Regression e SVM são sensíveis à escala dos dados

### 4.3 Balanceamento de Classes (SMOTE)
A técnica **SMOTE (Synthetic Minority Over-sampling Technique)** foi aplicada para gerar amostras sintéticas da classe minoritária (Alta Qualidade):
- **Antes do SMOTE**: 787 Baixa/Média vs 127 Alta
- **Depois do SMOTE**: 787 Baixa/Média vs 787 Alta

O SMOTE cria novas amostras sintéticas interpolando entre exemplos existentes da classe minoritária, permitindo que o modelo aprenda melhor os padrões de vinhos de alta qualidade.

---

## 5. Modelos Desenvolvidos

Foram treinados e avaliados **4 modelos de classificação**:

### 5.1 Logistic Regression
Modelo linear simples e interpretável. Serve como baseline (referência) para comparação com modelos mais complexos. Utiliza peso balanceado por classe (class_weight='balanced').

### 5.2 Random Forest
Ensemble de 200 árvores de decisão que votam de forma democrática. Robusto a overfitting e capaz de capturar relações não-lineares.

### 5.3 XGBoost
Gradient boosting otimizado. Constrói árvores sequencialmente, onde cada nova árvore corrige os erros das anteriores. Inclui regularização para evitar overfitting.

### 5.4 Gradient Boosting
Similar ao XGBoost, mas com implementação do scikit-learn. Também constrói árvores sequencialmente para minimizar o erro.

---

## 6. Resultados dos Modelos

### 6.1 Métricas de Desempenho

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **Gradient Boosting** | **89.1%** | **59.5%** | **68.8%** | **63.8%** | 86.1% |
| XGBoost | 88.6% | 57.9% | 68.8% | 62.9% | 89.1% |
| Random Forest | 87.8% | 55.6% | 62.5% | 58.8% | **90.7%** |
| Logistic Regression | 79.0% | 36.2% | 65.6% | 46.7% | 84.8% |

### 6.2 Interpretação das Métricas

**Accuracy (Acurácia):** Percentual geral de acertos. O Gradient Boosting acertou 89.1% das classificações.

**Precision (Precisão):** Dos vinhos classificados como "Alta Qualidade", quantos realmente eram. O Gradient Boosting acertou 59.5% das vezes que disse ser alta qualidade.

**Recall (Sensibilidade):** Dos vinhos que realmente são de Alta Qualidade, quantos o modelo identificou. O Gradient Boosting encontrou 68.8% dos vinhos de alta qualidade.

**F1-Score:** Média harmônica entre Precision e Recall — a métrica mais equilibrada para problemas desbalanceados. O Gradient Boosting obteve 63.8%.

**AUC-ROC:** Capacidade do modelo de distinguir entre as classes. O Random Forest obteve a maior AUC (90.7%), indicando excelente capacidade discriminativa.

### 6.3 Validação Cruzada
A validação cruzada (5-fold) confirmou a robustez dos resultados, com baixa variância entre os folds, garantindo que os modelos não estão sobreajustados aos dados.

### 6.4 Melhor Modelo: Gradient Boosting

O **Gradient Boosting** foi selecionado como melhor modelo pelo critério F1-Score, que é a métrica mais adequada para problemas desbalanceados. Ele oferece o melhor equilíbrio entre:
- Identificar corretamente vinhos de alta qualidade (Recall: 68.8%)
- Evitar classificar erroneamente vinhos medíocres como alta qualidade (Precision: 59.5%)

---

## 7. Variáveis Mais Influentes

Com base na análise de **Feature Importance** dos modelos, as variáveis com maior poder preditivo são:

### Top 5 Variáveis Mais Importantes

| Ranking | Variável | Impacto na Qualidade |
|---------|----------|---------------------|
| 1 | **Alcohol** (Teor Alcoólico) | Quanto maior, melhor a qualidade |
| 2 | **Volatile Acidity** (Acidez Volátil) | Quanto maior, pior a qualidade |
| 3 | **Sulphates** (Sulfatos) | Níveis adequados melhoram a qualidade |
| 4 | **Citric Acid** (Ácido Cítrico) | Contribui positivamente |
| 5 | **Total Sulfur Dioxide** (SO2 Total) | Excesso prejudica a qualidade |

---

## 8. Implicações para a Indústria

### 8.1 Recomendações para a Produção

Com base nos resultados, os produtores de vinho podem focar em:

1. **Monitorar o teor alcoólico**: Garantir que a fermentação atinja níveis ideais de álcool, pois esta é a variável mais associada à alta qualidade.

2. **Controlar a acidez volátil**: Minimizar a formação de ácido acético durante a fermentação. Acidez volátil elevada é o principal indicador de baixa qualidade.

3. **Dosar sulfatos adequadamente**: Adicionar sulfatos (antioxidante natural) em quantidades que preservem o vinho sem alterar negativamente o sabor.

4. **Equilibrar o ácido cítrico**: Manter níveis que contribuam para o frescor e a complexidade do vinho.

5. **Controlar o dióxido de enxofre**: Encontrar o equilíbrio entre a preservação do vinho e a qualidade sensorial, evitando excesso de SO2.

### 8.2 Aplicação Prática do Modelo

O modelo pode ser utilizado como ferramenta complementar na linha de produção:
- **Triagem rápida**: Analisar amostras em laboratório e usar o modelo para uma previsão preliminar de qualidade
- **Controle de qualidade**: Identificar lotes que podem precisar de ajustes no processo
- **Redução de custos**: Diminuir a necessidade de avaliações sensoriais para todos os lotes
- **Padronização**: Oferecer uma referência objetiva e reproduzível de qualidade

---

## 9. Limitações do Estudo

1. **Tamanho do dataset**: 1.143 amostras é um volume relativamente pequeno para Machine Learning. Mais dados poderiam melhorar o desempenho dos modelos.

2. **Apenas vinhos tintos**: Os resultados são específicos para vinhos tintos e podem não se aplicar a vinhos brancos, rosés ou espumantes.

3. **Classificação binária**: A simplificação para apenas duas classes (Alta vs Baixa/Média) perde as nuances da escala original de qualidade.

4. **Subjetividade residual**: A variável alvo (quality) foi definida por especialistas, que também carregam subjetividade em suas avaliações.

5. **Variáveis sensoriais ausentes**: O dataset não inclui informações sobre aroma, sabor, cor ou aparência visual, que são componentes importantes da avaliação de qualidade.

---

## 10. Conclusão

Este projeto demonstrou que é **viável utilizar modelos de Machine Learning para prever a qualidade de vinhos tintos** a partir de análises físico-químicas, alcançando um F1-Score de 63.8% e AUC-ROC de 86.1% com o modelo Gradient Boosting.

Os resultados validam que **as propriedades físico-químicas são preditores relevantes da qualidade percebida**, com destaque para o teor alcoólico, acidez volátil e sulfatos.

O modelo não substitui a avaliação sensorial humana, mas representa uma **ferramenta complementar valiosa** para triagem, controle de qualidade e tomada de decisão na produção de vinhos.

---

## Tecnologias Utilizadas

- Python 3.12
- pandas, numpy (manipulação de dados)
- matplotlib, seaborn (visualização)
- scikit-learn (modelos e métricas)
- XGBoost (gradient boosting)
- imbalanced-learn (SMOTE)
- scipy (testes estatísticos)

---
