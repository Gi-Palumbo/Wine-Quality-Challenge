# Relatorio Executivo
## Classificacao da Qualidade de Vinhos com Machine Learning

### Tech Challenge - Fase 2 | POSTECH FIAP

---

## 1. Contexto e Motivacao

A avaliacao da qualidade de vinhos e tradicionalmente realizada por paineis de degustacao compostos por especialistas (enologos e sommeliers). Este processo, embora valioso, apresenta limitacoes significativas:

- **Subjetividade**: diferentes avaliadores podem atribuir notas distintas ao mesmo vinho
- **Custo elevado**: manter um painel de especialistas e caro e demanda tempo
- **Escalabilidade limitada**: nao e viavel avaliar sensorialmente grandes volumes de producao

Com o avanco das tecnicas de ciencia de dados, tornou-se possivel utilizar **dados fisico-quimicos** — obtidos por analises laboratoriais rapidas e padronizadas — para auxiliar na previsao da qualidade final do produto.

**Proposta deste projeto:** Desenvolver um modelo de Machine Learning capaz de classificar vinhos como "Alta Qualidade" ou "Baixa/Media Qualidade" com base exclusivamente em suas propriedades fisico-quimicas.

---

## 2. Dados Utilizados

### 2.1 Fonte
O dataset **Wine Quality Dataset** foi obtido do Kaggle e contem **1.143 amostras de vinhos tintos** com 11 variaveis fisico-quimicas e uma nota de qualidade atribuida por especialistas (escala de 3 a 8).

### 2.2 Variaveis Analisadas

| Variavel | O que mede | Unidade |
|---|---|---|
| Fixed Acidity | Acidos nao-volateis (tartarico) | g/dm3 |
| Volatile Acidity | Acidos volateis (acetico) | g/dm3 |
| Citric Acid | Acido citrico (frescor) | g/dm3 |
| Residual Sugar | Acucar residual apos fermentacao | g/dm3 |
| Chlorides | Teor de cloretos (sal) | g/dm3 |
| Free Sulfur Dioxide | SO2 livre (conservante ativo) | mg/dm3 |
| Total Sulfur Dioxide | SO2 total | mg/dm3 |
| Density | Densidade do vinho | g/cm3 |
| pH | Nivel de acidez/alcalinidade | - |
| Sulphates | Sulfatos (conservante) | g/dm3 |
| Alcohol | Teor alcoolico | % vol. |
| Quality | Nota dos especialistas (3-8) | - |

### 2.3 Qualidade dos Dados
- **Zero valores faltantes** em todas as variaveis
- Dados consistentes e sem erros de digitacao aparentes
- Presenca de outliers em algumas variaveis (mantidos na analise por representarem composicoes reais)

### 2.4 Transformacao da Variavel Alvo
Para simplificar o problema, a nota de qualidade (3 a 8) foi transformada em **classificacao binaria**:
- **Alta Qualidade**: nota >= 7 (159 amostras, 13.9%)
- **Baixa/Media Qualidade**: nota < 7 (984 amostras, 86.1%)

**Distribuicao original das notas:**
| Nota | Quantidade | Percentual |
|------|-----------|------------|
| 3 | 6 | 0.5% |
| 4 | 33 | 2.9% |
| 5 | 483 | 42.3% |
| 6 | 462 | 40.4% |
| 7 | 143 | 12.5% |
| 8 | 16 | 1.4% |

> **Observacao importante:** As classes estao fortemente desbalanceadas (86% vs 14%), o que foi tratado na etapa de pre-processamento com a tecnica SMOTE.

---

## 3. Analise Exploratoria - Principais Insights

### 3.1 Variaveis que Mais Diferenciam Vinhos de Alta Qualidade

A analise estatistica (teste de Mann-Whitney U) revelou que **todas as 11 variaveis** apresentam diferenca estatisticamente significativa (p < 0.05) entre vinhos de alta e baixa/media qualidade. Porem, as mais relevantes sao:

**Indicadores de ALTA qualidade:**
1. **Teor Alcoolico (alcohol)** - Vinhos de alta qualidade tem, em media, maior teor alcoolico. A correlacao com qualidade e a mais forte entre todas as variaveis (+0.48).
2. **Sulfatos (sulphates)** - Niveis moderadamente mais altos de sulfatos estao associados a melhor qualidade (+0.27).
3. **Acido Citrico (citric acid)** - Contribui para o frescor e complexidade do sabor (+0.20).

**Indicadores de BAIXA qualidade:**
1. **Acidez Volatil (volatile acidity)** - E o principal indicador negativo (-0.31). Niveis elevados indicam presenca excessiva de acido acetico, que confere sabor de vinagre.
2. **Dioxido de Enxofre Total** - Excesso de SO2 prejudica o aroma e sabor (-0.14).
3. **Densidade** - Vinhos mais densos tendem a ter menor qualidade (-0.14), correlacionado com menor teor alcoolico.

### 3.2 Correlacoes Fisico-Quimicas Importantes

Foram identificadas correlacoes esperadas do ponto de vista quimico:
- **Acidez fixa vs pH** (correlacao negativa forte): quanto mais acido, menor o pH
- **Dioxido de enxofre livre vs total** (correlacao positiva forte): SO2 livre e subconjunto do total
- **Densidade vs alcohol** (correlacao negativa): alcool e menos denso que agua
- **Acidez fixa vs acido citrico** (correlacao positiva): ambos sao acidos presentes no vinho

### 3.3 Deteccao de Outliers

Foram detectados outliers pelo metodo IQR (Interquartile Range) em diversas variaveis. Os mais notaveis:
- **Residual sugar**: vinhos com acucar residual muito acima da media
- **Chlorides**: algumas amostras com niveis atipicos de cloretos
- **Total sulfur dioxide**: amostras com SO2 total muito elevado

**Decisao:** Os outliers foram **mantidos** na analise, pois representam composicoes reais de vinhos e podem conter informacoes valiosas para a classificacao.

---

## 4. Tratamento dos Dados

### 4.1 Divisao dos Dados
- **80% para treinamento** (914 amostras)
- **20% para teste** (229 amostras)
- Divisao estratificada para manter a proporcao de classes em ambos os conjuntos

### 4.2 Normalizacao
Aplicado **StandardScaler** para padronizar todas as variaveis para media 0 e desvio padrao 1. Isso e importante pois:
- Variaveis estao em escalas diferentes (ex: pH vai de 2.7 a 4.0, enquanto total sulfur dioxide vai de 6 a 289)
- Modelos como Logistic Regression e SVM sao sensiveis a escala dos dados

### 4.3 Balanceamento de Classes (SMOTE)
A tecnica **SMOTE (Synthetic Minority Over-sampling Technique)** foi aplicada para gerar amostras sinteticas da classe minoritaria (Alta Qualidade):
- **Antes do SMOTE**: 787 Baixa/Media vs 127 Alta
- **Depois do SMOTE**: 787 Baixa/Media vs 787 Alta

O SMOTE cria novas amostras sinteticas interpolando entre exemplos existentes da classe minoritaria, permitindo que o modelo aprenda melhor os padroes de vinhos de alta qualidade.

---

## 5. Modelos Desenvolvidos

Foram treinados e avaliados **4 modelos de classificacao**:

### 5.1 Logistic Regression
Modelo linear simples e interpretavel. Serve como baseline (referencia) para comparacao com modelos mais complexos. Utiliza peso balanceado por classe (class_weight='balanced').

### 5.2 Random Forest
Ensemble de 200 arvores de decisao que votam de forma democratica. Robusto a overfitting e capaz de capturar relacoes nao-lineares.

### 5.3 XGBoost
Gradient boosting otimizado. Constroi arvores sequencialmente, onde cada nova arvore corrige os erros das anteriores. Inclui regularizacao para evitar overfitting.

### 5.4 Gradient Boosting
Similar ao XGBoost, mas com implementacao do scikit-learn. Tambem constroi arvores sequencialmente para minimizar o erro.

---

## 6. Resultados dos Modelos

### 6.1 Metricas de Desempenho

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **Gradient Boosting** | **89.1%** | **59.5%** | **68.8%** | **63.8%** | 86.1% |
| XGBoost | 88.6% | 57.9% | 68.8% | 62.9% | 89.1% |
| Random Forest | 87.8% | 55.6% | 62.5% | 58.8% | **90.7%** |
| Logistic Regression | 79.0% | 36.2% | 65.6% | 46.7% | 84.8% |

### 6.2 Interpretacao das Metricas

**Accuracy (Acuracia):** Percentual geral de acertos. O Gradient Boosting acertou 89.1% das classificacoes.

**Precision (Precisao):** Dos vinhos classificados como "Alta Qualidade", quantos realmente eram. O Gradient Boosting acertou 59.5% das vezes que disse ser alta qualidade.

**Recall (Sensibilidade):** Dos vinhos que realmente sao de Alta Qualidade, quantos o modelo identificou. O Gradient Boosting encontrou 68.8% dos vinhos de alta qualidade.

**F1-Score:** Media harmonica entre Precision e Recall — a metrica mais equilibrada para problemas desbalanceados. O Gradient Boosting obteve 63.8%.

**AUC-ROC:** Capacidade do modelo de distinguir entre as classes. O Random Forest obteve a maior AUC (90.7%), indicando excelente capacidade discriminativa.

### 6.3 Validacao Cruzada
A validacao cruzada (5-fold) confirmou a robustez dos resultados, com baixa variancia entre os folds, garantindo que os modelos nao estao sobreajustados aos dados.

### 6.4 Melhor Modelo: Gradient Boosting

O **Gradient Boosting** foi selecionado como melhor modelo pelo criterio F1-Score, que e a metrica mais adequada para problemas desbalanceados. Ele oferece o melhor equilibrio entre:
- Identificar corretamente vinhos de alta qualidade (Recall: 68.8%)
- Evitar classificar erroneamente vinhos mediocres como alta qualidade (Precision: 59.5%)

---

## 7. Variaveis Mais Influentes

Com base na analise de **Feature Importance** dos modelos, as variaveis com maior poder preditivo sao:

### Top 5 Variaveis Mais Importantes

| Ranking | Variavel | Impacto na Qualidade |
|---------|----------|---------------------|
| 1 | **Alcohol** (Teor Alcoolico) | Quanto maior, melhor a qualidade |
| 2 | **Volatile Acidity** (Acidez Volatil) | Quanto maior, pior a qualidade |
| 3 | **Sulphates** (Sulfatos) | Niveis adequados melhoram a qualidade |
| 4 | **Citric Acid** (Acido Citrico) | Contribui positivamente |
| 5 | **Total Sulfur Dioxide** (SO2 Total) | Excesso prejudica a qualidade |

---

## 8. Implicacoes para a Industria

### 8.1 Recomendacoes para a Producao

Com base nos resultados, os produtores de vinho podem focar em:

1. **Monitorar o teor alcoolico**: Garantir que a fermentacao atinja niveis ideais de alcool, pois esta e a variavel mais associada a alta qualidade.

2. **Controlar a acidez volatil**: Minimizar a formacao de acido acetico durante a fermentacao. Acidez volatil elevada e o principal indicador de baixa qualidade.

3. **Dosar sulfatos adequadamente**: Adicionar sulfatos (antioxidante natural) em quantidades que preservem o vinho sem alterar negativamente o sabor.

4. **Equilibrar o acido citrico**: Manter niveis que contribuam para o frescor e a complexidade do vinho.

5. **Controlar o dioxido de enxofre**: Encontrar o equilibrio entre a preservacao do vinho e a qualidade sensorial, evitando excesso de SO2.

### 8.2 Aplicacao Pratica do Modelo

O modelo pode ser utilizado como ferramenta complementar na linha de producao:
- **Triagem rapida**: Analisar amostras em laboratorio e usar o modelo para uma previsao preliminar de qualidade
- **Controle de qualidade**: Identificar lotes que podem precisar de ajustes no processo
- **Reducao de custos**: Diminuir a necessidade de avaliacoes sensoriais para todos os lotes
- **Padronizacao**: Oferecer uma referencia objetiva e reproduzivel de qualidade

---

## 9. Limitacoes do Estudo

1. **Tamanho do dataset**: 1.143 amostras e um volume relativamente pequeno para Machine Learning. Mais dados poderiam melhorar o desempenho dos modelos.

2. **Apenas vinhos tintos**: Os resultados sao especificos para vinhos tintos e podem nao se aplicar a vinhos brancos, roses ou espumantes.

3. **Classificacao binaria**: A simplificacao para apenas duas classes (Alta vs Baixa/Media) perde as nuances da escala original de qualidade.

4. **Subjetividade residual**: A variavel alvo (quality) foi definida por especialistas, que tambem carregam subjetividade em suas avaliacoes.

5. **Variaveis sensoriais ausentes**: O dataset nao inclui informacoes sobre aroma, sabor, cor ou aparencia visual, que sao componentes importantes da avaliacao de qualidade.

---

## 10. Conclusao

Este projeto demonstrou que e **viavel utilizar modelos de Machine Learning para prever a qualidade de vinhos tintos** a partir de analises fisico-quimicas, alcancando um F1-Score de 63.8% e AUC-ROC de 86.1% com o modelo Gradient Boosting.

Os resultados validam que **as propriedades fisico-quimicas sao preditores relevantes da qualidade percebida**, com destaque para o teor alcoolico, acidez volatil e sulfatos.

O modelo nao substitui a avaliacao sensorial humana, mas representa uma **ferramenta complementar valiosa** para triagem, controle de qualidade e tomada de decisao na producao de vinhos.

---

## Tecnologias Utilizadas

- Python 3.12
- pandas, numpy (manipulacao de dados)
- matplotlib, seaborn (visualizacao)
- scikit-learn (modelos e metricas)
- XGBoost (gradient boosting)
- imbalanced-learn (SMOTE)
- scipy (testes estatisticos)

---

*Relatorio gerado como parte do Tech Challenge - Fase 2 | POSTECH FIAP*
