# Diabetes Risk Prediction 🩺

Repository for a notebook from a Machine Learning class (IBMEC).

Projeto de classificação que busca prever o **risco de diabetes** a partir de indicadores de saúde e estilo de vida, usando o dataset [Diabetes Health Indicators](https://raw.githubusercontent.com/cassiusf/datasets/refs/heads/main/desafios/diabetes_health_indicators.csv), comparando regressão logística e Random Forest.

Uma pequena aplicação com o modelo de Random Forrest está disponível no link: https://thiagobrandaorj.github.io/diabetes-risk-prediction/

## Arquivo

### `Cópia_de_03_Diabetes.ipynb`

Notebook único, organizado em duas grandes etapas:

**1. EDA (Análise Exploratória)**
- Carregamento do dataset e remoção de **23.899 linhas duplicadas** (não havia valores ausentes).
- **Reagrupamento do alvo**: a variável original `Diabetes_012` tinha 3 classes (sem diabetes / pré-diabetes / diabetes), mas a classe "pré-diabetes" era muito rara. As classes pré-diabetes e diabetes foram unificadas em um único rótulo positivo, transformando o problema em **classificação binária**.
- **Análise univariada** de praticamente todas as ~20 features do dataset (`HighBP`, `HighChol`, `CholCheck`, `BMI`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk`, `Sex`, `Age`, `Education`, `Income`), com distribuição percentual de cada variável categórica/discreta e histogramas para as contínuas (`BMI`, `MentHlth`, `PhysHlth`).
- Identificação de variáveis com classes raras/subrepresentadas (`CholCheck`, `Stroke`) como possíveis candidatas à remoção, e discussão sobre manter variáveis como `Age`, `Education` e `Income` por relevância de negócio mesmo com categorias menos frequentes.

**2. Pré-processamento e Modelagem**
- Split treino/teste (80/20), estratificado pela variável alvo.
- Pipeline com `ColumnTransformer`: `RobustScaler` aplicado apenas em `BMI` (única variável contínua que precisava de tratamento de escala), demais colunas passam direto (`passthrough`).
- Análise da matriz de correlação entre as features processadas e o alvo.
- **Modelo 1 — Regressão Logística**: `class_weight="balanced"` + threshold customizado (0.33) para priorizar recall na classe positiva.
- **Modelo 2 — Random Forest**: hiperparâmetros ajustados (indícios de busca via `HalvingRandomSearchCV`, comentada no notebook), com `class_weight={0: 1, 1: 15}` para reforçar ainda mais o peso da classe minoritária. Inclui análise de **importância de features** (impureza média / MDI), destacando quais indicadores mais pesam na predição de diabetes.

## Dataset

- **Fonte**: [diabetes_health_indicators.csv](https://raw.githubusercontent.com/cassiusf/datasets/refs/heads/main/desafios/diabetes_health_indicators.csv)
- **Variável alvo**: `Diabetes_012` (reagrupada em binária: 0 = sem diabetes, 1 = pré-diabetes ou diabetes)
- **Features**: ~20 indicadores clínicos e de estilo de vida (pressão alta, colesterol, IMC, tabagismo, atividade física, saúde geral, idade, renda, escolaridade, entre outros)

## Como rodar

Desenvolvido no Google Colab. Principais dependências:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap
```

## Principais decisões e aprendizados

- Classes muito raras dentro do próprio alvo (como "pré-diabetes") podem ser agrupadas com a classe de interesse mais próxima para tornar o problema mais tratável, sem perder o objetivo de negócio (identificar quem está em risco).
- Ajuste de **threshold de decisão** e **pesos de classe** (tanto em regressão logística quanto em Random Forest) são ferramentas complementares para lidar com desbalanceamento, sem depender de oversampling/undersampling.
- Nem toda variável rara deve ser descartada automaticamente — a decisão de manter `Age`, `Education` e `Income` mesmo com categorias pouco frequentes respeita a relevância de negócio dessas variáveis. 
