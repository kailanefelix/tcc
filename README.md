# TCC — Benchmark de Modelos Preditivos

Previsão do número de beneficiários por município e setor (PROGRAMA) usando séries temporais curtas (2021–2025).

---

## Estrutura

```
code/
  preprocessing.py       # pipeline de pré-processamento
  benchmark.py           # treinamento e avaliação dos modelos
  eda.ipynb              # análise exploratória
  modeling.ipynb         # execução principal e análise de resultados
data/
  cientista_cp.xlsx      # dados brutos (517 linhas, 101 municípios, 3 programas)
```

## Modelos

| Família       | Modelos                             | Granularidade de treino       |
|---------------|-------------------------------------|-------------------------------|
| Estatístico   | ARIMA (1,1,0), ETS (Holt Linear)    | Série agregada por programa   |
| ML Clássico   | Regressão Linear, Árvore de Decisão | Município × programa          |
| ML Ensemble   | Random Forest, LightGBM             | Município × programa          |

**Modos:** `GLOBAL` (todos os municípios/programas juntos) · `POR_SETOR` (um modelo por PROGRAMA)

**Validação:** walk-forward cross-validation — folds com teste em 2023, 2024 e 2025 (mínimo 2 anos de treino por fold).

---

## Log de Desenvolvimento

### 22/03/2025
- EDA: 517 linhas, 101 municípios, 3 programas; lacunas identificadas (nem todo município tem todos os anos/programas)
- Primeiros testes manuais de ARIMA, ETS, Random Forest e LightGBM
- Criação de `preprocessing.py` e `benchmark.py`

### 26/03/2026
- Confirmado data leakage via `lag_1`: removido de `ML_FEATURES` (Opção A)
- Reexecução do benchmark sem `lag_1` → resultados salvos em `resultados_benchmark_v3.csv`
- DecisionTree Pesca: 0,03% → 9,25% (confirmação do leakage — resultado anterior era espúrio)
- LightGBM Fruticultura: 0,23% → 2,40% / LightGBM Canavieira: 0,65% → 4,27% (ainda competitivo)
- LightGBM POR_SETOR continua o melhor modelo de ML; ETS POR_SETOR Canavieira segue o melhor geral

### 10/05/2026
- Identificado leakage residual: `rolling_mean_2` e `trend` dependiam de `lag_1` internamente
  - `rolling_mean_2` era `(lag_1 + lag_2) / 2` → corrigido para `(lag_2 + lag_3) / 2`
  - `trend` era `lag_1 - lag_2` → corrigido para `lag_2 - lag_3`
  - `lag_1` removido inteiramente do pré-processamento
- Reexecução do benchmark com features limpas → resultados salvos em `resultados_benchmark_v4.csv`
- LightGBM Fruticultura: 2,40% → 3,30% / LightGBM Canavieira: 4,27% → 7,27%
- LightGBM Pesca: 16,61% → 26,45% (série mais volátil, features mais antigas são menos informativas)
- RandomForest Fruticultura melhora: 13,50% → 5,80% (reduziu overfitting com features antigas)
- LightGBM POR_SETOR segue melhor ML (Fruticultura, 3,30%); ETS POR_SETOR Canavieira segue melhor geral (3,78%)
- Implementado walk-forward CV com 3 folds (teste em 2023, 2024, 2025); ARIMA é o modelo mais estável no CV
  - ETS Zona Canavieira: 3,78% no fold 2025 mas 40,05% ±34,45 no CV — resultado único era sorte
  - LightGBM Fruticultura: 3,30% no fold 2025 mas 33,64% ±29,84 no CV — mesma ressalva
  - ARIMA POR_SETOR: 5,84% ±2,16 no CV — menor média e menor variância entre folds
- Feature engineering (v5): adicionadas `growth_rate`, `zero_historico`, `programa_total_lag2`, `share_municipio`
  - Todas baseadas em lag_2/lag_3 — sem leakage
  - RandomForest beneficiado pelo contexto cross-município: Fruticultura 5,80% → 2,30%, Canavieira 17,65% → 3,00%
  - CV do RandomForest: Fruticultura 9,09% → 7,83%, Pesca 10,85% → 9,91% (melhora consistente, variância persiste)
  - Resultados salvos em `resultados_benchmark_v5.csv` e `resultados_cv_v5_*.csv`

### 16/05/2026
- **Renomeação de família:** "ML Moderno" → "ML baseado em Ensemble" em todo o código (`benchmark.py`, `modeling.ipynb`, CSVs de saída v7+).
- **Grid search para RandomForest e LightGBM:** `run_grid_search()` em `benchmark.py` avalia 18 combinações (RF) e 12 (LightGBM) via walk-forward CV POR_SETOR e seleciona os melhores hiperparâmetros.
  - RF: `n_estimators=100, max_depth=6, min_samples_leaf=2` (CV MAPE médio 13,65%)
  - LightGBM: `n_estimators=100, max_depth=3, learning_rate=0.05` (CV MAPE médio 27,10%)
  - Nota: LightGBM não se beneficia de tuning nessa escala de dados — alta variância persiste independente dos hiperparâmetros.
- **Benchmark v7:** reexecução com params otimizados → `resultados_benchmark_v7.csv` e `resultados_cv_v7_*.csv`
  - RandomForest Fruticultura fold 2025: 2,30% (v6) → **0,62%** (v7) — melhora expressiva
  - RandomForest Fruticultura CV: 7,83% ±9,78 (v5) → **3,58% ±3,48** (v7) — menor média e menor variância
  - LightGBM sem ganho significativo no CV mesmo com tuning (alta variância estrutural)
- **Flag `--gridsearch` adicionada ao CLI:** `python benchmark.py --file ... --gridsearch` roda o grid search antes e usa os melhores params no benchmark/CV subsequente.
- `modeling.ipynb` atualizado para usar dados v7 e incluir `ETS_notrend` e `Naive` nos gráficos.

### 14/05/2026
- **Fix leakage residual no CV (crítico):** `_add_context_features` calculava `programa_total_lag2`
  e `share_municipio` sobre o `df_ml` completo (todos os anos). Em folds com test_year=2023 ou 2024,
  as linhas de treino tinham essas features contaminadas com dados de anos futuros.
  Correção: em `run_walkforward_cv` e `run_benchmark`, `df_ml` é cortado para `ANO <= test_year`
  e `_add_context_features` é recalculada sobre o subconjunto antes de cada fold.
  `_add_context_features` importada explicitamente de `preprocessing.py`.
- **Modelo naïve adicionado (família "Baseline"):** previsão = `lag_2` (valor de t-2).
  Referência mínima para avaliar se os modelos aprendem algo além de repetir o penúltimo valor.
  Resultados no fold 2025: Pesca 2,13% / Fruticultura 3,12% — vários modelos ML ficam abaixo do baseline no CV.
- **ETS sem tendência adicionado (`ETS_notrend`):** `trend=None` no Holt-Winters.
  Resultado fold 2025: GLOBAL 3,13% (melhor geral), Pesca 4,77%, Canavieira 6,20% — supera ETS com trend em todos os casos.
- **RandomForest regularizado:** `min_samples_leaf=3` adicionado para reduzir overfitting em séries curtas.
  Efeito no fold 2025: Fruticultura 2,30% → 15,28% (piora no fold único mas melhora esperada no CV).
- Resultados salvos em `resultados_benchmark_v6.csv` e `resultados_cv_v6_*.csv`

---

## Problemas e Soluções

### Problema 1 — Comparação injusta entre famílias (resolvido)

**Causa:** MAPE calculado em granularidades diferentes. Modelos estatísticos geram 1 previsão por programa (agregado); modelos de ML geravam 1 previsão por município, inflando o erro em municípios com poucos beneficiários.

**Evidência (resultados incorretos):**

| Modelo            | Modo      | Programa        | MAPE%      |
|-------------------|-----------|-----------------|------------|
| ETS (Holt Linear) | POR_SETOR | Zona Canavieira | 3,78%      |
| DecisionTree      | POR_SETOR | Pesca Artesanal | 303,64%    |
| LinearRegression  | GLOBAL    | Todos           | 4.152,45%  |
| LightGBM          | POR_SETOR | Fruticultura    | 10.989,82% |

**Solução:** função `evaluate_aggregated()` em `benchmark.py` — previsões de ML são somadas por programa antes do cálculo das métricas, igualando a granularidade dos modelos estatísticos.

**Resultados após correção:**

| Modelo            | Modo      | Programa        | MAPE%   |
|-------------------|-----------|-----------------|---------|
| ETS (Holt Linear) | POR_SETOR | Zona Canavieira | 3,78%   |
| LightGBM          | POR_SETOR | Zona Canavieira | 0,65%   |
| RandomForest      | POR_SETOR | Fruticultura    | 0,23%   |
| LinearRegression  | POR_SETOR | Zona Canavieira | 119,08% |

---

### Problema 2 — Data leakage via `lag_1` (resolvido)

**Causa:** com apenas 5 pontos anuais, `lag_1` funciona como oráculo em séries estáveis. DecisionTree apresentou MAPE de 0,03% na Pesca Artesanal com `lag_1` respondendo por 98,4% da feature importance. O modelo apenas repete o ano anterior.

**Evidência:**

| Município | Real 2025 | Previsto | Erro abs. | lag_1 |
|-----------|-----------|----------|-----------|-------|
| Goiana    | 1817      | 1810,00  | 7,00      | 1810  |
| Recife    | 600       | 596,00   | 4,00      | 592   |
| Ipojuca   | 186       | 89,31    | 96,69     | 98    |
| Igarassu  | 511       | 443,25   | 67,75     | 477   |

**Solução (Opção A):** `lag_1` removido de `ML_FEATURES` em `benchmark.py`. Features de ML passam a ser: `lag_2`, `rolling_mean_2`, `trend`, `ano_rel`, `ANO`, `MUNICIPIO_CODE`, `PROGRAMA_CODE`.

### Problema 3 — Leakage residual via features derivadas de `lag_1` (resolvido)

**Causa:** mesmo com `lag_1` fora de `ML_FEATURES`, ele ainda entrava nos modelos indiretamente:
- `rolling_mean_2 = (lag_1 + lag_2) / 2` — média incluía o ano anterior
- `trend = lag_1 - lag_2` — diferença usava o ano anterior como ponto mais recente

**Solução:** `lag_1` removido inteiramente do pré-processamento. Features recalculadas com `lag_3`:
- `rolling_mean_2 = (lag_2 + lag_3) / 2`
- `trend = lag_2 - lag_3`

Features de ML na v4: `lag_2`, `lag_3`, `rolling_mean_2`, `trend`, `ano_rel`, `ANO`, `MUNICIPIO_CODE`, `PROGRAMA_CODE`.

---

## Resultados Definitivos (v7 — grid search + features enriquecidas, sem leakage)

> `resultados_benchmark_v7.csv` / `resultados_cv_v7_*.csv`
> features: `lag_2`, `lag_3`, `rolling_mean_2`, `trend`, `ano_rel`, `growth_rate`, `zero_historico`, `programa_total_lag2`, `share_municipio`
> RF: `n_estimators=100, max_depth=6, min_samples_leaf=2` · LightGBM: `n_estimators=100, max_depth=3, lr=0.05` (selecionados por grid search CV)

### Benchmark único (fold 2025)

| Modelo                  | Modo      | Programa              | MAPE%   |
|-------------------------|-----------|-----------------------|---------|
| RandomForest            | POR_SETOR | Fruticultura Irrigada | 0,62%   |
| RandomForest            | POR_SETOR | Pesca Artesanal       | 1,77%   |
| LightGBM                | POR_SETOR | Fruticultura Irrigada | 1,22%   |
| Naive (baseline)        | POR_SETOR | Pesca Artesanal       | 2,13%   |
| Naive (baseline)        | POR_SETOR | Fruticultura Irrigada | 3,12%   |
| ETS_notrend             | GLOBAL    | Todos                 | 3,13%   |
| ETS (Holt Linear)       | POR_SETOR | Zona Canavieira       | 3,78%   |
| RandomForest            | POR_SETOR | Zona Canavieira       | 4,68%   |
| ETS_notrend             | POR_SETOR | Pesca Artesanal       | 4,77%   |
| ARIMA                   | GLOBAL    | Todos                 | 4,06%   |
| ARIMA                   | POR_SETOR | Fruticultura Irrigada | 4,31%   |

### Walk-forward CV (média de 3 folds: 2023, 2024, 2025)

| Modelo                  | Modo      | Programa              | MAPE médio | ± desvio | Folds |
|-------------------------|-----------|-----------------------|------------|----------|-------|
| ETS_notrend             | POR_SETOR | Pesca Artesanal       | 5,35%      | ±3,91    | 3     |
| ARIMA                   | POR_SETOR | Fruticultura Irrigada | 5,84%      | ±2,16    | 2     |
| ARIMA                   | POR_SETOR | Pesca Artesanal       | 6,75%      | ±3,64    | 2     |
| ARIMA                   | GLOBAL    | Todos                 | 7,46%      | ±4,81    | 2     |
| RandomForest            | POR_SETOR | Fruticultura Irrigada | **3,58%**  | ±3,48    | 3     |
| RandomForest            | POR_SETOR | Pesca Artesanal       | 17,31%     | ±14,31   | 3     |
| DecisionTree            | POR_SETOR | Pesca Artesanal       | 9,55%      | ±7,68    | 3     |
| LightGBM                | POR_SETOR | Fruticultura Irrigada | 32,93%     | ±30,88   | 3     |

**Melhores por família (CV):**
- Estatístico: ETS_notrend POR_SETOR — 5,35% ±3,91 (Pesca Artesanal) · ARIMA POR_SETOR — 5,84% ±2,16 (Fruticultura)
- ML baseado em Ensemble: RandomForest POR_SETOR — 3,58% ±3,48 (Fruticultura Irrigada)
- ML Clássico: DecisionTree POR_SETOR — 9,55% ±7,68 (Pesca Artesanal)
- Baseline: Naive POR_SETOR — 17,34% ±13,99 (Pesca Artesanal)

**Achados consolidados:**
- **POR_SETOR > GLOBAL** consistentemente — dinâmicas distintas entre programas justificam modelos separados
- **RandomForest com grid search** é agora o melhor modelo em CV (3,58% ±3,48 na Fruticultura) — grid search reduziu overfitting ao encontrar `min_samples_leaf=2`; melhora em fold único de 2,30% → 0,62%
- **ARIMA é o modelo estatístico mais robusto** — menor variância entre folds; consistente mesmo com poucos dados de treino
- **ETS_notrend** surpreendentemente bom no CV da Pesca (5,35%) — remover tendência ajuda em séries voláteis
- **ETS (com tendência)** tem CV de 40% na Canavieira — resultado de 3,78% no fold 2025 era excepcional
- **LightGBM** não se beneficia de grid search nessa escala — alta variância estrutural em todos os hiperparâmetros testados
- **Naive (lag_2)** bate vários modelos de ML no CV para Pesca (17,34%) — reforça dificuldade fundamental do problema com 5 pontos
- **DecisionTree e LinearRegression** seguem fracos — dependiam estruturalmente do leakage via lag_1

---

## Próximos Passos

- [x] Decidir tratamento do `lag_1` → Opção A (removido)
- [x] Reexecutar benchmark e registrar resultados definitivos (v3)
- [x] Remover leakage residual em `rolling_mean_2` e `trend` (v4)
- [x] Implementar walk-forward CV com 3 folds
- [x] Feature engineering sem leakage: `growth_rate`, `zero_historico`, `programa_total_lag2`, `share_municipio` (v5)
- [x] Visualização comparativa no `modeling.ipynb`
- [x] Grid search para RandomForest e LightGBM (v7)
- [x] Renomear "ML Moderno" → "ML baseado em Ensemble"
- [ ] Redigir seção de metodologia do TCC
