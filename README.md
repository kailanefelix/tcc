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
| ML Moderno    | Random Forest, LightGBM             | Município × programa          |

**Modos:** `GLOBAL` (todos os municípios/programas juntos) · `POR_SETOR` (um modelo por PROGRAMA)

**Validação:** walk-forward temporal — treino 2021–2024, teste 2025, sem embaralhamento.

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

---

## Resultados Definitivos (v3 — sem `lag_1`)

> `resultados_benchmark_v3.csv` · features: `lag_2`, `rolling_mean_2`, `trend`, `ano_rel`

| Modelo            | Modo      | Programa              | MAPE%  |
|-------------------|-----------|-----------------------|--------|
| LightGBM          | POR_SETOR | Fruticultura Irrigada | 2,40%  |
| ETS (Holt Linear) | POR_SETOR | Zona Canavieira       | 3,78%  |
| ARIMA             | GLOBAL    | Todos                 | 4,06%  |
| ARIMA             | POR_SETOR | Fruticultura Irrigada | 4,31%  |
| LightGBM          | POR_SETOR | Zona Canavieira       | 4,27%  |
| ETS (Holt Linear) | POR_SETOR | Pesca Artesanal       | 9,14%  |
| DecisionTree      | POR_SETOR | Pesca Artesanal       | 9,25%  |
| ARIMA             | POR_SETOR | Pesca Artesanal       | 9,32%  |
| LinearRegression  | POR_SETOR | Fruticultura Irrigada | 10,12% |
| LightGBM          | GLOBAL    | Todos                 | 11,34% |

**Melhores por família:**
- Estatístico: ETS POR_SETOR — 3,78% (Zona Canavieira)
- ML Moderno: LightGBM POR_SETOR — 2,40% (Fruticultura Irrigada)
- ML Clássico: DecisionTree POR_SETOR — 9,25% (Pesca Artesanal)

**Achados consolidados:**
- **POR_SETOR > GLOBAL** consistentemente — dinâmicas distintas entre programas justificam modelos separados
- **LightGBM POR_SETOR** é competitivo ou superior aos modelos estatísticos sem leakage
- **DecisionTree** sem `lag_1` desempenho cai para 9–31% — dependia quase integralmente do leakage
- **Regressão Linear** segue fraca (41–105% POR_SETOR), insuficiência de pontos para estimar tendência linear
- **ETS global** continua o pior resultado (36,22%) — agregação perde informação setorial

---

## Próximos Passos

- [x] Decidir tratamento do `lag_1` → Opção A (removido)
- [x] Reexecutar benchmark e registrar resultados definitivos (v3)
- [ ] Visualização comparativa no `modeling.ipynb`
- [ ] Redigir seção de metodologia do TCC
