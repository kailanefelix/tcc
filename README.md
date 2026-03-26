# TCC — Benchmark de Modelos Preditivos
**Programa Cientista do Meu Estado · Pernambuco**

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

### Problema 2 — Data leakage via `lag_1` (em avaliação)

**Causa:** com apenas 5 pontos anuais, `lag_1` funciona como oráculo em séries estáveis. DecisionTree apresentou MAPE de 0,03% na Pesca Artesanal com `lag_1` respondendo por 98,4% da feature importance — o modelo apenas repete o ano anterior.

**Evidência:**

| Município | Real 2025 | Previsto | Erro abs. | lag_1 |
|-----------|-----------|----------|-----------|-------|
| Goiana    | 1817      | 1810,00  | 7,00      | 1810  |
| Recife    | 600       | 596,00   | 4,00      | 592   |
| Ipojuca   | 186       | 89,31    | 96,69     | 98    |
| Igarassu  | 511       | 443,25   | 67,75     | 477   |

**Opções em avaliação:**
- **A (preferida):** remover `lag_1`, manter `lag_2`, `rolling_mean_2`, `trend`, `ano_rel`
- **B:** manter `lag_1` e documentar como limitação explícita (ameaça à validade)
- **C:** leave-one-out temporal (prever 2023 com dados até 2022, prever 2024 com dados até 2023)

---

## Achados Preliminares

> Sujeitos à revisão do Problema 2.

- **POR_SETOR > GLOBAL** consistentemente — os 3 programas têm dinâmicas distintas
- **ETS** é o estatístico mais robusto: 3,78% (Canavieira), 9,14% (Pesca); ETS global foi o pior geral (36,22%)
- **LightGBM e Random Forest** competitivos com estatísticos quando avaliados corretamente
- **Regressão Linear** fraca em vários cenários (119% Canavieira, 42% Pesca) — poucos pontos para estimar tendência

---

## Próximos Passos

- [ ] Decidir tratamento do `lag_1` (Opções A, B ou C)
- [ ] Ajustar `preprocessing.py` e `benchmark.py`
- [ ] Reexecutar benchmark e registrar resultados definitivos
- [ ] Visualização comparativa no `modeling.ipynb`
- [ ] Redigir seção de metodologia do TCC
