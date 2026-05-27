"""
selection.py — Seleção estática de modelos para séries temporais curtas
Programa Chapéu de Palha / Pernambuco — TCC

Lê os CSVs produzidos pelo benchmark.py e aplica três critérios de
seleção sem retreinar modelos nem reimportar dados brutos.

Uso:
    python selection.py [folds_csv] [summary_csv]

Defaults (quando executado do diretório outputs/):
    python selection.py resultados_benchmark_v7_folds.csv \
                        resultados_benchmark_v7_summary.csv

Importável:
    from selection import run_static_selection
    df = run_static_selection("folds.csv", "summary.csv")
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FOLD_EVAL = 2025

# Pesos por fold para WeightedMAPE (recência crescente)
FOLD_WEIGHTS = {2023: 1 / 3, 2024: 2 / 3}

# Modelos/famílias excluídos da elegibilidade
EXCLUIR_FAMILIAS = {"Ensemble"}
EXCLUIR_MODELOS  = {"Naive"}

# Fator λ em StableMAPE: score = mean + λ × std
PENALIDADE_STD = 0.5


# ---------------------------------------------------------------------------
# Carregamento e limpeza
# ---------------------------------------------------------------------------
def _carregar(folds_path: str, summary_path: str):
    df_folds   = pd.read_csv(folds_path)
    df_summary = pd.read_csv(summary_path)

    # "MAPE%" → "mape" para evitar o símbolo especial em todo o código
    df_folds = df_folds.rename(columns={"MAPE%": "mape"})
    df_folds["mape"] = pd.to_numeric(df_folds["mape"], errors="coerce")

    return df_folds, df_summary


def _elegivel(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Ensemble, Naive, linhas sem MAPE válido e modelos com < 2 folds históricos."""
    df = df[
        ~df["familia"].isin(EXCLUIR_FAMILIAS) &
        ~df["modelo"].isin(EXCLUIR_MODELOS)  &
        df["mape"].notna()
    ].copy()
    # Exige mínimo de 2 folds históricos para elegibilidade.
    # ARIMA falha no fold 2023 (treino com apenas 2 pontos — abaixo
    # do mínimo do ARIMA(1,1,0)), acumulando só 1 fold válido.
    # Com menos de 2 folds não é possível calcular estabilidade
    # nem garantir comparação justa entre critérios.
    df["n_folds"] = df.groupby(["modelo", "programa", "modo"])["fold"].transform("nunique")
    df = df[df["n_folds"] >= 2]
    return df


# ---------------------------------------------------------------------------
# Critérios de seleção
# ---------------------------------------------------------------------------
def _mean_mape(grp: pd.DataFrame):
    """
    Critério 1 — MeanMAPE:
    score = MAPE médio nos folds históricos; elege o menor score.
    """
    scores = grp.groupby("modelo")["mape"].mean()
    m = scores.idxmin()
    return m, float(scores[m])


def _weighted_mape(grp: pd.DataFrame):
    """
    Critério 2 — WeightedMAPE:
    Pondera os folds por recência (2023 = 1/3, 2024 = 2/3).
    Se um fold estiver ausente para um modelo, exclui apenas esse fold
    do cálculo, mas mantém o modelo elegível se tiver ≥ 1 fold válido.
    """
    scores = {}
    for modelo, sub in grp.groupby("modelo"):
        num = den = 0.0
        for fold, w in FOLD_WEIGHTS.items():
            row = sub[sub["fold"] == fold]
            if not row.empty and not pd.isna(row["mape"].iloc[0]):
                num += w * row["mape"].iloc[0]
                den += w
        if den > 0:
            scores[modelo] = num / den   # normaliza pelos pesos presentes
    if not scores:
        return None, np.nan
    m = min(scores, key=scores.__getitem__)
    return m, scores[m]


def _stable_mape(grp: pd.DataFrame):
    """
    Critério 3 — StableMAPE:
    Penaliza variância: score = mean + 0.5 × std.
    Se std=NaN (apenas 1 fold disponível), score = mean sem penalidade
    — evitar punir modelo por falta de dados, não por instabilidade real.
    """
    stats = grp.groupby("modelo")["mape"].agg(["mean", "std"])
    # std NaN → 0 antes de multiplicar (sem penalidade)
    stats["score"] = stats["mean"] + PENALIDADE_STD * stats["std"].fillna(0)
    m = stats["score"].idxmin()
    return m, float(stats.loc[m, "score"])


CRITERIOS = {
    "MeanMAPE":     _mean_mape,
    "WeightedMAPE": _weighted_mape,
    "StableMAPE":   _stable_mape,
}


# ---------------------------------------------------------------------------
# Consultas no fold de avaliação (2025)
# ---------------------------------------------------------------------------
def _mape_no_teste(df_test: pd.DataFrame, modelo: str, programa: str, modo: str) -> float:
    """MAPE do modelo eleito no fold 2025 para o (programa, modo) dado."""
    rows = df_test[
        (df_test["modelo"]  == modelo)  &
        (df_test["programa"] == programa) &
        (df_test["modo"]     == modo)    &
        df_test["mape"].notna()
    ]
    return float(rows["mape"].iloc[0]) if not rows.empty else np.nan


def _mape_oracle(df_test: pd.DataFrame, programa: str, modo: str) -> float:
    """
    Menor MAPE entre todos os modelos individuais (não-Ensemble, não-Naive)
    no fold 2025 — representa a escolha ótima com informação perfeita.
    Não aplica o filtro n_folds: o oracle mede o que era atingível em 2025
    independente de quantos folds históricos o modelo possuía.
    """
    candidatos = df_test[
        (df_test["programa"] == programa) &
        (df_test["modo"]     == modo)     &
        ~df_test["familia"].isin(EXCLUIR_FAMILIAS) &
        ~df_test["modelo"].isin(EXCLUIR_MODELOS)   &
        df_test["mape"].notna()
    ]
    return float(candidatos["mape"].min()) if not candidatos.empty else np.nan


# ---------------------------------------------------------------------------
# Estatísticas históricas do modelo eleito (para referência no output)
# ---------------------------------------------------------------------------
def _hist_stats(grp: pd.DataFrame, modelo: str):
    """Retorna (mape_mean, mape_std, n_folds) do modelo no histórico."""
    vals = grp[grp["modelo"] == modelo]["mape"].dropna()
    mean  = float(vals.mean())
    std   = float(vals.std()) if len(vals) > 1 else np.nan
    return mean, std, int(len(vals))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _r(v):
    """Arredonda para 4 casas decimais; preserva NaN e None."""
    return round(float(v), 4) if not pd.isna(v) else np.nan


def _fmt(v, dec=2):
    """Formata número para exibição; retorna 'N/A' se ausente."""
    return f"{v:.{dec}f}" if not pd.isna(v) else "N/A"


def _linha_na(criterio, programa, modo, oracle=np.nan):
    """Linha de fallback quando não há histórico válido."""
    return {
        "criterio": criterio, "programa": programa, "modo": modo,
        "modelo_eleito": "N/A", "score_historico": np.nan,
        "mape_hist_mean": np.nan, "mape_hist_std": np.nan,
        "n_folds_hist": 0,
        "mape_realizado": np.nan,
        "mape_oracle": _r(oracle),
        "regret": np.nan,
    }


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------
def run_static_selection(folds_path: str, summary_path: str) -> pd.DataFrame:
    """
    Aplica os três critérios de seleção estática para cada (programa, modo).

    Expanding window estrita: fold 2025 nunca entra na decisão.
    Salva resultados_selecao_v2.csv no mesmo diretório dos CSVs de entrada.

    Parâmetros
    ----------
    folds_path   : caminho para resultados_benchmark_*_folds.csv
    summary_path : caminho para resultados_benchmark_*_summary.csv

    Retorno
    -------
    DataFrame com uma linha por (criterio, programa, modo).
    """
    df_folds, df_summary = _carregar(folds_path, summary_path)

    # Partição: histórico (decisão) vs fold de avaliação (resultado real)
    df_hist = _elegivel(df_folds[df_folds["fold"] < FOLD_EVAL])
    df_test = df_folds[df_folds["fold"] == FOLD_EVAL].copy()

    registros = []

    # Itera sobre todas as combinações (programa, modo) com histórico
    combos = (
        df_hist[["programa", "modo"]]
        .drop_duplicates()
        .sort_values(["modo", "programa"])
        .values.tolist()
    )

    for programa, modo in combos:
        grp = df_hist[
            (df_hist["programa"] == programa) &
            (df_hist["modo"]     == modo)
        ]

        if grp.empty:
            # Sem histórico elegível para este (programa, modo)
            for criterio in CRITERIOS:
                registros.append(_linha_na(criterio, programa, modo))
            continue

        oracle = _mape_oracle(df_test, programa, modo)

        for criterio, fn in CRITERIOS.items():
            modelo, score = fn(grp)

            if modelo is None:
                # Critério não conseguiu eleger nenhum modelo
                registros.append(_linha_na(criterio, programa, modo, oracle=oracle))
                continue

            mape_real          = _mape_no_teste(df_test, modelo, programa, modo)
            hist_mean, hist_std, n_folds = _hist_stats(grp, modelo)
            regret             = mape_real - oracle if not (pd.isna(mape_real) or pd.isna(oracle)) else np.nan

            registros.append({
                "criterio":       criterio,
                "programa":       programa,
                "modo":           modo,
                "modelo_eleito":  modelo,
                "score_historico": _r(score),
                "mape_hist_mean": _r(hist_mean),
                "mape_hist_std":  _r(hist_std),
                "n_folds_hist":   n_folds,
                "mape_realizado": _r(mape_real),
                "mape_oracle":    _r(oracle),
                "regret":         _r(regret),
            })

    df_sel = pd.DataFrame(registros)

    # Grava CSV no mesmo diretório dos inputs
    out_dir  = os.path.dirname(os.path.abspath(folds_path))
    out_path = os.path.join(out_dir, "resultados_selecao_v2.csv")
    df_sel.to_csv(out_path, index=False)
    print(f"\nSalvo: {out_path}")

    _relatorio(df_sel)
    return df_sel


# ---------------------------------------------------------------------------
# Relatório no terminal
# ---------------------------------------------------------------------------
def _relatorio(df: pd.DataFrame) -> None:
    SEP = "=" * 108

    print("\n" + SEP)
    print("  SELEÇÃO ESTÁTICA DE MODELOS — Programa Chapéu de Palha / Pernambuco")
    print(SEP)

    # ── [1] Tabela principal ─────────────────────────────────────────────────
    print("\n[1] Modelos eleitos por critério, programa e modo\n")
    hdr = (
        f"  {'Critério':<14} {'Programa':<25} {'Modo':<11}"
        f" {'Modelo eleito':<22} {'Hist%':>6} {'Real%':>6} {'Oracle%':>8} {'Regret':>7}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    prev_criterio = None
    for _, row in df.iterrows():
        if row["criterio"] != prev_criterio and prev_criterio is not None:
            print()
        prev_criterio = row["criterio"]
        print(
            f"  {row['criterio']:<14} {row['programa']:<25} {row['modo']:<11}"
            f" {row['modelo_eleito']:<22}"
            f" {_fmt(row['mape_hist_mean']):>6}"
            f" {_fmt(row['mape_realizado']):>6}"
            f" {_fmt(row['mape_oracle']):>8}"
            f" {_fmt(row['regret']):>7}"
        )

    # ── [2] Tabela comparativa realizado × oracle × regret ───────────────────
    print("\n\n[2] Comparativo por programa/modo: realizado vs oracle vs regret\n")
    hdr2 = (
        f"  {'Programa':<25} {'Modo':<11} {'Critério':<14}"
        f" {'Real%':>6} {'Oracle%':>8} {'Regret':>7}"
    )
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    pivot = df.sort_values(["programa", "modo", "criterio"])
    prev_pm = None
    for _, row in pivot.iterrows():
        pm = (row["programa"], row["modo"])
        if pm != prev_pm and prev_pm is not None:
            print()
        prev_pm = pm
        print(
            f"  {row['programa']:<25} {row['modo']:<11} {row['criterio']:<14}"
            f" {_fmt(row['mape_realizado']):>6}"
            f" {_fmt(row['mape_oracle']):>8}"
            f" {_fmt(row['regret']):>7}"
        )

    # ── [3] Resumo: regret médio por critério ────────────────────────────────
    print("\n\n[3] Resumo — regret médio por critério (perda vs oracle de informação perfeita)\n")
    resumo = (
        df.dropna(subset=["regret"])
          .groupby("criterio")["regret"]
          .mean()
          .sort_values()
    )
    melhor = resumo.idxmin()
    for criterio, val in resumo.items():
        tag = "  << MELHOR" if criterio == melhor else ""
        print(f"  {criterio:<14}: {val:7.4f}%{tag}")

    print("\n" + SEP + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    folds_path   = sys.argv[1] if len(sys.argv) > 1 else "resultados_benchmark_v7_folds.csv"
    summary_path = sys.argv[2] if len(sys.argv) > 2 else "resultados_benchmark_v7_summary.csv"
    run_static_selection(folds_path, summary_path)
