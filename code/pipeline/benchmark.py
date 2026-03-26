"""
benchmark.py
Pipeline de benchmark de modelos para o TCC.

Estrutura:
  - Família 1 – Estatísticos      : ARIMA, ETS
  - Família 2 – ML Clássico       : Regressão Linear, Árvore de Decisão
  - Família 3 – ML Moderno        : Random Forest, LightGBM

Cada modelo é avaliado em dois modos de treinamento:
  a) GLOBAL    – treinado com todos os dados de todos os municípios/programas
  b) POR_SETOR – treinado separadamente para cada PROGRAMA

Métrica principal: MAE, RMSE, MAPE (calculados sobre o ano de teste = último ano).
O ano de teste é sempre o último ano disponível; o treino usa todos os anteriores.

Uso:
    python benchmark.py --file cientista_cp.xlsx
    python benchmark.py --file cientista_cp.xlsx --test-year 2025
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from preprocessing import load_and_preprocess, TARGET, YEAR_COL

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ML_FEATURES = ["ANO", "MUNICIPIO_CODE", "PROGRAMA_CODE",
                "lag_2", "rolling_mean_2", "trend", "ano_rel"]

FAMILIES = {
    "Estatístico": ["ARIMA", "ETS"],
    "ML Clássico":  ["LinearRegression", "DecisionTree"],
    "ML Moderno":   ["RandomForest", "LightGBM"],
}


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas no nível linha (usado internamente e pelos modelos estatísticos)."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = _mape(y_true, y_pred)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE%": round(mape, 2)}


def evaluate_aggregated(y_true: np.ndarray, y_pred: np.ndarray,
                         test_df: pd.DataFrame, programa: str) -> dict:
    """
    Métricas calculadas sobre o TOTAL agregado por programa.
    Isso torna os modelos de ML comparáveis com os estatísticos,
    que também operam sobre séries agregadas.

    Para modo GLOBAL, agrega por programa e calcula MAE/RMSE/MAPE
    sobre os totais de cada programa (um ponto por programa).
    Para modo POR_SETOR, há apenas um programa — um único ponto de comparação.
    """
    df_agg = test_df.copy()
    df_agg["_pred"] = y_pred
    df_agg["_true"] = y_true

    if programa == "TODOS":
        # Agrega por programa → um valor por programa
        agg = df_agg.groupby("PROGRAMA").agg(
            total_true=("_true", "sum"),
            total_pred=("_pred", "sum")
        ).reset_index()
    else:
        # Já está filtrado por programa → soma tudo num único valor
        agg = pd.DataFrame({
            "total_true": [df_agg["_true"].sum()],
            "total_pred": [df_agg["_pred"].sum()],
        })

    y_t = agg["total_true"].values.astype(float)
    y_p = agg["total_pred"].values.astype(float)

    mae  = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mape = _mape(y_t, y_p)
    return {
        "MAE":   round(mae, 2),
        "RMSE":  round(rmse, 2),
        "MAPE%": round(mape, 2),
        "y_true_total": round(float(y_t.sum()), 0),
        "y_pred_total": round(float(y_p.sum()), 0),
    }


# ---------------------------------------------------------------------------
# Modelos estatísticos (operam em séries agregadas por programa)
# ---------------------------------------------------------------------------

def _fit_stat_model(series: pd.Series, model_name: str) -> float:
    """
    Treina modelo estatístico na série `series` (todos os pontos menos o último)
    e retorna a previsão para o próximo passo.
    """
    if len(series) < 3:
        return np.nan

    train = series.iloc[:-1]

    if model_name == "ARIMA":
        try:
            m = ARIMA(train, order=(1, 1, 0)).fit()
            return float(m.forecast(1).iloc[0])
        except Exception:
            return np.nan

    elif model_name == "ETS":
        try:
            m = ExponentialSmoothing(
                train, trend="add", seasonal=None,
                initialization_method="estimated"
            ).fit()
            return float(m.forecast(1).iloc[0])
        except Exception:
            return np.nan

    return np.nan


def run_stat_models(df_full: pd.DataFrame, test_year: int,
                    scope: str = "GLOBAL", programa: str | None = None) -> dict:
    """
    Para modelos estatísticos, agrega por programa e avalia a previsão
    do `test_year` usando o histórico anterior como treino.

    scope: "GLOBAL" (todos programas juntos) ou "POR_SETOR" (por programa separado)
    """
    results = {}

    if scope == "GLOBAL":
        # Série agregada de todos os programas juntos
        series = (df_full.groupby(YEAR_COL)[TARGET].sum()
                  .sort_index()
                  .rename("beneficiarios"))
        series.index = pd.to_datetime(series.index, format="%Y")
        series = series.asfreq("YS")
        real = series.iloc[-1]

        for name in FAMILIES["Estatístico"]:
            pred = _fit_stat_model(series, name)
            err  = evaluate(np.array([real]), np.array([pred]))
            results[name] = {**err, "y_true": real, "y_pred": pred,
                             "scope": "GLOBAL", "programa": "TODOS"}

    elif scope == "POR_SETOR":
        prog = programa or "DESCONHECIDO"
        series = (df_full[df_full["PROGRAMA"] == prog]
                  .groupby(YEAR_COL)[TARGET].sum()
                  .sort_index())
        series.index = pd.to_datetime(series.index, format="%Y")
        series = series.asfreq("YS")
        real = series.iloc[-1]

        for name in FAMILIES["Estatístico"]:
            pred = _fit_stat_model(series, name)
            err  = evaluate(np.array([real]), np.array([pred]))
            results[name] = {**err, "y_true": real, "y_pred": pred,
                             "scope": "POR_SETOR", "programa": prog}

    return results


# ---------------------------------------------------------------------------
# Modelos de ML (operam no nível de município × programa)
# ---------------------------------------------------------------------------

def _get_ml_splits(df_ml: pd.DataFrame, test_year: int):
    """Separa treino/teste garantindo que lag_1 existe no split de teste."""
    train = df_ml[df_ml[YEAR_COL] < test_year]
    test  = df_ml[df_ml[YEAR_COL] == test_year]

    # Remove linhas de teste onde lag_1 = 0 e real = 0 (série vazia)
    test = test[~((test["lag_1"] == 0) & (test[TARGET] == 0))]

    feats = [f for f in ML_FEATURES if f in df_ml.columns]
    X_train, y_train = train[feats], train[TARGET]
    X_test,  y_test  = test[feats],  test[TARGET]
    return X_train, y_train, X_test, y_test


def _get_ml_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "DecisionTree":     DecisionTreeRegressor(max_depth=4, random_state=42),
        "RandomForest":     RandomForestRegressor(n_estimators=200, max_depth=6,
                                                  random_state=42, n_jobs=-1),
        "LightGBM":         lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                               max_depth=4, random_state=42,
                                               verbose=-1),
    }


def run_ml_models(df_ml: pd.DataFrame, test_year: int,
                  scope: str = "GLOBAL", programa: str | None = None) -> dict:
    """
    Treina e avalia modelos de ML.

    scope = "GLOBAL"    → treina em todos os municípios/programas
    scope = "POR_SETOR" → treina só nos dados do programa especificado

    Métricas calculadas sobre totais AGREGADOS por programa,
    tornando os resultados comparáveis com os modelos estatísticos.
    """
    prog_label = programa or "TODOS"

    if scope == "GLOBAL":
        subset = df_ml
    else:
        subset = df_ml[df_ml["PROGRAMA"] == programa]

    X_train, y_train, X_test, y_test = _get_ml_splits(subset, test_year)

    if len(X_train) == 0 or len(X_test) == 0:
        return {}

    # Guarda o df de teste completo para agregar depois
    test_df = subset[subset[YEAR_COL] == test_year].copy()
    test_df = test_df[~((test_df["lag_1"] == 0) & (test_df[TARGET] == 0))]

    results = {}
    ml_model_names = FAMILIES["ML Clássico"] + FAMILIES["ML Moderno"]

    for name, model in _get_ml_models().items():
        if name not in ml_model_names:
            continue
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Métricas agregadas (mesma granularidade dos estatísticos)
            err = evaluate_aggregated(y_test.values, preds, test_df, prog_label)

            results[name] = {
                **err,
                "scope":    scope,
                "programa": prog_label,
            }
        except Exception as e:
            results[name] = {"erro": str(e), "scope": scope,
                             "programa": prog_label}

    return results


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_benchmark(file_path: str, test_year: int | None = None) -> pd.DataFrame:
    """
    Executa o pipeline completo de benchmark.

    Retorna um DataFrame com uma linha por (modelo, modo, programa).
    """
    data = load_and_preprocess(file_path)
    df_full     = data["df_full"]
    df_ml       = data["df_ml"]
    programas   = data["programas"]
    anos        = data["anos"]

    if test_year is None:
        test_year = max(anos)

    print(f"\n[benchmark] Ano de teste: {test_year}")
    print(f"[benchmark] Programas   : {programas}")
    print(f"[benchmark] Anos treino : {[a for a in anos if a < test_year]}\n")

    all_results = []

    # ---- MODO GLOBAL --------------------------------------------------------
    print("=" * 60)
    print("MODO GLOBAL (todos os dados)")
    print("=" * 60)

    stat_global = run_stat_models(df_full, test_year, scope="GLOBAL")
    for model_name, metrics in stat_global.items():
        row = {"modelo": model_name, "familia": "Estatístico",
               "modo": "GLOBAL", **metrics}
        all_results.append(row)
        print(f"  {model_name:20s} MAE={metrics['MAE']:>10.2f}  "
              f"RMSE={metrics['RMSE']:>10.2f}  MAPE%={metrics['MAPE%']:>6.2f}")

    ml_global = run_ml_models(df_ml, test_year, scope="GLOBAL")
    for model_name, metrics in ml_global.items():
        familia = ("ML Clássico" if model_name in FAMILIES["ML Clássico"]
                   else "ML Moderno")
        row = {"modelo": model_name, "familia": familia,
               "modo": "GLOBAL", **metrics}
        all_results.append(row)
        print(f"  {model_name:20s} MAE={metrics.get('MAE','N/A'):>10}  "
              f"RMSE={metrics.get('RMSE','N/A'):>10}  "
              f"MAPE%={metrics.get('MAPE%','N/A'):>6}")

    # ---- MODO POR SETOR -----------------------------------------------------
    print("\n" + "=" * 60)
    print("MODO POR SETOR")
    print("=" * 60)

    for prog in programas:
        print(f"\n  >> {prog}")

        stat_prog = run_stat_models(df_full, test_year,
                                    scope="POR_SETOR", programa=prog)
        for model_name, metrics in stat_prog.items():
            row = {"modelo": model_name, "familia": "Estatístico",
                   "modo": "POR_SETOR", **metrics}
            all_results.append(row)
            print(f"     {model_name:20s} MAE={metrics['MAE']:>10.2f}  "
                  f"RMSE={metrics['RMSE']:>10.2f}  MAPE%={metrics['MAPE%']:>6.2f}")

        ml_prog = run_ml_models(df_ml, test_year,
                                scope="POR_SETOR", programa=prog)
        for model_name, metrics in ml_prog.items():
            familia = ("ML Clássico" if model_name in FAMILIES["ML Clássico"]
                       else "ML Moderno")
            row = {"modelo": model_name, "familia": familia,
                   "modo": "POR_SETOR", **metrics}
            all_results.append(row)
            print(f"     {model_name:20s} MAE={metrics.get('MAE','N/A'):>10}  "
                  f"RMSE={metrics.get('RMSE','N/A'):>10}  "
                  f"MAPE%={metrics.get('MAPE%','N/A'):>6}")

    # ---- Consolidação -------------------------------------------------------
    df_results = pd.DataFrame(all_results)
    return df_results


# ---------------------------------------------------------------------------
# Relatório / Tabela resumo
# ---------------------------------------------------------------------------

def print_summary(df_results: pd.DataFrame):
    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)

    cols = ["familia", "modelo", "modo", "programa", "MAE", "RMSE", "MAPE%"]
    cols = [c for c in cols if c in df_results.columns]
    summary = df_results[cols].sort_values(["familia", "MAPE%"])
    print(summary.to_string(index=False))

    print("\n--- Melhor modelo por família (MAPE%) ---")
    for familia in df_results["familia"].unique():
        sub = df_results[df_results["familia"] == familia].copy()
        sub = sub.dropna(subset=["MAPE%"])
        if sub.empty:
            continue
        best = sub.loc[sub["MAPE%"].idxmin()]
        print(f"  {familia:15s}: {best['modelo']} "
              f"(modo={best['modo']}, MAPE%={best['MAPE%']:.2f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de modelos – TCC")
    parser.add_argument("--file",      type=str, required=True,
                        help="Caminho para o arquivo .xlsx")
    parser.add_argument("--test-year", type=int, default=None,
                        help="Ano de teste (padrão: último ano do dataset)")
    parser.add_argument("--output",    type=str, default="resultados_benchmark.csv",
                        help="Arquivo CSV de saída com os resultados")
    args = parser.parse_args()

    df_results = run_benchmark(args.file, test_year=args.test_year)
    print_summary(df_results)

    df_results.to_csv(args.output, index=False)
    print(f"\n[benchmark] Resultados salvos em: {args.output}")