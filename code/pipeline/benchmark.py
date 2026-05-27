"""
benchmark.py
Pipeline de benchmark de modelos para o TCC.

Estrutura:
  - Família 1 – Estatísticos      : ARIMA, ETS
  - Família 2 – ML Clássico       : Regressão Linear, Árvore de Decisão
  - Família 3 – ML baseado em Ensemble : Random Forest, LightGBM

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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from preprocessing import load_and_preprocess, _add_context_features, TARGET, YEAR_COL

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ML_FEATURES = ["ANO", "MUNICIPIO_CODE", "PROGRAMA_CODE",
               "lag_2", "lag_3", "rolling_mean_2", "trend", "ano_rel",
               "growth_rate", "zero_historico",
               "programa_total_lag2", "share_municipio"]

FAMILIES = {
    "Estatístico": ["ARIMA", "ETS", "ETS_notrend"],  # ETS_notrend: testa sem trend em séries voláteis
    "ML Clássico":  ["LinearRegression", "DecisionTree"],
    "ML baseado em Ensemble": ["RandomForest", "LightGBM"],
    "Baseline":     ["Naive"],  # baseline: repete lag_2 — referência mínima de aprendizado
    "Ensemble":     ["SimpleAverage", "WeightedAverage", "StackingMeta"],
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

    elif model_name == "ETS_notrend":
        # ETS sem componente de tendência — avalia se trend=None reduz erro em séries voláteis
        try:
            m = ExponentialSmoothing(
                train, trend=None, seasonal=None,
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

    nan_metrics = {"MAE": np.nan, "RMSE": np.nan, "MAPE%": np.nan}

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
            if np.isnan(pred):
                err = nan_metrics
            else:
                err = evaluate(np.array([real]), np.array([pred]))
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
            if np.isnan(pred):
                err = nan_metrics
            else:
                err = evaluate(np.array([real]), np.array([pred]))
            results[name] = {**err, "y_true": real, "y_pred": pred,
                             "scope": "POR_SETOR", "programa": prog}

    return results


# ---------------------------------------------------------------------------
# Modelos de ML (operam no nível de município × programa)
# ---------------------------------------------------------------------------

def _get_ml_splits(df_ml: pd.DataFrame, test_year: int):
    """Separa treino/teste. Remove linhas onde a série está vazia (lag_2 e target = 0)."""
    train = df_ml[df_ml[YEAR_COL] < test_year]
    test  = df_ml[df_ml[YEAR_COL] == test_year]

    # Remove linhas de teste onde lag_2 = 0 e real = 0 (série vazia)
    test = test[~((test["lag_2"] == 0) & (test[TARGET] == 0))]

    feats = [f for f in ML_FEATURES if f in df_ml.columns]
    X_train, y_train = train[feats], train[TARGET]
    X_test,  y_test  = test[feats],  test[TARGET]
    return X_train, y_train, X_test, y_test


def _get_ml_models() -> dict:
    """
    Instancia os modelos de ML. Usa _ML_BEST_PARAMS se disponível
    (preenchido por run_grid_search), caso contrário _ML_DEFAULT_PARAMS.
    """
    rf_params   = _ML_BEST_PARAMS.get("RandomForest",
                                       _ML_DEFAULT_PARAMS.get("RandomForest", {}))
    lgbm_params = _ML_BEST_PARAMS.get("LightGBM",
                                       _ML_DEFAULT_PARAMS.get("LightGBM", {}))
    return {
        "LinearRegression": LinearRegression(),
        "DecisionTree":     DecisionTreeRegressor(max_depth=4, random_state=42),
        "RandomForest":     RandomForestRegressor(random_state=42, n_jobs=-1, **rf_params),
        "LightGBM":         lgb.LGBMRegressor(random_state=42, verbose=-1, **lgbm_params),
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
    test_df = test_df[~((test_df["lag_2"] == 0) & (test_df[TARGET] == 0))]

    results = {}
    ml_model_names = FAMILIES["ML Clássico"] + FAMILIES["ML baseado em Ensemble"]

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
# Modelo baseline
# ---------------------------------------------------------------------------

def run_naive_model(df_ml: pd.DataFrame, test_year: int,
                    scope: str = "GLOBAL", programa: str | None = None) -> dict:
    """
    Baseline naïve: previsão = lag_2 (valor de t-2).
    Serve como referência mínima — modelos úteis devem superar esse baseline.
    Usa evaluate_aggregated para garantir comparabilidade com os demais modelos.
    """
    prog_label = programa or "TODOS"

    if scope == "GLOBAL":
        subset = df_ml
    else:
        subset = df_ml[df_ml["PROGRAMA"] == programa]

    # Mesmo filtro de linhas vazias aplicado pelos modelos de ML
    test_df = subset[subset[YEAR_COL] == test_year].copy()
    test_df = test_df[~((test_df["lag_2"] == 0) & (test_df[TARGET] == 0))]

    if test_df.empty:
        return {}

    y_true = test_df[TARGET].values
    y_pred = test_df["lag_2"].values  # previsão naïve: repete o valor de t-2

    err = evaluate_aggregated(y_true, y_pred, test_df, prog_label)
    return {"Naive": {**err, "scope": scope, "programa": prog_label}}


# ---------------------------------------------------------------------------
# Ensemble de modelos
# ---------------------------------------------------------------------------

def _collect_ml_predictions(df_ml: pd.DataFrame, test_year: int,
                             scope: str = "GLOBAL",
                             programa: str | None = None) -> dict:
    """
    Treina modelos de ML e devolve previsões no nível linha (município × programa).
    Retorna {model_name: {"y_pred": array, "y_true": array, "test_df": df}}.
    Usado exclusivamente pelo pipeline de ensemble — não altera run_ml_models().
    """
    if scope == "GLOBAL":
        subset = df_ml
    else:
        subset = df_ml[df_ml["PROGRAMA"] == programa]

    X_train, y_train, X_test, y_test = _get_ml_splits(subset, test_year)
    test_df = subset[subset[YEAR_COL] == test_year].copy()
    test_df = test_df[~((test_df["lag_2"] == 0) & (test_df[TARGET] == 0))]

    if len(X_train) == 0 or len(X_test) == 0:
        return {}

    ml_names = FAMILIES["ML Clássico"] + FAMILIES["ML baseado em Ensemble"]
    results = {}
    for name, model in _get_ml_models().items():
        if name not in ml_names:
            continue
        try:
            model.fit(X_train, y_train)
            results[name] = {
                "y_pred": model.predict(X_test),
                "y_true": y_test.values,
                "test_df": test_df,
            }
        except Exception:
            pass

    # Baseline Naive: previsão = lag_2
    if not test_df.empty:
        results["Naive"] = {
            "y_pred": test_df["lag_2"].values,
            "y_true": test_df[TARGET].values,
            "test_df": test_df,
        }

    return results


def _agg_predictions(y_pred: np.ndarray, y_true: np.ndarray,
                     test_df: pd.DataFrame, programa: str
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Agrega previsões ao nível de programa, replicando a lógica de evaluate_aggregated().
    Para GLOBAL (programa="TODOS"): um total por programa.
    Para POR_SETOR: um único total.
    """
    df = test_df.copy()
    df["_pred"] = y_pred
    df["_true"] = y_true

    if programa == "TODOS":
        agg = df.groupby("PROGRAMA").agg(
            total_true=("_true", "sum"),
            total_pred=("_pred", "sum"),
        ).reset_index()
    else:
        agg = pd.DataFrame({
            "total_true": [df["_true"].sum()],
            "total_pred": [df["_pred"].sum()],
        })

    return (agg["total_pred"].values.astype(float),
            agg["total_true"].values.astype(float))


def _eval_agg_array(y_pred_agg: np.ndarray, y_true_agg: np.ndarray) -> dict:
    """Calcula métricas sobre arrays já agregados (sem precisar de test_df)."""
    mae  = mean_absolute_error(y_true_agg, y_pred_agg)
    rmse = np.sqrt(mean_squared_error(y_true_agg, y_pred_agg))
    mape = _mape(y_true_agg, y_pred_agg)
    return {
        "MAE":   round(mae, 2),
        "RMSE":  round(rmse, 2),
        "MAPE%": round(mape, 2),
        "y_true_total": round(float(y_true_agg.sum()), 0),
        "y_pred_total": round(float(y_pred_agg.sum()), 0),
    }


def run_ensemble_models(
    fold_predictions: dict,
    cv_history: list,
    test_df: pd.DataFrame,
    programa: str,
) -> dict:
    """
    Avalia três estratégias de ensemble sobre previsões dos modelos individuais
    de ML para o fold corrente.

    Parâmetros
    ----------
    fold_predictions : {modelo: {"y_pred": array, "y_true": array, "test_df": df}}
        Previsões no nível linha do fold corrente (apenas modelos de ML + Naive).
    cv_history : list of dict
        Expanding window: um elemento por fold ANTERIOR ao corrente.
        Cada elemento: {modelo: {"mape": float, "y_pred_agg": array, "y_true_agg": array}}.
        Nunca inclui dados do fold corrente — garante leakage-free.
    test_df : pd.DataFrame
        Não usado diretamente (test_df já embutido em fold_predictions); mantido
        para compatibilidade com a assinatura especificada.
    programa : str
        "TODOS" para modo GLOBAL, nome do programa para POR_SETOR.

    Retorna
    -------
    {estratégia: métricas} no mesmo formato de evaluate_aggregated().
    """
    available_models = [m for m in fold_predictions if fold_predictions[m]]
    if not available_models:
        return {}

    # Agrega previsões de cada modelo ao nível de programa (igual a evaluate_aggregated)
    agg_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for m in available_models:
        fp = fold_predictions[m]
        y_pred_agg, y_true_agg = _agg_predictions(
            fp["y_pred"], fp["y_true"], fp["test_df"], programa
        )
        agg_preds[m] = (y_pred_agg, y_true_agg)

    y_true_ref = list(agg_preds.values())[0][1]
    results = {}

    # -------------------------------------------------------------------
    # Estratégia 1 — Média Simples
    # Janela: usa apenas previsões do fold corrente — sem histórico necessário.
    # -------------------------------------------------------------------
    preds_stack = np.stack([agg_preds[m][0] for m in available_models], axis=0)
    simple_pred = preds_stack.mean(axis=0)
    results["SimpleAverage"] = _eval_agg_array(simple_pred, y_true_ref)

    # -------------------------------------------------------------------
    # Estratégia 2 — Média Ponderada por CV (expanding window)
    # Pesos ∝ 1/MAPE médio calculado sobre folds ANTERIORES ao corrente.
    # Fold sem histórico (primeiro fold): fallback para pesos iguais.
    # -------------------------------------------------------------------
    if not cv_history:
        # Primeiro fold — sem histórico de CV disponível: pesos iguais
        results["WeightedAverage"] = _eval_agg_array(simple_pred.copy(), y_true_ref)
    else:
        # Acumula MAPEs de cada modelo nos folds anteriores (expanding window)
        mape_history: dict[str, list] = {}
        for hist_fold in cv_history:
            for m, info in hist_fold.items():
                mape_val = info.get("mape", np.nan)
                if not np.isnan(mape_val):
                    mape_history.setdefault(m, []).append(mape_val)

        # Modelos com histórico válido e presentes no fold corrente
        valid_models = [m for m in available_models if mape_history.get(m)]

        if not valid_models:
            results["WeightedAverage"] = _eval_agg_array(simple_pred.copy(), y_true_ref)
        else:
            avg_mapes   = {m: np.mean(mape_history[m]) for m in valid_models}
            inv_mapes   = {m: 1.0 / v for m, v in avg_mapes.items() if v > 0}
            total_inv   = sum(inv_mapes.values())

            if total_inv == 0:
                results["WeightedAverage"] = _eval_agg_array(simple_pred.copy(), y_true_ref)
            else:
                weights = {m: inv_mapes.get(m, 0.0) / total_inv for m in available_models}
                weighted_pred = sum(
                    agg_preds[m][0] * weights[m] for m in available_models
                )
                results["WeightedAverage"] = _eval_agg_array(weighted_pred, y_true_ref)

    # -------------------------------------------------------------------
    # Estratégia 3 — Stacking com Ridge (expanding window)
    # Meta-modelo treinado sobre previsões agregadas dos folds ANTERIORES.
    # Primeiro fold (sem histórico): retorna NaN — sem fallback artificial.
    # -------------------------------------------------------------------
    nan_metrics = {"MAE": np.nan, "RMSE": np.nan, "MAPE%": np.nan,
                   "y_true_total": np.nan, "y_pred_total": np.nan}

    if not cv_history:
        results["StackingMeta"] = nan_metrics
    else:
        X_meta_rows, y_meta_rows = [], []
        for hist_fold in cv_history:
            hist_models = [m for m in available_models if m in hist_fold]
            if not hist_models:
                continue
            n_pts = len(hist_fold[hist_models[0]]["y_pred_agg"])
            # Uma linha por ponto de agregação (programa); colunas = modelos
            row_block = np.stack(
                [hist_fold[m]["y_pred_agg"] if m in hist_fold
                 else np.full(n_pts, np.nan)
                 for m in available_models],
                axis=1,
            )  # shape: (n_pts, n_models)
            X_meta_rows.append(row_block)
            y_meta_rows.append(hist_fold[hist_models[0]]["y_true_agg"])

        if not X_meta_rows:
            results["StackingMeta"] = nan_metrics
        else:
            X_meta = np.vstack(X_meta_rows)
            y_meta = np.concatenate(y_meta_rows)

            # Remove linhas com NaN (modelo ausente em algum fold anterior)
            mask = ~np.isnan(X_meta).any(axis=1) & ~np.isnan(y_meta)
            X_meta, y_meta = X_meta[mask], y_meta[mask]

            n_samples = len(X_meta)
            if n_samples < 3:
                print(f"AVISO: Stacking treinado com apenas {n_samples} amostras")

            if n_samples == 0:
                results["StackingMeta"] = nan_metrics
            else:
                try:
                    meta = Ridge(alpha=1.0).fit(X_meta, y_meta)
                    X_curr = np.stack([agg_preds[m][0] for m in available_models], axis=1)
                    stacking_pred = meta.predict(X_curr)
                    results["StackingMeta"] = _eval_agg_array(stacking_pred, y_true_ref)
                except Exception:
                    results["StackingMeta"] = nan_metrics

    return results


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def _run_single_fold(df_full_fold: pd.DataFrame, df_ml: pd.DataFrame,
                     programas: list, test_year: int) -> pd.DataFrame:
    """
    Executa todos os modelos para um único fold (test_year).

    df_full_fold : df_full já filtrado para anos <= test_year
    df_ml        : df_ml filtrado para anos <= test_year e com context features
                   recalculadas sem dados futuros (fix: leakage residual no CV)
    """
    all_results = []

    # ---- MODO GLOBAL --------------------------------------------------------
    print("=" * 60)
    print("MODO GLOBAL (todos os dados)")
    print("=" * 60)

    stat_global = run_stat_models(df_full_fold, test_year, scope="GLOBAL")
    for model_name, metrics in stat_global.items():
        row = {"modelo": model_name, "familia": "Estatístico",
               "modo": "GLOBAL", **metrics}
        all_results.append(row)
        print(f"  {model_name:20s} MAE={metrics['MAE']:>10.2f}  "
              f"RMSE={metrics['RMSE']:>10.2f}  MAPE%={metrics['MAPE%']:>6.2f}")

    ml_global = run_ml_models(df_ml, test_year, scope="GLOBAL")
    for model_name, metrics in ml_global.items():
        familia = ("ML Clássico" if model_name in FAMILIES["ML Clássico"]
                   else "ML baseado em Ensemble")
        row = {"modelo": model_name, "familia": familia,
               "modo": "GLOBAL", **metrics}
        all_results.append(row)
        print(f"  {model_name:20s} MAE={metrics.get('MAE','N/A'):>10}  "
              f"RMSE={metrics.get('RMSE','N/A'):>10}  "
              f"MAPE%={metrics.get('MAPE%','N/A'):>6}")

    # Baseline naïve — GLOBAL
    for model_name, metrics in run_naive_model(df_ml, test_year, scope="GLOBAL").items():
        row = {"modelo": model_name, "familia": "Baseline", "modo": "GLOBAL", **metrics}
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

        stat_prog = run_stat_models(df_full_fold, test_year,
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
                       else "ML baseado em Ensemble")
            row = {"modelo": model_name, "familia": familia,
                   "modo": "POR_SETOR", **metrics}
            all_results.append(row)
            print(f"     {model_name:20s} MAE={metrics.get('MAE','N/A'):>10}  "
                  f"RMSE={metrics.get('RMSE','N/A'):>10}  "
                  f"MAPE%={metrics.get('MAPE%','N/A'):>6}")

        # Baseline naïve — POR_SETOR
        for model_name, metrics in run_naive_model(
                df_ml, test_year, scope="POR_SETOR", programa=prog).items():
            row = {"modelo": model_name, "familia": "Baseline",
                   "modo": "POR_SETOR", **metrics}
            all_results.append(row)
            print(f"     {model_name:20s} MAE={metrics.get('MAE','N/A'):>10}  "
                  f"RMSE={metrics.get('RMSE','N/A'):>10}  "
                  f"MAPE%={metrics.get('MAPE%','N/A'):>6}")

    return pd.DataFrame(all_results)


def run_benchmark(file_path: str, test_year: int | None = None) -> pd.DataFrame:
    """
    Executa o pipeline completo de benchmark para um único ano de teste.

    Retorna um DataFrame com uma linha por (modelo, modo, programa).
    """
    data = load_and_preprocess(file_path)
    df_full   = data["df_full"]
    df_ml     = data["df_ml"]
    programas = data["programas"]
    anos      = data["anos"]

    if test_year is None:
        test_year = max(anos)

    print(f"\n[benchmark] Ano de teste: {test_year}")
    print(f"[benchmark] Programas   : {programas}")
    print(f"[benchmark] Anos treino : {[a for a in anos if a < test_year]}\n")

    df_full_fold = df_full[df_full[YEAR_COL] <= test_year].copy()
    # fix: recalcula context features sem dados futuros (consistência com o CV)
    df_ml_fold = df_ml[df_ml[YEAR_COL] <= test_year].copy()
    df_ml_fold = _add_context_features(df_ml_fold)
    return _run_single_fold(df_full_fold, df_ml_fold, programas, test_year)


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def run_walkforward_cv(file_path: str,
                       min_train_years: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward cross-validation sobre todos os modelos do benchmark.

    Para cada ano t com pelo menos min_train_years de treino antes dele,
    treina em [t_min, ..., t-1] e avalia em t. Os modelos estatísticos retornam
    NaN automaticamente quando a série de treino é curta demais (< 3 pontos).

    Parâmetros
    ----------
    file_path       : caminho para o arquivo .xlsx
    min_train_years : mínimo de anos de treino para incluir um fold (padrão: 2)

    Retorna
    -------
    df_folds   – resultados brutos por fold (inclui colunas "fold" e "n_train")
    df_summary – MAPE médio, desvio padrão e nº de folds válidos por
                 (modelo, familia, modo, programa)
    """
    data = load_and_preprocess(file_path)
    df_full   = data["df_full"]
    df_ml     = data["df_ml"]
    programas = data["programas"]
    anos      = data["anos"]

    test_years = [a for i, a in enumerate(anos) if i >= min_train_years]

    print(f"\n[cv] Walk-forward CV  |  min_train_years={min_train_years}")
    print(f"[cv] Folds ({len(test_years)}): anos de teste = {test_years}\n")

    fold_dfs = []
    # Expanding window: acumula histórico de previsões de folds anteriores para o ensemble.
    # cv_history_global[i] e cv_history_ps[prog][i] contêm dados do i-ésimo fold passado;
    # o fold corrente nunca entra no histórico antes de ser avaliado.
    cv_history_global: list[dict] = []
    cv_history_ps: dict[str, list] = {prog: [] for prog in programas}

    for test_year in test_years:
        n_train = anos.index(test_year)
        train_anos = anos[:n_train]
        print(f"\n{'#'*60}")
        print(f"# FOLD  test_year={test_year}  |  treino: {train_anos}")
        print(f"{'#'*60}")

        df_full_fold = df_full[df_full[YEAR_COL] <= test_year].copy()
        # fix: recalcula context features sem dados futuros (leakage residual no CV)
        df_ml_fold = df_ml[df_ml[YEAR_COL] <= test_year].copy()
        df_ml_fold = _add_context_features(df_ml_fold)
        fold_df = _run_single_fold(df_full_fold, df_ml_fold, programas, test_year)
        fold_df["fold"]    = test_year
        fold_df["n_train"] = n_train

        # ----------------------------------------------------------------
        # Ensemble — coleta previsões do fold corrente e avalia estratégias
        # ----------------------------------------------------------------
        ens_rows: list[dict] = []

        # GLOBAL
        fold_preds_g = _collect_ml_predictions(df_ml_fold, test_year, "GLOBAL")
        if fold_preds_g:
            print(f"\n  [Ensemble] GLOBAL  (histórico: {len(cv_history_global)} fold(s))")
            ens_g = run_ensemble_models(fold_preds_g, cv_history_global,
                                        test_df=None, programa="TODOS")
            for strategy, metrics in ens_g.items():
                ens_rows.append({"modelo": strategy, "familia": "Ensemble",
                                  "modo": "GLOBAL", "programa": "TODOS", **metrics})
                print(f"     {strategy:20s} MAPE%={metrics.get('MAPE%', 'nan'):>6}")

            # Acumula histórico APÓS avaliação (expanding window — sem leakage)
            hist_g: dict = {}
            for m, fp in fold_preds_g.items():
                mape_row = fold_df[(fold_df["modelo"] == m) & (fold_df["modo"] == "GLOBAL")]
                mape_val = (float(mape_row["MAPE%"].iloc[0])
                            if not mape_row.empty else np.nan)
                y_pred_agg, y_true_agg = _agg_predictions(
                    fp["y_pred"], fp["y_true"], fp["test_df"], "TODOS"
                )
                hist_g[m] = {"mape": mape_val,
                              "y_pred_agg": y_pred_agg,
                              "y_true_agg": y_true_agg}
            cv_history_global.append(hist_g)

        # POR_SETOR
        for prog in programas:
            fold_preds_ps = _collect_ml_predictions(
                df_ml_fold, test_year, "POR_SETOR", prog
            )
            if fold_preds_ps:
                ens_ps = run_ensemble_models(fold_preds_ps, cv_history_ps[prog],
                                             test_df=None, programa=prog)
                for strategy, metrics in ens_ps.items():
                    ens_rows.append({"modelo": strategy, "familia": "Ensemble",
                                      "modo": "POR_SETOR", "programa": prog, **metrics})

                # Acumula histórico para este programa APÓS avaliação
                hist_ps: dict = {}
                for m, fp in fold_preds_ps.items():
                    mape_row = fold_df[(fold_df["modelo"] == m) &
                                       (fold_df["modo"] == "POR_SETOR") &
                                       (fold_df["programa"] == prog)]
                    mape_val = (float(mape_row["MAPE%"].iloc[0])
                                if not mape_row.empty else np.nan)
                    y_pred_agg, y_true_agg = _agg_predictions(
                        fp["y_pred"], fp["y_true"], fp["test_df"], prog
                    )
                    hist_ps[m] = {"mape": mape_val,
                                   "y_pred_agg": y_pred_agg,
                                   "y_true_agg": y_true_agg}
                cv_history_ps[prog].append(hist_ps)

        if ens_rows:
            ens_df = pd.DataFrame(ens_rows)
            ens_df["fold"]    = test_year
            ens_df["n_train"] = n_train
            fold_df = pd.concat([fold_df, ens_df], ignore_index=True)

        fold_dfs.append(fold_df)

    df_folds = pd.concat(fold_dfs, ignore_index=True)

    # Resumo agregado por (modelo, familia, modo, programa)
    grp_cols = ["modelo", "familia", "modo", "programa"]
    df_summary = (
        df_folds
        .dropna(subset=["MAPE%"])
        .groupby(grp_cols, sort=False)["MAPE%"]
        .agg(mape_mean="mean", mape_std="std", n_folds="count")
        .reset_index()
        .sort_values(["familia", "mape_mean"])
        .reset_index(drop=True)
    )

    return df_folds, df_summary


# ---------------------------------------------------------------------------
# Grid search de hiperparâmetros (Random Forest e LightGBM)
# ---------------------------------------------------------------------------

_ML_DEFAULT_PARAMS: dict = {
    "RandomForest": {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 3},
    "LightGBM":     {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
}

_ML_BEST_PARAMS: dict = {}  # preenchido por run_grid_search()

_RF_GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
    for n in [100, 200] for d in [4, 6, 8] for l in [2, 3, 5]
]  # 18 combinações

_LGBM_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in [100, 200] for d in [3, 4, 6] for lr in [0.05, 0.1]
]  # 12 combinações


def run_grid_search(file_path: str, min_train_years: int = 2) -> dict:
    """
    Grid search de hiperparâmetros para RandomForest e LightGBM usando
    walk-forward CV (POR_SETOR, média de MAPE sobre todos os folds e programas).

    Atualiza _ML_BEST_PARAMS globalmente e retorna o dicionário de melhores params.

    Parâmetros
    ----------
    file_path       : caminho para o arquivo .xlsx
    min_train_years : mínimo de anos de treino por fold (padrão: 2)
    """
    global _ML_BEST_PARAMS

    data = load_and_preprocess(file_path)
    df_ml     = data["df_ml"]
    programas = data["programas"]
    anos      = data["anos"]
    test_years = [a for i, a in enumerate(anos) if i >= min_train_years]

    grids = {
        "RandomForest": _RF_GRID,
        "LightGBM":     _LGBM_GRID,
    }
    model_factories = {
        "RandomForest": lambda p: RandomForestRegressor(
            random_state=42, n_jobs=-1, **p
        ),
        "LightGBM": lambda p: lgb.LGBMRegressor(
            random_state=42, verbose=-1, **p
        ),
    }

    best_params: dict = {}

    for model_name, param_grid in grids.items():
        n_combos = len(param_grid)
        print(f"\n[gridsearch] {model_name} — {n_combos} combinações × "
              f"{len(test_years)} folds × {len(programas)} programas")

        best_mape = float("inf")
        best_p: dict = _ML_DEFAULT_PARAMS[model_name]

        for params in param_grid:
            model = model_factories[model_name](params)
            mapes: list[float] = []

            for test_year in test_years:
                df_ml_fold = df_ml[df_ml[YEAR_COL] <= test_year].copy()
                df_ml_fold = _add_context_features(df_ml_fold)

                for prog in programas:
                    subset = df_ml_fold[df_ml_fold["PROGRAMA"] == prog]
                    X_train, y_train, X_test, y_test = _get_ml_splits(subset, test_year)
                    if len(X_train) == 0 or len(X_test) == 0:
                        continue
                    test_df = subset[subset[YEAR_COL] == test_year].copy()
                    test_df = test_df[~((test_df["lag_2"] == 0) & (test_df[TARGET] == 0))]
                    try:
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        err = evaluate_aggregated(y_test.values, preds, test_df, prog)
                        if not np.isnan(err["MAPE%"]):
                            mapes.append(err["MAPE%"])
                    except Exception:
                        pass

            mean_mape = float(np.mean(mapes)) if mapes else float("inf")
            if mean_mape < best_mape:
                best_mape = mean_mape
                best_p = params
            print(f"  {params}  ->  CV MAPE medio = {mean_mape:.2f}%")

        print(f"\n  >> Melhor {model_name}: {best_p}  (CV MAPE = {best_mape:.2f}%)")
        best_params[model_name] = best_p

    _ML_BEST_PARAMS.update(best_params)
    return best_params


# ---------------------------------------------------------------------------
# Relatórios
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


def print_cv_summary(df_summary: pd.DataFrame):
    print("\n" + "=" * 80)
    print("RESUMO WALK-FORWARD CV  (MAPE médio ± desvio padrão entre folds)")
    print("=" * 80)

    # Tabela geral sem os ensembles (mais legível)
    df_ind = df_summary[df_summary["familia"] != "Ensemble"]
    print(df_ind.to_string(index=False, float_format="%.2f"))

    print("\n--- Melhor modelo por família (MAPE médio) ---")
    for familia in df_ind["familia"].unique():
        sub = df_ind[df_ind["familia"] == familia]
        best = sub.loc[sub["mape_mean"].idxmin()]
        std_str = f"±{best['mape_std']:.2f}" if not np.isnan(best["mape_std"]) else "±n/a"
        print(f"  {familia:15s}: {best['modelo']} "
              f"(modo={best['modo']}, programa={best['programa']}, "
              f"MAPE={best['mape_mean']:.2f}% {std_str}, folds={int(best['n_folds'])})")

    # ----------------------------------------------------------------
    # Seção separada: Ensemble
    # ----------------------------------------------------------------
    df_ens = df_summary[df_summary["familia"] == "Ensemble"]
    if df_ens.empty:
        return

    print("\n" + "=" * 80)
    print("ENSEMBLE — MAPE médio ± desvio padrão entre folds")
    print("=" * 80)
    print(df_ens.to_string(index=False, float_format="%.2f"))

    print("\n--- Melhor estratégia de ensemble (MAPE médio) ---")
    df_ens_valid = df_ens.dropna(subset=["mape_mean"])
    if df_ens_valid.empty:
        print("  (nenhuma estratégia com folds válidos suficientes)")
        return
    best_ens = df_ens_valid.loc[df_ens_valid["mape_mean"].idxmin()]
    std_str  = (f"±{best_ens['mape_std']:.2f}"
                if not np.isnan(best_ens["mape_std"]) else "±n/a")
    print(f"  {best_ens['modelo']} "
          f"(modo={best_ens['modo']}, programa={best_ens['programa']}, "
          f"MAPE={best_ens['mape_mean']:.2f}% {std_str}, "
          f"folds={int(best_ens['n_folds'])})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de modelos – TCC")
    parser.add_argument("--file",            type=str, required=True,
                        help="Caminho para o arquivo .xlsx")
    parser.add_argument("--test-year",       type=int, default=None,
                        help="Ano de teste para benchmark único (padrão: último ano)")
    parser.add_argument("--output",          type=str, default="resultados_benchmark.csv",
                        help="Arquivo CSV de saída")
    parser.add_argument("--cv",              action="store_true",
                        help="Executa walk-forward cross-validation")
    parser.add_argument("--gridsearch",      action="store_true",
                        help="Roda grid search antes do benchmark e usa melhores params")
    parser.add_argument("--min-train-years", type=int, default=2,
                        help="Mínimo de anos de treino nos folds do CV (padrão: 2)")
    args = parser.parse_args()

    if args.gridsearch:
        best = run_grid_search(args.file, min_train_years=args.min_train_years)
        print("\n[gridsearch] Parâmetros selecionados:")
        for name, params in best.items():
            print(f"  {name}: {params}")

    if args.cv:
        df_folds, df_summary = run_walkforward_cv(
            args.file, min_train_years=args.min_train_years
        )
        print_cv_summary(df_summary)

        stem = args.output.replace(".csv", "")
        df_folds.to_csv(f"{stem}_folds.csv", index=False)
        df_summary.to_csv(f"{stem}_summary.csv", index=False)
        print(f"\n[cv] Resultados por fold salvos em : {stem}_folds.csv")
        print(f"[cv] Resumo salvo em               : {stem}_summary.csv")
    else:
        df_results = run_benchmark(args.file, test_year=args.test_year)
        print_summary(df_results)
        df_results.to_csv(args.output, index=False)
        print(f"\n[benchmark] Resultados salvos em: {args.output}")