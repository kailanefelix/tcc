"""
preprocessing.py
Pipeline de pré-processamento para os dados do TCC.

Responsabilidades:
  1. Leitura e limpeza dos dados brutos
  2. Tratamento de datas / coluna ANO
  3. Label encoding de variáveis categóricas (MUNICIPIO, PROGRAMA)
  4. Expansão do grid completo (município × programa × ano) e fill com 0
  5. Engenharia de features para modelos de ML (lags, médias móveis, tendência)
  6. Separação em splits: global e por setor

Uso:
    from preprocessing import load_and_preprocess
    data = load_and_preprocess("cientista_cp.xlsx")
    df_full    = data["df_full"]          # grid completo, sem features de ML
    df_ml      = data["df_ml"]            # dataset com features para ML
    df_by_prog = data["df_by_programa"]   # dict { programa: df_ml filtrado }
    encoders   = data["encoders"]         # { "MUNICIPIO": le, "PROGRAMA": le }
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from itertools import product


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
TARGET = "beneficiarios"
NUMERIC_COLS = ["cadastros", "beneficiarios", "valor", "total_pcp",
                "total_pcp_corrigido", "valor_total"]
CATEGORICAL_COLS = ["MUNICIPIO", "PROGRAMA"]
YEAR_COL = "ANO"


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _clean_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas numéricas e preenche NaN com 0."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _clean_year(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que ANO seja inteiro e cria coluna datetime para séries temporais."""
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").fillna(0).astype(int)
    # Coluna auxiliar datetime (início do ano) – útil para statsmodels
    df["DATE"] = pd.to_datetime(df[YEAR_COL], format="%Y")
    return df


def _label_encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Aplica LabelEncoder em MUNICIPIO e PROGRAMA.
    Retorna o df transformado e um dicionário com os encoders para uso futuro.
    """
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[f"{col}_CODE"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def _expand_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria o produto cartesiano completo de (MUNICIPIO, PROGRAMA, ANO) e
    faz left join com os dados reais. Combinações sem dados recebem 0.

    Isso garante que todos os modelos vejam séries completas sem lacunas.
    """
    municipios = df["MUNICIPIO"].unique()
    programas  = df["PROGRAMA"].unique()
    anos       = sorted(df[YEAR_COL].unique())

    grid = pd.DataFrame(
        list(product(municipios, programas, anos)),
        columns=["MUNICIPIO", "PROGRAMA", YEAR_COL]
    )

    # Verifica se o par (município, programa) de fato existe no dataset original
    # Para não inflar combinações impossíveis (ex: município que só tem pesca)
    valid_pairs = df[["MUNICIPIO", "PROGRAMA"]].drop_duplicates()
    grid = grid.merge(valid_pairs, on=["MUNICIPIO", "PROGRAMA"], how="inner")

    # Join com dados reais
    df_full = grid.merge(
        df,
        on=["MUNICIPIO", "PROGRAMA", YEAR_COL],
        how="left"
    )

    # Preenche lacunas com 0
    for col in NUMERIC_COLS:
        if col in df_full.columns:
            df_full[col] = df_full[col].fillna(0)

    return df_full


def _add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features de ML baseadas no histórico de cada série (município × programa):
      - lag_1, lag_2: beneficiários do ano anterior e do anterior ao anterior
      - rolling_mean_2: média dos 2 anos anteriores
      - trend: diferença entre lag_1 e lag_2 (captura aceleração/desaceleração)
      - ano_rel: ano relativo a 2021 (tendência linear simples)
    """
    df = df.sort_values(["MUNICIPIO", "PROGRAMA", YEAR_COL]).copy()
    grp = df.groupby(["MUNICIPIO", "PROGRAMA"])[TARGET]

    df["lag_1"]         = grp.shift(1)
    df["lag_2"]         = grp.shift(2)
    df["rolling_mean_2"] = (df["lag_1"] + df["lag_2"]) / 2
    df["trend"]         = df["lag_1"] - df["lag_2"]
    df["ano_rel"]       = df[YEAR_COL] - df[YEAR_COL].min()

    # Preenche NaN gerados pelos lags com 0
    lag_cols = ["lag_1", "lag_2", "rolling_mean_2", "trend"]
    df[lag_cols] = df[lag_cols].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def load_and_preprocess(file_path: str) -> dict:
    """
    Executa o pipeline completo de pré-processamento.

    Parâmetros
    ----------
    file_path : str
        Caminho para o arquivo .xlsx com os dados brutos.

    Retorna
    -------
    dict com as chaves:
        df_full        – DataFrame com grid completo, sem features de ML
        df_ml          – DataFrame com features de ML (inclui lags etc.)
        df_by_programa – dict { nome_programa: df_ml filtrado }
        encoders       – dict { coluna: LabelEncoder treinado }
        programas      – lista de programas únicos
        anos           – lista de anos únicos
    """
    # 1. Leitura
    df = pd.read_excel(file_path, engine="openpyxl")
    print(f"[preproc] Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # 2. Limpeza básica
    df = _clean_numerics(df)
    df = _clean_year(df)
    print(f"[preproc] Anos disponíveis: {sorted(df[YEAR_COL].unique())}")
    print(f"[preproc] Programas: {df['PROGRAMA'].unique().tolist()}")
    print(f"[preproc] Municípios únicos: {df['MUNICIPIO'].nunique()}")

    # 3. Label encoding
    df, encoders = _label_encode(df)

    # 4. Expansão do grid e fill com 0
    df_full = _expand_grid(df)
    # Re-aplica encoding no grid expandido (novos encoders usando os já treinados)
    for col, le in encoders.items():
        df_full[f"{col}_CODE"] = le.transform(df_full[col].astype(str))
    df_full = _clean_year(df_full)
    df_full["ano_rel"] = df_full[YEAR_COL] - df_full[YEAR_COL].min()
    print(f"[preproc] Grid expandido: {df_full.shape[0]} linhas "
          f"(adicionadas {df_full.shape[0] - df.shape[0]} combinações sem dados)")

    # 5. Features de ML
    df_ml = _add_ml_features(df_full)

    # 6. Splits por setor (programa)
    programas = sorted(df_ml["PROGRAMA"].unique())
    df_by_programa = {prog: df_ml[df_ml["PROGRAMA"] == prog].copy()
                      for prog in programas}

    anos = sorted(df_ml[YEAR_COL].unique())

    return {
        "df_full":        df_full,
        "df_ml":          df_ml,
        "df_by_programa": df_by_programa,
        "encoders":       encoders,
        "programas":      programas,
        "anos":           anos,
    }


# ---------------------------------------------------------------------------
# Exemplo de uso direto
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "cientista_cp.xlsx"
    data = load_and_preprocess(path)
    print("\n--- Resumo do pré-processamento ---")
    print(f"df_full shape : {data['df_full'].shape}")
    print(f"df_ml shape   : {data['df_ml'].shape}")
    print(f"Programas     : {data['programas']}")
    print(f"Anos          : {data['anos']}")
    print("\nPrimeiras linhas do df_ml:")
    print(data["df_ml"].head(10).to_string(index=False))