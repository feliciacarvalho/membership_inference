import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """
    Carrega o dataset do caminho especificado.
    """
    return pd.read_csv(file_path)

### --------------------------------------------------------------------------------------------

def drop_correlated_columns(df, columns_to_drop):
    """
    Descartar colunas altamente correlacionadas.
    """
    return df.drop(columns=columns_to_drop, axis=1)

def encode_categorical_columns(df):
    """
    Codificar variáveis categóricas usando OneHotEncoder.
    """
    categoricals = [column for column in df.columns if df[column].dtype == "O"]
    oh_encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded_categoricals = oh_encoder.fit_transform(df[categoricals])

    # Converte as variáveis categóricas codificadas para um DataFrame
    encoded_categoricals = pd.DataFrame(encoded_categoricals, 
                                        columns=oh_encoder.get_feature_names_out().tolist())

    # Junta os dados codificados com o dataset original e remove as colunas categóricas antigas
    df_encoded = df.join(encoded_categoricals)
    df_encoded.drop(columns=categoricals, inplace=True)

    return df_encoded

### --------------------------------------------------------------------------------------------

def split_dataset(df, target_col):
    """
    Separa o dataset em dois: um para o modelo alvo e outro para o modelo sombra.
    A divisão é feita ao meio, sem sobreposição de dados.
    """
    df.dropna(inplace=True)
    
    
    target_dataset = df.sample(frac=0.5, random_state=42) 
    shadow_dataset = df.drop(target_dataset.index)  

    return target_dataset, shadow_dataset

### --------------------------------------------------------------------------------------------

def prepare_training_data(dataset, target_col, test_size=0.2, val_size=0.2, random_state=1):
    """
    Separa as variáveis preditoras e a variável alvo e divide os dados em treino, validação e teste.
    Retorna arrays numpy ao invés de DataFrames pandas.
    """
    
    X = dataset.drop(columns=[target_col]).values 
    y = dataset[target_col].values  

    # Dividindo em treino + validação e teste
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, 
                                                                random_state=random_state, stratify=y)

    # Dividindo treino em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, 
                                                      random_state=random_state, stratify=y_train_val)

    return X_train, X_val, X_test, y_train, y_val, y_test

### --------------------------------------------------------------------------------------------

def prepare_shadow_data(shadow_dataset, target_col, test_size=0.2, random_state=1):
    """
    Divide o dataset shadow em treino e teste.
    """
    X_shadow = shadow_dataset.drop(columns=[target_col]).values
    y_shadow = shadow_dataset[target_col].values

    X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = train_test_split(
        X_shadow, y_shadow, test_size=test_size, random_state=random_state, stratify=y_shadow
    )

    return X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test

### --------------------------------------------------------------------------------------------

def apply_smote(X_train, y_train):
    """
    Aplica SMOTE para balancear o conjunto de dados de treino.
    """
    smote = SMOTE(sampling_strategy="auto")
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

### --------------------------------------------------------------------------------------------
def apply_minmax_scaling_target_model(X_train, X_val, X_test):
    """
    Aplica MinMaxScaler ao conjunto de dados do modelo alvo (com treino, validação e teste).
    """
    scaler = MinMaxScaler()
    
    if np.any(np.isnan(X_train)):
        print("Aviso: Valores NaN encontrados em X_train. Substituindo por 0.")
        X_train = np.nan_to_num(X_train)

    if np.any(np.isnan(X_val)):
        print("Aviso: Valores NaN encontrados em X_val. Substituindo por 0.")
        X_val = np.nan_to_num(X_val)

    if np.any(np.isnan(X_test)):
        print("Aviso: Valores NaN encontrados em X_test. Substituindo por 0.")
        X_test = np.nan_to_num(X_test)
    

    X_train = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train
    X_val = X_val.reshape(-1, 1) if X_val.ndim == 1 else X_val
    X_test = X_test.reshape(-1, 1) if X_test.ndim == 1 else X_test
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def apply_minmax_scaling_shadow_model(X_train, X_test):
    """
    Aplica MinMaxScaler ao conjunto de dados do modelo sombra.
    """
    scaler = MinMaxScaler()
    
    if np.any(np.isnan(X_train)):
        print("Aviso: Valores NaN encontrados em X_train. Substituindo por 0.")
        X_train = np.nan_to_num(X_train)

    if np.any(np.isnan(X_test)):
        print("Aviso: Valores NaN encontrados em X_test. Substituindo por 0.")
        X_test = np.nan_to_num(X_test)
    
    X_train = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train
    X_test = X_test.reshape(-1, 1) if X_test.ndim == 1 else X_test
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled



### --------------------------------------------------------------------------------------------

def preprocess_data(file_path, target_col, apply_smote_option=False, apply_scaling_option=False):
    """
    Função completa de pré-processamento. Executa:
    1. Carregamento dos dados
    2. Descartar colunas correlacionadas
    3. Codificação de variáveis categóricas
    4. Separação dos dados em target e shadow datasets (sem sobreposição de dados)
    5. Divisão em treino, validação e teste para o modelo alvo e em treino e teste para o modelo shadow
    6. Opcional: Aplicação de SMOTE e MinMaxScaler
    """
    
    df = load_data(file_path)
    print("Dataset carregado!!!")

    # Descartar colunas correlacionadas
    columns_to_drop = ['relationship', 'education']
    df = drop_correlated_columns(df, columns_to_drop)
    print("Colunas correlacionadas deletadas!!!")

    # Codificar variáveis categóricas
    df_encoded = encode_categorical_columns(df)
    print("Variáveis categóricas codificadas!!!")

    # Separar datasets para o modelo alvo e sombra
    target_dataset, shadow_dataset = split_dataset(df_encoded, target_col)
    print("Separando dados do modelo alvo e shadow!!!")

    # Preparar os dados para treinamento, teste e validação do modelo alvo
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_training_data(target_dataset, target_col)
    print("Dados de treinamento do modelo alvo preparados!!!")

    # Preparar os dados para o modelo shadow 
    X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = prepare_shadow_data(shadow_dataset, target_col)
    print("Dados de treinamento do modelo shadow preparados!!!")

    # Aplicar SMOTE se selecionado
    if apply_smote_option:
        X_train, y_train = apply_smote(X_train, y_train)
        print("SMOTE aplicado no modelo alvo!!!")

    # Aplicar MinMaxScaler se selecionado
    if apply_scaling_option:
        # Aplicar MinMaxScaler no modelo alvo
        X_train, X_val, X_test = apply_minmax_scaling_target_model(X_train, X_val, X_test)
        print("MinMaxScaler aplicado no modelo alvo!!!")

        # Aplicar MinMaxScaler no modelo shadow
        X_shadow_train, X_shadow_test = apply_minmax_scaling_shadow_model(X_shadow_train, X_shadow_test)
        print("MinMaxScaler aplicado no modelo shadow!!!")

    # Retornar dados para o modelo alvo e dados para o modelo shadow
    return X_train, X_val, X_test, y_train, y_val, y_test, X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test



