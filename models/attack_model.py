import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def generate_attack_data_with_in_out(shadow_models, X_train, X_test, y_train, y_test):
    """
    Gera os dados de entrada para o modelo de ataque, criando os rótulos 'in' e 'out' para os exemplos de treino e teste
    dos modelos sombra.

    :param shadow_models: Lista de modelos sombra treinados, onde cada modelo é uma tupla (modelo, histórico)
    :param X_train: Dados de treino do modelo alvo
    :param X_test: Dados de teste do modelo alvo
    :param y_train: Labels de treino do modelo alvo
    :param y_test: Labels de teste do modelo alvo
    :return: Dados de entrada e rótulos para o modelo de ataque
    """
    shadow_train_outputs = []
    shadow_test_outputs = []
    
    # Gerando as previsões para o conjunto de treino e teste dos modelos sombra
    for model, _ in shadow_models:  
        shadow_train_outputs.append(model.predict(X_train)) 
        shadow_test_outputs.append(model.predict(X_test))    

    # Criando os rótulos 'in' (membros) e 'out' (não membros) para treino e teste
    attack_X = []
    attack_y = []

    
    for i in range(len(X_train)):
        for j, model_output in enumerate(shadow_train_outputs):
            attack_X.append(X_train[i])  
            attack_y.append(1)  # Rótulo 'in' (membro) se a previsão do modelo corresponde a este dado

    
    for i in range(len(X_test)):
        for j, model_output in enumerate(shadow_test_outputs):
            attack_X.append(X_test[i]) 
            attack_y.append(0)  # Rótulo 'out' (não membro) se a previsão do modelo não corresponde

    
    attack_X = np.array(attack_X)
    attack_y = np.array(attack_y)

    return attack_X, attack_y


def train_attack_model(X_attack, y_attack):
    """
    Treina um modelo de ataque para inferência de membros usando um classificador.
    """
   
    attack_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    
    attack_model.fit(X_attack, y_attack)
    
    
    y_pred = attack_model.predict(X_attack)
    
   
    accuracy = accuracy_score(y_attack, y_pred)
    precision = precision_score(y_attack, y_pred)
    recall = recall_score(y_attack, y_pred)
    f1 = f1_score(y_attack, y_pred)
    conf_matrix = confusion_matrix(y_attack, y_pred)
    
    
    y_prob = attack_model.predict_proba(X_attack)[:, 1] 
    roc_auc = roc_auc_score(y_attack, y_prob)


    print("\nMétricas do modelo de ataque:")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Revocação: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Matriz de Confusão: \n{conf_matrix}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    
    return attack_model

import pandas as pd

def visualizar_conjunto_dados_ataque(attack_X, attack_y, num_shadow_models):
    """
    Visualiza o conjunto de dados gerado para o modelo de ataque em formato tabular.

    :param attack_X: Conjunto de características gerado para o modelo de ataque
    :param attack_y: Rótulos (0 - não membro, 1 - membro) gerados para o modelo de ataque
    :param num_shadow_models: Número de modelos sombra usados no experimento (não utilizado diretamente aqui, mas mantido para referência)
    """
    
    feature_columns = [f"Feature_{i}" for i in range(attack_X.shape[1])] + ["Membro"]

    
    attack_data = pd.DataFrame(
        data=np.hstack((attack_X, attack_y.reshape(-1, 1))),
        columns=feature_columns
    )

    
    print("\nConjunto de Dados para o Modelo de Ataque:")
    print(attack_data.head())
    print(f"\n[{len(attack_data)} rows x {len(feature_columns)} columns]")



def salvar_predicoes_csv(attack_model, shadow_models, X_train, X_test, y_train, y_test, num_shadow_models=5):
    """
    Gera um arquivo CSV contendo as classificações feitas pelo modelo de ataque para cada registro.

    :param attack_model: Modelo de ataque treinado.
    :param shadow_models: Lista de modelos sombra treinados.
    :param X_train: Dados de treino do modelo alvo.
    :param X_test: Dados de teste do modelo alvo.
    :param y_train: Labels de treino do modelo alvo.
    :param y_test: Labels de teste do modelo alvo.
    :param num_shadow_models: Número de modelos sombra.
    """
    
    attack_X, attack_y = generate_attack_data_with_in_out(shadow_models, X_train, X_test, y_train, y_test)
    
    
    attack_predictions = attack_model.predict(attack_X)
    attack_probabilities = attack_model.predict_proba(attack_X)[:, 1]
    
    
    # Assumindo que attack_X foi construído como [X_train * num_shadow_models] + [X_test * num_shadow_models]
    num_train = len(X_train)
    num_test = len(X_test)
    
    
    origem = ['treino'] * (num_train * num_shadow_models) + ['teste'] * (num_test * num_shadow_models)
    
    
    ids_treino = np.tile(np.arange(num_train), num_shadow_models)
    ids_teste = np.tile(np.arange(num_test), num_shadow_models)
    ids = np.concatenate([ids_treino, ids_teste])
    
    df_pred = pd.DataFrame({
        'ID_Instancia': ids,
        'Origem': origem,
        'Predicao_Ataque': attack_predictions,
        'Probabilidade_Ataque': attack_probabilities,
        'Rótulo_Original': np.concatenate([np.ones(num_train * num_shadow_models), np.zeros(num_test * num_shadow_models)])
    })
    
    # Agrupar por ID_Instancia e Origem para obter uma predição agregada
    df_aggregated = df_pred.groupby(['ID_Instancia', 'Origem']).agg(
        Predicao_Ataque_Media=('Predicao_Ataque', 'mean'),
        Probabilidade_Ataque_Media=('Probabilidade_Ataque', 'mean'),
        Rótulo_Original=('Rótulo_Original', 'first') 
    ).reset_index()
    
    
    df_aggregated['Predicao_Ataque_Binaria'] = (df_aggregated['Predicao_Ataque_Media'] >= 0.5).astype(int)
    
    
    df_aggregated['Acerto_Ataque'] = df_aggregated['Predicao_Ataque_Binaria'] == df_aggregated['Rótulo_Original']
    
    
    if isinstance(X_train, pd.DataFrame):
        X_train_df = X_train.copy()
    else:
        X_train_df = pd.DataFrame(X_train)
    
    if isinstance(X_test, pd.DataFrame):
        X_test_df = X_test.copy()
    else:
        X_test_df = pd.DataFrame(X_test)
    
    
    X_train_df['ID_Instancia'] = np.arange(num_train)
    X_test_df['ID_Instancia'] = np.arange(num_test)
    
    
    X_all_df = pd.concat([X_train_df, X_test_df], ignore_index=True)
    
    
    df_final = pd.merge(df_aggregated, X_all_df, on='ID_Instancia', how='left', suffixes=('_Ataque', '_Original'))
    
    
    df_final.to_csv("results/output/predicoes_modelo_ataque.csv", index=False)
    
    print("Arquivo CSV com as predições do modelo de ataque foi salvo em 'results/output/predicoes_modelo_ataque.csv'")
