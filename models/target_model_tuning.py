import numpy as np
import os
import matplotlib.pyplot as plt
from models.cnn_model_tuning import run_random_search
from models.model import plot_training_history, PrintDot, evaluate_model


metrics_dir = "results/metrics"
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)

def tune_target_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Aplica Keras Tuner para encontrar o melhor modelo e hiperparâmetros.
    
    :param X_train: Dados de treino
    :param y_train: Labels de treino
    :param X_val: Dados de validação
    :param y_val: Labels de validação
    :param X_test: Dados de teste
    :param y_test: Labels de teste
    :return: Melhor modelo, histórico de treinamento e melhores parâmetros
    """
    # Expandindo as dimensões para a CNN 1D
    X_train = np.expand_dims(X_train, axis=2)  
    X_val = np.expand_dims(X_val, axis=2)     
    X_test = np.expand_dims(X_test, axis=2)    

    # Executa a busca de hiperparâmetros
    best_model, best_params = run_random_search(X_train, y_train, X_val, y_val)

    # Treinar o melhor modelo encontrado no conjunto de validação
    history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                             callbacks=[PrintDot()], epochs=20, batch_size=32, verbose=0)

    # Salvar gráficos do histórico de treinamento
    plot_training_history(history, model_name='target_model_tunning')
    # plt.savefig(f"{metrics_dir}/target_model_tuned_history.png")

    # Avaliar o modelo no conjunto de teste e salvar o gráfico da curva ROC
    evaluate_model(best_model, X_test, y_test, model_name='target_model_tunning')
    # plt.savefig(f"{metrics_dir}/target_model_tuned_roc.png")

    # Retornar o melhor modelo, histórico e parâmetros
    print(f"Melhores parâmetros encontrados: {best_params.values}")

    return best_model, history, best_params
