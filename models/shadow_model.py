import numpy as np
from models.model import build_shadow_dense_model, PrintDot, plot_training_history, evaluate_model

def train_multiple_shadow_models(X_train, y_train, X_test, y_test, num_models=5, epochs=20, batch_size=32):
    """
    Treina múltiplos modelos shadow usando uma rede neural densa.
    
    :param X_train: Dados de treino do modelo shadow
    :param y_train: Labels de treino do modelo shadow
    :param X_test: Dados de teste do modelo shadow
    :param y_test: Labels de teste do modelo shadow
    :param num_models: Número de modelos shadow a serem treinados
    :param epochs: Número de épocas
    :param batch_size: Tamanho do batch
    :return: Lista de modelos shadow treinados e seus históricos de treinamento
    """
    shadow_models = []

    for i in range(num_models):
        print(f"\nTreinando o modelo shadow {i + 1}/{num_models}...")

        
        input_shape = (X_train.shape[1],)
        model = build_shadow_dense_model(input_shape)

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=0.2, callbacks=[PrintDot()], verbose=0)

    
        shadow_models.append((model, history))

        plot_training_history(history, model_name=f'shadow_model_{i + 1}')
        evaluate_model(model, X_test, y_test, model_name=f'shadow_model_{i + 1}')
        print("----------------------------------------------------------------------")

    print("\nTreinamento de todos os modelos shadow concluído!")
    return shadow_models
