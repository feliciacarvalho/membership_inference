import numpy as np
from models.model import build_dense_model, PrintDot, plot_training_history, evaluate_model

def train_target_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=20, batch_size=32):
    """
    Treina o modelo alvo usando uma rede neural densa.
    
    :param X_train: Dados de treino
    :param y_train: Labels de treino
    :param X_test: Dados de teste
    :param y_test: Labels de teste
    :param epochs: Número de épocas
    :param batch_size: Tamanho do batch
    :return: O modelo treinado e o histórico do treinamento
    """

    input_shape = (X_train.shape[1],)  
    model = build_dense_model(input_shape)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=[PrintDot()], verbose=0)

    print("\nTreinamento do modelo alvo concluído!")
    plot_training_history(history, model_name='target_model') 
    evaluate_model(model, X_test, y_test, model_name='target_model')

    return model, history
