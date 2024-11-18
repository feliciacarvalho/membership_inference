import numpy as np
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, RMSprop

def build_cnn_1d_model_tuner(hp):
    model = Sequential()
    
    input_length = hp.Int('input_length', min_value=76, max_value=128, step=4)
    model.add(Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32),
                     kernel_size=hp.Choice('kernel_size', values=[2, 3, 4]),
                     activation='relu',
                     input_shape=(input_length, 1)))
    
    model.add(MaxPooling1D(pool_size=hp.Choice('pool_size', values=[2, 3])))
    model.add(Conv1D(filters=hp.Int('filters2', min_value=64, max_value=128, step=32),
                     kernel_size=hp.Choice('kernel_size2', values=[2, 3]),
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=hp.Choice('pool_size2', values=[2, 3])))

    # GlobalAveragePooling1D para evitar problemas de dimensão
    model.add(GlobalAveragePooling1D())
    
    model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model



def run_random_search(X_train, y_train, X_val, y_val, max_trials=10, epochs=20, batch_size=32):
    """
    Executa o Random Search usando Keras Tuner para encontrar os melhores hiperparâmetros.

    :param X_train: Dados de treino
    :param y_train: Labels de treino
    :param X_val: Dados de validação
    :param y_val: Labels de validação
    :param max_trials: Número máximo de tentativas de busca
    :param epochs: Número de épocas para cada tentativa
    :param batch_size: Tamanho do batch
    :return: Melhor modelo e melhores hiperparâmetros
    """
    tuner = kt.RandomSearch(
        build_cnn_1d_model_tuner,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='results/hp_tuning',
        project_name='cnn_tuning'
    )

    # Expandindo as dimensões para a CNN 1D
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)

    tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size)

    # Obter o melhor modelo
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model, tuner.get_best_hyperparameters(num_trials=1)[0]
