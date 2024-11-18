# import numpy as np
# from sklearn.model_selection import RandomizedSearchCV
# from scikeras.wrappers import KerasClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
# from tensorflow.keras.optimizers import Adam, RMSprop

# def build_cnn_1d_model_tuning(input_shape, optimizer='adam', kernel_size=2, pool_size=2, dense_units=64):
#     """
#     Constrói uma CNN 1D para classificação binária com suporte a parâmetros variáveis para tuning.
    
#     :param input_shape: Tuple que representa a forma do input para o modelo
#     :param optimizer: Otimizador usado no modelo
#     :param kernel_size: Tamanho do kernel na camada Conv1D
#     :param pool_size: Tamanho do MaxPooling1D
#     :param dense_units: Unidades na camada densa
#     :return: Um modelo Keras Sequential
#     """
#     model = Sequential([
#         Conv1D(64, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
#         MaxPooling1D(pool_size=pool_size),
#         Conv1D(128, kernel_size=kernel_size, activation='relu'),
#         MaxPooling1D(pool_size=pool_size),
#         Flatten(),
#         Dense(dense_units, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])

#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def create_model_for_random_search(input_shape):
#     """
#     Função que retorna o modelo Keras em um formato compatível com RandomizedSearchCV.
    
#     :param input_shape: Tuple que representa a forma do input para o modelo
#     :return: Função de criação do modelo
#     """
#     def model(optimizer='adam', kernel_size=2, dense_units=64):
#         return build_cnn_1d_model_tuning(input_shape=input_shape, optimizer=optimizer, kernel_size=kernel_size, dense_units=dense_units)
    
#     return model

# def perform_random_search(X_train, y_train, param_distributions, n_iter=10):
#     """
#     Executa o Random Search para encontrar os melhores hiperparâmetros.

#     :param X_train: Dados de treino (já ajustados)
#     :param y_train: Labels de treino
#     :param param_distributions: Distribuição de parâmetros para Random Search
#     :param n_iter: Número de iterações do Random Search
#     :return: Melhor modelo e melhores parâmetros encontrados
#     """
#     input_shape = (X_train.shape[1], 1)
#     model = KerasClassifier(build_fn=create_model_for_random_search(input_shape), verbose=0)

#     random_search = RandomizedSearchCV(
#         estimator=model,
#         param_distributions=param_distributions,
#         n_iter=n_iter,
#         cv=3,
#         verbose=2,
#         n_jobs=-1
#     )

#     random_search.fit(X_train, y_train)

#     best_model = random_search.best_estimator_.model
#     best_params = random_search.best_params_

#     return best_model, best_params
