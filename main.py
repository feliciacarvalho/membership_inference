import os
import joblib 
from preprocess.preprocessing import preprocess_data
from models.target_model import train_target_model
from models.shadow_model import train_multiple_shadow_models
from models.attack_model import generate_attack_data_with_in_out, train_attack_model, visualizar_conjunto_dados_ataque, salvar_predicoes_csv
from XAI.shap import explain_attack_model_with_shap

DATA_PATH = "data/adult_clean.csv"
TARGET_COL = "income"
NUM_SHADOW_MODELS = 5  
APPLY_SMOTE = False    
APPLY_SCALING = True 


def main():
    print("Carregando e pré-processando o conjunto de dados...")
    

    X_train, X_val, X_test, y_train, y_val, y_test, X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = preprocess_data(
        DATA_PATH, TARGET_COL, apply_smote_option=APPLY_SMOTE, apply_scaling_option=APPLY_SCALING)
    
    print(f"Tamanho do conjunto de dados do modelo alvo (train/val/test): {len(X_train)}, {len(X_val)}, {len(X_test)}")
    print(f"Tamanho do conjunto de dados do modelo shadow (train/test): {len(X_shadow_train)}, {len(X_shadow_test)}")
    
    shadow_models = [] 
    
    while True:
        print("\nEscolha uma opção:")
        print("1. Treinar o modelo alvo")
        print("2. Treinar modelos shadow")
        print("3. Treinar o modelo de ataque")
        print("4. Eexplicações com SHAP")
        print("5. Sair")

        option = input("Digite o número da opção desejada: ")

        if option == '1':
            
            target_model, _ = train_target_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=20, batch_size=32)

            
            if target_model:
                target_model.save("results/output/target_model.h5")
                print("Modelo alvo treinado e salvo com sucesso!")

        elif option == '2':

            shadow_models = train_multiple_shadow_models(X_shadow_train, y_shadow_train, X_shadow_test, y_shadow_test, num_models=NUM_SHADOW_MODELS, epochs=20, batch_size=32)

            if shadow_models:
                for idx, (model, _) in enumerate(shadow_models):  # Desempacota a tupla para obter o modelo
                    model.save(f"results/output/shadow_model_{idx}.h5") 
                print("Modelos shadow treinados e salvos com sucesso!")

        elif option == '3':
            if not shadow_models:
                print("Os modelos sombra ainda não foram treinados. Por favor, treine os modelos sombra primeiro.")
            else:
                # Gerar dados para o ataque usando 'in' e 'out'
                attack_X, attack_y = generate_attack_data_with_in_out(shadow_models, X_train, X_test, y_train, y_test)

                visualizar_conjunto_dados_ataque(attack_X, attack_y, NUM_SHADOW_MODELS)

                attack_model = train_attack_model(attack_X, attack_y)

                if attack_model:
                    joblib.dump(attack_model, "results/output/attack_model.pkl")
                    print("Modelo de ataque treinado e salvo com sucesso!")

                    salvar_predicoes_csv(attack_model, shadow_models, X_train, X_test, y_train, y_test, num_shadow_models=NUM_SHADOW_MODELS)
        elif option == '4':
            explain_attack_model_with_shap()
        elif option == '5':
            print("Saindo do programa...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    os.makedirs("results/output", exist_ok=True) 
    os.makedirs("results/metrics", exist_ok=True)  
    main()
