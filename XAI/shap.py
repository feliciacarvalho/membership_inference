import shap
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def explain_attack_model_with_shap():

    print("Carregando o modelo de ataque...")
    attack_model_path = "results/output/attack_model.pkl"

    if not os.path.exists(attack_model_path):
        print("Modelo de ataque não encontrado. Treine o modelo de ataque primeiro.")
        return

    
    with open(attack_model_path, "rb") as file:
        attack_model = joblib.load(file)

    
    attack_X = pd.read_csv("results/output/predicoes_modelo_ataque.csv").filter(regex='^\d+$')  

   
    print("Gerando explicações SHAP...")
    explainer = shap.Explainer(attack_model.predict, attack_X)
    shap_values = explainer(attack_X)


    print("Salvando gráficos SHAP...")
    shap.summary_plot(shap_values, attack_X, show=False)
    plt.savefig("results/output/shap_summary_plot.png")
    plt.close()

    # Force Plot (exemplo de uma instância específica)
    instance_index = 0 
    shap.force_plot(explainer.expected_value[1], shap_values[instance_index], attack_X.iloc[instance_index, :], matplotlib=True).savefig("results/output/shap_force_plot.png")

    print("Gráficos SHAP salvos em 'results/output/'.")
