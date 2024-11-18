import shutil
import os

def limpar_diretorio(diretorio):
    
    if os.path.exists(diretorio):
        shutil.rmtree(diretorio)
        print(f"O diretório '{diretorio}' foi limpo.")
    else:
        print(f"O diretório '{diretorio}' não existia.")
