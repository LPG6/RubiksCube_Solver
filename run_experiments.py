import time
import pandas as pd
import torch
from tqdm import tqdm
from cube import RubiksCube
from solver_ia import NeuralHeuristic, BuscaGulosaNeural
import numpy as np

def run_experiment_1():
    print("Iniciando Experimento 1: MLP vs CNN vs Profundidade")
    
    # Carrega os dois modelos
    modelos = {
        "MLP": NeuralHeuristic('modelo_mlp.pth', tipo_modelo="MLP"),
        "CNN": NeuralHeuristic('modelo_cnn.pth', tipo_modelo="CNN")
    }
    
    profundidades =[1, 2, 3, 5, 8, 10, 12, 15, 18, 20]
    n_testes_por_profundidade = 100
    
    resultados =[]

    for nome_modelo, heuristica in modelos.items():
        print(f"\n--- Testando Modelo: {nome_modelo} ---")
        solver = BuscaGulosaNeural(heuristica, max_moves=25) # Limite para não ficar infinito
        
        for prof in profundidades:
            sucessos = 0
            tempos = []
            movimentos_excedentes =[]
            
            # Barra de progresso para você acompanhar
            for _ in tqdm(range(n_testes_por_profundidade), desc=f"Profundidade {prof}"):
                cube = RubiksCube(n=3)
                cube.shuffle(l_rot=prof, u_rot=prof) # Embaralha exatamente 'prof' vezes
                estado_inicial = cube.stringify()
                
                # Inicia a contagem de tempo
                start_time = time.perf_counter()
                
                # Roda as 3 fases em sequência
                m_l1 = solver.run(estado_inicial, 'L1')
                
                # Simula fisicamente os movimentos da L1 para passar pra F2L
                for a in m_l1:
                    if a[0] == 'h': cube.horizontal_twist(a[1], a[2])
                    elif a[0] == 'v': cube.vertical_twist(a[1], a[2])
                    elif a[0] == 's': cube.side_twist(a[1], a[2])
                
                m_f2l = solver.run(cube.stringify(), 'F2L')
                for a in m_f2l:
                    if a[0] == 'h': cube.horizontal_twist(a[1], a[2])
                    elif a[0] == 'v': cube.vertical_twist(a[1], a[2])
                    elif a[0] == 's': cube.side_twist(a[1], a[2])
                
                m_full = solver.run(cube.stringify(), 'FULL')
                for a in m_full:
                    if a[0] == 'h': cube.horizontal_twist(a[1], a[2])
                    elif a[0] == 'v': cube.vertical_twist(a[1], a[2])
                    elif a[0] == 's': cube.side_twist(a[1], a[2])
                
                end_time = time.perf_counter()
                
                movimentos_totais = len(m_l1) + len(m_f2l) + len(m_full)
                resolvido = cube.solved()
                
                if resolvido:
                    sucessos += 1
                    tempos.append(end_time - start_time)
                    # Excesso = Movimentos feitos - Profundidade ideal teórica
                    movimentos_excedentes.append(movimentos_totais - prof)

            # Salva o resumo daquela profundidade
            tempo_medio = sum(tempos)/len(tempos) if tempos else 0
            tempo_std = np.std(tempos) if tempos else 0

            excesso_medio = sum(movimentos_excedentes)/len(movimentos_excedentes) if movimentos_excedentes else 0
            excesso_std = np.std(movimentos_excedentes) if movimentos_excedentes else 0

            
            resultados.append({
            "Modelo": nome_modelo,
            "Profundidade": prof,
            "Taxa_Sucesso_%": (sucessos / n_testes_por_profundidade) * 100,
            "Tempo_Medio_s": tempo_medio,
            "Tempo_Std_s": tempo_std,            # NOVO
            "Excesso_Medio_Movs": excesso_medio,
            "Excesso_Std_Movs": excesso_std      # NOVO
        })

    # Salva tudo no final
    df = pd.DataFrame(resultados)
    df.to_csv('resultados_experimento_1.csv', index=False)
    print("\nResultados salvos em 'resultados_experimento_1.csv'!")
    print(df)

if __name__ == '__main__':
    run_experiment_1()