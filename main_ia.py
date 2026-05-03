import time
from cube import RubiksCube
from solver_ia import NeuralHeuristic, BuscaGulosaNeural

def main():
    print("Iniciando Cubo de Rubik com Inteligência Artificial Multi-Head")
    print("==============================================================\n")
    
    # 1. Carregando o Cérebro (Vamos usar a MLP que teve o menor Erro)
    print("Carregando pesos da Rede Neural MLP...")
    ia_heuristic = NeuralHeuristic('modelo_mlp.pth', tipo_modelo="MLP")
    
    # 2. Criando e Embaralhando o cubo
    cube = RubiksCube(n=3)
    
    # Vamos embaralhar em 8 movimentos. Como a IA vai tentar resolver em 3 fases,
    # será muito interessante ver como ela monta o caminho!
    print("\nEmbaralhando o cubo em 8 movimentos...")
    cube.shuffle(l_rot=8, u_rot=8)
    cube.show()
    print("--------------------------------------------------")

    # 3. Solucionador Layer-by-Layer
    solver = BuscaGulosaNeural(ia_heuristic, max_moves=15)
    movimentos_totais =[]
    
    start_time = time.time()
    
    # --- FASE 1: Montar a Base (L1) ---
    print("\n[Fase 1] IA pensando: Montando a 1ª Camada...")
    estado_atual = cube.stringify()
    moves_l1 = solver.run(estado_atual, target_phase='L1')
    print(f"Movimentos L1: {moves_l1}")
    for a in moves_l1: # Aplica fisicamente no cubo
        if a[0] == 'h': cube.horizontal_twist(a[1], a[2])
        elif a[0] == 'v': cube.vertical_twist(a[1], a[2])
        elif a[0] == 's': cube.side_twist(a[1], a[2])
        movimentos_totais.append(a)
    
    # --- FASE 2: Montar F2L (L1 + L2) ---
    print("\n[Fase 2] IA pensando: Montando as duas camadas (F2L)...")
    estado_atual = cube.stringify()
    moves_f2l = solver.run(estado_atual, target_phase='F2L')
    print(f"Movimentos F2L: {moves_f2l}")
    for a in moves_f2l:
        if a[0] == 'h': cube.horizontal_twist(a[1], a[2])
        elif a[0] == 'v': cube.vertical_twist(a[1], a[2])
        elif a[0] == 's': cube.side_twist(a[1], a[2])
        movimentos_totais.append(a)

    # --- FASE 3: Finalizar o Cubo (FULL) ---
    print("\n[Fase 3] IA pensando: Finalizando o Cubo...")
    estado_atual = cube.stringify()
    moves_full = solver.run(estado_atual, target_phase='FULL')
    print(f"Movimentos Finais: {moves_full}")
    for a in moves_full:
        if a[0] == 'h': cube.horizontal_twist(a[1], a[2])
        elif a[0] == 'v': cube.vertical_twist(a[1], a[2])
        elif a[0] == 's': cube.side_twist(a[1], a[2])
        movimentos_totais.append(a)

    # 4. Resultados
    print("\n==============================================================")
    print(f"Cubo Resolvido! Tempo total: {time.time() - start_time:.2f} segundos")
    print(f"Solução final completa ({len(movimentos_totais)} movimentos):")
    print(movimentos_totais)
    print("\nEstado Final do Cubo:")
    cube.show()

if __name__ == '__main__':
    main()