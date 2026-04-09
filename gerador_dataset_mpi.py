import csv
import time
from random import choice, randint
from mpi4py import MPI
from tqdm import tqdm
from cube import RubiksCube

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- CONFIGURAÇÕES DA GERAÇÃO ---
    EPISODES = 20_000   # Quantos cubos cada núcleo vai resolver/embaralhar
    MAX_MOVES = 20      # Profundidade máxima de embaralhamento 
    CSV_FILE = 'dataset_rubiks.csv'
    # --------------------------------

    actions =[(r, n, d) for r in ['h', 'v', 's'] for d in[0, 1] for n in range(3)]
    
    
    local_data = {}

    if rank == 0:
        print(f"Iniciando MPI com {size} processos.")
        print(f"Meta: {EPISODES * size} episodios totais. Ate {MAX_MOVES} de profundidade.")
        start_time = time.time()
        pbar = tqdm(total=EPISODES, desc='Progresso (Apenas Rank 0)')

    # Cada núcleo Roda seus episódios independentemente
    for _ in range(EPISODES):
        cube = RubiksCube(n=3)
        dist_l1, dist_f2l, dist_full = 0, 0, 0
        
        # Embaralhamento de até MAX_MOVES
        for _ in range(MAX_MOVES):
            # Aplica movimento aleatório
            a = choice(actions)
            j = randint(0, cube.n - 1)
            
            if a[0] == 'h': cube.horizontal_twist(a[1], a[2])
            elif a[0] == 'v': cube.vertical_twist(a[1], a[2])
            elif a[0] == 's': cube.side_twist(a[1], a[2])

            
            # Se a camada estiver resolvida, distância reseta para 0. Se quebrou, soma 1.
            dist_l1 = 0 if cube.solved_l1() else dist_l1 + 1
            dist_f2l = 0 if cube.solved_f2l() else dist_f2l + 1
            dist_full = 0 if cube.solved() else dist_full + 1
            
            state = cube.stringify()
            
            # Salva sempre a MENOR distância encontrada para aquele estado
            if state in local_data:
                old_l1, old_f2l, old_full = local_data[state]
                local_data[state] = (min(old_l1, dist_l1), min(old_f2l, dist_f2l), min(old_full, dist_full))
            else:
                local_data[state] = (dist_l1, dist_f2l, dist_full)
                
        if rank == 0:
            pbar.update(1)

    if rank == 0:
        pbar.close()
        print("\nProcessamento dos nos finalizado. Juntando os dados no No Mestre...")

    # O Mestre puxa os dicionários de todos os Ranks
    gathered_data = comm.gather(local_data, root=0)

    
    if rank == 0:
        global_data = {}
        print("Consolidando dicionario global e removendo duplicatas...")
        for data in gathered_data:
            for state, (l1, f2l, full) in data.items():
                if state in global_data:
                    old_l1, old_f2l, old_full = global_data[state]
                    global_data[state] = (min(old_l1, l1), min(old_f2l, f2l), min(old_full, full))
                else:
                    global_data[state] = (l1, f2l, full)

        print(f"Total de estados unicos gerados: {len(global_data)}")
        
        print(f"Escrevendo arquivo {CSV_FILE}...")
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['state', 'dist_l1', 'dist_f2l', 'dist_full']) # Cabeçalho
            
            for state, (l1, f2l, full) in global_data.items():
                writer.writerow([state, l1, f2l, full])
                
        print(f"Finalizado em {time.time() - start_time:.2f} segundos!")

if __name__ == '__main__':
    main() 