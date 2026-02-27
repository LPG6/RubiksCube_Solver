import json
import time
from mpi4py import MPI
from cube import RubiksCube
from tqdm import tqdm

def chunk_list(lst, n_chunks):
    """Divide uma lista em 'n_chunks' pedaços mais ou menos iguais."""
    k, m = divmod(len(lst), n_chunks)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n_chunks)]

def expand_states(states_chunk, actions):
    """Dado um pedaço de estados, aplica todos os movimentos e retorna os novos estados."""
    new_states = set()
    for state_str in states_chunk:
        for a in actions:
            # Instancia o cubo no estado atual
            cube = RubiksCube(state=state_str)
            
            # Aplica o movimento
            if a[0] == 'h':
                cube.horizontal_twist(a[1], a[2])
            elif a[0] == 'v':
                cube.vertical_twist(a[1], a[2])
            elif a[0] == 's':
                cube.side_twist(a[1], a[2])
            
            # Adiciona o novo estado gerado
            new_states.add(cube.stringify())
            
    return new_states

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Configurações
    MAX_MOVES = 6 # Tente aumentar para 6 ou 7 agora que é paralelo!
    HEURISTIC_FILE = 'heuristic_mpi.json'
    
    # Gera o estado resolvido para iniciar (n=3 padrão)
    solved_cube = RubiksCube(n=3)
    initial_state = solved_cube.stringify()

    # Gera a lista de todas as ações possíveis (18 ações para um cubo 3x3)
    actions =[(r, n, d) for r in ['h', 'v', 's'] for d in[0, 1] for n in range(3)]

    # Variáveis globais (apenas o Mestre vai mantê-las completas)
    global_heuristic = {}
    frontier =[]

    if rank == 0:
        print(f"Iniciando MPI com {size} processos.")
        print(f"Construindo heurística até a profundidade: {MAX_MOVES}")
        global_heuristic[initial_state] = 0
        frontier = [initial_state]
        start_time = time.time()
        pbar = tqdm(total=MAX_MOVES, desc='Progresso por Profundidade')

    # Loop pelos níveis de profundidade (BFS)
    for depth in range(MAX_MOVES):
        if rank == 0:
            print(f"--> Nível {depth} | Estados na fronteira: {len(frontier)}")
            pbar.set_postfix({"Fronteira": len(frontier)})
            # Divide a fronteira pelo número de processos
            chunks = chunk_list(frontier, size)
        else:
            chunks = None

        # Espalha (Scatter) os pedaços para todos os Ranks
        local_chunk = comm.scatter(chunks, root=0)

        # Cada processo expande sua parte da fronteira
        local_new_states = expand_states(local_chunk, actions)

        # Junta (Gather) todos os novos estados de volta no Mestre
        gathered_states = comm.gather(list(local_new_states), root=0)

        # Mestre atualiza o dicionário global e cria a próxima fronteira
        if rank == 0:
            frontier =[]
            # gathered_states é uma lista de listas
            for process_states in gathered_states:
                for state in process_states:
                    # Se o estado for novo, adiciona à heurística e à próxima fronteira
                    if state not in global_heuristic:
                        global_heuristic[state] = depth + 1
                        frontier.append(state)
            pbar.update(1)

    # Ao final, apenas o Nó Mestre salva o arquivo JSON
    if rank == 0:
        pbar.close()
        print(f"Concluído! Total de estados gerados: {len(global_heuristic)}")
        print(f"Tempo total: {time.time() - start_time:.2f} segundos.")
        
        with open(HEURISTIC_FILE, 'w', encoding='utf-8') as f:
            json.dump(global_heuristic, f, ensure_ascii=False, indent=4)
        print(f"Heurística salva em {HEURISTIC_FILE}.")

if __name__ == '__main__':
    main()