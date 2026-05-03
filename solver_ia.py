import torch
import numpy as np
from random import choice
from cube import RubiksCube
from treinar_modelos import RubiksMLP, RubiksCNN, COLOR_MAP

class NeuralHeuristic:
    def __init__(self, model_path, tipo_modelo="MLP"):
        self.tipo_modelo = tipo_modelo
        self.device = torch.device('cpu') # Usamos CPU para inferência rápida de um único cubo
        
        if tipo_modelo == "CNN":
            self.model = RubiksCNN()
        else:
            self.model = RubiksMLP()
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Coloca a rede em modo de "Adivinhação" (Inferência)

    def predict(self, cube_str, target_phase):
        """Transforma a string na hora e pergunta a distância para a Rede Neural"""
        state_num =[COLOR_MAP[c] for c in cube_str]
        one_hot = np.zeros((54, 6), dtype=np.float32)
        one_hot[np.arange(54), state_num] = 1.0

        if self.tipo_modelo == "CNN":
            cnn_input = one_hot.reshape(6, 3, 3, 6).transpose(0, 3, 1, 2).reshape(36, 3, 3)
            inputs = torch.tensor(cnn_input).unsqueeze(0) # Adiciona a dimensão do batch (1 cubo)
        else:
            mlp_input = one_hot.flatten()
            inputs = torch.tensor(mlp_input).unsqueeze(0)

        with torch.no_grad(): # Desliga o aprendizado para ficar ultra rápido
            out_l1, out_f2l, out_full = self.model(inputs)

        # Escolhe qual "Cabeça" usar baseada no objetivo atual
        if target_phase == 'L1':
            return out_l1.item()
        elif target_phase == 'F2L':
            return out_f2l.item()
        else:
            return out_full.item()

class BuscaGulosaNeural(object):
    def __init__(self, neural_heuristic, max_moves=20):
        self.neural_h = neural_heuristic
        self.max_moves = max_moves # Evita loop infinito. Se não achar em 20 passos, desiste.

    def run(self, state, target_phase):
        cube = RubiksCube(state=state)
        moves =[]
        
        # O algoritmo Guloso: Sempre escolhe o melhor passo imediato!
        for _ in range(self.max_moves):
            # Verifica se já chegou no objetivo
            if target_phase == 'L1' and cube.solved_l1(): break
            if target_phase == 'F2L' and cube.solved_f2l(): break
            if target_phase == 'FULL' and cube.solved(): break
            
            melhor_movimento = None
            menor_h = float('inf')
            
            actions =[(r, n, d) for r in ['h', 'v', 's'] for d in [0, 1] for n in range(cube.n)]
            
            # Testa todos os 18 movimentos possíveis mentalmente
            for a in actions:
                temp_cube = RubiksCube(state=cube.stringify())
                if a[0] == 'h': temp_cube.horizontal_twist(a[1], a[2])
                elif a[0] == 'v': temp_cube.vertical_twist(a[1], a[2])
                elif a[0] == 's': temp_cube.side_twist(a[1], a[2])
                
                # Se o movimento matar o desafio de cara, escolhe ele!
                if target_phase == 'L1' and temp_cube.solved_l1(): return moves + [a]
                if target_phase == 'F2L' and temp_cube.solved_f2l(): return moves + [a]
                if target_phase == 'FULL' and temp_cube.solved(): return moves +[a]
                
                # Pergunta a nota para a Inteligência Artificial
                h_score = self.neural_h.predict(temp_cube.stringify(), target_phase)
                
                # Guarda o movimento que tem a menor distância
                if h_score < menor_h:
                    menor_h = h_score
                    melhor_movimento = a
            
            # Executa fisicamente o melhor movimento e vai para o próximo passo
            if melhor_movimento:
                moves.append(melhor_movimento)
                if melhor_movimento[0] == 'h': cube.horizontal_twist(melhor_movimento[1], melhor_movimento[2])
                elif melhor_movimento[0] == 'v': cube.vertical_twist(melhor_movimento[1], melhor_movimento[2])
                elif melhor_movimento[0] == 's': cube.side_twist(melhor_movimento[1], melhor_movimento[2])
            else:
                break
                
        return moves