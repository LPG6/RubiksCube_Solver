import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.optim as optim
import time


# 1. Mapa de cores para números
COLOR_MAP = {'w': 0, 'o': 1, 'g': 2, 'r': 3, 'b': 4, 'y': 5}

# 2. Classe que lê o Dataset (CSV) e transforma em Matrizes (Tensores)
class RubiksDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Carregando dados de {csv_file}...")
        self.data = pd.read_csv(csv_file)
        print(f"Dataset carregado com {len(self.data)} estados únicos!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        state_str = row['state']
        
        # Transformando a string de 54 caracteres em números (0 a 5)
        state_num =[COLOR_MAP[c] for c in state_str]
        
        # --- ONE-HOT ENCODING ---
        # Cria uma matriz de zeros (54 peças, 6 cores)
        one_hot = np.zeros((54, 6), dtype=np.float32)
        # Coloca "1" na coluna correspondente à cor da peça
        one_hot[np.arange(54), state_num] = 1.0 
        
        # Para a MLP: Achatamos tudo num vetor de 324 posições (54 x 6)
        mlp_input = one_hot.flatten()
        
        # Para a CNN: Remodelamos para 6 faces, 3x3, com 6 canais de cores.
        # Depois organizamos para o formato PyTorch:[Canais(36), Altura(3), Largura(3)]
        cnn_input = one_hot.reshape(6, 3, 3, 6) # 6 faces, 3x3 peças, 6 cores
        cnn_input = cnn_input.transpose(0, 3, 1, 2) # Formato PyTorch: (faces, cores, H, W)
        cnn_input = cnn_input.reshape(36, 3, 3) # Junta faces e cores nos Canais -> (36, 3, 3)
        
        # As três distâncias (Labels / Respostas Certas)
        labels = np.array([row['dist_l1'], row['dist_f2l'], row['dist_full']], dtype=np.float32)
        
        return torch.tensor(mlp_input), torch.tensor(cnn_input), torch.tensor(labels)

# 3. Modelo 1: A Rede Densa (MLP Multi-Head)
class RubiksMLP(nn.Module):
    def __init__(self):
        super(RubiksMLP, self).__init__()
        # Tronco cego (só enxerga a tripa de números)
        self.tronco = nn.Sequential(
            nn.Linear(324, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256)
        )
        # As 3 cabeças independentes
        self.head_l1 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_f2l = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_full = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        features = self.tronco(x)
        return self.head_l1(features), self.head_f2l(features), self.head_full(features)

# 4. Modelo 2: A Rede Convolucional (CNN Multi-Head)
class RubiksCNN(nn.Module):
    def __init__(self):
        super(RubiksCNN, self).__init__()
        # Tronco Geométrico: Reduz as faces 3x3 perfeitamente para 1x1
        self.tronco = nn.Sequential(
            # Entrada:[36, 3, 3] -> Saída: [128, 2, 2]
            nn.Conv2d(in_channels=36, out_channels=128, kernel_size=2, padding=0),
            nn.ReLU(), nn.BatchNorm2d(128),
            
            # Entrada: [128, 2, 2] -> Saída:[256, 1, 1]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=0),
            nn.ReLU(), nn.BatchNorm2d(256),
            
            # Achata o[256, 1, 1] para um vetor de 256
            nn.Flatten(),
            
            # Camada Densa final do tronco
            nn.Linear(256, 256), nn.ReLU()
        )
        # As mesmas 3 cabeças independentes
        self.head_l1 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_f2l = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_full = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        features = self.tronco(x)
        return self.head_l1(features), self.head_f2l(features), self.head_full(features)
    

def treinar_modelo(tipo_modelo="MLP", epochs=3, batch_size=1024, lr=0.001):
    # Verifica se tem Placa de Vídeo (NVIDIA CUDA), se não, usa o Processador (CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Preparando treinamento no dispositivo: {device}")

    # Carrega os dados (Pode demorar uns segundinhos porque são 4 milhões de linhas)
    dataset = RubiksDataset('dataset_rubiks.csv')
    
    # O DataLoader quebra os 4 milhões de dados em "pacotinhos" (batches) de 1024
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Escolhe qual "Cérebro" vamos treinar
    if tipo_modelo == "CNN":
        modelo = RubiksCNN().to(device)
    else:
        modelo = RubiksMLP().to(device)

    # Otimizador (O "Professor" que atualiza as sinapses da rede)
    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    
    # Função de Erro (Erro Quadrático Médio)
    criterion = nn.MSELoss() 

    print(f"\n--- Iniciando Treinamento da Rede {tipo_modelo} ---")
    print(f"Total de épocas: {epochs} | Tamanho do Lote: {batch_size}")
    
    inicio_treino = time.time()

    for epoch in range(epochs):
        modelo.train()
        running_loss = 0.0

        for batch_idx, (x_mlp, x_cnn, labels) in enumerate(dataloader):
            # 1. Pega os dados e envia pra memória correta (CPU/GPU)
            inputs = x_cnn if tipo_modelo == "CNN" else x_mlp
            inputs, labels = inputs.to(device), labels.to(device)

            # Zera a memória de aprendizado da rodada anterior
            optimizer.zero_grad()

            # 2. Forward (A Rede chuta a distância)
            out_l1, out_f2l, out_full = modelo(inputs)

            # Separa o gabarito real do CSV. O .unsqueeze(1) ajusta o formato da matriz
            target_l1 = labels[:, 0].unsqueeze(1)
            target_f2l = labels[:, 1].unsqueeze(1)
            target_full = labels[:, 2].unsqueeze(1)

            # 3. Calcula o Erro (Loss) das 3 cabeças e soma tudo!
            loss_l1 = criterion(out_l1, target_l1)
            loss_f2l = criterion(out_f2l, target_f2l)
            loss_full = criterion(out_full, target_full)
            
            loss = loss_l1 + loss_f2l + loss_full

            # 4. Backward (Calcula o ajuste necessário) e Otimização (Aplica o ajuste)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Mostra no painel a cada 500 lotes processados
            if batch_idx % 500 == 0:
                print(f"Época[{epoch+1}/{epochs}] | Lote [{batch_idx}/{len(dataloader)}] | Perda (Erro): {loss.item():.4f}")

        # Resumo da Época
        perda_media = running_loss / len(dataloader)
        print(f"=== Fim da Época {epoch+1} | Perda Média: {perda_media:.4f} ===")

    # Terminou o treino! Salva o "cérebro" em um arquivo .pth
    tempo_total = (time.time() - inicio_treino) / 60
    nome_arquivo = f"modelo_{tipo_modelo.lower()}.pth"
    torch.save(modelo.state_dict(), nome_arquivo)
    print(f"\nTreinamento Finalizado em {tempo_total:.2f} minutos!")
    print(f"Pesos neurais salvos no arquivo '{nome_arquivo}'!")

if __name__ == '__main__':
    # Escolha qual modelo treinar mudando o parâmetro: "MLP" ou "CNN"
    treinar_modelo(tipo_modelo="CNN", epochs=3, batch_size=2048)