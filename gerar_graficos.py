import pandas as pd
import matplotlib.pyplot as plt

def carregar_dados_experimento():
    try:
        df = pd.read_csv('resultados_experimento_1.csv')
        df_mlp = df[df['Modelo'] == 'MLP']
        df_cnn = df[df['Modelo'] == 'CNN']
        return df_mlp, df_cnn
    except FileNotFoundError:
        print("Erro: O arquivo 'resultados_experimento_1.csv' não foi encontrado na pasta.")
        return None, None

def plotar_taxa_sucesso():
    df_mlp, df_cnn = carregar_dados_experimento()
    if df_mlp is None: return

    plt.figure(figsize=(8, 5))

    plt.plot(df_mlp['Profundidade'], df_mlp['Taxa_Sucesso_%'], 
             marker='o', linestyle='--', color='#1f77b4', linewidth=2, markersize=8, label='MLP (Densa)')
    
    plt.plot(df_cnn['Profundidade'], df_cnn['Taxa_Sucesso_%'], 
             marker='s', linestyle='-', color='#d62728', linewidth=2, markersize=8, label='CNN (Convolucional)')

    plt.title('Taxa de Sucesso vs. Complexidade de Embaralhamento', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Profundidade (Nº de Movimentos Iniciais)', fontsize=12, fontweight='bold')
    plt.ylabel('Taxa de Sucesso de Resolução (%)', fontsize=12, fontweight='bold')

    plt.ylim(-5, 105)
    plt.xticks([1, 2, 3, 5, 8, 10, 12, 15, 18, 20])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('grafico_sucesso.png', dpi=300)
    print("Gráfico de Sucesso gerado ('grafico_sucesso.png')!")
    plt.close()

def plotar_excesso_movimentos():
    df_mlp, df_cnn = carregar_dados_experimento()
    if df_mlp is None: return

    plt.figure(figsize=(8, 5))

    # MLP: Linha + Banda Sombreada de Desvio Padrão
    plt.plot(df_mlp['Profundidade'], df_mlp['Excesso_Medio_Movs'], 
             marker='o', linestyle='--', color='#1f77b4', linewidth=2, markersize=8, label='MLP (Média)')
    plt.fill_between(df_mlp['Profundidade'], 
                     df_mlp['Excesso_Medio_Movs'] - df_mlp['Excesso_Std_Movs'], 
                     df_mlp['Excesso_Medio_Movs'] + df_mlp['Excesso_Std_Movs'], 
                     color='#1f77b4', alpha=0.15, label='MLP (± Desvio Padrão)')
    
    # CNN: Linha + Banda Sombreada de Desvio Padrão
    plt.plot(df_cnn['Profundidade'], df_cnn['Excesso_Medio_Movs'], 
             marker='s', linestyle='-', color='#d62728', linewidth=2, markersize=8, label='CNN (Média)')
    plt.fill_between(df_cnn['Profundidade'], 
                     df_cnn['Excesso_Medio_Movs'] - df_cnn['Excesso_Std_Movs'], 
                     df_cnn['Excesso_Medio_Movs'] + df_cnn['Excesso_Std_Movs'], 
                     color='#d62728', alpha=0.15, label='CNN (± Desvio Padrão)')

    # Linha do ZERO (Marca a diferença entre Excesso e Atalho)
    plt.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.8)
    
    # Preenchendo a área de "Atalhos" (abaixo de zero)
    # Como o desvio padrão vai até o infinito, ampliamos a margem do axhspan
    plt.axhspan(-40, 0, facecolor='green', alpha=0.05, label='Zona de Atalhos (< 0)')

    plt.title('Sub-otimalidade e Variabilidade: Excesso de Movimentos', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Profundidade (Nº de Movimentos Iniciais)', fontsize=12, fontweight='bold')
    plt.ylabel('Excesso de Movimentos (μ ± σ)', fontsize=12, fontweight='bold')

    plt.xticks([1, 2, 3, 5, 8, 10, 12, 15, 18, 20])
    
    # Ajustando o limite Y dinamicamente para mostrar a loucura do desvio padrão da MLP
    plt.ylim(-15, 35)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Ajustando a legenda para caber as áreas sombreadas
    plt.legend(fontsize=9, loc='upper left', framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig('grafico_excesso.png', dpi=300)
    print("Gráfico de Excesso gerado ('grafico_excesso.png')!")
    plt.close()

def plotar_tempo_inferencia():
    df_mlp, df_cnn = carregar_dados_experimento()
    if df_mlp is None: return

    plt.figure(figsize=(8, 5))

    # MLP: Tempo + Desvio Padrão
    plt.plot(df_mlp['Profundidade'], df_mlp['Tempo_Medio_s'], 
             marker='o', linestyle='--', color='#1f77b4', linewidth=2, markersize=8, label='MLP')
    plt.fill_between(df_mlp['Profundidade'], 
                     df_mlp['Tempo_Medio_s'] - df_mlp['Tempo_Std_s'], 
                     df_mlp['Tempo_Medio_s'] + df_mlp['Tempo_Std_s'], 
                     color='#1f77b4', alpha=0.15)
    
    # CNN: Tempo + Desvio Padrão
    plt.plot(df_cnn['Profundidade'], df_cnn['Tempo_Medio_s'], 
             marker='s', linestyle='-', color='#d62728', linewidth=2, markersize=8, label='CNN')
    plt.fill_between(df_cnn['Profundidade'], 
                     df_cnn['Tempo_Medio_s'] - df_cnn['Tempo_Std_s'], 
                     df_cnn['Tempo_Medio_s'] + df_cnn['Tempo_Std_s'], 
                     color='#d62728', alpha=0.15)

    plt.title('Estabilidade de Execução: Tempo de Inferência', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Profundidade (Nº de Movimentos Iniciais)', fontsize=12, fontweight='bold')
    plt.ylabel('Tempo de Resolução (segundos)', fontsize=12, fontweight='bold')

    plt.xticks([1, 2, 3, 5, 8, 10, 12, 15, 18, 20])
    plt.ylim(0, 0.8) # Limite para focar nos milissegundos
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    plt.savefig('grafico_tempo.png', dpi=300)
    print("Gráfico de Tempo gerado ('grafico_tempo.png')!")
    plt.close()

def plotar_curva_aprendizado_real():
    try:
        df = pd.read_csv('historico_treinamento.csv')
    except FileNotFoundError:
        print("Erro: 'historico_treinamento.csv' não encontrado.")
        return

    df_mlp = df[df['Modelo'] == 'MLP']
    df_cnn = df[df['Modelo'] == 'CNN']

    plt.figure(figsize=(8, 5))

    # O Pulo do Gato: Ignorar a Época 0 (Erro inicial > 500)
    # Pegamos apenas onde a 'Epoca' é maior ou igual a 1
    # Note que se o seu CSV salvou as epocas de 1 a 10 direto, basta plotar!
    
    plt.plot(df_mlp['Epoca'], df_mlp['MSE'], marker='o', linestyle='--', 
             color='#1f77b4', linewidth=2, markersize=6, label='MLP (Densa)')
    
    plt.plot(df_cnn['Epoca'], df_cnn['MSE'], marker='s', linestyle='-', 
             color='#d62728', linewidth=2, markersize=6, label='CNN (Convolucional)')

    # Marcação da Zona de Estabilização (Ponto Ótimo de Parada)
    plt.axvline(x=3, color='gray', linestyle=':', linewidth=2, alpha=0.8)
    plt.text(3.15, df_cnn['MSE'].max() - 1, 'Início dos Retornos Decrescentes', 
             color='#333333', fontsize=10, fontweight='bold', ha='left')

    plt.title('Curva de Aprendizado e Estabilização Assintótica (10 Épocas)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Épocas de Treinamento', fontsize=12, fontweight='bold')
    plt.ylabel('Erro Quadrático Médio (MSE)', fontsize=12, fontweight='bold')

    plt.xticks(range(1, 11)) # Eixo X de 1 a 10
    
    # Ajuste dinâmico do eixo Y para focar na variação sutil
    plt.ylim(15, 36)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig('grafico_aprendizado.png', dpi=300)
    print("Gráfico da Curva de Aprendizado Real gerado (sem o salto inicial)!")
    plt.show()

if __name__ == '__main__':
    plotar_taxa_sucesso()
    plotar_excesso_movimentos() 
    plotar_tempo_inferencia() # <--- NOVO GRÁFICO GERADO AQUI!
    plotar_curva_aprendizado_real()