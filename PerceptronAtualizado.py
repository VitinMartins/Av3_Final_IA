import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Função de ativação de degrau bipolar
def ativacao_degrau_bipolar(x):
    return 1 if x > 0 else -1

# Função para treinar o Perceptron com bias
def treinar_perceptron(X, y, epocas=1000, taxa_aprendizado=0.1):
    # Inicialização dos pesos (incluindo o peso do bias)
    pesos = np.random.random_sample(X.shape[1]) - 1
    for _ in range(epocas):
        for i in range(X.shape[0]):
            # Cálculo da predição (incluindo o bias)
            y_pred = ativacao_degrau_bipolar(np.dot(X[i], pesos))
            erro = y[i] - y_pred
            # Atualização dos pesos (incluindo o bias)
            pesos += taxa_aprendizado * erro * X[i]
    return pesos

# Função para calcular métricas de desempenho
def calcular_metricas(pesos, X_teste, y_teste):
    y_pred = np.array([ativacao_degrau_bipolar(np.dot(X_teste[i], pesos)) for i in range(X_teste.shape[0])])
    acuracia = np.mean(y_pred == y_teste)
    sensibilidade = np.sum((y_pred == 1) & (y_teste == 1)) / np.sum(y_teste == 1) if np.sum(y_teste == 1) > 0 else 0
    especificidade = np.sum((y_pred == -1) & (y_teste == -1)) / np.sum(y_teste == -1) if np.sum(y_teste == -1) > 0 else 0
    return acuracia, sensibilidade, especificidade, y_pred

# Função para criar a matriz de confusão
def matriz_de_confusao(y_real, y_pred):
    TP = np.sum((y_real == 1) & (y_pred == 1))
    TN = np.sum((y_real == -1) & (y_pred == -1))
    FP = np.sum((y_real == -1) & (y_pred == 1))
    FN = np.sum((y_real == 1) & (y_pred == -1))
    return np.array([[TP, FP], [FN, TN]])

# Carregar os dados
data_np = np.loadtxt('spiral.csv', delimiter=',')
X = data_np[:, :-1]  # Características      
y = data_np[:, -1]   # Rótulos

# Adicionar a coluna de bias (uma coluna de 1s)
X = np.hstack((X, np.ones((X.shape[0], 1))))

# Normalizar os dados (sem afetar a coluna de bias)
X_min = X[:, :-1].min(axis=0)
X_max = X[:, :-1].max(axis=0)
X[:, :-1] = (X[:, :-1] - X_min) / (X_max - X_min)

# Parâmetros de simulação
num_simulacoes = 500
epocas = 1000
taxa_aprendizado = 0.1

# Tempo de início
start_time = time.time()

# Listas para armazenar as métricas de todas as rodadas
acuracias = []
sensibilidades = []
especificidades = []
modelos = []

# Simulação de Monte Carlo
for i in range(num_simulacoes):
    # Imprime o progresso da simulação atual
    print(f"Rodada {i + 1}/{num_simulacoes} em andamento...")

    # Particionamento dos dados em 80% treinamento e 20% teste
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index = int(0.8 * X.shape[0])
    X_treino, X_teste = X[indices[:split_index]], X[indices[split_index:]]
    y_treino, y_teste = y[indices[:split_index]], y[indices[split_index:]]

    # Treinamento do perceptron
    pesos = treinar_perceptron(X_treino, y_treino, epocas, taxa_aprendizado)

    # Cálculo das métricas de desempenho
    acuracia, sensibilidade, especificidade, y_pred = calcular_metricas(pesos, X_teste, y_teste)

    # Armazenando as métricas
    acuracias.append(acuracia)
    sensibilidades.append(sensibilidade)
    especificidades.append(especificidade)
    modelos.append((pesos, y_pred))

    # Imprime as métricas de cada rodada
    if i % 100 == 0 or i == num_simulacoes - 1:
        print(f"Rodada {i + 1}: Acurácia={acuracia:.2f}, Sensibilidade={sensibilidade:.2f}, Especificidade={especificidade:.2f}")

# Tempo total de execução
end_time = time.time()
print(f"Execução total concluída em {end_time - start_time:.2f} segundos")

# Encontrando as rodadas com maior e menor acurácia
indice_max_acuracia = np.argmax(acuracias)
indice_min_acuracia = np.argmin(acuracias)

# Matriz de confusão para maior e menor acurácia
matriz_max_acuracia = matriz_de_confusao(y[indices[split_index:]], modelos[indice_max_acuracia][1])
matriz_min_acuracia = matriz_de_confusao(y[indices[split_index:]], modelos[indice_min_acuracia][1])

# Plotando a matriz de confusão com Seaborn
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(matriz_max_acuracia, annot=True, cmap='Blues', fmt='d')
plt.title(f'Matriz de Confusão - Maior Acurácia ({acuracias[indice_max_acuracia]:.2f})')

plt.subplot(1, 2, 2)
sns.heatmap(matriz_min_acuracia, annot=True, cmap='Reds', fmt='d')
plt.title(f'Matriz de Confusão - Menor Acurácia ({acuracias[indice_min_acuracia]:.2f})')

plt.tight_layout()
plt.show()

# Estatísticas de desempenho
print("\nEstatísticas de desempenho após 500 rodadas de Monte Carlo:")
print("Acurácia")
print(f"Média: {np.mean(acuracias):.2f}, Desvio Padrão: {np.std(acuracias):.2f}, Máximo: {np.max(acuracias):.2f}, Mínimo: {np.min(acuracias):.2f}")
print("Sensibilidade")
print(f"Média: {np.mean(sensibilidades):.2f}, Desvio Padrão: {np.std(sensibilidades):.2f}, Máximo: {np.max(sensibilidades):.2f}, Mínimo: {np.min(sensibilidades):.2f}")
print("Especificidade")
print(f"Média: {np.mean(especificidades):.2f}, Desvio Padrão: {np.std(especificidades):.2f}, Máximo: {np.max(especificidades):.2f}, Mínimo: {np.min(especificidades):.2f}")

# Visualização dos resultados (boxplots)
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.boxplot(acuracias, vert=False)
plt.title('Boxplot da Acurácia')

plt.subplot(1, 3, 2)
plt.boxplot(sensibilidades, vert=False)
plt.title('Boxplot da Sensibilidade')

plt.subplot(1, 3, 3)
plt.boxplot(especificidades, vert=False)
plt.title('Boxplot da Especificidade')

plt.tight_layout()
plt.show()
