import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do seaborn para plotagens
sns.set(style="whitegrid")

# Função de ativação linear para o ADALINE
def ativacao_linear(x):
    return x  # Retorna o valor de entrada como está (sem alteração)

# Função para treinar o ADALINE
def treinar_adaline(X, y, epocas=1000, taxa_aprendizado=.1):
    pesos = np.random.randn(X.shape[1])
    eqm_por_epoca = []  # Lista para armazenar o erro quadrático médio (EQM) por época
    for _ in range(epocas):
        eqm = 0
        for i in range(X.shape[0]):
            y_pred = ativacao_linear(np.dot(X[i], pesos))
            erro = y[i] - y_pred
            pesos += taxa_aprendizado * erro * X[i]
            eqm += erro ** 2
        eqm_por_epoca.append(eqm / X.shape[0])
    return pesos, eqm_por_epoca

# Função para calcular métricas de desempenho
def calcular_metricas(pesos, X_teste, y_teste):
    y_pred = np.array([ativacao_linear(np.dot(X_teste[i], pesos)) for i in range(X_teste.shape[0])])
    # Convertendo para valores binários com um limiar de 0 (para calcular a acurácia, sensibilidade e especificidade)
    y_pred_binario = np.where(y_pred > 0, 1, -1)
    acuracia = np.mean(y_pred_binario == y_teste)
    sensibilidade = np.sum((y_pred_binario == 1) & (y_teste == 1)) / np.sum(y_teste == 1) if np.sum(y_teste == 1) > 0 else 0
    especificidade = np.sum((y_pred_binario == -1) & (y_teste == -1)) / np.sum(y_teste == -1) if np.sum(y_teste == -1) > 0 else 0
    return acuracia, sensibilidade, especificidade, y_pred_binario

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

# Normalizar os dados
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)

# Adicionar a coluna de bias (uma coluna de 1s)
X_bias = np.hstack((X_normalized, np.ones((X_normalized.shape[0], 1))))

# Parâmetros de simulação
num_simulacoes = 500
epocas = 1000
taxa_aprendizado = .1

# Tempo de início
start_time = time.time()

# Listas para armazenar as métricas de todas as rodadas
acuracias = []
sensibilidades = []
especificidades = []
eqm_histories = []
predicoes_teste = []

# Simulação de Monte Carlo
for i in range(num_simulacoes):
    # Imprime o progresso da simulação atual
    print(f"Rodada {i}/{num_simulacoes} em andamento...")

    # Particionamento dos dados em 80% treinamento e 20% teste
    indices = np.arange(X_bias.shape[0])
    np.random.shuffle(indices)
    split_index = int(0.8 * X_bias.shape[0])
    X_treino, X_teste = X_bias[indices[:split_index]], X_bias[indices[split_index:]]
    y_treino, y_teste = y[indices[:split_index]], y[indices[split_index:]]

    # Treinamento do ADALINE
    pesos, eqm_por_epoca = treinar_adaline(X_treino, y_treino, epocas, taxa_aprendizado)

    # Cálculo das métricas de desempenho
    acuracia, sensibilidade, especificidade, y_pred_binario = calcular_metricas(pesos, X_teste, y_teste)

    # Armazenando as métricas
    acuracias.append(acuracia)
    sensibilidades.append(sensibilidade)
    especificidades.append(especificidade)
    eqm_histories.append(eqm_por_epoca)
    predicoes_teste.append((y_pred_binario, y_teste))

    # Imprime as métricas de cada rodada
    if i % 100 == 0 or i == num_simulacoes - 1:
        print(f"Rodada {i}: Acurácia={acuracia:.2f}, Sensibilidade={sensibilidade:.2f}, Especificidade={especificidade:.2f}")

# Tempo total de execução
end_time = time.time()
print(f"Execução total concluída em {end_time - start_time:.2f} segundos")

# Cálculo das estatísticas
media_acuracia = np.mean(acuracias)
desvio_padrao_acuracia = np.std(acuracias)
maximo_acuracia = np.max(acuracias)
minimo_acuracia = np.min(acuracias)

media_sensibilidade = np.mean(sensibilidades)
desvio_padrao_sensibilidade = np.std(sensibilidades)
maximo_sensibilidade = np.max(sensibilidades)
minimo_sensibilidade = np.min(sensibilidades)

media_especificidade = np.mean(especificidades)
desvio_padrao_especificidade = np.std(especificidades)
maximo_especificidade = np.max(especificidades)
minimo_especificidade = np.min(especificidades)

# Impressão das estatísticas
print("\nEstatísticas de desempenho após 500 rodadas de Monte Carlo:")
print(f"Acurácia - Média: {media_acuracia:.2f}, Desvio Padrão: {desvio_padrao_acuracia:.2f}, Máximo: {maximo_acuracia:.2f}, Mínimo: {minimo_acuracia:.2f}")
print(f"Sensibilidade - Média: {media_sensibilidade:.2f}, Desvio Padrão: {desvio_padrao_sensibilidade:.2f}, Máximo: {maximo_sensibilidade:.2f}, Mínimo: {minimo_sensibilidade:.2f}")
print(f"Especificidade - Média: {media_especificidade:.2f}, Desvio Padrão: {desvio_padrao_especificidade:.2f}, Máximo: {maximo_especificidade:.2f}, Mínimo: {minimo_especificidade:.2f}")

# Plotando a curva de aprendizado para maior e menor EQM
indice_max_acuracia = np.argmax(acuracias)
indice_min_acuracia = np.argmin(acuracias)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(eqm_histories[indice_max_acuracia], label="Maior Acurácia")
plt.title("Curva de Aprendizado - Maior Acurácia")
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(eqm_histories[indice_min_acuracia], label="Menor Acurácia")
plt.title("Curva de Aprendizado - Menor Acurácia")
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.legend()

plt.tight_layout()
plt.show()

# Plotando o gráfico de dispersão com os dados normalizados
plt.figure(figsize=(8, 5))
plt.scatter(X_normalized[y == 1][:, 0], X_normalized[y == 1][:, 1], label="Classe 1", alpha=0.7)
plt.scatter(X_normalized[y == -1][:, 0], X_normalized[y == -1][:, 1], label="Classe -1", alpha=0.7)
plt.title("Gráfico de Dispersão com Dados Normalizados")
plt.xlabel("Característica 1 (normalizada)")
plt.ylabel("Característica 2 (normalizada)")
plt.legend()
plt.show()

# Plotando a matriz de confusão para maior e menor acurácia
y_pred_max, y_teste_max = predicoes_teste[indice_max_acuracia]
y_pred_min, y_teste_min = predicoes_teste[indice_min_acuracia]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
matriz_max = matriz_de_confusao(y_teste_max, y_pred_max)
sns.heatmap(matriz_max, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe -1", "Classe 1"], yticklabels=["Classe -1", "Classe 1"])
plt.title(f"Matriz de Confusão - Maior Acurácia ({maximo_acuracia:.2f})")
plt.xlabel("Predito")
plt.ylabel("Real")

plt.subplot(1, 2, 2)
matriz_min = matriz_de_confusao(y_teste_min, y_pred_min)
sns.heatmap(matriz_min, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe -1", "Classe 1"], yticklabels=["Classe -1", "Classe 1"])
plt.title(f"Matriz de Confusão - Menor Acurácia ({minimo_acuracia:.2f})")
plt.xlabel("Predito")
plt.ylabel("Real")

plt.tight_layout()
plt.show()
