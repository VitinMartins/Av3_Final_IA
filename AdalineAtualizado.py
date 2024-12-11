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
    return acuracia, sensibilidade, especificidade

# Carregar os dados
data_np = np.loadtxt('spiral.csv', delimiter=',')
X = data_np[:, :-1]  # Características
y = data_np[:, -1]   # Rótulos

# Normalizar os dados
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)

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

# Simulação de Monte Carlo
for i in range(num_simulacoes):
    # Imprime o progresso da simulação atual
    print(f"Rodada {i}/{num_simulacoes} em andamento...")

    # Particionamento dos dados em 80% treinamento e 20% teste
    indices = np.arange(X_normalized.shape[0])
    np.random.shuffle(indices)
    split_index = int(0.8 * X_normalized.shape[0])
    X_treino, X_teste = X_normalized[indices[:split_index]], X_normalized[indices[split_index:]]
    y_treino, y_teste = y[indices[:split_index]], y[indices[split_index:]]

    # Treinamento do ADALINE
    pesos, eqm_por_epoca = treinar_adaline(X_treino, y_treino, epocas, taxa_aprendizado)

    # Cálculo das métricas de desempenho
    acuracia, sensibilidade, especificidade = calcular_metricas(pesos, X_teste, y_teste)

    # Armazenando as métricas
    acuracias.append(acuracia)
    sensibilidades.append(sensibilidade)
    especificidades.append(especificidade)
    eqm_histories.append(eqm_por_epoca)

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

# Plotando boxplots e violin plots para cada métrica
fig, axes = plt.subplots(3, 1, figsize=(14, 18))

# Boxplot de acurácia
sns.boxplot(data=acuracias, ax=axes[0])
axes[0].set_title('Boxplot de Acurácia')
axes[0].set_ylabel('Acurácia')

# Boxplot de sensibilidade
sns.boxplot(data=sensibilidades, ax=axes[1])
axes[1].set_title('Boxplot de Sensibilidade')
axes[1].set_ylabel('Sensibilidade')

# Boxplot de especificidade
sns.boxplot(data=especificidades, ax=axes[2])
axes[2].set_title('Boxplot de Especificidade')
axes[2].set_ylabel('Especificidade')

plt.tight_layout()
plt.show()

# Plotando violin plots para cada métrica
fig, axes = plt.subplots(3, 1, figsize=(14, 18))

# Violin plot de acurácia
sns.violinplot(data=acuracias, ax=axes[0])
axes[0].set_title('Violin Plot de Acurácia')
axes[0].set_ylabel('Acurácia')

# Violin plot de sensibilidade
sns.violinplot(data=sensibilidades, ax=axes[1])
axes[1].set_title('Violin Plot de Sensibilidade')
axes[1].set_ylabel('Sensibilidade')

# Violin plot de especificidade
sns.violinplot(data=especificidades, ax=axes[2])
axes[2].set_title('Violin Plot de Especificidade')
axes[2].set_ylabel('Especificidade')

plt.tight_layout()
plt.show()

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
