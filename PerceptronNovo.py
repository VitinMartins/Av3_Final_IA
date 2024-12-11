import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Função de ativação de degrau bipolar
def ativacao_degrau_bipolar(x):
    return 1 if x > 0 else -1

# Função para treinar o Perceptron e calcular a curva de aprendizado (acurácia por época)
def treinar_perceptron_com_curva(X, y, X_teste, y_teste, epocas=1000, taxa_aprendizado=0.1):
    pesos = np.random.random_sample(X.shape[1]) - 1
    acuracias_por_epoca = []  # Para registrar a acurácia em cada época

    for _ in range(epocas):
        for i in range(X.shape[0]):
            y_pred = ativacao_degrau_bipolar(np.dot(X[i], pesos))
            erro = y[i] - y_pred
            pesos += taxa_aprendizado * erro * X[i]

        # Cálculo da acurácia no conjunto de teste após cada época
        y_pred_teste = np.array([ativacao_degrau_bipolar(np.dot(X_teste[i], pesos)) for i in range(X_teste.shape[0])])
        acuracia = np.mean(y_pred_teste == y_teste)
        acuracias_por_epoca.append(acuracia)

    return pesos, acuracias_por_epoca

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

# Função para plotar o gráfico de dispersão com os dados normalizados
def plotar_dados_normalizados(X, y):
    plt.figure(figsize=(8, 5))
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Classe 1", alpha=0.7)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], label="Classe -1", alpha=0.7)
    plt.title("Gráfico de Dispersão com Dados Normalizados")
    plt.xlabel("Característica 1 (normalizada)")
    plt.ylabel("Característica 2 (normalizada)")
    plt.legend()
    plt.show()

# Função para plotar a matriz de confusão
def plotar_matriz_confusao(y_real, y_pred, titulo):
    matriz = matriz_de_confusao(y_real, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe -1", "Classe 1"], yticklabels=["Classe -1", "Classe 1"])
    plt.title(titulo)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()

# Função para exibir as estatísticas de métricas de desempenho
def exibir_estatisticas(metricas, nome_metrica):
    media = np.mean(metricas)
    desvio_padrao = np.std(metricas)
    valor_maximo = np.max(metricas)
    valor_minimo = np.min(metricas)

    print(f"\nEstatísticas para {nome_metrica}:")
    print(f"Média: {media:.4f}")
    print(f"Desvio Padrão: {desvio_padrao:.4f}")
    print(f"Maior Valor: {valor_maximo:.4f}")
    print(f"Menor Valor: {valor_minimo:.4f}")

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

# Listas para armazenar os resultados de cada simulação
acuracias = []
sensibilidades = []
especificidades = []
curvas_acuracia = []
predicoes_teste = []

# Simulação de Monte Carlo
for i in range(num_simulacoes):
    print(f"Rodada {i + 1}/{num_simulacoes} em andamento...")  # Mensagem de progresso
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index = int(0.8 * X.shape[0])
    X_treino, X_teste = X[indices[:split_index]], X[indices[split_index:]]
    y_treino, y_teste = y[indices[:split_index]], y[indices[split_index:]]

    # Treinamento do Perceptron com curva de aprendizado
    pesos, acuracias_por_epoca = treinar_perceptron_com_curva(X_treino, y_treino, X_teste, y_teste, epocas, taxa_aprendizado)

    # Cálculo das métricas finais
    acuracia, sensibilidade, especificidade, y_pred = calcular_metricas(pesos, X_teste, y_teste)

    # Armazenar métricas, curva de aprendizado e predições
    acuracias.append(acuracia)
    sensibilidades.append(sensibilidade)
    especificidades.append(especificidade)
    curvas_acuracia.append(acuracias_por_epoca)
    predicoes_teste.append((y_pred, y_teste))

# Exibir estatísticas para cada métrica
exibir_estatisticas(acuracias, "Acurácia")
exibir_estatisticas(sensibilidades, "Sensibilidade")
exibir_estatisticas(especificidades, "Especificidade")

# Encontrando as rodadas com maior e menor acurácia
indice_max_acuracia = np.argmax(acuracias)
indice_min_acuracia = np.argmin(acuracias)

# Curva de aprendizado para maior e menor acurácia
curva_max_acuracia = curvas_acuracia[indice_max_acuracia]
curva_min_acuracia = curvas_acuracia[indice_min_acuracia]

# Plotando a curva de aprendizado
plt.figure(figsize=(12, 6))
plt.plot(range(epocas), curva_max_acuracia, label=f"Maior Acurácia ({acuracias[indice_max_acuracia]:.2f})", color='blue')
plt.plot(range(epocas), curva_min_acuracia, label=f"Menor Acurácia ({acuracias[indice_min_acuracia]:.2f})", color='red')
plt.title("Curva de Aprendizado (Baseada na Acurácia)")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()
plt.show()

# Plotando o gráfico de dispersão com os dados normalizados
plotar_dados_normalizados(X, y)

# Plotando a matriz de confusão para maior e menor acurácia
y_pred_max, y_teste_max = predicoes_teste[indice_max_acuracia]
plotar_matriz_confusao(y_teste_max, y_pred_max, titulo=f"Matriz de Confusão - Maior Acurácia ({acuracias[indice_max_acuracia]:.2f})")

y_pred_min, y_teste_min = predicoes_teste[indice_min_acuracia]
plotar_matriz_confusao(y_teste_min, y_pred_min, titulo=f"Matriz de Confusão - Menor Acurácia ({acuracias[indice_min_acuracia]:.2f})")
