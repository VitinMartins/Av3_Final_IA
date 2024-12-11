import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def mlp(L, q, m, lr, maxEpoch, pr, Xtrain, Ytrain, Xtest, Ytest, C):
    W = []
    u = [0] * (L + 1)
    y = [0] * (L + 1)
    delta = [0] * (L + 1)

    p = Xtrain.shape[0] - 1

    for i in range(0, L + 1):
        if i == 0:
            W.append(np.random.random_sample((q[0], p + 1)) - 0.5)
        elif i == L:
            W.append(np.random.random_sample((m, q[L - 1] + 1)) - 0.5)
        else:
            W.append(np.random.random_sample((q[i], q[i - 1] + 1)) - 0.5)

    def g(x):
        return np.tanh(x)

    def gprime(x):
        return 0.5 * (1 - np.tanh(x) ** 2)

    def forward(x):
        for i in range(0, L + 1):
            if i == 0:
                u[i] = W[i] @ x
                y[i] = g(u[i])
            else:
                ybias = np.concatenate((-np.ones(1, ), y[i - 1]), axis=0)
                u[i] = W[i] @ ybias
                y[i] = g(u[i])

    def backward(x, d):
        i = L
        while i >= 0:
            if i == L:
                delta[i] = gprime(u[i]) * (d - y[i])
                ybias = np.concatenate((-np.ones(1,), y[i - 1]), axis=0)
                W[i] += lr * (np.outer(delta[i], ybias))

            elif i == 0:
                Wnobias = W[i + 1][:, 1:]
                delta[i] = gprime(u[i]) * (Wnobias.T @ delta[i + 1])
                W[i] += lr * (np.outer(delta[i], x))

            else:
                Wnobias = W[i + 1][:, 1:]
                delta[i] = gprime(u[i]) * (Wnobias.T @ delta[i + 1])
                ybias = np.concatenate((-np.ones(1,), y[i - 1]), axis=0)
                W[i] += lr * (np.outer(delta[i], ybias))

            i -= 1

    def EQM():
        eqm = 0
        for t in range(0, Xtrain.shape[1]):
            x_t = Xtrain[:, t]
            forward(x_t)
            d = Ytrain[:, t]
            eqm += np.sum((d - y[L]) ** 2)
        return eqm / (2 * Xtrain.shape[1])

    def group(x):
        return 0 if x >= 0 else 1

    eqm = 1
    epoch = 0
    aprendizagem = []
    while eqm > pr and epoch < maxEpoch:
        for t in range(0, Xtrain.shape[1]):
            x_t = Xtrain[:, t]
            forward(x_t)
            d = Ytrain[:, t]
            backward(x_t, d)

        eqm = EQM()
        aprendizagem.append(eqm)
        epoch += 1

    confMatrix = np.zeros((C, C))

    for t in range(0, Xtest.shape[1]):
        x_t = Xtest[:, t]
        forward(x_t)
        d = Ytest[:, t]
        c = d
        c_hat = y[L]
        confMatrix[group(c_hat), group(c)] += 1

    return (confMatrix, aprendizagem)

# Substituindo o pandas por numpy para leitura do CSV
data = np.loadtxt("spiral.csv", delimiter=",")
X = data[:, 0:2].T
Y = data[:, 2].reshape(1, -1)

print(Y.shape)

def normalize(X):
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    return 2 * (X - X_min) / (X_max - X_min) - 1

X = normalize(X)

p = 2
N = 2000
C = 2

acuracias = []
sensibilidades = []
especificidades = []
confs = []
aprendizagem = []

def split_data(X, Y, train_size=0.8):
    N = X.shape[1]
    indices = np.random.permutation(N)
    X_train = X[:, indices[:int(N * train_size)]]
    Y_train = Y[:, indices[:int(N * train_size)]]
    X_test = X[:, indices[int(N * train_size):]]
    Y_test = Y[:, indices[int(N * train_size):]]
    return X_train, Y_train, X_test, Y_test

R = 500
for i in range(R):
    X_train, Y_train, X_test, Y_test = split_data(X, Y)
    X_train = np.concatenate((-np.ones((1, X_train.shape[1])), X_train), axis=0)
    X_test = np.concatenate((-np.ones((1, X_test.shape[1])), X_test), axis=0)

    print(i)
    # normal
    (conf, aprend) = mlp(L=2, q=[2, 2], m=1, lr=0.01, maxEpoch=50, pr=0.1, Xtrain=X_train, Ytrain=Y_train, Xtest=X_test, Ytest=Y_test, C=C)

    # overfitting
    # (conf, aprend) = mlp(L=1, q=[1000], m=1, lr=0.01, maxEpoch=20, pr=0.36, Xtrain=X_train, Ytrain=Y_train, Xtest=X_test, Ytest=Y_test, C=C)

    # underfitting
    #(conf, aprend) = mlp(L=1, q=[1], m=1, lr=0.01, maxEpoch=20, pr=0.1, Xtrain=X_train, Ytrain=Y_train, Xtest=X_test, Ytest=Y_test, C=C)

    conf = conf.astype(int)
    confs.append(conf)
    print(conf)

    acuracia = np.trace(conf) / np.sum(conf)
    sensibilidade = np.diag(conf) / np.sum(conf, axis=1)
    especificidade = np.diag(conf) / np.sum(conf, axis=0)

    acuracias.append(acuracia)
    sensibilidades.append(sensibilidade)
    especificidades.append(especificidade)
    aprendizagem.append(aprend)

confusion_max = confs[np.argmax(acuracias)]
confusion_min = confs[np.argmin(acuracias)]

labels = ["Predito: +1", "Predito: -1"]
categories = ["Classe: +1", "Classe: -1"]

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_max, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=categories)
plt.title("Matriz de Confusão - Rodada com Maior Acurácia")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_min, annot=True, fmt="d", cmap="Reds", xticklabels=labels, yticklabels=categories)
plt.title("Matriz de Confusão - Rodada com Menor Acurácia")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(aprendizagem[np.argmax(acuracias)], label='Maior Acurácia', linewidth=2)
plt.plot(aprendizagem[np.argmin(acuracias)], label='Menor Acurácia', linewidth=2)
plt.title("Curvas de Aprendizagem")
plt.xlabel("Época")
plt.ylabel("Erro Quadrático Médio")
plt.legend()
plt.grid()
plt.show()

print("MLP")
print("Acurácia média: ", np.mean(acuracias))
print("Acurácia desvio padrão: ", np.std(acuracias))
print("Acurácia maior valor: ", np.max(acuracias))
print("Acurácia menor valor: ", np.min(acuracias))

print("Sensibilidade média: ", np.mean(sensibilidades))
print("Sensibilidade desvio padrão: ", np.std(sensibilidades))
print("Sensibilidade maior valor: ", np.max(sensibilidades))
print("Sensibilidade menor valor: ", np.min(sensibilidades))

print("Especificidade média: ", np.mean(especificidades))
print("Especificidade desvio padrão: ", np.std(especificidades))
print("Especificidade maior valor: ", np.max(especificidades))
print("Especificidade menor valor: ", np.min(especificidades))
