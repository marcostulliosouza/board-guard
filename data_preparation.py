import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification


def prepare_data():
    # Gerar um dataset sintético com características mais realistas
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    n_informative = 10  # Número de características informativas
    n_clusters_per_class = 2  # Número de clusters por classe

    # Gerar dados com características mais realistas
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_clusters_per_class=n_clusters_per_class,
        weights=[0.7, 0.3],  # Desbalanceamento entre classes
        flip_y=0.1,  # Introduzir algum ruído
        random_state=42
    )

    # Converter para DataFrame para melhor visualização
    test_columns = [f'test_{i}' for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=test_columns)
    df['failure'] = y

    # Dividir os dados em treino e teste
    X = df.drop('failure', axis=1)
    y = df['failure']
    X_train, X_test, y_train, y_test = train_test_split(
        # Garantir que a divisão respeite a proporção das classes
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Padronizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Aplicar PCA para redução dimensional
    # Reduzir para no máximo 10 componentes principais
    pca = PCA(n_components=min(X_train.shape[1], 10))
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Balancear os dados usando SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Salvar os dados preparados
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)


if __name__ == "__main__":
    prepare_data()
