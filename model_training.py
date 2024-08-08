import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib


def train_model():
    # Carregar os dados preparados
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    # Configurar o modelo RandomForest
    rf = RandomForestClassifier(random_state=42)

    # Definir hiperparâmetros para ajuste
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Grid Search com validação cruzada
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Melhor modelo
    best_model = grid_search.best_estimator_

    # Fazer previsões
    y_pred = best_model.predict(X_test)

    # Avaliar o modelo
    report = classification_report(y_test, y_pred)
    print("Relatório de Classificação:\n", report)

    # Salvar o modelo treinado
    joblib.dump(best_model, 'model.pkl')


if __name__ == "__main__":
    train_model()
