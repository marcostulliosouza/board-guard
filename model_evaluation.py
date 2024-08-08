import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model():
    # Carregar os dados e o modelo
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    clf = joblib.load('model.pkl')

    # Fazer previsões
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Matriz de Confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[
                'No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')

    # Cálculo das Métricas
    accuracy = np.mean(y_pred == y_test) * 100
    precision = conf_matrix[1, 1] / \
        (conf_matrix[1, 1] + conf_matrix[0, 1]) * 100
    recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) * 100
    specificity = conf_matrix[0, 0] / \
        (conf_matrix[0, 0] + conf_matrix[0, 1]) * 100

    print(f"Acurácia: {accuracy:.2f}%")
    print(f"Precisão: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"Especificidade: {specificity:.2f}%")
    plt.show()

    # Relatório de Classificação
    report = classification_report(y_test, y_pred)
    print("Relatório de Classificação:\n", report)

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'Curva ROC (Área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show()

    # Importância das características
    if hasattr(clf, 'feature_importances_'):
        feature_importances = clf.feature_importances_
        features = [f'test_{i}' for i in range(
            1, len(feature_importances) + 1)]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=features, y=feature_importances * 100, palette='viridis')
        plt.xlabel('Testes Funcionais')
        plt.ylabel('Importância (%)')
        plt.title('Importância dos Testes Funcionais na Previsão de Falhas')
        plt.xticks(rotation=90)
        plt.show()


if __name__ == "__main__":
    evaluate_model()
