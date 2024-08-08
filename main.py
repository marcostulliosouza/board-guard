import data_preparation
import model_training
import model_evaluation


def main():
    data_preparation.prepare_data()  # Preparar os dados
    model_training.train_model()  # Treinar o modelo
    model_evaluation.evaluate_model()  # Avaliar o modelo


if __name__ == "__main__":
    main()
