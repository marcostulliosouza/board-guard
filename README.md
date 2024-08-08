# BoardGuard: Inteligência para Detecção de Falhas em Testes Eletrônicos

## Descrição

Este projeto utiliza `scikit-learn` para prever falhas durante os testes funcionais de placas eletrônicas. Inclui scripts para preparar dados, treinar e avaliar modelos, e gerar visualizações.

## Estrutura do Projeto

- **data_preparation.py**: Prepara e carrega os dados.
- **model_training.py**: Treina o modelo de machine learning.
- **model_evaluation.py**: Avalia o modelo e gera uma matriz de confusão.
- **visualizations.py**: Gera visualizações da importância das características.
- **main.py**: Executa o fluxo completo do projeto.
- **requirements.txt**: Lista as dependências do projeto.

## Instruções

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt