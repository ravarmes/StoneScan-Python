<h1 align="center">
    <img alt="Fapes" src="assets/logo-fapes.png" width="300" />
    <img alt="Granimaster" src="assets/logo-granimaster.png" width="300" />
</h1>

<h3 align="center">
  StoneScan: Identificação Inteligente de Rochas Ornamentais
</h3>

<p align="center">Aplicativo de reconhecimento de rochas ornamentais utilizando inteligência artificial e visão computacional.</p>

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/ravarmes/StoneScan-Python?color=%2304D361">

  <a href="http://www.linkedin.com/in/rafael-vargas-mesquita">
    <img alt="Made by Rafael Vargas Mesquita" src="https://img.shields.io/badge/made%20by-Rafael%20Vargas%20Mesquita-%2304D361">
  </a>

  <img alt="License" src="https://img.shields.io/badge/license-MIT-%2304D361">

  <a href="https://github.com/ravarmes/StoneScan-Python/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/ravarmes/StoneScan-Python?style=social">
  </a>
</p>

## :page_with_curl: Sobre o Projeto <a name="-about"/></a>

O StoneScan é um aplicativo inovador que utiliza inteligência artificial para identificar rochas ornamentais através de fotografias tiradas por smartphones. Desenvolvido para atender tanto profissionais do setor quanto consumidores finais, o aplicativo emprega redes neurais avançadas para reconhecer e classificar diferentes tipos de rochas, como granitos e mármores, a partir de imagens de superfícies como pias, pisos e paredes.

O projeto é resultado de uma colaboração multidisciplinar entre estudantes de Sistemas de Informação e Técnico em Mineração, sob orientação docente. As informações sobre as rochas são baseadas em fontes autorizadas, incluindo o manual de rochas ornamentais do SindRochas e dados fornecidos por empresas parceiras do setor.

### :notebook_with_decorative_cover: Arquitetura do Sistema <a name="-architecture"/></a>

<img src="assets/architecture.png" width="700">

### Arquivos

| Arquivo                              | Descrição                                                                                                                                                           |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_training.py                    | Implementação principal do modelo de deep learning usando TensorFlow e MobileNetV2 para classificação de imagens de rochas. Inclui pipeline completo de treinamento. |
| model_development.ipynb              | Notebook Jupyter para desenvolvimento, experimentação e validação do modelo de classificação de rochas. |
| .gitignore                          | Configurações para ignorar arquivos e diretórios específicos no controle de versão.                                                                                  |

### Características Principais

- Utilização do MobileNetV2 pré-treinado para extração de características
- Data augmentation para melhor generalização do modelo
- Pipeline de treinamento em duas fases (transfer learning e fine-tuning)
- Sistema de logging para monitoramento do treinamento
- Callbacks para early stopping e checkpointing
- Visualização do histórico de treinamento

### Requisitos do Sistema

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Logging

### Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/StoneScan-Python.git
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o treinamento:
```bash
python model_training.py
```

### Resultados

O modelo gera gráficos de acurácia e perda durante o treinamento, salvos como 'training_history.png'. O melhor modelo é automaticamente salvo como 'best_model.keras'. 