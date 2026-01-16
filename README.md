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

## :rock: Classes de Rochas <a name="-classes"/></a>

O modelo é treinado para classificar as seguintes 5 classes de rochas ornamentais:

| Código da Pasta | Nome Comercial |
|-----------------|----------------|
| g-03 | Granito Branco Itaúnas |
| m-02 | Mármore Matarazzo |
| q-01 | Quartzito Perla |
| q-02 | Quartzito Wakanda |
| q-03 | Quartzito Verde Gaya |

## :notebook_with_decorative_cover: Arquitetura do Sistema <a name="-architecture"/></a>

<img src="assets/architecture.png" width="700">

## :file_folder: Estrutura do Projeto <a name="-structure"/></a>

```
StoneScan-Python/
├── dataset/                    # Pasta com as imagens de treinamento
│   ├── g-03/                   # Granito Branco Itaúnas
│   ├── m-02/                   # Mármore Matarazzo
│   ├── q-01/                   # Quartzito Perla
│   ├── q-02/                   # Quartzito Wakanda
│   └── q-03/                   # Quartzito Verde Gaya
├── results/                    # Resultados do treinamento (gerado automaticamente)
│   ├── confusion_matrix.png
│   ├── learning_curves.png
│   ├── classification_report.txt
│   └── best_model.pth
├── assets/                     # Logos e imagens do projeto
├── train_rock_classifier.py    # Script principal de treinamento
├── requirements.txt            # Dependências do projeto
└── README.md
```

## :gear: Características Técnicas <a name="-features"/></a>

### Modelo de Deep Learning
- **Arquitetura**: ResNet18 com Transfer Learning (pré-treinado no ImageNet)
- **Framework**: PyTorch
- **Validação**: K-Fold Cross-Validation estratificada (5 folds por padrão)
- **Otimização de Hiperparâmetros**: Grid Search

### Data Augmentation
- Random Resize Crop
- Flips horizontais e verticais
- Rotação aleatória
- Color Jitter (brilho, contraste, saturação)
- Affine transformations
- Gaussian Blur

### Métricas de Avaliação
- Acurácia
- Precisão, Recall e F1-Score
- Matriz de Confusão
- Cohen's Kappa

## :computer: Requisitos do Sistema <a name="-requirements"/></a>

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- NumPy, Pandas
- scikit-learn
- Matplotlib, Seaborn
- tqdm

GPU com suporte CUDA é recomendada para treinamento mais rápido.

## :rocket: Como Executar <a name="-how-to-run"/></a>

### 1. Clone o repositório:
```bash
git clone https://github.com/ravarmes/StoneScan-Python.git
cd StoneScan-Python
```

### 2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac
```

### 3. Instale as dependências:
```bash
pip install -r requirements.txt
```

### 4. Baixe o dataset do Kaggle:

O dataset de imagens está hospedado no Kaggle. Você pode baixá-lo de duas formas:

**Opção A - Via Kaggle API (recomendado):**
```bash
# Instale a API do Kaggle (se ainda não tiver)
pip install kaggle

# Configure suas credenciais do Kaggle (~/.kaggle/kaggle.json)
# Veja: https://github.com/Kaggle/kaggle-api#api-credentials

# Baixe e extraia o dataset
kaggle datasets download -d ravarmes2/stonescan -p dataset/ --unzip
```

**Opção B - Download manual:**
1. Acesse: https://www.kaggle.com/datasets/ravarmes2/stonescan
2. Clique em "Download"
3. Extraia o conteúdo para a pasta `dataset/`

Após o download, a estrutura deve ficar:
```
dataset/
├── g-03/   # imagens de Granito Branco Itaúnas
├── m-02/   # imagens de Mármore Matarazzo
├── q-01/   # imagens de Quartzito Perla
├── q-02/   # imagens de Quartzito Wakanda
└── q-03/   # imagens de Quartzito Verde Gaya
```

### 5. Execute o treinamento:
```bash
# Treinamento padrão com validação cruzada
python train_rock_classifier.py

# Com Grid Search para otimização de hiperparâmetros
python train_rock_classifier.py --grid-search

# Especificando parâmetros customizados
python train_rock_classifier.py --epochs 50 --batch-size 32 --learning-rate 0.0005
```

## :bar_chart: Resultados <a name="-results"/></a>

Após o treinamento, os seguintes arquivos são gerados na pasta `results/`:

| Arquivo | Descrição |
|---------|-----------|
| `confusion_matrix.png` | Matriz de confusão do modelo |
| `learning_curves.png` | Gráficos de loss e acurácia por época |
| `classification_report.txt` | Relatório detalhado com métricas por classe |
| `best_model.pth` | Pesos do melhor modelo treinado |
| `training_config.json` | Configurações utilizadas no treinamento |
| `grid_search_results.csv` | Resultados do Grid Search (se executado) |

## :page_facing_up: Licença <a name="-license"/></a>

Este projeto está sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

---

<p align="center">
  Feito com :purple_heart: por Rafael Vargas Mesquita
</p>