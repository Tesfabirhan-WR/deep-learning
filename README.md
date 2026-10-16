# Deep Learning Mini Projects

A collection of 6 hands-on mini projects covering the fundamentals of deep learning: from a single perceptron to image classification and text modelling. Each project lives in its own Jupyter notebook and shares helper functions from `libs.py`.

## Project Structure

```
.
├── libs.py                    # Shared helpers (data loading, plotting, metrics)
├── mnist_learning.ipynb       # Handwritten digit classification (MNIST)
├── neuron_network.ipynb       # Neural network built step by step
├── fashion-mnist.ipynb        # Clothing image classification (Fashion-MNIST)
├── perception_simple.ipynb    # Single perceptron from scratch
├── titanic.ipynb              # Survival prediction on tabular data
├── victor_hugo.ipynb          # Text modelling on Victor Hugo's works
├── data/                      # Datasets (see "Datasets" below)
│   ├── victor-hugo-texts.txt
│   └── victor-hugo-titles.txt
└── README.md
```

## The 6 Projects

### 1. Perceptron (`perception_simple.ipynb`)
Implementation of the simplest neural unit: weighted sum, activation function and weight updates. Illustrates how a perceptron learns a linearly separable problem and where it fails.

**Concepts:** weights and bias, activation function, learning rate, perceptron learning rule.

### 2. Neural Network (`neuron_network.ipynb`)
Building a multi-layer neural network and understanding what happens during training.

**Concepts:** hidden layers, forward propagation, loss function, backpropagation, gradient descent.

### 3. MNIST (`mnist_learning.ipynb`)
Classification of handwritten digits (0 to 9) from 28x28 grayscale images.

**Concepts:** image normalization, dense network for classification, softmax, accuracy and confusion matrix.

### 4. Fashion-MNIST (`fashion-mnist.ipynb`)
A harder drop-in replacement for MNIST: 10 categories of clothing items (t-shirt, trouser, sneaker, bag, etc.).

**Concepts:** model comparison with MNIST, overfitting, regularization (dropout), optionally convolutional layers.

### 5. Titanic (`titanic.ipynb`)
Predicting passenger survival from tabular data (age, sex, class, fare, etc.).

**Concepts:** data cleaning, missing values, categorical encoding, feature scaling, neural networks on tabular data, train/validation split.

### 6. Victor Hugo (`victor_hugo.ipynb`)
Next-word prediction and text generation with a recurrent neural network, trained on a corpus of Victor Hugo's poetry (`victor-hugo-texts.txt`).

**How it works:**
- **Preprocessing:** each non-empty line of the corpus is read, then tokenized with the Keras `Tokenizer` (vocabulary limited to 1000 words).
- **Training examples:** a sliding window of the **3 previous words** is used to predict the next word, with zero padding at the start when fewer than 3 words are available (11,541 examples).
- **Model:** `Embedding` (16 dimensions) → `SimpleRNN` (32 units) → `Dense` (1000, softmax), trained with sparse categorical cross-entropy and Adam. `EarlyStopping` on the loss (patience 10) stops training automatically.
- **Generation:** starting from a seed word, the model predicts a probability distribution over the next word and samples from it, one word at a time. A `temperature` parameter controls the result (lower = safer and more repetitive, higher = more varied).

**Concepts:** tokenization, word embeddings, sliding windows, recurrent neural networks, softmax sampling, temperature.

**Example output** (seed "Poème"): the generated text has a poetic vocabulary ("fleurs", "soleil", "âme", "coeurs") but little grammar, which is expected with a simple RNN, a 3-word context and a small corpus.

## Datasets

| Project | Dataset | Source |
|---|---|---|
| MNIST | Handwritten digits | Provided / loaded via Keras |
| Fashion-MNIST | Clothing images | Provided / loaded via Keras |
| Titanic | Passenger records (CSV) | Provided locally |
| Victor Hugo | `victor-hugo-texts.txt` (training corpus), `victor-hugo-titles.txt` (list of the 48 poem titles) | Provided locally |
| Perceptron / Neural Network | Small toy datasets | Generated in the notebooks |

Place the local files in the `data/` folder (or update the paths at the top of each notebook).

## Getting Started

### Requirements

- Python 3.9+
- Jupyter Notebook or JupyterLab
- numpy, pandas, matplotlib, scikit-learn
- tensorflow / keras

### Installation

```bash
git clone https://github.com/Tesfabirhan-WR/deep-learning
cd projects/deep-learning

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install numpy pandas matplotlib scikit-learn tensorflow jupyter
```

### Run

```bash
jupyter notebook
```

Then open any notebook. They are independent, but a good learning order is:

1. `perception_simple.ipynb`
2. `neuron_network.ipynb`
3. `mnist_learning.ipynb`
4. `fashion-mnist.ipynb`
5. `titanic.ipynb`
6. `victor_hugo.ipynb`

## About `libs.py`

Common utilities imported by the notebooks, so the code is not duplicated:

```python
from libs import *
```

## What I Learned

- How a neuron works and how networks learn through backpropagation
- Applying neural networks to three data types: images, tabular data and text (with an RNN for next-word generation)
- Evaluating models and diagnosing overfitting
- Structuring a small ML project with reusable code

## Author

Tesfabirhan REDIE
Data Engineer
