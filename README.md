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
│   └── victor-hugo-titles.txt  # poem titles, used to clean the corpus
└── README.md
```

## The 6 Projects

### 1. Perceptron (`perception_simple.ipynb`)
Implementation of the simplest neural unit: weighted sum, activation function and weight updates. Illustrates how a perceptron learns a linearly separable problem and where it fails.

**Concepts:** weights and bias, activation function, learning rate, perceptron learning rule.

### 2. Neural Network (`neuron_network.ipynb`)
Multi-class classification of the **Iris** flowers (3 species, 4 measurements) with a small MLP (4-8-3, 67 parameters), built step by step.

**Concepts:** stratified train/test split, standardization (scaler fitted on the train set only), one-hot encoding, hidden layers, softmax, categorical cross-entropy, epochs and batch size, confusion matrix, predicting new samples.

### 3. MNIST (`mnist_learning.ipynb`)
Classification of handwritten digits (0 to 9) from 28x28 grayscale images.

**Concepts:** image normalization, dense network for classification, softmax, accuracy and confusion matrix.

### 4. Fashion-MNIST (`fashion-mnist.ipynb`)
A harder drop-in replacement for MNIST: 10 categories of clothing items (t-shirt, trouser, sneaker, bag, etc.).

**Concepts:** model comparison with MNIST, overfitting, regularization (dropout), optionally convolutional layers.

### 5. Titanic (`titanic.ipynb`)
Predicting passenger survival from tabular data (age, sex, class, fare, etc.) with a small network: 9 input features after encoding, 27 hidden neurons, 1 sigmoid output.

**Concepts:** missing values (imputation with statistics computed on the train set), one-hot encoding applied identically to train and test, feature scaling, binary cross-entropy, early stopping on the validation loss, 0.5 decision threshold, confusion matrix.

### 6. Victor Hugo (`victor_hugo.ipynb`)
Next-word prediction and text generation with a recurrent neural network, trained on a corpus of Victor Hugo's poetry (`victor-hugo-texts.txt`).

**How it works:**
- **Cleaning:** metadata lines (`Victor Hugo: ...`, collection, date), poem titles (listed in `victor-hugo-titles.txt`) and section numbers are removed; apostrophes and quotation marks are normalized; elisions are split (`l'ombre` → `l' ombre`) to get a denser vocabulary. An optional `split_verses` setting cuts lines that contain several verses.
- **Preprocessing:** each remaining line is tokenized with the Keras `Tokenizer` (vocabulary limited to 1000 words).
- **Train / validation / test split:** 80 % / 10 % / 10 %, split by **lines** (not by windows) so that overlapping windows from the same line never end up in different sets.
- **Training examples:** a sliding window of the **3 previous words** is used to predict the next word, with zero padding at the start when fewer than 3 words are available.
- **Model:** `Embedding` (16 dimensions) → `SimpleRNN` (32 units) → `Dense` (1000, softmax), trained with sparse categorical cross-entropy and Adam. `EarlyStopping` on the **validation loss** (patience 5) stops training and restores the best weights.
- **Evaluation:** Top-1 and Top-5 accuracy on train, validation and test sets, compared with a "most frequent word" baseline, plus a confusion matrix and a classification report on the test set.
- **Generation:** starting from a seed word, the model predicts a probability distribution over the next word and samples from it, one word at a time. A `temperature` parameter controls the result (lower = safer and more repetitive, higher = more varied).

**Concepts:** tokenization, word embeddings, sliding windows, recurrent neural networks, softmax sampling, temperature.

**Example output** (seed "Ce siècle avait deux ans", first run): *"... sont il avec un pas l'enfant pitié un bêtes le il a que petit depuis navire son pavé de vont diable ..."*. The vocabulary is poetic but the grammar is loose, which is expected with a simple RNN, a 3-word context and a small corpus.

**Limitations:** the corpus is very small (about 17,000 words, 1,400 lines, with 62 % of the words appearing only once), so a word-level model cannot learn much: Top-1 accuracy stays in the single digits and should be read against the "most frequent word" baseline, and Top-5 and the validation loss are fairer metrics. A character-level model or a larger corpus would give much better results.

## Datasets

| Project | Dataset | Source |
|---|---|---|
| MNIST | Handwritten digits | Provided / loaded via Keras |
| Fashion-MNIST | Clothing images | Provided / loaded via Keras |
| Titanic | Passenger records | Loaded with `seaborn.load_dataset("titanic")` |
| Victor Hugo | `victor-hugo-texts.txt` (training corpus), `victor-hugo-titles.txt` (list of the 48 poem titles) | Provided locally |
| Neural Network | Iris flowers | Loaded with `sklearn.datasets.load_iris()` |
| Perceptron | Small toy dataset | Generated in the notebook |

Place the Victor Hugo files in the `data/` folder (or update the paths at the top of each notebook).

## Getting Started

### Requirements

- Python 3.9+
- Jupyter Notebook or JupyterLab
- numpy, pandas, matplotlib, seaborn, scikit-learn, networkx
- tensorflow / keras

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install numpy pandas matplotlib seaborn scikit-learn networkx tensorflow jupyter
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
### Example of loading dataset

```
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(x_train.shape, y_train.shape)
```


### Data Classification -example

```
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = titanic[features]
y = titanic["survived"].values

-----

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```
### One-hot encoding of the labels

```
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))
print("Sortie encodées :\n", y_train[:5])
```

## What I Learned

- How a neuron works and how networks learn through backpropagation
- Applying neural networks to three data types: images, tabular data and text (with an RNN for next-word generation)
- Evaluating models and diagnosing overfitting
- Structuring a small ML project with reusable code

## Author

Tesfabirhan REDIE
Data Engineer
