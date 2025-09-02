Here is a comprehensive README for your project

-----

NER-WITH-BERT

The objective of this project is to develop a Named Entity Recognition (NER) system using BERT, a state-of-the-art pre-trained transformer model, and the Hugging Face Transformers library. The NER system will be able to extract and classify entities (such as names of people, organizations, locations, dates, etc.) from unstructured text data.

-----

Features

  * **Named Entity Recognition (NER)**: Identifies and classifies entities like persons, locations, and organizations within a sentence. This is done by attributing a specific label to each token (word).
  * **Part-of-Speech Tagging (POS)**: Marks each word in a sentence with its corresponding part of speech (e.g., noun, verb, adjective).
  * **Multi-task Learning Model**: The `MultiTaskBERT` model is a custom PyTorch module that uses a BERT base model with two separate linear layers (heads) for NER and POS tasks. The model is trained to minimize a combined loss from both tasks.
  * **Flask API**: The `app.py` file creates a Flask application that uses the trained models to perform NER and POS predictions via a web interface.

-----

Setup Guide

1. Prerequisites

  * Python 3.6 or higher
  * PyTorch
  * Flask
  * Hugging Face `transformers` library
  * `datasets` library
  * `tokenizers` library
  * `seqeval` library

You can install the necessary Python packages using the following command:

```bash
!pip install transformers datasets tokenizers seqeval -q
```

2. Project Structure

  * `app.py`: Contains the Flask application for serving the NER and POS models.
  * `dataset.py`: Defines the `MultiTaskDataset` class for preparing data for training.
  * `label_maps.py`: Stores the label-to-ID mappings for both NER and POS tasks.
  * `main.py`: The main script for training the `MultiTaskBERT` model.
  * `model.py`: Defines the `MultiTaskBERT` model architecture.
  * `NER_WITH_BERT.ipynb`: A Jupyter Notebook that demonstrates the data loading, tokenization, and model training process using the `conll2003` dataset.
  * `README.md`: This file.

3. Training the Model

The `main.py` script handles the training process. You can run it with the following command:

```bash
python main.py
```

The script will load a dataset from `"data/train.json"` and train the `MultiTaskBERT` model for a specified number of epochs, printing the loss after each epoch.

#### **4. Running the Flask Application**

To run the web application, execute `app.py`:

```bash
python app.py
```

This will start a local server, and you can access the web interface at `http://127.0.0.1:5000` to make predictions on text input.

-----

Dataset Details

The project uses the `conll2003` dataset, which is designed for Named Entity Recognition and contains tokens, Part-of-Speech (POS) tags, syntactic chunk tags, and NER tags. The NER tags are in the IOB2 format, where:

  * `O`: The word does not correspond to any entity.
  * `B-PER/I-PER`: The word corresponds to the beginning of/is inside a person entity.
  * `B-ORG/I-ORG`: The word corresponds to the beginning of/is inside an organization entity.
  * `B-LOC/I-LOC`: The word corresponds to the beginning of/is inside a location entity.
  * `B-MISC/I-MISC`: The word corresponds to the beginning of/is inside a miscellaneous entity.

To prepare the dataset for the BERT model, a function `tokenize_and_align_labels` is used to align the NER tags with the tokenized words, adding a label of -100 for special tokens like `[CLS]` and `[SEP]`.
