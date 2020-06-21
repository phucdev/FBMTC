import json
import sys

from torch.optim.adam import Adam

from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

from fbmtc import utils

if __name__ == "__main__":
    """
    python bert_text_classifier.py config_file model_dir
    Adopted from: "https://github.com/flairNLP/flair/releases/tag/v0.5" with a few changes
    """
    with open(sys.argv[1], 'rb') as f:
        config = json.load(f)

    # 1. get the corpus
    column_name_map = {0: config["label_name"], 1: "text"}
    corpus: Corpus = CSVClassificationCorpus(config["data_folder"],
                                             column_name_map,
                                             skip_header=True,
                                             delimiter='\t',  # tab-separated files
                                             )
    print(corpus)

    # 2. create the label dictionary
    label_dict = corpus.make_label_dictionary()
    class_weights = utils.get_inverted_class_balance(corpus.train.dataset)

    # 3. initialize transformer document embeddings (many models are available)
    document_embeddings = TransformerDocumentEmbeddings('allenai/scibert_scivocab_uncased', fine_tune=True)

    # 4. create the text classifier
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, loss_weights=class_weights)

    # 5. initialize the text classifier trainer with Adam optimizer
    trainer = ModelTrainer(classifier, corpus, optimizer=Adam)

    # 6. start the training
    trainer.train(sys.argv[2],
                  learning_rate=3e-5,  # use very small learning rate
                  mini_batch_size=16,
                  mini_batch_chunk_size=4,  # optionally set this if transformer is too much for your machine
                  max_epochs=5,  # terminate after 5 epochs
                  )
