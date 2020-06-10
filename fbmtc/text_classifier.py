from typing import Dict

from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import FastTextEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, TokenEmbeddings, WordEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import sys
import json


embedding_mapping: Dict[str, TokenEmbeddings] = {
    # FastText variants
    "en-crawl": WordEmbeddings("en-crawl"),
    "biowordvec": FastTextEmbeddings("https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec"
                                     "/BioWordVec_PubMed_MIMICIII_d200.bin", use_local=False),
    # Flair variants
    "en-forward": FlairEmbeddings("en-forward"),
    "en-backward": FlairEmbeddings("en-backrward"),
    "pubmed_forward": FlairEmbeddings("pubmed-forward"),
    "pubmed_backward": FlairEmbeddings("pubmed-backward")
}


if __name__ == "__main__":
    """
    python text_classifier.py config_file model_dir
    """
    with open(sys.argv[1], 'rb') as f:
        config = json.load(f)

    # get the corpus
    column_name_map = {0: config["label_name"], 1: "text"}
    corpus: Corpus = CSVClassificationCorpus(config["data_folder"],
                                             column_name_map,
                                             skip_header=True,
                                             delimiter='\t',  # tab-separated files
                                             )
    print(corpus)

    # make the label dictionary from the corpus
    label_dictionary = corpus.make_label_dictionary()
    # TODO calculate inverted class frequencies to pass as loss weights to the text classifier

    # initialize embeddings
    word_embeddings = [embedding_mapping[emb_name] for emb_name in config["word_embeddings"]]
    # TODO keep in mind that state of the art models use the fine tuned transformer approach:
    #  https://github.com/flairNLP/flair/issues/1527#issuecomment-616638837
    #  Corresponding example code: https://github.com/flairNLP/flair/issues/1527#issuecomment-616095945
    document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=256)

    # initialize text classifier

    classifier = TextClassifier(document_embeddings,
                                label_dictionary=corpus.make_label_dictionary(),
                                multi_label=False)

    # initialize trainer
    trainer = ModelTrainer(classifier, corpus)

    trainer.train(sys.argv[2],
                  learning_rate=config["learning_rate"],
                  mini_batch_size=config["batch_size"],
                  max_epochs=config["max_epochs"],
                  embeddings_in_memory=True)
