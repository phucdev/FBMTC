from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus, CSVClassificationDataset
from flair.embeddings import DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import sys
import json
from fbmtc import utils


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
    class_weights = utils.get_inverted_class_balance(corpus.train.dataset)

    # initialize embeddings
    word_embeddings = None
    chosen_embeddings = config["word_embeddings"]
    if chosen_embeddings == "general":
        word_embeddings = utils.get_general_embeddings()
    elif chosen_embeddings == "mixed_bio":
        word_embeddings = utils.get_mixed_bio_embeddings()
    elif chosen_embeddings == "bio":
        word_embeddings = utils.get_bio_embeddings()
    elif chosen_embeddings == "scibert_flair":
        word_embeddings = utils.get_scibert_flair_embeddings()

    # TODO keep in mind that state of the art models use the fine tuned transformer approach:
    #  https://github.com/flairNLP/flair/issues/1527#issuecomment-616638837
    #  Corresponding example code: https://github.com/flairNLP/flair/issues/1527#issuecomment-616095945
    document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=256)

    # initialize text classifier

    classifier = TextClassifier(document_embeddings,
                                label_dictionary=label_dictionary,
                                multi_label=False, loss_weights=class_weights)

    # initialize trainer
    trainer = ModelTrainer(classifier, corpus)

    # train_with_dev: use option for final model
    trainer.train(sys.argv[2],
                  learning_rate=config["learning_rate"],
                  mini_batch_size=config["batch_size"],
                  max_epochs=config["max_epochs"],
                  embeddings_storage_mode="gpu")
