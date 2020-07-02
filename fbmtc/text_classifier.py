from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import DocumentRNNEmbeddings, WordEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import sys
import json
from fbmtc import utils
from fbmtc.custom_document_embeddings import CustomDocumentRNNEmbeddings

if __name__ == "__main__":
    """
    python text_classifier.py config_file model_dir
    """
    with open(sys.argv[1], 'rb') as f:
        config = json.load(f)

    # set parameters from config
    data_folder = config["data_folder"]
    hidden_size = config["hidden_size"] if "hidden_size" in config else 256
    bidirectional = config["bidirectional"] == "True" if "bidirectional" in config else True
    rnn_layers = config["rnn_layers"] if "rnn_layers" in config else 1
    dropout = config["dropout"] if "dropout" in config else 0.5
    doc_embedding = config["doc_embedding"] if "doc_embedding" in config else "normal_rnn"
    attention_size = config["attention_size"] if "attention_size" in config else 100
    use_loss_weights = config["use_loss_weights"] if "use_loss_weights" in config else False
    learning_rate = config["learning_rate"] if "learning_rate" in config else 0.1
    mini_batch_size = config["batch_size"] if "batch_size" in config else 32
    max_epochs = config["max_epochs"] if "max_epochs" in config else 150

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
    # TODO check calculate inverted class frequencies to pass as loss weights to the text classifier
    class_weights = utils.get_inverted_class_balance(corpus.train.dataset)

    # initialize embeddings
    chosen_embeddings = config["word_embeddings"]
    if chosen_embeddings == "general":
        word_embeddings = utils.get_general_embeddings()
    elif chosen_embeddings == "mixed_bio":
        word_embeddings = utils.get_mixed_bio_embeddings()
    elif chosen_embeddings == "bio":
        word_embeddings = utils.get_bio_embeddings()
    else:
        word_embeddings = [WordEmbeddings('glove')]

    if doc_embedding == "custom":
        document_embeddings = CustomDocumentRNNEmbeddings(embeddings=word_embeddings,
                                                          hidden_size=hidden_size,
                                                          rnn_layers=rnn_layers,
                                                          bidirectional=bidirectional,
                                                          dropout=dropout,
                                                          attention_size=attention_size)
    else:
        document_embeddings = DocumentRNNEmbeddings(embeddings=word_embeddings,
                                                    hidden_size=hidden_size,
                                                    rnn_layers=rnn_layers,
                                                    bidirectional=bidirectional,
                                                    dropout=dropout)

    # initialize text classifier
    if use_loss_weights:
        classifier = TextClassifier(document_embeddings,
                                    label_dictionary=label_dictionary,
                                    multi_label=False, loss_weights=class_weights)
    else:
        classifier = TextClassifier(document_embeddings,
                                    label_dictionary=label_dictionary,
                                    multi_label=False)

    # initialize trainer
    trainer = ModelTrainer(classifier, corpus)

    # train_with_dev: use option for final model
    trainer.train(sys.argv[2],
                  learning_rate=learning_rate,
                  mini_batch_size=mini_batch_size,
                  max_epochs=max_epochs,
                  embeddings_storage_mode="gpu")
