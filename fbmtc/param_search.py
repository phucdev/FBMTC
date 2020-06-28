import json
import sys

from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.hyperparameter.param_selection import TextClassifierParamSelector, OptimizationValue
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from fbmtc import utils

if __name__ == "__main__":
    """
    python param_search.py config_file
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
    word_embeddings = [utils.get_general_embeddings(), utils.get_mixed_bio_embeddings(), utils.get_bio_embeddings()]
    search_space = SearchSpace()
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=word_embeddings)
    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[128, 256])
    search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
    search_space.add(Parameter.BIDIRECTIONAL, hp.choice, options=[False, True])
    search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
    search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[16, 32])
    param_selector = TextClassifierParamSelector(
        corpus=corpus,
        multi_label=False,
        base_path='resources/results'+config['task'],
        document_embedding_type='lstm',
        max_epochs=10,
        training_runs=1,
        optimization_value=OptimizationValue.DEV_SCORE
    )
    param_selector.optimize(search_space, max_evals=100)
