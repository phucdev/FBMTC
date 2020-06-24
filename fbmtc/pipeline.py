from pathlib import Path
from typing import Union, Optional

import pandas as pd
from fbmtc import predictor, NEGATIVE_IS_CANCER, NEGATIVE_DOID
from flair.data import Sentence


def predict_doc(doc, binary_classifier, multi_class_classifier):
    binary_result = predictor.predict_instance(binary_classifier, Sentence(doc['text']))['prediction']
    if binary_result != NEGATIVE_IS_CANCER:
        doc['is_cancer'] = binary_result
        multi_class_result = predictor.predict_instance(multi_class_classifier, Sentence(doc['text']))['prediction']
        doc['doid'] = multi_class_result
    else:
        doc['is_cancer'] = NEGATIVE_IS_CANCER
        doc['doid'] = NEGATIVE_DOID
    return doc


def predict_docs(docs: Union[str, Path],
                 binary_classifier_model_dir: Union[str, Path],
                 multi_class_classifier_model_dir: Union[str, Path],
                 binary_classifier_model_archive: str = 'best-model.pt',
                 multi_class_classifier_model_archive: str = 'best-model.pt'
                 ):
    """
    :param docs: Path to data file
    :param binary_classifier_model_dir: Path to binary classifier model directory
    :param multi_class_classifier_model_dir: Path to multi class classifier model directory
    :param binary_classifier_model_archive: Model archive name of binary classifier
    :param multi_class_classifier_model_archive: Model archive name of multi class classifier
    :return:
    """
    pred_docs = pd.read_csv(docs, sep='\t', header=0)

    binary_classifier = predictor.load_predictor(model_dir=binary_classifier_model_dir,
                                                 archive_filename=binary_classifier_model_archive)
    multi_class_classifier = predictor.load_predictor(model_dir=multi_class_classifier_model_dir,
                                                      archive_filename=multi_class_classifier_model_archive)

    labeled_docs = pred_docs.apply(lambda doc: predict_doc(doc, binary_classifier, multi_class_classifier), axis=1)
    return labeled_docs
