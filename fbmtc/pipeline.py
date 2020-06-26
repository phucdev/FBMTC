from pathlib import Path
from typing import Union, Optional

import pandas as pd
from fbmtc import predictor, NEGATIVE_IS_CANCER, NEGATIVE_DOID
from flair.data import Sentence
from sklearn.metrics import f1_score

import time
import sys
import json

def predict_doc(doc, binary_classifier, multi_class_classifier):

    binary_result = predictor.predict_instance(binary_classifier, Sentence(doc['text']))['prediction']

    if multi_class_classifier is not None:
        if binary_result != NEGATIVE_IS_CANCER:
            doc['is_cancer'] = binary_result
            multi_class_result = predictor.predict_instance(multi_class_classifier, Sentence(doc['text']))['prediction']
            doc['doid'] = multi_class_result
        else:
            doc['is_cancer'] = NEGATIVE_IS_CANCER
            doc['doid'] = NEGATIVE_DOID
    else:
        doc['is_cancer']= binary_result
    return doc


def predict_docs(docs: Union[str, Path],
                 binary_classifier_model_dir: Union[str, Path],
                 multi_class_classifier_model_dir: Union[str, Path] = None,
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
    multi_class_classifier = None
    if multi_class_classifier_model_dir is not None and multi_class_classifier_model_archive is not None:
        multi_class_classifier = predictor.load_predictor(model_dir=multi_class_classifier_model_dir,
                                                          archive_filename=multi_class_classifier_model_archive)

    labeled_docs = pred_docs.apply(lambda doc: predict_doc(doc, binary_classifier, multi_class_classifier), axis=1)
    return labeled_docs

def evaluate_predictions(docs: Union[str, Path],
                        pred_frame: pd.DataFrame,
                        target_attribute: str,
                        average: str = 'micro'):
    '''

    :param docs: Path to data file that contains true labels for target_attribute
    :param pred_frame: Pandas.DataFrame that contains target_attribute as column
    :param target_attribute: Target attribute
    :param average: Method to calculate f1-score. From https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html:
        'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
        'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
        'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
                    This alters 'macro' to account for label imbalance; it can result in an F-score that is not between precision and recall.
    :return: F1-score
    '''

    doc_true = pd.read_csv(docs, sep='\t', header=0)
    target_true = doc_true[target_attribute].tolist()

    target_pred = pred_frame[target_attribute].tolist()

    return f1_score(target_true, target_pred, average=average)

if __name__ == "__main__":
    """
    python pipeline.py config_file
    """

    with open(sys.argv[1], 'rb') as f:
        config = json.load(f)

    prediction_file = config['prediction_file']
    binary_model_dir = config['binary_model_dir']
    multi_class_model_dir = config.get('multi_class_model_dir', None)
    export_columns = config['export_column']
    prediction_output_files = config['prediction_output_files']

    start = time.time()
    df = predict_docs(prediction_file,
                        binary_classifier_model_dir=binary_model_dir,
                        multi_class_classifier_model_dir= multi_class_model_dir
                        )
    print('runtime:', '{0:.2f}'.format(time.time() - start), 'seconds')

    for i, column in enumerate(export_columns):
        df[['pmid', column]].to_csv(prediction_output_files[i], sep=',', index=False)
        print('Exported output to:', prediction_output_files[i])
