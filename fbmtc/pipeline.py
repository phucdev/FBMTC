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


def predict_docs(docs, binary_classifier_path, multi_class_classifier_path):
    pred_docs = pd.read_csv(docs, sep='\t', header=0)

    binary_classifier = predictor.load_predictor(binary_classifier_path)
    multi_class_classifier = predictor.load_predictor(multi_class_classifier_path)

    labeled_docs = docs.apply(lambda doc: pred_docs(doc, binary_classifier, multi_class_classifier), axis=1)
    return labeled_docs
