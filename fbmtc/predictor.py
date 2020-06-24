from pathlib import Path

from flair.models import TextClassifier
from flair.data import Sentence
from typing import Dict, List, Any, Union


def load_predictor(model_dir: Union[str, Path], archive_filename: str = "best-model.pt") -> TextClassifier:
    archive_path = Path(model_dir).joinpath(archive_filename)
    return TextClassifier.load(archive_path)


def predict_instance(classifier: TextClassifier, example: Sentence) -> Dict:
    flair_result = classifier.predict(example)[0]
    result = {
                "text": " ".join([token.text for token in flair_result.tokens]),
                "prediction": flair_result.labels[0].value
    }
    return result


def batched_predict_instances(
        classifier: TextClassifier,
        examples: List[Sentence],
        batch_size: int = 16) -> List[Dict[str, Any]]:
    results = []  # type: List[Dict[str, Any]]
    for i in range(0, len(examples), batch_size):
        batch_examples = examples[i: i + batch_size]
        flair_batch_results = classifier.predict(batch_examples, mini_batch_size=batch_size)

        batch_results = [{"text": " ".join([token.text for token in r.tokens]),
                          "prediction": r}
                         for r in flair_batch_results]
        results.extend(batch_results)
    return results
