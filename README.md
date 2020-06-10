# FlairBioMedTextClassification
Seminar "Klassifikation Biomedizinischer Texte" with Flair

Training Flair
```
CUDA_VISIBLE_DEVICES=0 python fbmtc/text_classifier.py configs/task1.json task1_classifier 2> stderr.log 1> stdout.log
```