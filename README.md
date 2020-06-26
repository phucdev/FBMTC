# FlairBioMedTextClassification
Seminar "Klassifikation Biomedizinischer Texte" with Flair

Training Flair
```
CUDA_VISIBLE_DEVICES=0 python fbmtc/text_classifier.py configs/task1.json task1_classifier
```

Do predictions on hold-out data
```
CUDA_VISIBLE_DEVICES=0 python3 fbmtc/pipeline.py configs/predictions/task1_general.json 
```
