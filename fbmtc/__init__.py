from typing import Dict

from flair.embeddings import TokenEmbeddings, WordEmbeddings, FastTextEmbeddings, FlairEmbeddings, \
    TransformerWordEmbeddings

embedding_mapping: Dict[str, TokenEmbeddings] = {
    # GloVe
    "en-glove": WordEmbeddings("en-glove"),

    # FastText variants
    "en-crawl": WordEmbeddings("en-crawl"),
    "biowordvec": FastTextEmbeddings("https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec"
                                     "/BioWordVec_PubMed_MIMICIII_d200.bin", use_local=False),
    # Flair variants
    "en-forward": FlairEmbeddings("en-forward"),
    "en-backward": FlairEmbeddings("en-backward"),
    "en-forward-fast": FlairEmbeddings("en-forward-fast"),
    "en-backward-fast": FlairEmbeddings("en-backward-fast"),
    "pubmed_forward": FlairEmbeddings("pubmed-forward"),
    "pubmed_backward": FlairEmbeddings("pubmed-backward"),

    # BERT variants
    "bert-base-cased": TransformerWordEmbeddings(model="bert-base-cased", fine_tune=True),
    # TODO check cased vs. uncased: mix and match problem?
    # https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_cased.tar
    # https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar
    "scibert": TransformerWordEmbeddings(model="PLACEHOLDER", fine_tune=True)
}
