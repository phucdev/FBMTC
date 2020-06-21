from flair.embeddings import WordEmbeddings, FastTextEmbeddings, FlairEmbeddings, \
    TransformerWordEmbeddings


def get_general_embeddings():
    return [
        WordEmbeddings("en-crawl"),
        FlairEmbeddings("en-forward"),
        FlairEmbeddings("en-backward")
    ]


def get_mixed_bio_embeddings():
    return [
        FastTextEmbeddings("data/BioWordVec_PubMed_MIMICIII_d200.bin"),
        FlairEmbeddings("en-forward"),
        FlairEmbeddings("en-backward")
    ]


def get_bio_embeddings():
    return [
        FastTextEmbeddings("data/BioWordVec_PubMed_MIMICIII_d200.bin"),
        FlairEmbeddings("pubmed-forward"),
        FlairEmbeddings("pubmed-backward")
    ]


def get_scibert_flair_embeddings():
    return [
        TransformerWordEmbeddings(model="allenai/scibert_scivocab_uncased", fine_tune=True),
        FlairEmbeddings("pubmed-forward"),
        FlairEmbeddings("pubmed-backward")
    ]

# pd.read_csv(DATAPATH+'/train.tsv', sep='\t', header=0)
#
# train, dev, train_y, dev_y = train_test_split(train_df, train_df['is_cancer'], stratify=train_df['is_cancer'], test_size=0.2)
#
# train[['is_cancer', 'text']].to_csv(DATAPATH+'/task1/train.csv', sep='\t', index = False, header = True)
# dev[['is_cancer', 'text']].to_csv(DATAPATH+'/task1/dev.csv', sep='\t', index = False, header = True)
#
# train[['doid', 'text']].to_csv(DATAPATH+'/task2/train.csv', sep='\t', index = False, header = True)
# dev[['doid', 'text']].to_csv(DATAPATH+'/task2/dev.csv', sep='\t', index = False, header = True)

