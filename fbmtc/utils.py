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
        WordEmbeddings("en-crawl"),
        FlairEmbeddings("pubmed-forward"),
        FlairEmbeddings("pubmed-backward")
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


def get_class_balance(dataset):
    labels = [doc[0] for doc in dataset.raw_data]
    label_freqs = {}
    for clazz in set(labels):
        label_freqs[clazz] = float(labels.count(clazz))
    return label_freqs


def get_inverted_class_balance(dataset):
    class_balance = get_class_balance(dataset)
    # Invert
    for k, v in class_balance.items():
        class_balance[k] = (1 / v)
    weights_sum = sum(class_balance.values())
    # Normalize
    for k, v in class_balance.items():
        class_balance[k] = v / weights_sum
    return class_balance


# pd.read_csv(DATAPATH+'/train.tsv', sep='\t', header=0)
#
# train, dev, train_y, dev_y = train_test_split(train_df, train_df['is_cancer'], stratify=train_df['is_cancer'], test_size=0.2)
#
# train[['is_cancer', 'text']].to_csv(DATAPATH+'/task1/train.csv', sep='\t', index = False, header = True)
# dev[['is_cancer', 'text']].to_csv(DATAPATH+'/task1/dev.csv', sep='\t', index = False, header = True)
#
# train[['doid', 'text']].to_csv(DATAPATH+'/task2/train.csv', sep='\t', index = False, header = True)
# dev[['doid', 'text']].to_csv(DATAPATH+'/task2/dev.csv', sep='\t', index = False, header = True)

