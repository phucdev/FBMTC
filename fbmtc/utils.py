import pandas as pd
from sklearn.model_selection import train_test_split


pd.read_csv(DATAPATH+'/train.tsv', sep='\t', header=0)

train, dev, train_y, dev_y = train_test_split(train_df, train_df['is_cancer'], stratify=train_df['is_cancer'], test_size=0.2)

train[['is_cancer', 'text']].to_csv(DATAPATH+'/task1/train.csv', sep='\t', index = False, header = True)
dev[['is_cancer', 'text']].to_csv(DATAPATH+'/task1/dev.csv', sep='\t', index = False, header = True)

train[['doid', 'text']].to_csv(DATAPATH+'/task2/train.csv', sep='\t', index = False, header = True)
dev[['doid', 'text']].to_csv(DATAPATH+'/task2/dev.csv', sep='\t', index = False, header = True)
