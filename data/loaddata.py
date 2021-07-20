from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import DataLoader
import pandas as pd

"""Takes in filename of CSV in data/ folder and returns pytorch dataset for further use.
    inputs: filename (str), train index (starting from document number of the CSV), test_idx (last x documents from the CSV), batch_size (batch size of documents while training)
    return: vocab: pandas DataFrame of vocab, docs: all the documents in the CSV in form of vectors, train_dl, test_dl: train and test DataLoader for PyTorch, bow: bag of words for each doc, texts:
"""
def text_data(filename, train_idx, test_idx, val_idx, batch_size):
    path = 'data/' + filename
    with open(path) as f:
        lines = f.read().splitlines()
    vectorizer = CountVectorizer(max_df=0.8, min_df=50, lowercase=True, stop_words='english')
    bow = vectorizer.fit_transform(lines)
    docs = torch.from_numpy(bow.toarray()).float()
    boww = vectorizer.inverse_transform(docs)
    texts = []
    for i in boww:
        texts.append(i.tolist())
    vocab = pd.DataFrame(columns=['word', 'index'])
    vocab['word'] = vectorizer.get_feature_names()
    vocab['index'] = vocab.index
    train_docs, val_docs, test_docs = docs[train_idx:], docs[-test_idx: -val_idx], docs[-val_idx: -1]
    train_dl = DataLoader([[train_docs[i]] for i in range(len(train_docs))], shuffle=True, batch_size=batch_size)
    val_dl = DataLoader([[val_docs[i]] for i in range(len(val_docs))], batch_size=batch_size)
    test_dl = DataLoader([[test_docs[i]] for i in range(len(test_docs))], batch_size=batch_size)
    return vocab, docs, train_dl, val_dl, test_dl, bow, texts