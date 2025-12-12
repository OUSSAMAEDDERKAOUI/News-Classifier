from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

def run() :
    df_train = pd.read_csv("./data/raw/df_train.csv")
    df_test  = pd.read_csv("./data/raw/df_test.csv")

    df_train["text"] = df_train["text"].str.replace(r'\s+', ' ', regex=True)
    df_test["text"]  = df_test["text"].str.replace(r'\s+', ' ', regex=True)

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    train_embeddings = model.encode(df_train["text"].tolist(), show_progress_bar=True, normalize_embeddings=True)
    test_embeddings  = model.encode(df_test["text"].tolist(), show_progress_bar=True, normalize_embeddings=True)

    df_train["embedding"] = list(train_embeddings)
    df_test["embedding"]  = list(test_embeddings)



    np.save('./data/embeddings/train_embeddings.npy', train_embeddings)
    np.save('./data/embeddings/test_embeddings.npy', test_embeddings)


    df_train["id"] = df_train.index.astype(str)
    df_test["id"]  = df_test.index.astype(str)

    train_metadata = df_train[["id", "label", "label_text"]]
    test_metadata  = df_test[["id", "label", "label_text"]]


    train_metadata.to_csv("./data/metadata/train_metadata.csv", index=False)
    test_metadata.to_csv("./data/metadata/test_metadata.csv", index=False)
