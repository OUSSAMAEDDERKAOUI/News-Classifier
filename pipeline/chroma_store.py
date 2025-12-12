import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings
df_train = pd.read_csv("./data/raw/df_train.csv")
df_test  = pd.read_csv("./data/raw/df_test.csv")
train_metadata=pd.read_csv("./data/metadata/train_metadata.csv")
test_metadata=pd.read_csv("./data/metadata/test_metadata.csv")

train_embeddings=np.load('./data/embeddings/train_embeddings.npy')
test_embeddings=np.load('./data/embeddings/test_embeddings.npy')


# client = chromadb.Client(Settings(persist_directory="./data/chroma_db"))
client = chromadb.PersistentClient(path='./data/chroma_DB')
train_collection = client.create_collection("news_train", get_or_create=True)
test_collection  = client.create_collection("news_test",get_or_create=True)

batch_size = 5000
n = len(train_embeddings)

for i in range(0, n, batch_size):
    end = min(i + batch_size, n)

    batch_ids = [str(j) for j in list(train_metadata['id'][i: end])]
    batch_embeddings = train_embeddings[i:end].tolist()
    batch_metadatas = [{"label": int(train_metadata['label'][j])} for j in range(i, end)]

    train_collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas
    )

    print(f"Inserted {end} / {n}")



batch_size = 1000
n = len(test_embeddings)

for i in range(0, n, batch_size):
    end = min(i + batch_size, n)

    batch_ids = [str(j) for j in list(test_metadata['id'][i: end])]
    batch_embeddings = test_embeddings[i:end].tolist()
    batch_metadatas = [{"label": int(test_metadata['label'][j])} for j in range(i, end)]

    test_collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas
    )

    print(f"Inserted {end} / {n}")
