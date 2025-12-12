from datasets import load_dataset
import pandas as pd

dataset=load_dataset("SetFit/ag_news")

df_train=pd.DataFrame(dataset["train"])
df_train.to_csv('./data/raw/df_train.csv',index=False)


df_test=pd.DataFrame(dataset["test"])
df_test.to_csv('./data/raw/df_test.csv',index=False)
