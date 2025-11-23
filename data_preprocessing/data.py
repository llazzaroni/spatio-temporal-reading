import pandas as pd
from data_preprocessing import preproc_funcs
from pathlib import Path

def include_index(datapath):
    datapath = Path(datapath)
    meco_df = pd.read_csv(datapath / "hp_augmented_meco_100_1000_1_10.csv")
    texts_df = pd.read_csv(datapath / "hp_eng_texts_100_1000_1_10.csv")

    df_indices = preproc_funcs.include_index(meco_df, texts_df)
    df_indices.to_csv(datapath / "hp_augmented_meco_100_1000_1_10_sacc_idx.csv", index=False)

# df_indices = preproc_funcs.include_indices(meco_df, texts_df)
# df_indices.to_csv("/Users/lorenzolazzaroni/Documents/Programming/Python/Research in DS/spatio-temporal-reading-proj/data/df1.csv", index=False)
#df_indices = pd.read_csv("/Users/lorenzolazzaroni/Documents/Programming/Python/Research in DS/spatio-temporal-reading-proj/data/df1.csv")
#print(df_indices.iloc[0])

#df_indices = pd.read_csv("/Users/lorenzolazzaroni/Documents/Programming/Python/Research in DS/spatio-temporal-reading-proj/data/df1.csv")