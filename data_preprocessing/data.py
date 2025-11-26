import pandas as pd
from data_preprocessing import preproc_funcs
from pathlib import Path

def include_index(datapath):
    meco_df = pd.read_csv(datapath / "hp_augmented_meco_100_1000_1_10.csv")
    texts_df = pd.read_csv(datapath / "hp_eng_texts_100_1000_1_10.csv")

    df_indices = preproc_funcs.include_index(meco_df, texts_df)
    df_indices.to_csv(datapath / "hp_augmented_meco_100_1000_1_10_sacc_idx.csv", index=False)
    print("Saved indices to", datapath)