import pandas as pd
from data_preprocessing import preproc_funcs
from pathlib import Path
import json

def include_index(datapath):
    meco_df = pd.read_csv(datapath / "hp_augmented_meco_100_1000_1_10.csv")
    texts_df = pd.read_csv(datapath / "hp_eng_texts_100_1000_1_10.csv")

    df_indices = preproc_funcs.include_index(meco_df, texts_df)
    df_nans = preproc_funcs.handle_nans(df_indices)
    df_indices.to_csv(datapath / "hp_augmented_meco_100_1000_1_10_sacc_idx.csv", index=False)
    df_nans.to_csv(datapath / "hp_augmented_meco_100_1000_1_10_model.csv", index=False)
    d = preproc_funcs.compute_variances(df_indices)
    with open(datapath / "variances.json", "w") as f:
        json.dump(d, f, indent=4)
    print("Saved indices to", datapath)