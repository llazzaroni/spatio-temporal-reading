import pandas as pd
from data_preprocessing import preproc_funcs
from pathlib import Path
import numpy as np
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

def include_tokens(datapath):
    meco_df = pd.read_csv(datapath / "hp_augmented_meco_100_1000_1_10_model.csv")
    texts_df = pd.read_csv(datapath / "hp_eng_texts_100_1000_1_10.csv")

    df_token_indices, hidden_states = preproc_funcs.hidden_states(texts_df)
    df_token_indices.to_csv(datapath / "hp_eng_texts_100_1000_1_10_tokens.csv", index=False)
    np.save(datapath / "hidden_states.npy", hidden_states)

    meco_df = preproc_funcs.include_tokens_indices(df_token_indices, meco_df)
    meco_df.to_csv(datapath / "hp_augmented_meco_100_1000_1_10_model_tokens.csv", index=False)