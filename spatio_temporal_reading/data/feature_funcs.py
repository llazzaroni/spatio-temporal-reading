import torch

def feature_func(subset, texts_df, reader_to_idx):

    subset = subset.copy()

    # Potentially add a text embedding; also a text embedding up to that point

    reader_id = subset["reader"].iloc[0]
    reader_idx = reader_to_idx[reader_id]
    reader_emb = torch.zeros(len(subset), len(reader_to_idx), dtype=torch.float32)
    reader_emb[:, reader_idx] = 1

    # How are nans handled?
    features = torch.tensor(subset[["char_level_surp", "word_level_surprisal", "len", "freq"]].values, dtype=torch.float32)

    result = torch.concatenate([reader_emb, features], axis=1)

    return result