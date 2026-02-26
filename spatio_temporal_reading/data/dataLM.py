from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


from .data import MecoDataset

class MecoDatasetLM(MecoDataset):

    def __init__(
            self,
            mode,
            filtering,
            datadir
    ):

        # Read the csvs
        self.meco_df = pd.read_csv(datadir / "hp_augmented_meco_100_1000_1_10_model_tokens.csv").copy()
        self.texts_df = pd.read_csv(datadir / "hp_eng_texts_100_1000_1_10_tokens.csv").copy()
        self.embeddings = np.load(datadir / "hidden_states.npy")

        #print(self.embeddings.shape)

        if filtering == "filtered":
            self.meco_df = self.meco_df[self.meco_df["ianum_word"].isna() == False]
            print("gangang")

        sizes = self.meco_df.groupby(["text", "reader"]).size().to_numpy()
        self.max_len = sizes.max()

        self.meco_df["freq"] = -np.log(self.meco_df["freq"] + 1e-9) # unigram surprisal (log frequency)
        self.meco_df["dur"] = np.log(self.meco_df["dur"] + 1e-9)

        # To understand later what for
        unique_chars = list(self.texts_df["character"].apply(lambda x : x.lower()).unique())
        unique_chars.append("lb")
        char_to_idx = {char: i for i, char in enumerate(unique_chars)}
        self.idx_to_char = {i: char for i, char in enumerate(iterable=unique_chars)}
        self.reader_to_idx = {
            reader: idx
            for idx, reader in enumerate(self.meco_df["reader"].sort_values().unique())
        }
        self.texts_df["char_idx"] = self.texts_df["character"].apply(lambda x: char_to_idx[x.lower()])
        self.texts_df["c_value"] = 1
        self.texts_df["is_capitalized"] = self.texts_df["character"].apply(lambda x: x.isupper())

        self.texts_df["x_diff"] = self.texts_df["bbox_x2"] - self.texts_df["bbox_x1"]
        self.texts_df["y_diff"] = self.texts_df["bbox_y2"] - self.texts_df["bbox_y1"]

        # Fix a random seed for the dataset, so that it will be split the same way for each run
        self.rnd_seed = 1242
        self.rnd = np.random.RandomState(self.rnd_seed)

        train_items, valid_items, test_items = self.get_splitting()

        if mode == "train":
            self.items = train_items
        elif mode == "valid":
            self.items = valid_items
        elif mode == "test":
            self.items = test_items

        self.d_in_saccade = 2 + 2 + len(self.reader_to_idx) + 8 + 1 + 1 + 4


    def __getitem__(self, index):
        item = self.items[index]

        text = item[0]
        reader = item[1]

        subset = self.meco_df[(self.meco_df["text"] == text) & (self.meco_df["reader"] == reader)].copy()

        # Spatial information
        # No nans in history points
        history_points = torch.tensor(subset[["dx", "dy"]].values, dtype=torch.float32)
        zero_row = torch.zeros((1, history_points.shape[1]), dtype=torch.float32)
        history_points = torch.cat([zero_row, history_points], dim=0)

        # Temporal information
        # No nans in temporal information
        dur_tensor = torch.tensor(subset["dur"].values, dtype=torch.float32)
        dur_tensor = torch.cat([torch.zeros(1), dur_tensor], dim=0)
        start_tensor = torch.tensor(subset["start"].values, dtype=torch.float32)
        start_tensor = torch.cat([torch.zeros(1), start_tensor], dim=0)
        sacc_tensor = torch.tensor(subset["saccade"].values, dtype=torch.float32)
        sacc_tensor = torch.cat([torch.zeros(1), sacc_tensor], dim=0)

        # Reader information
        # No nans in reader information
        reader_id = subset["reader"].iloc[0]
        reader_idx = self.reader_to_idx[reader_id]
        reader_emb = torch.zeros(len(subset) + 1, len(self.reader_to_idx), dtype=torch.float32)
        reader_emb[:, reader_idx] = 1

        # Features
        features = torch.tensor(subset[["char_level_surp", "word_level_surprisal", "len", "freq", "char_level_surp_nan", "word_level_surprisal_nan", "len_nan", "freq_nan"]].values, dtype=torch.float32)
        zero_row = torch.zeros((1, features.shape[1]), dtype=torch.float32)
        features = torch.cat([zero_row, features], dim=0)

        # BOS token
        BOS_token = torch.zeros((features.shape[0], 1), dtype=torch.float32)
        BOS_token[0] = 1

        # LM embeddings
        text_specific_embeddings = self.embeddings[text - 1]
        emb_indices = subset["token_index"].values

        emb_dim = text_specific_embeddings.shape[1]
        lm_emb = np.zeros((len(emb_indices), emb_dim), dtype=text_specific_embeddings.dtype)
        valid = emb_indices >= 0
        lm_emb[valid] = text_specific_embeddings[emb_indices[valid]]
        lm_emb = torch.tensor(lm_emb, dtype=torch.float32)

        empty_token = emb_indices < 0
        empty_token = torch.tensor(empty_token, dtype=torch.float32)
        empty_token = torch.cat([torch.zeros(1), empty_token], dim=0)

        zero_row = torch.zeros((1, lm_emb.shape[1]), dtype=torch.float32)
        lm_emb = torch.cat([zero_row, lm_emb], dim=0)


        input_model = torch.cat([
            history_points[:-1, :],
            dur_tensor[:-1].unsqueeze(-1),
            start_tensor[:-1].unsqueeze(-1),
            reader_emb[:-1, :],
            features[:-1, :],
            BOS_token[:-1, :],
            lm_emb[:-1, :],
            empty_token[:-1].unsqueeze(-1)],
            dim=-1
        )

        positions_target = history_points[1:, :]
        saccades_target = sacc_tensor[1:]

        return (
            input_model,
            positions_target,
            saccades_target
        )
    
class MecoDatasetLM_conv(MecoDatasetLM):
    CONTEXT_WINDOW = 3

    def __getitem__(self, index):
        item = self.items[index]

        text = item[0]
        reader = item[1]

        subset = self.meco_df[(self.meco_df["text"] == text) & (self.meco_df["reader"] == reader)].copy()

        # Spatial information
        # No nans in history points
        history_points = torch.tensor(subset[["dx", "dy"]].values, dtype=torch.float32)
        zero_row = torch.zeros((1, history_points.shape[1]), dtype=torch.float32)
        history_points = torch.cat([zero_row, history_points], dim=0)

        # Temporal information
        # No nans in temporal information
        dur_tensor = torch.tensor(subset["dur"].values, dtype=torch.float32)
        dur_tensor = torch.cat([torch.zeros(1), dur_tensor], dim=0)
        start_tensor = torch.tensor(subset["start"].values, dtype=torch.float32)
        start_tensor = torch.cat([torch.zeros(1), start_tensor], dim=0)
        sacc_tensor = torch.tensor(subset["saccade"].values, dtype=torch.float32)
        sacc_tensor = torch.cat([torch.zeros(1), sacc_tensor], dim=0)

        # Reader information
        # No nans in reader information
        reader_id = subset["reader"].iloc[0]
        reader_idx = self.reader_to_idx[reader_id]
        reader_emb = torch.zeros(len(subset) + 1, len(self.reader_to_idx), dtype=torch.float32)
        reader_emb[:, reader_idx] = 1

        # Features
        features = torch.tensor(subset[["char_level_surp", "word_level_surprisal", "len", "freq", "char_level_surp_nan", "word_level_surprisal_nan", "len_nan", "freq_nan"]].values, dtype=torch.float32)
        zero_row = torch.zeros((1, features.shape[1]), dtype=torch.float32)
        features = torch.cat([zero_row, features], dim=0)

        # BOS token
        BOS_token = torch.zeros((features.shape[0], 1), dtype=torch.float32)
        BOS_token[0] = 1

        # LM embeddings
        text_specific_embeddings = self.embeddings[text - 1]
        emb_indices = subset["token_index"].values

        empty_token = emb_indices < 0
        empty_token = torch.tensor(empty_token, dtype=torch.float32)
        empty_token = torch.cat([torch.zeros(1), empty_token], dim=0)

        lm_emb, ctx_valid = self.get_LM_embeddings(emb_indices, text_specific_embeddings)

        zero_row = torch.zeros((1, lm_emb.shape[1], lm_emb.shape[2]), dtype=torch.float32)
        lm_emb = torch.cat([zero_row, lm_emb], dim=0)

        T = lm_emb.shape[0]
        lm_emb_flat = lm_emb.reshape(T, 768*3)

        zero_row = torch.zeros((1, ctx_valid.shape[1]), dtype=torch.float32)
        ctx_valid = torch.cat([zero_row, ctx_valid], dim=0)

        input_model = torch.cat([
            history_points[:-1, :],
            dur_tensor[:-1].unsqueeze(-1),
            start_tensor[:-1].unsqueeze(-1),
            reader_emb[:-1, :],
            features[:-1, :],
            BOS_token[:-1, :],
            empty_token[:-1].unsqueeze(-1),
            lm_emb_flat[:-1, :],
            ctx_valid[:-1, :]],
            dim=-1
        )

        positions_target = history_points[1:, :]
        saccades_target = sacc_tensor[1:]

        return (
            input_model,
            positions_target,
            saccades_target
        )
    
    def get_LM_embeddings(self, emb_indices, text_specific_embeddings):
        N = emb_indices.shape[0]
        T_text, d = text_specific_embeddings.shape

        offsets = np.arange(-self.CONTEXT_WINDOW, self.CONTEXT_WINDOW + 1, dtype=np.int64)
        ctx_idx = emb_indices[:, None] + offsets[None, :]

        center_valid = emb_indices >= 0
        ctx_valid = (
            center_valid[:, None]
            & (ctx_idx >= 0)
            & (ctx_idx < T_text)
        )

        lm_emb = np.zeros((N, d, offsets.shape[0]), dtype=text_specific_embeddings.dtype)
        rows, cols = np.nonzero(ctx_valid)
        lm_emb[rows, :, cols] = text_specific_embeddings[ctx_idx[rows, cols]]

        lm_emb = torch.tensor(lm_emb, dtype=torch.float32)
        ctx_valid = torch.tensor(ctx_valid, dtype=torch.float32)

        return lm_emb, ctx_valid
