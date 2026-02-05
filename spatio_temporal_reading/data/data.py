from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class MecoDataset(Dataset):

    def __init__(
            self,
            mode,
            filtering,
            datadir
    ):

        # Read the csvs
        self.meco_df = pd.read_csv(datadir / "hp_augmented_meco_100_1000_1_10_model.csv").copy()
        self.texts_df = pd.read_csv(datadir / "hp_eng_texts_100_1000_1_10.csv").copy()

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

        self.d_in_saccade = 2 + 2 + len(self.reader_to_idx) + 8

    
    def __getitem__(self, index):
        item = self.items[index]

        text = item[0]
        reader = item[1]

        subset = self.meco_df[(self.meco_df["text"] == text) & (self.meco_df["reader"] == reader)].copy()

        # Spatial information
        # No nans in history points
        history_points = torch.tensor(subset[["dx", "dy"]].values, dtype=torch.float32)

        # Temporal information
        # No nans in temporal information
        dur_tensor = torch.tensor(subset["dur"].values, dtype=torch.float32)
        start_tensor = torch.tensor(subset["start"].values, dtype=torch.float32)
        sacc_tensor = torch.tensor(subset["saccade"].values, dtype=torch.float32)

        # Reader information
        # No nans in reader information
        reader_id = subset["reader"].iloc[0]
        reader_idx = self.reader_to_idx[reader_id]
        reader_emb = torch.zeros(len(subset), len(self.reader_to_idx), dtype=torch.float32)
        reader_emb[:, reader_idx] = 1

        # Features
        features = torch.tensor(subset[["char_level_surp", "word_level_surprisal", "len", "freq", "char_level_surp_nan", "word_level_surprisal_nan", "len_nan", "freq_nan"]].values, dtype=torch.float32)


        #result = (torch.cat([history_points, dur_tensor, reader_emb, features], dim=-1))

        return (
            history_points,
            dur_tensor,
            start_tensor,
            sacc_tensor,
            reader_emb,
            features
        )


    def __len__(self):
        return len(self.items)


    def get_splitting(self):
        # Load the texts and readers associated with them
        texts_readers_df = self.meco_df[["text", "reader"]].value_counts().reset_index()
        
        # Turn them into a list
        item_corpus = list(texts_readers_df.itertuples(index=False, name=None))

        # Shuffle the list randomly
        self.rnd.shuffle(item_corpus)

        # Build train, val and test splits
        train_items = []
        len_train_items = 0
        for i in range(len(item_corpus)):
            train_items.append((item_corpus[i][0], item_corpus[i][1]))
            len_train_items += item_corpus[i][2]
            if len_train_items >= 0.8 * len(self.meco_df):
                break
        
        val_items = []
        len_val_items = 0
        for i in range(len(train_items), len(item_corpus)):
            val_items.append((item_corpus[i][0], item_corpus[i][1]))
            len_val_items += item_corpus[i][2]
            if len_val_items >= 0.1 * len(self.meco_df):
                break
        
        test_items = item_corpus[len(train_items) + len(val_items):]

        return train_items, val_items, test_items