from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

import spatio_temporal_reading.data.utils as utils
from spatio_temporal_reading.data.feature_funcs import feature_func

class MecoDataset(Dataset):

    def __init__(
            self,
            mode,
            datadir
    ):

        # Read the csvs
        self.meco_df = pd.read_csv(datadir / "hp_augmented_meco_100_1000_1_10_sacc_idx.csv").copy()
        self.texts_df = pd.read_csv(datadir / "hp_eng_texts_100_1000_1_10.csv").copy()

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

        # To understand later what for
        self.boxes, self.one_hot, self.boxes_centroid, text_ids, self.char_info = (
            utils.create_boxes_tensor_from_dataframe(self.texts_df)
        )

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
        
        # Save information about splits in the meco_df field

        self.indices = {idx: item for idx, item in enumerate(self.items)}
        dic = {
            **dict.fromkeys(train_items, "train"),
            **dict.fromkeys(valid_items, "valid"),
            **dict.fromkeys(test_items, "test"),
        }
        self.meco_df["split"] = [
            (dic[(text, reader, fixid - 1)])
            for text, reader, fixid in zip(
                self.meco_df.text, self.meco_df.reader, self.meco_df.fixid
            )
        ]

    
    def __get_item__(self, index):
        item = self.indices[index]

        text = item[0]
        reader = item[1]
        current_observation = item[2]

        subset = self.meco_df[(self.meco_df["text"] == text) & (self.meco_df["reader"] == reader)].copy()

        history_points = torch.tensor(subset[["saccade_intervals", "x", "y"]].values, dtype=torch.float32) # why saccade intervals?
        dur_tensor = torch.tensor(subset[["dur", "start"]].values, dtype=torch.float32)
        input_features_tensor_stpp = feature_func(
            subset, self.texts_df, self.reader_to_idx
        )

        result = torch.concatenate([history_points, dur_tensor, input_features_tensor_stpp], axis=1)

        return result




    def __len__(self):
        return len(self.items)


    def get_splitting(self):
        # Load the texts and readers associated with them
        texts_readers_df = self.meco_df[["text", "reader"]].value_counts().reset_index()
        
        # Turn them into a list
        item_corpus = list(texts_readers_df.itertuples(index=False, name=None))

        # Shuffle the list randomly
        self.rnd.shuffle(item_corpus)

        # Extend the list to sample the correct percentage
        sample_list = [
            (idx_text, idx_reader, single_count)
            for idx_text, idx_reader, abs_count in item_corpus
            for single_count in range(abs_count)
        ]

        train_items = sample_list[:int(0.8 * len(sample_list))]
        last_text_train = train_items[-1][0]
        last_reader_train = train_items[-1][1]

        i = len(train_items)

        while sample_list[i][0] == last_text_train and sample_list[i][1] == last_reader_train:
            if i == len(sample_list):
                raise Exception
            train_items.append(sample_list[i])
            i += 1

        train_share = len(train_items)
        
        val_items = sample_list[train_share:int(0.9 * len(sample_list))]
        last_text_val = val_items[-1][0]
        last_reader_val = val_items[-1][1]

        i = train_share + len(val_items)

        while sample_list[i][0] == last_text_val and sample_list[i][1] == last_reader_val:
            if i == len(sample_list):
                raise Exception
            val_items.append(sample_list[i])
            i += 1

        val_share = len(val_items)

        test_items = sample_list[train_share + val_share:]

        test_share = len(test_items)

        return train_items, val_items, test_items