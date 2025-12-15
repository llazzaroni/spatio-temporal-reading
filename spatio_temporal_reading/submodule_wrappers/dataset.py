import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from submodule.src.dataset.dataset import MecoDataset
from submodule.src.consts import (
    ENABLE_DATA_VISUALIZATION,
)
from submodule.src.dataset.dataset_visuals import batch_plot_fixations
from submodule.src.paths import (
    CHARACTER_SURPS_PATH,
    IMAGES_MACO_DIR,
    WORDS_SURPS_PATH,
    DATA_DIR,
    MACO_DATASET_DIR,
    TEXTS_DF_PATH,
)
from submodule.src.dataset.preprocess_funcs import *

class MecoDatasetW(MecoDataset):
    def __init__(
        self,
        mode,
        filtering,
        splitting_procedure,
        feature_func_stpp,
        feature_func_dur,
        division_factor_space,
        division_factor_time,
        division_factor_durations,
        past_timesteps_duration_baseline_k,
        cfg,
        datadir,
        language="en",
    ):

        if filtering not in ["filtered", "raw"]:
            raise ValueError("Filtering must be either 'filtered' or 'raw'")

        if splitting_procedure != "random_shuffle":
            raise ValueError("splitting_procedure not supported")

        if language != "en":
            raise ValueError("Only English is supported")

        ##########################
        # LOAD MACO DATAFRAME     #
        ##########################

        self.meco_df = pd.read_csv(datadir / "hp_augmented_meco_100_1000_1_10_model.csv").copy()
        texts_df = pd.read_csv(datadir / "hp_eng_texts_100_1000_1_10.csv").copy()

        total_na_values_freq = self.meco_df["freq"].isna().sum()
        self.meco_df["freq"] = -np.log(self.meco_df["freq"] + 1e-9)
        self.meco_df["dur"] = np.log(self.meco_df["dur"] + 1e-9)

        assert total_na_values_freq == self.meco_df["freq"].isna().sum()

        if ENABLE_DATA_VISUALIZATION and mode == "train":
            print(
                "Visualizing scanpaths on the Meco dataset texts. This may take a while."
            )
            batch_plot_fixations(self.meco_df, texts_df, IMAGES_MACO_DIR)

        self.reader_to_idx = {
            reader: idx
            for idx, reader in enumerate(self.meco_df["reader"].sort_values().unique())
        }

        ###############################
        # ENCODE CHARACTERS AS NUMBERS #
        ###############################

        unique_chars = sorted(
            "".join(texts_df["character"].apply(lambda x: x.lower()).unique())
        )
        self.filtering = filtering
        unique_chars = list(set(unique_chars))
        unique_chars.append("lb")
        char_to_idx = {char: i for i, char in enumerate(unique_chars)}
        self.idx_to_char = {i: char for i, char in enumerate(iterable=unique_chars)}

        ###############################
        # CREATE BOXES TENSOR         #
        ###############################

        self.texts_df = texts_df
        self.texts_df["c_value"] = 1
        self.texts_df["char_idx"] = self.texts_df["character"].apply(
            lambda x: char_to_idx[x.lower()]
        )
        self.texts_df["is_capitalized"] = self.texts_df["character"].apply(
            lambda x: x.isupper()
        )
        # check well processing of one hot encoders
        # ids = (self.one_hot == 1).nonzero(as_tuple = False)[:, -1]
        # "".join([idx_to_char[id.item()] for id in ids])

        self.boxes, self.one_hot, self.boxes_centroid, text_ids, self.char_info = (
            create_boxes_tensor_from_dataframe(self.texts_df)
        )

        ###############################
        # CHECK CHARACTER ORDER       #
        ###############################
        for row in self.char_info:
            mask = row != -1  # Identify values before reaching -1
            valid_values = row[mask]  # Extract values before -1
            if not torch.all(
                valid_values[1:] == valid_values[:-1] + 1
            ):  # Check increment condition
                raise ValueError("Invalid character order")

        ###############################
        # CHECK MASK CONSISTENCY      #
        ###############################
        mask_centroid = (self.boxes_centroid == -1).all(axis=2)
        mask_boxes = (self.boxes == -1).all(axis=2)
        mask_one_hot = (self.one_hot == -1).all(axis=2)
        if (mask_centroid != mask_boxes).any() or (mask_centroid != mask_one_hot).any():
            raise ValueError(
                "Inconsistent masks: boxes, one_hot, and centroid masks do not match."
            )

        self.texts_df["x_diff"] = self.texts_df["bbox_x2"] - self.texts_df["bbox_x1"]
        self.texts_df["y_diff"] = self.texts_df["bbox_y2"] - self.texts_df["bbox_y1"]

        ###############################
        # CREATE DATASET SPLIT        #
        ###############################

        # WE FIX A RANDOM SEED FOR THE DATASET, THAT ENSURES THAT THE SPLIT IS CONSISTENT AMONG DIFFERENT DATASET INSTANCES AND RUNS
        self.rnd_seed = 1242
        self.rnd = np.random.RandomState(self.rnd_seed)
        self.feature_func_stpp = feature_func_stpp
        self.feature_func_dur = feature_func_dur

        if self.filtering == "filtered":
            self.meco_df = self.meco_df[self.meco_df["ianum_word"].isna() == False]
            self.meco_df = self.meco_df.sort_values(
                ["text", "reader", "fixid"], ascending=True
            )
            self.meco_df["fixid"] = (
                self.meco_df.groupby(["text", "reader"]).cumcount() + 1
            )

        train_items, valid_items, test_items = self.get_splitting(
            splitting_procedure=splitting_procedure
        )

        self.normalize_predictors = True

        if mode == "train":
            self.items = train_items
        elif mode == "valid":
            self.items = valid_items
        elif mode == "test":
            self.items = test_items
        self.indices = {idx: item for idx, item in enumerate(self.items)}
        dic = {
            **dict.fromkeys(train_items, "train"),
            **dict.fromkeys(valid_items, "valid"),
            **dict.fromkeys(test_items, "test"),
        }
        self.meco_df["split"] = [
            (
                "held_out_session"
                if (text == 3 and reader == 70)
                else dic[(text, reader, fixid - 1)]
            )
            for text, reader, fixid in zip(
                self.meco_df.text, self.meco_df.reader, self.meco_df.fixid
            )
        ]
        # Make sure the target directory exists before saving.
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.meco_df.to_csv(
            path_or_buf=DATA_DIR / f"meco_df_{self.filtering}.csv", index=False
        )

        if self.normalize_predictors:
            # Work on a copy to avoid modifying the original
            self.meco_df = self.meco_df.copy()

            # Split into train, valid, test
            train_df = self.meco_df[self.meco_df["split"] == "train"]

            # Columns to normalize and mapping for output names
            predictor_cols = [
                "freq",
                "len",
                "char_level_surp",
                "word_level_surprisal",
                "dur",
            ]
            output_cols = [
                "freq",
                "len",
                "char_level_surp",
                "word_level_surprisal",
                "norm_dur",
            ]

            # Compute training set min and max
            train_min = train_df[predictor_cols].min()
            train_max = train_df[predictor_cols].max()

            # Apply normalization using train stats
            for orig_col, out_col in zip(predictor_cols, output_cols):
                self.meco_df[out_col] = (
                    self.meco_df[orig_col] - train_min[orig_col]
                ) / (train_max[orig_col] - train_min[orig_col])

    def get_splitting(self, splitting_procedure):

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

        # Test set is built like in the simple transformer
        test_items_form = [
            (idx_text, idx_reader, single_count)
            for idx_text, idx_reader, obs_count in test_items
            for single_count in range(obs_count)
        ]

        # Shuffle again the rest
        non_test = item_corpus[: len(train_items) + len(val_items)]

        sample_list = [
            (idx_text, idx_reader, single_count)
            for idx_text, idx_reader, obs_count in non_test
            for single_count in range(obs_count)
        ]

        self.rnd.shuffle(sample_list)
        train_items_form = sample_list[: int(0.89 * len(sample_list))]
        valid_items_form = sample_list[int(0.89 * len(sample_list)) :]
        return train_items_form, valid_items_form, test_items_form
        
        



        '''
        texts_readers_df = self.meco_df[["text", "reader"]].value_counts().reset_index()
        assert self.meco_df.shape[0] == texts_readers_df["count"].sum()

        if splitting_procedure == "random_shuffle":
            HOLD_OUT_SESSION = (3, 70)
            self.held_out_reader = tuple((HOLD_OUT_SESSION[0], HOLD_OUT_SESSION[1], 1))
            texts_readers_df = texts_readers_df.query(
                f"not (text == {HOLD_OUT_SESSION[0]} and reader == {HOLD_OUT_SESSION[1]})"
            )

        item_corpus = list(texts_readers_df.itertuples(index=False, name=None))

        if splitting_procedure == "random_shuffle":
            self.rnd.shuffle(item_corpus)
            sample_list = [
                (idx_text, idx_reader, single_count)
                for idx_text, idx_reader, obs_count in item_corpus
                for single_count in range(obs_count)
            ]

            self.rnd.shuffle(sample_list)
            train_items = sample_list[: int(0.8 * len(sample_list))]
            valid_items = sample_list[
                int(0.8 * len(sample_list)) : int(0.9 * len(sample_list))
            ]
            test_items = sample_list[int(0.9 * len(sample_list)) :]
            return train_items, valid_items, test_items

        '''
