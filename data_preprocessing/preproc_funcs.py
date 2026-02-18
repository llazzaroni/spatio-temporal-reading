import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def include_index(meco_df, texts_df):

    # Split by text and readers
    texts = meco_df["text"].unique()
    readers = meco_df["reader"].unique()

    dfs = []
    step = 0

    for text in texts:
        for reader in readers:
            df_loop = meco_df[(meco_df["text"] == text) & (meco_df["reader"] == reader)].copy()
            df_text = texts_df[texts_df["text_id"] == text]

            if df_loop.empty or df_text.empty:
                continue

            # Compute the durations
            dur_raw = df_loop["dur"] / 1000

            # Compute the saccades
            prev_end = df_loop["start"].shift(1) + dur_raw.shift(1)
            sacc_dur = df_loop["start"] - prev_end

            # Append the saccades
            df_loop["saccade"] = sacc_dur
            df_loop["saccade"].iloc[0] = df_loop["start"].iloc[0]

            # Append dx and dy
            df_loop["dx"] = df_loop["x"].diff(1)
            df_loop["dy"] = df_loop["y"].diff(1)

            # Handle the first fixation
            df_loop.loc[df_loop.index[0], "dx"] = df_loop.loc[df_loop.index[0], "x"]
            df_loop.loc[df_loop.index[0], "dy"] = df_loop.loc[df_loop.index[0], "y"]

            # Append the character id in the rows where the fixation fell inside of the bounding box
            idx_to_char = df_text.set_index("idx")["ia_word"]
            idx_to_word = df_text.set_index("idx")["character"]
            df_loop["char_idx"] = df_loop.apply(lambda r: pick_id_bbox(df_text, r["x"], r["y"]), axis=1)

            step += 1

            dfs.append(df_loop)
            if step % 50 == 0:
                print("Processed", step, "scanpaths")
    
    out = pd.concat(dfs, ignore_index=True)

    return out



def pick_id_bbox(df_text, x, y):
    inside = (df_text["bbox_x1"] <= x) & (df_text["bbox_x2"] >= x) & (df_text["bbox_y1"] <= y) & (df_text["bbox_y2"] >= y)
    if inside.any():
        return int(df_text.loc[inside, "idx"].iloc[0])
    else:
        return pd.NA

def handle_nans(df_meco: pd.DataFrame) -> pd.DataFrame:
    df = df_meco.copy()

    cols = ["char_level_surp", "word_level_surprisal", "len", "freq"]

    for col in cols:
        nans = df[col].isna()
        df[f"{col}_nan"] = nans.astype(int)
        df.loc[nans, col] = 0
        
    return df

def compute_variances(meco_df) -> dict:
    # Compute variance for displacement along the x and y axes
    valid = ~pd.isna(meco_df["saccade"])

    var_x = meco_df[valid]["dx"].var()
    var_y = meco_df[valid]["dy"].var()
    var_sacc = meco_df[valid]["saccade"].var()
    return {
        "var_x": var_x,
        "var_y": var_y,
        "var_sacc": var_sacc,
    }





def include_indices(meco_df, texts_df):

    # Split by text and reader
    texts = meco_df["text"].unique()
    readers = meco_df["reader"].unique()

    # Initialize the result
    dfs = []

    step = 0

    # Start the main loop
    for text in texts:
        for reader in readers:
            df_loop = meco_df[(meco_df["reader"] == reader) & (meco_df["text"] == text)].copy()
            df_text = texts_df[texts_df["text_id"] == text]

            if df_loop.empty or df_text.empty:
                continue

            # Compute the durations
            dur_raw = df_loop["dur"] / 1000

            # Compute the saccades
            prev_end = df_loop["start"].shift(1) + dur_raw.shift(1)
            sacc_dur = df_loop["start"] - prev_end

            # Append the saccades
            df_loop["saccade"] = sacc_dur

            # Comple df_loop with closest index information
            idx_to_char = df_text.set_index("idx")["ia_word"]
            idx_to_word = df_text.set_index("idx")["character"]
            for i in range(5):
                df_loop["char_idx_" + str(i)] = df_loop.apply(lambda r: pick_id(df_text, r["x"], r["y"])[i][0], axis=1)
                df_loop["char_" + str(i)] = df_loop["char_idx_" + str(i)].map(idx_to_char)
                df_loop["word_" + str(i)] = df_loop["char_idx_" + str(i)].map(idx_to_word)
                df_loop["dist_" + str(i)] = df_loop.apply(lambda r: pick_id(df_text, r["x"], r["y"])[i][1], axis=1)

            dfs.append(df_loop)
            step += 1
            if step % 25 == 0:
                print("Reached step", step)
    
    out = pd.concat(dfs, ignore_index=True)

    return out

# pick_id return an array with the following information:

def pick_id(df_text, x, y):
    inside = (df_text["bbox_x1"] <= x) & (df_text["bbox_x2"] >= x) & (df_text["bbox_y1"] <= y) & (df_text["bbox_y2"] >= y)
    if inside.any():
        result = []
        result.append([int(df_text.loc[inside, "idx"].iloc[0]), pd.NA])
        for i in range(4):
            result.append([pd.NA, pd.NA])
        return result
    else:
        result = []
        rows = []
        for i in range(5):
            mask = pd.Series(True, index=df_text.index)
            for row in rows:
                mask = mask & (df_text["line"] != row)
            text_wo_row = df_text[mask]
            distance = (text_wo_row["center_x"] - x) ** 2 + (text_wo_row["center_y"] - y)**2
            row_label = distance.idxmin()
            row = df_text.loc[row_label, "line"]
            rows.append(row)
            result.append([df_text.loc[row_label, "idx"], float(distance.loc[row_label])])
        return result
    
# Try to come up with an algorithm to detect to which row a fixation really belongs
def choose_line(meco_df, texts_df):

    # Split by text and reader
    texts = meco_df["text"].unique()
    readers = meco_df["reader"].unique()

    # Initialize the result
    dfs = []

    average_line_len = texts_df.groupby(["text_id", "line"]).size().mean()
    left_bound = 0.8 * average_line_len
    right_bound = 1.2 * average_line_len

    step = 0

    # Start the main loop
    for text in texts:
        for reader in readers:
            df_loop = meco_df[(meco_df["reader"] == reader) & (meco_df["text"] == text)].copy()

            df_loop = df_loop.reset_index(drop=True)

            if df_loop.empty:
                continue

            # Initialize with 0
            df_loop["final_idx"] = pd.Series([0] * len(df_loop), index=df_loop.index)

            # Assume the first line is always correct
            df_loop.loc[0, "final_idx"] = df_loop.loc[0, "char_idx_0"]

            # Apply choosing criterion on the previous index
            for i in range(1, len(df_loop)):
                # If the fixation is inside of the bounding box, consider it correct and continue
                if pd.isna(df_loop.loc[i, "dist_0"]):
                    continue
                # Compute the char dist
                char_dist = df_loop.loc[i, "char_idx_0"] - df_loop.loc[i-1, "final_idx"]
                if char_dist >= left_bound and char_dist <= right_bound:
                    print("hello world")



            df_loop["up_or_down"] = pd.Series([pd.NA] * len(df_loop), index=df_loop.index)

            # Assume the first fixation is correct, then measure the uncertain fixations as a function of the char_dist from the last sure fixation
            for i in range(1, len(df_loop)):
                if pd.isna(df_loop.loc[i, "dist_closest"]):
                    continue
                # Fixation in the middle of a line
                # Find the closest fixation that happened between bounding boxes
                for j in range(1, 10):
                    # Impose 10 as upper bound; if there is a gap of more than 10 fixations, treat this as "unknown"
                    if i - j < 0:
                        df_loop.loc[i, "up_or_down"] = "unknown"
                        break
                    if pd.isna(df_loop.loc[i-j, "dist_closest"]):
                        char_dist = df_loop.loc[i, "closest_idx"] - df_loop.loc[i-j, "closest_idx"]
                        if char_dist >= left_bound and char_dist <= right_bound:
                            df_loop.loc[i, "up_or_down"] = "down"
                        elif char_dist >= -right_bound and char_dist <= -left_bound:
                            df_loop.loc[i, "up_or_down"] = "up"
                        else:
                            df_loop.loc[i, "up_or_down"] = "same line"
                        break

            dfs.append(df_loop)
            step += 1
            if step % 50 == 0:
                print("Reached step", step)
    
    out = pd.concat(dfs, ignore_index=True)
    return out


def up_or_down_final(meco_df):

    # Split by text and reader
    texts = meco_df["text"].unique()
    readers = meco_df["reader"].unique()

    # Initialize the result
    dfs = []

    step = 0

    # Start the main loop
    for text in texts:
        for reader in readers:
            df_loop = meco_df[(meco_df["reader"] == reader) & (meco_df["text"] == text)].copy()

            df_loop = df_loop.reset_index(drop=True)

            if df_loop.empty:
                continue

            df_loop["closest_idx_proc"] = df_loop["closest_idx"]

            for i in range(1, len(df_loop)):
                if df_loop.loc[i, "up_or_down"] == "up" or df_loop.loc[i, "up_or_down"] == "down":
                    # Check that we are not dumb
                    if df_loop.loc[i, "up_or_down"] == "up":
                        if df_loop.loc[i, "closest_idx"] < df_loop.loc[i, "second_closest_idx"]:
                            print("DUMB")
                    if df_loop.loc[i, "up_or_down"] == "down":
                        if df_loop.loc[i, "closest_idx"] > df_loop.loc[i, "second_closest_idx"]:
                            print("DUMB")
                    df_loop.loc[i, "closest_idx_proc"] = df_loop.loc[i, "second_closest_idx"]
                
            dfs.append(df_loop)
    
    out = pd.concat(dfs, ignore_index=True)
    return out

def fix_unknowns(meco_df, texts_df):
    # Split by text and reader
    texts = meco_df["text"].unique()
    readers = meco_df["reader"].unique()

    # Initialize the result
    dfs = []

    average_line_len = texts_df.groupby(["text_id", "line"]).size().mean()
    left_bound = 0.8 * average_line_len
    right_bound = 1.2 * average_line_len

    step = 0

    # Start the main loop
    for text in texts:
        for reader in readers:
            df_loop = meco_df[(meco_df["reader"] == reader) & (meco_df["text"] == text)].copy()

            df_loop = df_loop.reset_index(drop=True)

            if df_loop.empty:
                continue

            # Assume the first fixation is correct, then measure the uncertain fixations as a function of the char_dist from the last sure fixation
            for i in range(1, len(df_loop)):
                if df_loop.loc[i, "up_or_down"] == "unknown":
                    char_dist = df_loop.loc[i, "closest_idx"] - df_loop.loc[i-1, "closest_idx_proc"]
                    if char_dist >= left_bound and char_dist <= right_bound:
                        df_loop.loc[i, "up_or_down"] = "down"
                    elif char_dist >= -right_bound and char_dist <= -left_bound:
                        df_loop.loc[i, "up_or_down"] = "up"
                    else:
                        df_loop.loc[i, "up_or_down"] = "same line"
                    break
                

                if pd.isna(df_loop.loc[i, "dist_closest"]):
                    continue
                # Fixation in the middle of a line
                # Find the closest fixation that happened between bounding boxes
                for j in range(1, 10):
                    # Impose 10 as upper bound; if there is a gap of more than 10 fixations, treat this as "unknown"
                    if i - j < 0:
                        df_loop.loc[i, "up_or_down"] = "unknown"
                        break
                    if pd.isna(df_loop.loc[i-j, "dist_closest"]):
                        char_dist = df_loop.loc[i, "closest_idx"] - df_loop.loc[i-j, "closest_idx"]
                        if char_dist >= left_bound and char_dist <= right_bound:
                            df_loop.loc[i, "up_or_down"] = "down"
                        elif char_dist >= -right_bound and char_dist <= -left_bound:
                            df_loop.loc[i, "up_or_down"] = "up"
                        else:
                            df_loop.loc[i, "up_or_down"] = "same line"
                        break

            dfs.append(df_loop)
            step += 1
            if step % 50 == 0:
                print("Reached step", step)
    
    out = pd.concat(dfs, ignore_index=True)
    return out

# def forward_backward(meco_df):
#     texts = meco_df["text"].unique()
#     readers = meco_df["reader"].unique()
#     dfs = []

#     print(meco_df.columns)

#     for text in texts:
#         for reader in readers:
#             df_loop = meco_df[(meco_df["reader"] == reader) & (meco_df["text"] == text)].copy()
#             df_loop = df_loop.reset_index(drop=True)

#             #print(df_loop)

#             if df_loop.empty:
#                 continue

#             # ensure numeric
#             df_loop["closest_idx_proc"] = pd.to_numeric(df_loop["closest_idx_proc"], errors="coerce")

#             # compute direction
#             delta = df_loop["closest_idx_proc"].diff()
#             df_loop["forward_proc"] = delta >= 0
#             df_loop["backward_proc"] = delta < 0

#             # first fixation — no diff available
#             df_loop.loc[0, "forward_proc"] = True
#             df_loop.loc[0, "backward_proc"] = False

#             dfs.append(df_loop)

#     result = pd.concat(dfs, ignore_index=True)

#     return result

# Clean everything and turn the table in the same format as before
def final(meco_df, texts_df):
    # Drop columns
    meco_df = meco_df.drop(["saccade_intervals",
                            "character",
                            "ia_word",
                            "ianum_word",
                            "line",
                            "char_level_surp",
                            "word_level_surprisal",
                            "word_bbox_x1",
                            "word_bbox_x2",
                            "word_bbox_y1",
                            "word_bbox_y2",
                            "len",
                            "word_len",
                            "backward_fixation",
                            "forward_fixation",
                            "same_word_fixation",
                            "first_word_fixation",
                            "new_line_fixation",
                            "backward_fixation_same_line_on_words",
                            "forward_fixation_same_line_on_words"], axis=1)
    
    meco_df = meco_df.rename(columns={"closest_idx_proc": "char_idx"})

    # Rebuild the columns
    texts = meco_df["text"].unique()
    readers = meco_df["reader"].unique()

    dfs = []

    for text in texts:
        for reader in readers:
            df_loop = meco_df[(meco_df["reader"] == reader) & (meco_df["text"] == text)].copy()
            df_text = texts_df[texts_df["text_id"] == text]

            df_loop = df_loop.reset_index(drop=True)

            if df_loop.empty:
                continue

            for i in range(len(df_loop)):
                # character, ia_word
                df_loop.loc[i, "character"] = df_text[df_text["idx"] == df_loop.loc[i, "char_idx"]].iloc[0]["character"]
                df_loop.loc[i, "ia_word"] = df_text[df_text["idx"] == df_loop.loc[i, "char_idx"]].iloc[0]["ia_word"]
                


            dfs.append(df_loop)

    out = pd.concat(dfs, ignore_index=True)
    
    return out

def hidden_states(texts_df):

    # Texts are already ordered (should be)
    text_indices = texts_df["text_id"].unique()
    encodings_list = []
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model_gpt = AutoModel.from_pretrained("gpt2")

    dfs = []

    for text_id in text_indices:
        text_df = texts_df[texts_df["text_id"] == text_id].copy()

        text_df["token_id"] = 0
        token_id = 0
        words = []
        current_word = ""
        for i in range(len(text_df)):
            char = text_df["character"].iloc[i]
            text_df.loc[text_df.index[i], "token_id"] = token_id
            if char != " ":
                current_word += char
            else:
                words.append(current_word)
                current_word = ""
                token_id += 1

            if i == len(text_df) - 1:
                words.append(current_word)

        dfs.append(text_df)

        model_gpt.eval()

        all_ids = []
        word_token_spans = []
        for i, w in enumerate(words):
            w_text = w if i == 0 else " " + w
            ids = tokenizer.encode(w_text, add_special_tokens=False)
            start = len(all_ids)
            all_ids.extend(ids)
            end = len(all_ids) - 1
            word_token_spans.append((start, end))

        input_ids = torch.tensor([all_ids])

        with torch.no_grad():
            out = model_gpt(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            H = out.hidden_states[-1][0]

        # prefix encoding at each word boundary = hidden state at last subtoken of that word
        word_states = [H[end] for (start, end) in word_token_spans]

        word_states_np = torch.stack(word_states).cpu().numpy()
        print(word_states_np.shape[0])
        encodings_list.append(word_states_np)

    df_token_index = pd.concat(dfs)

    max_len = max(arr.shape[0] for arr in encodings_list)
    padded = np.zeros((len(encodings_list), max_len, 768))
    for i, seq in enumerate((encodings_list)):
        L = seq.shape[0]
        padded[i, :L, :] = seq

    return df_token_index, padded

def include_tokens_indices(df_token_indices, meco_df):
    meco_df["token_index"] = 0
    for i in range(len(meco_df)):
        if pd.isna(meco_df.loc[meco_df.index[i], "ianum_word"]):
            meco_df.loc[meco_df.index[i], "token_index"] = -1
        else:
            char_idx = int(meco_df.loc[meco_df.index[i], "char_idx"])
            text_id = int(meco_df.loc[meco_df.index[i], "text"])
            token_index = df_token_indices[(df_token_indices["idx"] == char_idx) & (df_token_indices["text_id"] == text_id)]["token_id"].iloc[0]
            meco_df.loc[meco_df.index[i], "token_index"] = token_index
    return meco_df