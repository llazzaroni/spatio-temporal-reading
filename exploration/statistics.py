import numpy as np
import pandas as pd

def saccade_vs_chars(meco_df, texts_df):
    # Make sure the dataset is filtered
    df = meco_df.sort_values(["text", "reader", "fixid"]).copy()
    tdf = texts_df.sort_values(["idx", "text_id"]).copy()

    tdf["cx"] = 0.5 * (tdf["bbox_x1"] + tdf["bbox_x2"])
    tdf["cy"] = 0.5 * (tdf["bbox_y1"] + tdf["bbox_y2"])

    # Split by text and reader
    texts = df["text"].unique()
    readers = df["reader"].unique()

    char_dist = []
    saccade_durs = []

    i = 0
    # Main computation; split by text_id and reader_id
    for text in texts:
        for reader in readers:
            # Get the specific df
            df_loop = df[(df["text"] == text) & (df["reader"] == reader)]
            df_loop = df_loop.reset_index(drop=True).copy()
            df_text = tdf[(tdf["text_id"] == text)]
            df_text = df_text.reset_index(drop=True).copy()

            if df_loop.empty or df_text.empty:
                continue

            # Compute the durations
            dur_raw = df_loop["dur"] / 1000

            # Compute the saccades
            prev_end = df_loop["start"].shift(1) + dur_raw.shift(1)
            sacc_dur = df_loop["start"] - prev_end

            # Finally, compute the char distances
            df_loop["char_idx"] = df_loop.apply(lambda r: pick_id(df_text, r["x"], r["y"]), axis=1)
            df_loop["char_dist"] = df_loop["char_idx"].diff()

            
            '''
            idx_to_word = df_text.set_index("idx")["ia_word"]          # or whatever the column is called
            df_loop["word_from_idx"] = df_loop["char_idx"].map(idx_to_word)
            #print(df_loop["char_idx"].map(idx_to_word).head(100).to_list())

            #for j in range(len(df_loop)):
                #print(df_loop["word_from_idx"].iloc[j], df_loop["ia_word"].iloc[j])

            # compare to the word column in df_loop (adjust column name!)
            df_loop["word_match"] = (
                df_loop["word_from_idx"].fillna("").str.lower()
                == df_loop["ia_word"].fillna("").str.lower()
            )
            #print(df_loop["word_match"].head(100).to_list())

            # optional: ignore punctuation/whitespace differences
            # norm = lambda s: s.str.replace(r"\W+", "", regex=True).str.lower()
            # df_loop["word_match"] = norm(df_loop["word_from_idx"].fillna("")) == norm(df_loop["word"].fillna(""))
            # count matches for this (text, reader)
            n_match = int(df_loop["word_match"].notna().sum())
            n_checked = int(df_loop["word_match"].notna().sum())

            print(n_match/n_checked * 100)
            '''


            valid = sacc_dur.notna() & df_loop["char_dist"].notna() & df_loop["ia_word"].notna()
            #valid = sacc_dur.notna() & df_loop["char_dist"].notna()
            valid[1] = False
            saccade_durs.extend(sacc_dur[valid].to_numpy())
            char_dist.extend(df_loop.loc[valid, "char_dist"].to_numpy())

            
    return saccade_durs, char_dist

# Helper function for saccade_vs_chars
def pick_id(df_text, x, y):
    inside = (df_text["bbox_x1"] <= x) & (df_text["bbox_x2"] >= x) & (df_text["bbox_y1"] <= y) & (df_text["bbox_y2"] >= y)
    if inside.any():
        return int(df_text.loc[inside, "idx"].iloc[0])
    else:
        distance = (df_text["cx"] - x)**2 + (df_text["cy"] - y)**2
        row_label = distance.idxmin()
        return int(df_text.loc[row_label, "idx"])
    

def duration_vs_chars(meco_df, texts_df):
    # Make sure the dataset is filtered
    df = meco_df.sort_values(["text", "reader", "fixid"]).copy()
    tdf = texts_df.sort_values(["idx", "text_id"]).copy()

    tdf["cx"] = 0.5 * (tdf["bbox_x1"] + tdf["bbox_x2"])
    tdf["cy"] = 0.5 * (tdf["bbox_y1"] + tdf["bbox_y2"])

    # Split by text and reader
    texts = df["text"].unique()
    readers = df["reader"].unique()

    char_dist = []
    durs = []

    i = 0
    # Main computation; split by text_id and reader_id
    for text in texts:
        for reader in readers:
            # Get the specific df
            df_loop = df[(df["text"] == text) & (df["reader"] == reader)]
            df_loop = df_loop.reset_index(drop=True).copy()
            df_text = tdf[(tdf["text_id"] == text)]
            df_text = df_text.reset_index(drop=True).copy()

            if df_loop.empty or df_text.empty:
                continue

            # Compute the durations
            dur_raw = df_loop["dur"] / 1000            

            # Finally, compute the char distances
            df_loop["char_idx"] = df_loop.apply(lambda r: pick_id(df_text, r["x"], r["y"]), axis=1)
            df_loop["char_dist"] = df_loop["char_idx"].diff()

            valid = dur_raw.notna() & df_loop["char_dist"].notna() & df_loop["ia_word"].notna()
            #valid = dur_raw.notna() & df_loop["char_dist"].notna()
            valid[1] = False
            durs.extend(dur_raw[valid].to_numpy())
            char_dist.extend(df_loop.loc[valid, "char_dist"].to_numpy())

            
    return durs, char_dist

def duration_distributions(meco_df, texts_df):
    durations_backwards = meco_df[meco_df["backward_fixation"] == True]
    durations_backwards = np.log(durations_backwards["dur"].to_numpy() + 1e-9)
    durations_forward = meco_df[meco_df["forward_fixation"] == True]
    durations_forward = np.log(durations_forward["dur"].to_numpy() + 1e-9)
    return durations_backwards, durations_forward

def empirical_regression_probability_vs_chars(meco_df, texts_df):
    # Preprocess: create a column with the number of characters between the jumps

    # Copy of the code from durations vs chars
    df = meco_df.sort_values(["text", "reader", "fixid"]).copy()
    tdf = texts_df.sort_values(["idx", "text_id"]).copy()

    tdf["cx"] = 0.5 * (tdf["bbox_x1"] + tdf["bbox_x2"])
    tdf["cy"] = 0.5 * (tdf["bbox_y1"] + tdf["bbox_y2"])

    # Split by text and reader
    texts = df["text"].unique()
    readers = df["reader"].unique()

    # Collect the results
    frames = []

    # Main computation; split by text_id and reader_id
    for text in texts:
        for reader in readers:
            # Get the specific df
            df_loop = df[(df["text"] == text) & (df["reader"] == reader)]
            df_loop = df_loop.reset_index(drop=True).copy()
            df_text = tdf[(tdf["text_id"] == text)]
            df_text = df_text.reset_index(drop=True).copy()

            if df_loop.empty or df_text.empty:
                continue

            # Finally, compute the char distances
            df_loop["char_idx"] = df_loop.apply(lambda r: pick_id(df_text, r["x"], r["y"]), axis=1)
            df_loop["char_dist"] = df_loop["char_idx"].diff().abs()

            valid = df_loop["char_dist"].notna() & df_loop["ia_word"].notna()
            #valid = sacc_dur.notna() & df_loop["char_dist"].notna()
            #valid[1] = False
            if valid.any():
                frames.append(
                    df_loop[valid].copy()
                )
    
    df = pd.concat(frames, ignore_index=True)

    probabilities = []
    indices = []
    lo = 1
    hi = 1
    while hi < 1300:
        running_sum = 0
        hi = lo
        while running_sum <= 500:
            running_sum += (df["char_dist"] == hi).to_numpy().sum()
            hi += 1
            if hi >= 1300:
                break
        df_filtered = df[(df["char_dist"] >= lo) & (df["char_dist"] < hi)].copy()
        if len(df_filtered) >= 1:
            print(len(df_filtered))
            backward = (df_filtered["backward_fixation"] == True).to_numpy().sum()
            front = (df_filtered["forward_fixation"] == True).to_numpy().sum()
            probabilities.append(backward / (backward + front))
            indices.append([lo, hi])
        lo = hi
    return probabilities, indices

def empirical_regression_probability_vs_dur(meco_df):
    df = meco_df.sort_values(["text", "reader", "fixid"]).copy()
    df["dur"] = np.log(df["dur"])

    probabilities = []
    indices = []
    lo = 0
    hi = 0
    while hi < 8:
        running_sum = 0
        hi = lo
        while running_sum <= 900:
            running_sum += ((df["dur"] >= hi) & (df["dur"] < hi + 0.01)).to_numpy().sum()
            hi += 0.01
            if hi >= 8:
                break
        df_filtered = df[(df["dur"] >= lo) & (df["dur"] < hi)].copy()
        if len(df_filtered) >= 1:
            print(len(df_filtered))
            backward = (df_filtered["backward_fixation"] == True).to_numpy().sum()
            front = (df_filtered["forward_fixation"] == True).to_numpy().sum()
            probabilities.append(backward / (backward + front))
            indices.append([lo, hi])
        lo = hi
    return probabilities, indices

def empirical_regression_probability_vs_dur_pre(meco_df):
    # Copy of the code from durations vs chars
    df = meco_df.sort_values(["text", "reader", "fixid"]).copy()
    df["dur"] = np.log(df["dur"])

    # Split by text and reader
    texts = df["text"].unique()
    readers = df["reader"].unique()

    # Collect the results
    frames = []

    # Main computation; split by text_id and reader_id
    for text in texts:
        for reader in readers:
            # Get the specific df
            df_loop = df[(df["text"] == text) & (df["reader"] == reader)]
            df_loop = df_loop.reset_index(drop=True).copy()

            if df_loop.empty:
                continue

            df_loop["dur"] = df_loop["dur"].shift(1)

            valid = df_loop["dur"].notna()
            #valid = sacc_dur.notna() & df_loop["char_dist"].notna()
            #valid[1] = False
            if valid.any():
                frames.append(
                    df_loop[valid].copy()
                )
    
    df = pd.concat(frames, ignore_index=True)


    probabilities = []
    indices = []
    lo = 0
    hi = 0
    while hi < 8:
        running_sum = 0
        hi = lo
        while running_sum <= 900:
            running_sum += ((df["dur"] >= hi) & (df["dur"] < hi + 0.01)).to_numpy().sum()
            hi += 0.01
            if hi >= 8:
                break
        df_filtered = df[(df["dur"] >= lo) & (df["dur"] < hi)].copy()
        if len(df_filtered) >= 1:
            print(len(df_filtered))
            backward = (df_filtered["backward_fixation"] == True).to_numpy().sum()
            front = (df_filtered["forward_fixation"] == True).to_numpy().sum()
            probabilities.append(backward / (backward + front))
            indices.append([lo, hi])
        lo = hi
    return probabilities, indices


def jump_vs_surprisal_rank(meco_df, texts_df):
    # Make sure the dataset is filtered
    df = meco_df.sort_values(["text", "reader", "fixid"]).copy()
    tdf = texts_df.sort_values(["text_id", "idx"]).copy()

    tdf["cx"] = 0.5 * (tdf["bbox_x1"] + tdf["bbox_x2"])
    tdf["cy"] = 0.5 * (tdf["bbox_y1"] + tdf["bbox_y2"])

    # Split by text and reader
    texts = df["text"].unique()
    readers = df["reader"].unique()

    frames = []
    # Main computation; split by text_id and reader_id
    for text in texts:
        for reader in readers:
            # Get the specific df
            df_loop = df[(df["text"] == text) & (df["reader"] == reader)]
            df_loop = df_loop.reset_index(drop=True).copy()
            df_text = tdf[(tdf["text_id"] == text)]
            df_text = df_text.reset_index(drop=True).copy()

            if df_loop.empty or df_text.empty:
                continue

            # Finally, compute the char distances
            df_loop["char_idx"] = df_loop.apply(lambda r: pick_id(df_text, r["x"], r["y"]), axis=1)
            df_loop["char_dist"] = df_loop["char_idx"].diff().abs()

            df_loop["prev_char_idx"] = df_loop["char_idx"].shift(1)

            pairs = zip(df_loop["prev_char_idx"], df_loop["char_idx"])
            ranks_spans = [get_rank_and_span(df_text, int(a), int(b)) if pd.notna(a) and pd.notna(b) else (np.nan, np.nan)
                        for a, b in pairs]
            df_loop["surprisal_rank"] = [r for r, s in ranks_spans]
            df_loop["span_len"]       = [s for r, s in ranks_spans]

            df_loop = df_loop[df_loop["span_len"].notna() & (df_loop["span_len"] > 0)]
            df_loop["norm_rank"] = df_loop["surprisal_rank"] / df_loop["span_len"]

            consecutive = df_loop["fixid"].diff().eq(1)

            valid = df_loop["surprisal_rank"].notna() & df_loop["char_dist"].notna() & df_loop["ia_word"].notna() & consecutive
            if valid.any():
                frames.append(
                    df_loop[valid].copy()
                )
    
    df = pd.concat(frames, ignore_index=True)

    d = df.loc[df["norm_rank"].notna()].copy()
    d = d.sort_values("span_len")
    min_n = 1000
    d["bin_id"] = (np.arange(len(d)) // min_n)

    agg = (d.groupby("bin_id")
            .agg(
                span_min=("span_len", "min"),
                span_max=("span_len", "max"),
                n=("norm_rank", "size"),
                mean_norm_rank=("norm_rank", "mean"),
                std_norm_rank=("norm_rank", "std"),
            )
            .reset_index(drop=True))

    x = ((agg["span_min"] + agg["span_max"]) / 2).to_numpy()
    y = agg["mean_norm_rank"].to_numpy()

    return x, y


# Helper functions to get the rank of the surprisal
def get_rank_and_span(df_text, idx1, idx2):
    if pd.isna(idx1) or pd.isna(idx2) or idx1 == idx2:
        return np.nan, np.nan
    lo, hi = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)

    span = df_text[(df_text["idx"] > lo) & (df_text["idx"] <= hi)]
    if span.empty:
        return np.nan, np.nan

    landing = df_text.loc[df_text["idx"] == idx2, "char_level_surp"]
    if landing.empty:
        return np.nan, np.nan
    surprise = float(landing.iloc[0])

    surpr = span["char_level_surp"].to_numpy()
    rank = 1 + (surpr > surprise).sum()
    return int(rank), int(len(span))





                                  

