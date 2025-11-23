import numpy as np
import pandas as pd

def saccades_vs_chars(meco_df):

    # Split by text and reader
    texts = meco_df["text"].unique()
    readers = meco_df["reader"].unique()

    char_dists = []
    saccade_durs = []

    step = 0
    # Main computation; split by text_id and reader_id
    for text in texts:
        for reader in readers:
            # Get the specific df
            df_loop = meco_df[(meco_df["text"] == text) & (meco_df["reader"] == reader)]
            df_loop = df_loop.reset_index(drop=True).copy()

            if df_loop.empty:
                continue

            saccades = []
            char_dist = []

            for i in range(1, len(df_loop)):
                if pd.isna(df_loop.loc[i, "char_idx"]) or pd.isna(df_loop.loc[i-1, "char_idx"]):
                    continue
                saccades.append(df_loop.loc[i, "saccade"])
                char_dist.append(df_loop.loc[i, "char_idx"] - df_loop.loc[i-1, "char_idx"])
            
            saccade_durs.extend(saccades)
            char_dists.extend(char_dist)

    return saccade_durs, char_dists
    

def durations_vs_chars(meco_df, pre):

    # Split by text and reader
    texts = meco_df["text"].unique()
    readers = meco_df["reader"].unique()

    char_dists = []
    durs = []

    # Main computation; split by text_id and reader_id
    for text in texts:
        for reader in readers:
            # Get the specific df
            df_loop = meco_df[(meco_df["text"] == text) & (meco_df["reader"] == reader)]
            df_loop = df_loop.reset_index(drop=True).copy()

            if df_loop.empty:
                continue

            durations = []
            char_dist = []

            for i in range(1, len(df_loop)):
                if pd.isna(df_loop.loc[i, "char_idx"]) or pd.isna(df_loop.loc[i-1, "char_idx"]):
                    continue
                if pre:
                    durations.append(df_loop.loc[i-1, "dur"] / 1000)
                else:
                    durations.append(df_loop.loc[i, "dur"] / 1000)
                char_dist.append(df_loop.loc[i, "char_idx"] - df_loop.loc[i-1, "char_idx"])         

            durs.extend(durations)
            char_dists.extend(char_dist)
           
    return durs, char_dists


def duration_distributions(meco_df):
    durations_backwards = meco_df[meco_df["backward_fixation"] == True]
    durations_backwards = np.log(durations_backwards["dur"].to_numpy() + 1e-9)
    durations_forward = meco_df[meco_df["forward_fixation"] == True]
    durations_forward = np.log(durations_forward["dur"].to_numpy() + 1e-9)
    return durations_backwards, durations_forward


def empirical_regression_probability_vs_chars(meco_df):
    df = meco_df.copy()

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

            df_loop["char_dist"] = df_loop["char_idx"].diff().abs()

            valid = df_loop["char_dist"].notna()

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
            backward = (df_filtered["backward_fixation"] == True).to_numpy().sum()
            front = (df_filtered["forward_fixation"] == True).to_numpy().sum()
            probabilities.append(backward / (backward + front))
            indices.append([lo, hi])
        lo = hi
    return probabilities, indices


def empirical_regression_probability_vs_dur(meco_df):
    df = meco_df.copy()
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
            backward = (df_filtered["backward_fixation"] == True).to_numpy().sum()
            front = (df_filtered["forward_fixation"] == True).to_numpy().sum()
            probabilities.append(backward / (backward + front))
            indices.append([lo, hi])
        lo = hi
    return probabilities, indices

def empirical_regression_probability_vs_dur_pre(meco_df):
    df = meco_df.copy()
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
            backward = (df_filtered["backward_fixation"] == True).to_numpy().sum()
            front = (df_filtered["forward_fixation"] == True).to_numpy().sum()
            probabilities.append(backward / (backward + front))
            indices.append([lo, hi])
        lo = hi
    return probabilities, indices


def saccade_length_forward_vs_backward(meco_df):
    # Make sure the dataset is filtered
    df = meco_df.copy()

    # Split by text and reader
    texts = df["text"].unique()
    readers = df["reader"].unique()

    forward = []
    backward = []

    # Main computation; split by text_id and reader_id
    for text in texts:
        for reader in readers:
            # Get the specific df
            df_loop = df[(df["text"] == text) & (df["reader"] == reader)]
            df_loop = df_loop.reset_index(drop=True).copy()

            if df_loop.empty:
                continue

            for i in range(1, len(df_loop)):
                if pd.isna(df_loop.loc[i, "char_idx"]) or pd.isna(df_loop.loc[i-1, "char_idx"]):
                    continue
                if df_loop.loc[i-1, "forward_fixation"]:
                    forward.append(df_loop.loc[i, "char_idx"] - df_loop.loc[i-1, "char_idx"])
                if df_loop.loc[i-1, "backward_fixation"]:
                    backward.append(df_loop.loc[i, "char_idx"] - df_loop.loc[i-1, "char_idx"])

    return forward, backward


def jump_vs_surprisal_rank(meco_df, texts_df):

    # Split by text and reader
    texts = meco_df["text"].unique()
    readers = meco_df["reader"].unique()

    char_spans_ranks = []

    step = 0

    # Main computation; split by text_id and reader_id
    for text in texts:
        for reader in readers:
            # Get the specific df
            df_loop = meco_df[(meco_df["text"] == text) & (meco_df["reader"] == reader)]
            df_loop = df_loop.reset_index(drop=True).copy()
            df_text = texts_df[(texts_df["text_id"] == text)]
            df_text = df_text.reset_index(drop=True).copy()

            if df_loop.empty or df_text.empty:
                continue

            df_loop["char_dist"] = df_loop["char_idx"].diff().abs()

            for i in range(1, len(df_loop)):
                if pd.isna(df_loop.loc[i, "char_idx"]) or pd.isna(df_loop.loc[i-1, "char_idx"]):
                    continue
                rank, dist = get_rank_and_span(df_text, df_loop.loc[i, "char_idx"], df_loop.loc[i-1, "char_idx"])
                char_spans_ranks.append([dist, rank])
            
            step += 1
            if step % 50 == 0:
                print("Reached step", step)
    
    char_spans_ranks = sorted(char_spans_ranks, key=lambda x: x[0])
    char_spans_df = pd.DataFrame(char_spans_ranks, columns=["span_len", "surprisal_rank"])

    mean_rank = []
    indices = []
    lo = 6
    hi = 6
    while hi < 1178:
        running_sum = 0
        hi = lo
        while running_sum <= 500:
            running_sum += (char_spans_df["span_len"] == hi).to_numpy().sum()
            hi += 1
            if hi >= 1178:
                break
        df_filtered = char_spans_df[(char_spans_df["span_len"] >= lo) & (char_spans_df["span_len"] < hi)].copy()
        if len(df_filtered) >= 1:
            indices.append(lo)
            df_filtered["norm_rank"] = df_filtered["surprisal_rank"] / (df_filtered["span_len"] - 1)
            mean_rank.append(df_filtered["norm_rank"].mean())
        lo = hi

    return indices, mean_rank

# Helper functions to get the rank of the surprisal
def get_rank_and_span(df_text, idx1, idx2):
    lo, hi = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)

    # Take the next 5 chars
    span = df_text[(df_text["idx"] > lo) & (df_text["idx"] <= hi+5)]
    if span.empty:
        return np.nan, np.nan

    landing = df_text.loc[df_text["idx"] == idx2, "char_level_surp"]
    if landing.empty:
        return np.nan, np.nan
    surprise = float(landing.iloc[0])

    surpr = span["char_level_surp"].to_numpy()
    rank = (surpr > surprise).sum()
    return int(rank), int(len(span))