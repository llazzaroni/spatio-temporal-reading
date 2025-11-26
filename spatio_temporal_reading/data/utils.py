import torch
import numpy as np

def create_boxes_tensor_from_dataframe(
    df,
    x1_col="bbox_x1",
    x2_col="bbox_x2",
    y1_col="bbox_y1",
    y2_col="bbox_y2",
    c_col="is_capitalized",
    textid_col="text_id",
):
    """
    Reads bounding-box info from a pandas DataFrame for many text_ids,
    and transforms them into a single 3D torch.Tensor of shape [T, N_max, 5].
    The last dimension is [x1_min, x1_max, y1_min, y1_max, c].

    Returns:
        boxes_3d : FloatTensor of shape [T, N_max, 5]
                   (zero-padded for texts with fewer boxes)
        text_ids : list of unique text_ids in sorted order
    """
    text_ids = sorted(df[textid_col].unique())
    boxes_by_text = []
    lengths = []

    for tid in text_ids:
        sub_df = df[df[textid_col] == tid]

        x1_np = sub_df[x1_col].values
        x2_np = sub_df[x2_col].values
        y1_np = sub_df[y1_col].values
        y2_np = sub_df[y2_col].values

        centroid_x = (x1_np + x2_np) / 2
        centroid_y = (y1_np + y2_np) / 2

        c_np = sub_df[c_col].values
        char_idx = sub_df["char_idx"].values
        char_order_idx = sub_df["idx"]

        # shape [N_i, 5]
        data_np = np.stack(
            arrays=[
                x1_np,
                x2_np,
                y1_np,
                y2_np,
                c_np,
                char_idx,
                centroid_x,
                centroid_y,
                char_order_idx,
            ],
            axis=1,
        )
        data_t = torch.from_numpy(data_np).float()  # => shape [N_i, 5]

        boxes_by_text.append(data_t)
        lengths.append(data_t.shape[0])

    T = len(text_ids)
    N_max = max(lengths) if T > 0 else 0

    # tensor of all -1
    boxes_3d = torch.full(size=(T, N_max, 9), fill_value=-1, dtype=torch.float32)
    for i, data_t in enumerate(boxes_by_text):
        n_i = data_t.shape[0]
        boxes_3d[i, :n_i, :] = data_t

    centroids = boxes_3d[:, :, 6:8]
    char_order_id = boxes_3d[:, :, 8]
    boxes_3d = boxes_3d[:, :, :6]

    char_ids = boxes_3d[:, :, 5].to(torch.int64)
    N = char_ids.max() + 1
    mask = boxes_3d[:, :, 5] == -1

    one_hot = torch.zeros(
        boxes_3d.shape[0], boxes_3d.shape[1], int(N), dtype=torch.float32
    )

    char_ids[mask] = 0
    one_hot_encodings = one_hot.scatter(2, char_ids.unsqueeze(-1), 1)
    one_hot_encodings[mask] = -1

    return (
        boxes_3d[:, :, :-1],
        one_hot_encodings,
        centroids,
        text_ids,
        char_order_id,
    )