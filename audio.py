import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PMEmoDataset(Dataset):

    def __init__(
        self,
        feature_path="/home/yashkale/MER_VER3/PMEmo2019/features/dynamic_features.csv",
        va_path="/home/yashkale/MER_VER3/PMEmo2019/annotations/dynamic_annotations.csv",
    ):

        # ===============================
        # ---- Load audio features ------
        # ===============================

        feat_df = pd.read_csv(feature_path)
        feat_df = feat_df.sort_values(["musicId", "frameTime"])
        feat_df["musicId"] = feat_df["musicId"].astype(int)

        # ===============================
        # ---- Load VA annotations ------
        # ===============================

        va_df = pd.read_csv(va_path)
        va_df = va_df.sort_values(["musicId", "frameTime"])
        va_df["musicId"] = va_df["musicId"].astype(int)

        # ===============================
        # ---- Find common music IDs ----
        # ===============================

        feature_ids = set(feat_df["musicId"].unique())
        va_ids = set(va_df["musicId"].unique())

        common_ids = sorted(feature_ids.intersection(va_ids))
        self.music_ids = np.array(common_ids)

        # Keep only common songs
        feat_df = feat_df[feat_df["musicId"].isin(common_ids)]
        va_df = va_df[va_df["musicId"].isin(common_ids)]

        # ===============================
        # ---- Compute Feature Normalization ----
        # ===============================

        feature_values = feat_df.iloc[:, 2:].values.astype(np.float32)

        self.feat_mean = feature_values.mean(axis=0)
        self.feat_std = feature_values.std(axis=0) + 1e-8

        print("Feature normalization computed:")
        print("Feature dim:", self.feat_mean.shape[0])

        # ===============================
        # ---- Group by musicId ---------
        # ===============================

        self.feat_groups = {k: v for k, v in feat_df.groupby("musicId")}
        self.va_groups = {k: v for k, v in va_df.groupby("musicId")}

        print(f"Total songs loaded: {len(self.music_ids)}")

    def __len__(self):
        return len(self.music_ids)

    def __getitem__(self, idx):

        music_id = int(self.music_ids[idx])

        # ===============================
        # ---- Audio Features -----------
        # ===============================

        feat_song = self.feat_groups[music_id]

        feature_times = feat_song["frameTime"].values
        feat = feat_song.iloc[:, 2:].values.astype(np.float32)

        # Apply normalization
        feat = (feat - self.feat_mean) / self.feat_std

        # ===============================
        # ---- VA annotations -----------
        # ===============================

        va_song = self.va_groups[music_id]

        va_times = va_song["frameTime"].values
        valence = va_song["Valence(mean)"].values
        arousal = va_song["Arousal(mean)"].values

        # ===============================
        # ---- Align by Common Time -----
        # ===============================

        common_times = np.intersect1d(feature_times, va_times)

        feat_mask = np.isin(feature_times, common_times)
        va_mask = np.isin(va_times, common_times)

        aligned_feat = feat[feat_mask]

        aligned_va = np.stack(
            [valence[va_mask], arousal[va_mask]],
            axis=1
        ).astype(np.float32)

        # ===============================
        # ---- Safety Check -------------
        # ===============================

        assert len(aligned_feat) == len(aligned_va), \
            f"Mismatch for musicId {music_id}"

        return {
            "features": torch.from_numpy(aligned_feat),   # (T, 260)
            "va_curve": torch.from_numpy(aligned_va)      # (T, 2)
        }