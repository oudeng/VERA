# KNN_v1.py
# -*- coding: utf-8 -*-
"""Gower-distance KNN baseline, ported from ``project_sni_R0/sni/baselines/KNN_v1.py``.

R1 changes (P1 T1.4)
--------------------
* R0 normalized the Gower distance by continuous ranges computed from the
  ground-truth table (``KNN_v1.py:40-48`` called from ``:110`` with the frame the
  registry passed at ``registry.py:81``, which was ``X_complete``). In R1 the
  ranges come from :class:`~baselines.schema.ObservedStats`, i.e. from the
  observed cells only. The distance function, the neighbor selection and the
  sequential in-frame update order are untouched.
* :meth:`fit` / :meth:`transform` are new. ``fit`` imputes the training fold with
  the R0 in-frame procedure and keeps it as the donor pool; ``transform`` scores
  each new row against that fixed pool. No information flows from the new frame
  back into the pool, which is the property reviewer R1-3 asks us to demonstrate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import Counter

class knnImputer:
    """
    基于 Gower 距离的 KNN 插补器，支持处理混合型数据（连续 + 分类）。
    接口与 MissForest、MIWAE 等保持一致，可在 JupyterLab 中直接：
        from KNN_v1 import knnImputer
        knn = knnImputer(categorical_vars=[...], continuous_vars=[...], k=5)
        X_imp, _ = knn.impute(X_incomplete_df, X_missing_df)
    """

    def __init__(
        self,
        categorical_vars,
        continuous_vars,
        k=5
    ):
        """
        参数：
            categorical_vars: list of str
                分类特征列名列表（必须与 DataFrame 中列名一致）。
            continuous_vars: list of str
                连续特征列名列表。
            k: int, default=5
                KNN 中选取的邻居数量。
        """
        self.categorical_vars = categorical_vars
        self.continuous_vars = continuous_vars
        self.k = k

        # 以下在 impute 时初始化
        self.continuous_ranges = {}  # {col: max-min}
        self.feature_list = self.continuous_vars + self.categorical_vars

    def _compute_continuous_ranges(self, df_complete):
        """
        根据完整数据计算每个连续变量的 (max - min)，供 Gower 距离归一化使用。
        """
        for col in self.continuous_vars:
            col_vals = df_complete[col].values.astype(float)
            col_min = np.nanmin(col_vals)
            col_max = np.nanmax(col_vals)
            self.continuous_ranges[col] = col_max - col_min if (col_max - col_min) > 0 else 1.0

    def _gower_distance(self, row_i, row_j):
        """
        计算数据集中两个样本 row_i、row_j 之间的 Gower 距离（只针对已观测的特征维度）。
        row_i, row_j: pandas.Series，索引包括 self.feature_list。
        返回一个标量距离 ∈ [0,1]。
        """
        sum_d = 0.0
        sum_w = 0.0  # 加权总数（可观测特征个数）

        # 连续特征部分
        for col in self.continuous_vars:
            xi = row_i[col]
            xj = row_j[col]
            # 如果任意一方缺失，则该特征在相似度计算中忽略
            if pd.isna(xi) or pd.isna(xj):
                continue
            rng = self.continuous_ranges[col]
            d_ij = abs(xi - xj) / rng
            sum_d += d_ij
            sum_w += 1.0

        # 分类特征部分
        for col in self.categorical_vars:
            xi = row_i[col]
            xj = row_j[col]
            # 同理，如果缺失则跳过
            if pd.isna(xi) or pd.isna(xj):
                continue
            d_ij = 0.0 if (xi == xj) else 1.0
            sum_d += d_ij
            sum_w += 1.0

        if sum_w == 0:
            # 如果所有特征在计算时都缺失，则距离定义为 1（最不相似）
            return 1.0
        return sum_d / sum_w

    def impute(self, X_incomplete, X_missing):
        """
        对混合型数据进行 KNN 插补。

        参数：
            X_incomplete: pandas.DataFrame
                原始“完整”数据（不含缺失），仅作形状校验使用。
            X_missing: pandas.DataFrame
                带缺失的数据，形状和列顺序与 X_incomplete 必须一致。

        返回：
            X_imputed: pandas.DataFrame
                KNN 插补完成后的 DataFrame，分类列恢复原始标签，连续列为 float。
            None: 本方法不返回额外模型。
        """
        # 1. 校验 shape (行数和列数) 是否一致
        if X_incomplete.shape != X_missing.shape:
            raise ValueError("X_incomplete 和 X_missing 的形状必须一致！")

        df_incomplete = X_incomplete.copy().reset_index(drop=True)
        df_missing = X_missing.copy().reset_index(drop=True)

        # 2. 计算连续列的范围 (max - min)，使用完整数据进行统计
        self._compute_continuous_ranges(df_incomplete)

        # 3. 准备一个拷贝 df_current，用于逐步填充
        df_current = df_missing.copy()

        n_samples = df_current.shape[0]

        # 4. 对每一个缺失的单元格，执行单次 KNN 插补
        #    逐行逐列扫描：对于 df_current.loc[i, col] 如果缺失，则在其他行中寻找该列非缺失的最近 K 邻居
        for i in range(n_samples):
            # 获取第 i 行
            row_i = df_current.loc[i]

            for col in self.feature_list:
                if pd.isna(row_i[col]):
                    # 需要插补的位置，先构造候选邻居集合：行 l 必须在 col 上有观测值
                    candidate_indices = df_current[~df_current[col].isna()].index.values
                    # 排除自己（如果自己也在候选里）
                    candidate_indices = candidate_indices[candidate_indices != i]
                    if candidate_indices.size == 0:
                        # 如果没有可用邻居，跳过本单元格（保持 NaN）
                        continue

                    # 计算 row_i 与每个候选行 row_l 之间的 Gower 距离
                    distances = []
                    for l in candidate_indices:
                        row_l = df_current.loc[l]
                        d_il = self._gower_distance(row_i, row_l)
                        distances.append((l, d_il))
                    # 按距离升序排序，取前 K 个
                    distances.sort(key=lambda x: x[1])
                    k_neigh = distances[: min(self.k, len(distances))]

                    neighbor_indices = [t[0] for t in k_neigh]

                    # 根据 col 的类型（连续 or 分类），分别取 neighbors 的平均或众数
                    if col in self.continuous_vars:
                        # 取 neighbors 在 col 上的观测值，计算平均
                        vals = df_current.loc[neighbor_indices, col].values.astype(float)
                        # 若 neighbors 中也可能存在 NaN（? 一般不会），先去掉
                        vals = vals[~np.isnan(vals)]
                        if vals.size > 0:
                            imp_val = np.mean(vals)
                            df_current.at[i, col] = imp_val
                    else:
                        # 分类列：取 neighbors 在 col 上的众数（mode）
                        vals = df_current.loc[neighbor_indices, col].values
                        # 仅保留非 NaN
                        vals = [v for v in vals if not pd.isna(v)]
                        if len(vals) > 0:
                            most_common = Counter(vals).most_common(1)[0][0]
                            df_current.at[i, col] = most_common
                    # 填充完后，row_i[col] 在下一列判断时已经更新

        # 5. 返回插补后的 DataFrame，分类列自动保持原始的 dtype 或 object
        return df_current, None

    # ------------------------------------------------------------------
    # R1 additions
    # ------------------------------------------------------------------
    def set_ranges_from_stats(self, stats):
        """Install de-leaked continuous ranges (observed min/max) before imputing."""
        self.continuous_ranges = {c: float(stats.cont_range[c]) for c in self.continuous_vars}
        return self

    def impute_deleaked(self, X_missing, stats):
        """R0's in-frame KNN with the oracle range source replaced.

        Identical to :meth:`impute` except that ``_compute_continuous_ranges`` is
        not called on a ground-truth frame; the ranges come from ``stats``.
        """
        self.set_ranges_from_stats(stats)

        df_current = X_missing.copy().reset_index(drop=True)
        n_samples = df_current.shape[0]

        for i in range(n_samples):
            row_i = df_current.loc[i]
            for col in self.feature_list:
                if pd.isna(row_i[col]):
                    candidate_indices = df_current[~df_current[col].isna()].index.values
                    candidate_indices = candidate_indices[candidate_indices != i]
                    if candidate_indices.size == 0:
                        continue

                    distances = []
                    for l in candidate_indices:
                        row_l = df_current.loc[l]
                        d_il = self._gower_distance(row_i, row_l)
                        distances.append((l, d_il))
                    distances.sort(key=lambda x: x[1])
                    k_neigh = distances[: min(self.k, len(distances))]
                    neighbor_indices = [t[0] for t in k_neigh]

                    if col in self.continuous_vars:
                        vals = df_current.loc[neighbor_indices, col].values.astype(float)
                        vals = vals[~np.isnan(vals)]
                        if vals.size > 0:
                            df_current.at[i, col] = np.mean(vals)
                    else:
                        vals = df_current.loc[neighbor_indices, col].values
                        vals = [v for v in vals if not pd.isna(v)]
                        if len(vals) > 0:
                            df_current.at[i, col] = Counter(vals).most_common(1)[0][0]

        return df_current, None

    def fit(self, X_train_missing, stats):
        """Fit the donor pool on the training fold only."""
        self.set_ranges_from_stats(stats)
        donors, _ = self.impute_deleaked(X_train_missing, stats)
        self.donor_frame_ = donors.reset_index(drop=True)
        self.stats_ = stats
        return self

    def transform(self, X_new_missing):
        """Impute a disjoint frame using only the fitted donor pool.

        Each new row is compared against the stored (already imputed) training
        rows; new rows are never used as donors for one another, so there is no
        test -> test or test -> train information flow.
        """
        if getattr(self, "donor_frame_", None) is None:
            raise RuntimeError("knnImputer.transform called before fit")

        donors = self.donor_frame_
        df_out = X_new_missing.copy().reset_index(drop=True)

        for i in range(df_out.shape[0]):
            row_i = df_out.loc[i]
            miss_cols = [c for c in self.feature_list if pd.isna(row_i[c])]
            if not miss_cols:
                continue

            dists = np.array(
                [self._gower_distance(row_i, donors.loc[j]) for j in donors.index],
                dtype=float,
            )
            order = np.argsort(dists, kind="stable")

            for col in miss_cols:
                col_vals = donors.iloc[order][col]
                col_vals = col_vals[col_vals.notna()]
                if col_vals.shape[0] == 0:
                    continue
                sel = col_vals.iloc[: min(self.k, col_vals.shape[0])]
                if col in self.continuous_vars:
                    df_out.at[i, col] = float(np.mean(sel.values.astype(float)))
                else:
                    df_out.at[i, col] = Counter(list(sel.values)).most_common(1)[0][0]

        return df_out, None