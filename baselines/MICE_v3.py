# MICE_v3.py
# -*- coding: utf-8 -*-
"""
MICE (Multiple Imputation by Chained Equations) - 修正版
基于官方R包和论文: Van Buuren & Groothuis-Oudshoorn (2011), JSS
updated on Jan 4, 2026.

主要修正点（相比v2）：
1. 连续变量使用PMM（Predictive Mean Matching）而非直接预测
2. PMM从donor pool（默认5个）中随机选择观测值作为插补
3. 使用贝叶斯线性回归进行参数采样（proper imputation）
4. 迭代次数默认改为5（与官方一致）

Reference:
    Van Buuren, S. & Groothuis-Oudshoorn, K. (2011).
    mice: Multivariate Imputation by Chained Equations in R.
    Journal of Statistical Software, 45(3), 1-67.
    doi:10.18637/jss.v045.i03
    
    Van Buuren, S. (2018). Flexible Imputation of Missing Data.
    Second Edition. Chapman & Hall/CRC.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings


class MICEImputer:
    """
    MICE Imputer - 忠于原论文实现
    
    关键特性：
    - 连续变量使用PMM（Predictive Mean Matching）
    - 分类变量使用Logistic Regression（二分类）或Polytomous Regression（多分类）
    - 支持贝叶斯参数采样以反映不确定性
    """

    def __init__(
        self,
        categorical_vars,
        continuous_vars,
        max_iter=5,              # 官方默认 maxit=5
        seed=42,
        # PMM参数（官方默认）
        donors=5,                # donor pool大小（官方默认5）
        matchtype=1,             # 匹配类型（1=Type I matching）
        ridge=1e-5,              # 岭回归正则化参数
        # 向后兼容：接受但忽略v2的旧参数
        **kwargs,                # 忽略 use_probabilistic_sampling, add_noise_to_continuous, noise_level 等
    ):
        """
        参数：
            categorical_vars: list of str
                分类特征列名列表
            continuous_vars: list of str
                连续特征列名列表
            max_iter: int, default=5
                链式迭代次数（官方默认5）
            seed: int, default=42
                随机种子
            donors: int, default=5
                PMM的donor pool大小（官方默认5，推荐3-10）
            matchtype: int, default=1
                PMM匹配类型（1=Type I matching，官方默认）
            ridge: float, default=1e-5
                岭回归正则化参数（防止多重共线性）
            **kwargs: 
                向后兼容，忽略v2版本的旧参数
        """
        # 忽略旧版参数（静默）
        # 旧参数包括: use_probabilistic_sampling, add_noise_to_continuous, noise_level
        
        self.categorical_vars = categorical_vars if categorical_vars else []
        self.continuous_vars = continuous_vars if continuous_vars else []
        self.max_iter = max_iter
        self.seed = seed
        self.donors = donors
        self.matchtype = matchtype
        self.ridge = ridge

        np.random.seed(seed)

        self.label_encoders = {}
        self.models_ = {}

    def _pmm_match(self, yhat_obs, yhat_mis, y_obs, donors=5, seed=None):
        """
        Predictive Mean Matching (PMM) 核心算法
        
        对于每个缺失值，从预测值最接近的donors个观测值中随机选择一个
        
        Args:
            yhat_obs: 观测样本的预测值 (n_obs,)
            yhat_mis: 缺失样本的预测值 (n_mis,)
            y_obs: 观测样本的真实值 (n_obs,)
            donors: donor pool大小
            seed: 随机种子
            
        Returns:
            y_imp: 插补值 (n_mis,)，来自y_obs中的实际观测值
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_mis = len(yhat_mis)
        n_obs = len(yhat_obs)
        y_imp = np.zeros(n_mis)
        
        # 确保donors不超过观测样本数
        d = min(donors, n_obs)
        
        for j in range(n_mis):
            # 计算缺失样本j与所有观测样本的预测值距离
            distances = np.abs(yhat_obs - yhat_mis[j])
            
            # 找到d个最近的观测样本
            donor_indices = np.argsort(distances)[:d]
            
            # 从donor pool随机选择一个
            selected_donor = np.random.choice(donor_indices)
            
            # 使用该donor的观测值作为插补值
            y_imp[j] = y_obs[selected_donor]
        
        return y_imp

    def _bayesian_ridge_draw(self, X_train, y_train, X_pred, ridge=1e-5, seed=None):
        """
        贝叶斯线性回归参数采样（用于PMM的proper imputation）
        
        实现van Buuren (2012, p.73)的算法：
        1. 计算 β̂ = (X'X + λI)^(-1) X'y
        2. 从后验分布采样 β̇
        3. 返回预测值
        
        Args:
            X_train: 训练特征 (n_obs, p)
            y_train: 训练目标 (n_obs,)
            X_pred: 预测特征 (n_mis, p)
            ridge: 岭回归参数
            seed: 随机种子
            
        Returns:
            yhat_obs: 观测样本预测值（使用β̂）
            yhat_mis: 缺失样本预测值（使用β̇，采样的参数）
        """
        if seed is not None:
            np.random.seed(seed)
        
        n, p = X_train.shape
        
        # 添加截距项
        X_train_aug = np.column_stack([np.ones(n), X_train])
        X_pred_aug = np.column_stack([np.ones(X_pred.shape[0]), X_pred])
        p_aug = p + 1
        
        # 计算 S = X'X
        S = X_train_aug.T @ X_train_aug
        
        # 添加岭正则化: V = (S + diag(S)*κ)^(-1)
        S_diag = np.diag(np.diag(S)) * ridge
        try:
            V = np.linalg.inv(S + S_diag)
        except np.linalg.LinAlgError:
            # 如果仍然奇异，增加正则化
            V = np.linalg.inv(S + np.eye(p_aug) * 0.01)
        
        # 计算 β̂ = V X'y
        beta_hat = V @ X_train_aug.T @ y_train
        
        # 计算残差方差
        y_pred_train = X_train_aug @ beta_hat
        residuals = y_train - y_pred_train
        df = max(n - p_aug, 1)
        sigma_sq = np.sum(residuals ** 2) / df
        
        # 从后验分布采样σ² ~ σ²_hat * χ²_df / df
        # 简化：使用σ_dot ≈ σ_hat * sqrt(df / χ²_df)
        chi2_sample = np.random.chisquare(df)
        sigma_dot = np.sqrt(sigma_sq * df / chi2_sample)
        
        # 从后验分布采样β: β̇ = β̂ + σ̇ * V^(1/2) * z
        # 使用Cholesky分解计算V^(1/2)
        try:
            V_sqrt = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            # 如果V不是正定的，使用SVD
            U, s, Vt = np.linalg.svd(V)
            V_sqrt = U @ np.diag(np.sqrt(np.maximum(s, 0)))
        
        z = np.random.randn(p_aug)
        beta_dot = beta_hat + sigma_dot * V_sqrt @ z

        # 计算预测值
        # Type 1 matching: yhat_obs使用β̂，yhat_mis使用β̇
        yhat_obs = X_train_aug @ beta_hat
        yhat_mis = X_pred_aug @ beta_dot

        # R1 addition: keep the draw's state so that fit()/transform() can apply
        # the same fitted regression to a disjoint frame. Purely a side channel;
        # the returned values and therefore the R0 numerics are unchanged.
        self._last_draw_state = {
            "ok": True,
            "beta_hat": np.asarray(beta_hat, dtype=float),
            "sigma_dot": float(sigma_dot),
            "V_sqrt": np.asarray(V_sqrt, dtype=float),
            "yhat_obs": np.asarray(yhat_obs, dtype=float),
        }

        return yhat_obs, yhat_mis

    def impute(self, X_incomplete, X_missing, stats=None, record_sequence=False):
        """
        对缺失数据执行MICE插补

        Args:
            X_incomplete: 用于获取schema的完整数据（LabelEncoder需要）
            X_missing: 带缺失的数据
            stats: R1 addition. When given, an :class:`ObservedStats` fitted on
                the *incomplete* table supplies the LabelEncoder vocabulary and
                the initial mean/mode fill, and ``X_incomplete`` is ignored
                entirely (callers pass ``None``). When ``stats is None`` the
                original R0 behavior is reproduced exactly: the vocabulary
                (``:226-228``) and the initial fill (``:250``, ``:254``) are read
                off the ground-truth table.
            record_sequence: R1 addition. Store the per-(iteration, column)
                fitted state in ``self.sequence_`` so that :meth:`transform` can
                replay it on a disjoint frame.

        Returns:
            X_imputed: 插补后的DataFrame
            models_: 训练好的模型字典
        """
        deleaked = stats is not None
        if not deleaked:
            if X_incomplete.shape != X_missing.shape:
                raise ValueError("X_incomplete 和 X_missing 的 shape 必须一致！")

        df = X_missing.copy().reset_index(drop=True)
        n = df.shape[0]

        if record_sequence:
            self.sequence_ = []
            self.init_fill_ = {}

        # 准备LabelEncoder
        for col in self.categorical_vars:
            le = LabelEncoder()
            if deleaked:
                # R1: vocabulary from the OBSERVED cells of the incomplete table.
                unique_vals = list(stats.categories[col])
            else:
                non_null = X_incomplete[col].dropna().values
                unique_vals = sorted(pd.unique(non_null))
            le.fit(unique_vals)
            self.label_encoders[col] = le

            codes = np.full(n, -1, dtype=int)
            non_missing_idx = df[col].notna()
            if non_missing_idx.sum() > 0:
                known = set(le.classes_.tolist())
                vals = df.loc[non_missing_idx, col].values
                if deleaked and not set(pd.unique(vals)).issubset(known):
                    # Unseen label under fit/transform: treat as missing rather
                    # than crashing. Counted by ObservedStats.unseen_category_count.
                    keep = np.array([v in known for v in vals])
                    idx_positions = np.flatnonzero(non_missing_idx.values)
                    codes[idx_positions[keep]] = le.transform(vals[keep])
                else:
                    try:
                        transformed = le.transform(vals)
                    except ValueError as e:
                        raise ValueError(
                            f"列 '{col}' 在 X_missing 中出现了 X_incomplete 未见过的标签: {e}"
                        )
                    codes[non_missing_idx] = transformed
            df[col] = codes.astype(float)
            df.loc[df[col] < 0, col] = np.nan

        # 连续列确保为float
        for col in self.continuous_vars:
            df[col] = df[col].astype(float)

        # 初始填充：均值（连续）和众数（分类）
        for col in self.continuous_vars:
            mean_val = stats.cont_mean[col] if deleaked else X_incomplete[col].mean()
            if record_sequence:
                self.init_fill_[col] = float(mean_val)
            df[col] = df[col].fillna(mean_val)

        for col in self.categorical_vars:
            if deleaked:
                mode_val = stats.cat_mode[col]
                try:
                    mode_code = self.label_encoders[col].transform([mode_val])[0]
                except Exception:
                    mode_code = 0
            else:
                value_counts = Counter(X_incomplete[col].dropna().values)
                if len(value_counts) > 0:
                    mode_val = max(value_counts, key=value_counts.get)
                    mode_code = self.label_encoders[col].transform([mode_val])[0]
                else:
                    mode_code = 0
            if record_sequence:
                self.init_fill_[col] = float(mode_code)
            df[col] = df[col].fillna(mode_code)

        # 计算待插补列（按缺失数量排序）
        all_cols = self.continuous_vars + self.categorical_vars
        missing_counts = X_missing.isna().sum()
        ordered_cols = sorted(
            [c for c in all_cols if missing_counts[c] > 0],
            key=lambda x: missing_counts[x]
        )

        # 链式迭代
        for it in range(self.max_iter):
            for col in ordered_cols:
                mask_missing = X_missing[col].isna()
                if mask_missing.sum() == 0:
                    continue

                other_cols = [c for c in all_cols if c != col]
                X_train = df.loc[~mask_missing, other_cols].values
                y_train = df.loc[~mask_missing, col].values
                X_pred = df.loc[mask_missing, other_cols].values

                # 迭代特定的随机种子
                iter_seed = self.seed + it * 1000 + hash(col) % 1000

                if col in self.continuous_vars:
                    # PMM (Predictive Mean Matching) - 官方默认方法
                    try:
                        # 贝叶斯回归参数采样
                        yhat_obs, yhat_mis = self._bayesian_ridge_draw(
                            X_train, y_train, X_pred,
                            ridge=self.ridge,
                            seed=iter_seed
                        )
                        
                        # PMM匹配：从donor pool选择观测值
                        y_pred = self._pmm_match(
                            yhat_obs, yhat_mis, y_train,
                            donors=self.donors,
                            seed=iter_seed + 1
                        )
                    except Exception as e:
                        # 回退到BayesianRidge
                        warnings.warn(f"PMM failed for {col}, falling back to BayesianRidge: {e}")
                        model = BayesianRidge()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_pred)
                        self.models_[col] = model

                else:
                    # 分类变量：Logistic Regression
                    unique_classes = np.unique(y_train)
                    n_classes = len(unique_classes)
                    
                    if n_classes == 1:
                        # 边缘情况：观测样本只有一个类别
                        # 直接用该类别填充所有缺失值（无法训练分类器）
                        # 这在高度不平衡的数据集（如eICU死亡率）中可能发生
                        only_class = unique_classes[0]
                        y_pred = np.full(len(X_pred), only_class)
                        self.models_[col] = None  # 无模型
                        
                    elif n_classes == 2:
                        # 二分类：logreg
                        model = LogisticRegression(
                            solver='lbfgs',
                            max_iter=2000,
                            random_state=iter_seed
                        )
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.fit(X_train, y_train)
                        
                        # 使用概率采样（proper imputation）
                        proba = model.predict_proba(X_pred)
                        y_pred = np.array([
                            np.random.choice(model.classes_, p=p)
                            for p in proba
                        ])
                        
                        self.models_[col] = model
                    else:
                        # 多分类：polyreg (multinomial)
                        model = LogisticRegression(
                            multi_class='multinomial',
                            solver='lbfgs',
                            max_iter=2000,
                            random_state=iter_seed
                        )
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.fit(X_train, y_train)
                        
                        # 使用概率采样（proper imputation）
                        proba = model.predict_proba(X_pred)
                        y_pred = np.array([
                            np.random.choice(model.classes_, p=p)
                            for p in proba
                        ])
                        
                        self.models_[col] = model

                # R1: record the fitted state so transform() can replay it.
                if record_sequence:
                    step = {"iter": it, "col": col, "other_cols": list(other_cols),
                            "iter_seed": int(iter_seed)}
                    if col in self.continuous_vars:
                        state = getattr(self, "_last_draw_state", None)
                        if state is not None and state.get("ok"):
                            step.update({"kind": "pmm", "beta_hat": state["beta_hat"],
                                         "sigma_dot": state["sigma_dot"],
                                         "V_sqrt": state["V_sqrt"],
                                         "yhat_obs": state["yhat_obs"],
                                         "y_obs": np.asarray(y_train, dtype=float)})
                        else:
                            step.update({"kind": "ridge", "model": self.models_.get(col)})
                    else:
                        mdl = self.models_.get(col)
                        if mdl is None:
                            step.update({"kind": "const",
                                         "value": float(np.unique(y_train)[0])})
                        else:
                            step.update({"kind": "clf", "model": mdl})
                    self.sequence_.append(step)

                # 更新缺失值
                df.loc[mask_missing, col] = y_pred

        # 将分类列还原为原始类别
        df_imp = df.copy()
        for col in self.categorical_vars:
            df_imp[col] = df_imp[col].round().astype(int)
            le = self.label_encoders[col]
            
            # 处理可能的越界值
            max_class = len(le.classes_) - 1
            df_imp[col] = df_imp[col].clip(0, max_class)
            
            inv_labels = le.inverse_transform(df_imp[col].values)
            df_imp[col] = pd.Series(inv_labels, index=df_imp.index)

        return df_imp, self.models_

    # ------------------------------------------------------------------
    # R1 additions: fit / transform
    # ------------------------------------------------------------------
    def fit(self, X_train_missing, stats):
        """Run the chain on the training fold and keep the imputation sequence.

        This mirrors scikit-learn's ``IterativeImputer``, which stores
        ``imputation_sequence_`` at fit time and replays it in ``transform``.
        For the PMM branch the stored state is the posterior mean ``beta_hat``,
        the drawn residual scale, the Cholesky factor of V, the training-fold
        predicted means and the training-fold donor values -- everything the
        matching step needs, and nothing from the frame being transformed.
        """
        self.stats_ = stats
        _ = self.impute(None, X_train_missing, stats=stats, record_sequence=True)
        return self

    def transform(self, X_new_missing):
        if getattr(self, "sequence_", None) is None:
            raise RuntimeError("MICEImputer.transform called before fit")

        stats = self.stats_
        df = X_new_missing.copy().reset_index(drop=True)
        n = df.shape[0]

        # Encode with the FITTED vocabulary.
        for col in self.categorical_vars:
            le = self.label_encoders[col]
            known = set(le.classes_.tolist())
            codes = np.full(n, -1, dtype=int)
            nm = df[col].notna()
            if nm.sum() > 0:
                vals = df.loc[nm, col].values
                keep = np.array([v in known for v in vals])
                pos = np.flatnonzero(nm.values)
                if keep.any():
                    codes[pos[keep]] = le.transform(vals[keep])
            df[col] = codes.astype(float)
            df.loc[df[col] < 0, col] = np.nan

        for col in self.continuous_vars:
            df[col] = df[col].astype(float)

        mask_new = {c: df[c].isna().values.copy() for c in self.continuous_vars + self.categorical_vars}

        # Initial fill with the FITTED values.
        for col, val in self.init_fill_.items():
            df[col] = df[col].fillna(val)

        base_seed = int(self.seed)

        for step_idx, step in enumerate(self.sequence_):
            col = step["col"]
            m = mask_new.get(col)
            if m is None or not m.any():
                continue
            X_pred = df.loc[m, step["other_cols"]].values
            row_pos = np.flatnonzero(m)

            # One RNG per (step, row). Sampling must not be coupled across rows:
            # if it were, imputing 20 rows and imputing those same 20 rows inside
            # a batch of 40 would give different answers, and "transform" would
            # secretly still be transductive. tests/test_baselines.py asserts this.
            rs_step = np.random.RandomState(base_seed + 9176 + step_idx)

            if step["kind"] == "pmm":
                beta_hat = step["beta_hat"]
                p_aug = beta_hat.shape[0]
                X_pred_aug = np.column_stack([np.ones(X_pred.shape[0]), X_pred])
                z = rs_step.randn(p_aug)
                beta_dot = beta_hat + step["sigma_dot"] * step["V_sqrt"] @ z
                yhat_mis = X_pred_aug @ beta_dot
                yhat_obs = step["yhat_obs"]
                y_obs = step["y_obs"]
                d = min(self.donors, len(yhat_obs))
                y_pred = np.empty(len(yhat_mis))
                for j in range(len(yhat_mis)):
                    donor_idx = np.argsort(np.abs(yhat_obs - yhat_mis[j]))[:d]
                    rs_row = np.random.RandomState(
                        (base_seed + 31337 + step_idx * 1000003 + int(row_pos[j])) % (2 ** 31 - 1)
                    )
                    y_pred[j] = y_obs[donor_idx[rs_row.randint(len(donor_idx))]]
            elif step["kind"] == "ridge":
                y_pred = step["model"].predict(X_pred)
            elif step["kind"] == "const":
                y_pred = np.full(X_pred.shape[0], step["value"])
            else:  # clf
                model = step["model"]
                proba = model.predict_proba(X_pred)
                y_pred = np.empty(proba.shape[0], dtype=model.classes_.dtype)
                for j in range(proba.shape[0]):
                    rs_row = np.random.RandomState(
                        (base_seed + 31337 + step_idx * 1000003 + int(row_pos[j])) % (2 ** 31 - 1)
                    )
                    y_pred[j] = rs_row.choice(model.classes_, p=proba[j])

            df.loc[m, col] = y_pred

        df_imp = df.copy()
        for col in self.categorical_vars:
            df_imp[col] = df_imp[col].round().astype(int)
            le = self.label_encoders[col]
            df_imp[col] = df_imp[col].clip(0, len(le.classes_) - 1)
            df_imp[col] = pd.Series(le.inverse_transform(df_imp[col].values), index=df_imp.index)

        return df_imp, self.models_


# ============================================================
# 兼容性包装器
# ============================================================

class MICEWrapper:
    """
    包装器类，提供与其他baseline一致的接口
    """
    def __init__(self,
                 categorical_vars=None,
                 continuous_vars=None,
                 seed=42,
                 max_iter=5,
                 donors=5,
                 **kwargs):
        
        self.categorical_vars = categorical_vars if categorical_vars else []
        self.continuous_vars = continuous_vars if continuous_vars else []
        self.seed = seed
        self.max_iter = max_iter
        self.donors = donors
        
    def impute(self, X_complete, X_missing):
        """执行插补"""
        imputer = MICEImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            max_iter=self.max_iter,
            seed=self.seed,
            donors=self.donors,
        )
        
        X_imputed, _ = imputer.impute(X_complete, X_missing)
        return X_imputed


if __name__ == "__main__":
    print("MICE v3 (with PMM) 测试")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n = 200
    
    # 连续变量（有相关性）
    x1 = np.random.randn(n) * 10 + 50
    x2 = x1 * 0.5 + np.random.randn(n) * 5
    x3 = np.random.randn(n) * 3 + 10
    
    # 分类变量
    cat1 = np.random.choice(['A', 'B', 'C'], n, p=[0.5, 0.3, 0.2])
    cat2 = np.random.choice(['X', 'Y'], n)
    
    df_complete = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'cat1': cat1,
        'cat2': cat2
    })
    
    # 引入缺失（MCAR, 20%）
    df_missing = df_complete.copy()
    miss_rate = 0.2
    
    for col in df_missing.columns:
        miss_idx = np.random.choice(n, size=int(n * miss_rate), replace=False)
        df_missing.loc[miss_idx, col] = np.nan
    
    print(f"缺失比例: {df_missing.isna().mean().mean():.1%}")
    
    # 测试插补
    imputer = MICEImputer(
        categorical_vars=['cat1', 'cat2'],
        continuous_vars=['x1', 'x2', 'x3'],
        max_iter=5,
        donors=5,
        seed=42,
    )
    
    df_imputed, models = imputer.impute(df_complete, df_missing)
    
    # 计算误差
    print("\n" + "=" * 60)
    print("插补结果评估")
    print("=" * 60)
    
    # NRMSE for continuous
    for col in ['x1', 'x2', 'x3']:
        mask = df_complete[col].index.isin(
            df_missing[df_missing[col].isna()].index
        )
        if mask.sum() > 0:
            true_vals = df_complete.loc[mask, col].values
            imp_vals = df_imputed.loc[mask, col].values
            rmse = np.sqrt(np.mean((true_vals - imp_vals) ** 2))
            nrmse = rmse / np.std(true_vals)
            print(f"  {col}: NRMSE = {nrmse:.4f}")
            
            # 检查插补值是否来自观测值（PMM特性）
            obs_vals = set(df_complete.loc[~df_missing[col].isna(), col].values)
            from_obs = sum(1 for v in imp_vals if v in obs_vals)
            print(f"       来自观测值: {from_obs}/{len(imp_vals)} ({from_obs/len(imp_vals)*100:.1f}%)")
    
    # PFC for categorical
    for col in ['cat1', 'cat2']:
        mask = df_complete[col].index.isin(
            df_missing[df_missing[col].isna()].index
        )
        if mask.sum() > 0:
            true_vals = df_complete.loc[mask, col].values
            imp_vals = df_imputed.loc[mask, col].values
            pfc = np.mean(true_vals != imp_vals)
            print(f"  {col}: PFC = {pfc:.4f}")