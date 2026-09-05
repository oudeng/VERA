# MissForest_v2.py
# -*- coding: utf-8 -*-
"""
MissForest - 非参数缺失值插补方法（修正版）
基于官方R包和论文: Stekhoven & Bühlmann (2012), Bioinformatics

主要修正点（相比v1）：
1. 实现官方停止准则γ：当连续和分类变量的差异都首次增加时停止
2. 收敛时返回前一次迭代的结果（而非最后一次）
3. 正确设置mtry = floor(sqrt(p))，对所有变量类型一致
4. 正确设置nodesize：回归=5，分类=1（与官方一致）
5. 添加OOB误差估计功能

Reference:
    Stekhoven, D.J. & Bühlmann, P. (2012).
    MissForest—non-parametric missing value imputation for mixed-type data.
    Bioinformatics, 28(1), 112-118. doi:10.1093/bioinformatics/btr597
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings


class MissForestImputer:
    """
    MissForest Imputer - 忠于原论文实现
    
    关键特性：
    - 迭代式随机森林插补
    - 支持混合型数据（连续+分类）
    - 自动停止准则（差异首次增加时停止）
    - OOB误差估计
    """

    def __init__(
        self,
        categorical_vars,
        continuous_vars,
        n_estimators=100,        # 论文默认 ntree=100
        max_iter=10,             # 论文默认 maxiter=10
        seed=42,
        n_jobs=4,                # 并行线程数，-1表示使用所有CPU
        verbose=False,
        # 新增参数（与官方对齐）
        decreasing=False,        # 变量排序：False=按缺失数升序（官方默认）
    ):
        """
        参数：
            categorical_vars: list of str
                分类特征的列名列表
            continuous_vars: list of str
                连续特征的列名列表
            n_estimators: int, default=100
                随机森林中树的数量（论文默认100）
            max_iter: int, default=10
                最大迭代次数（论文默认10）
            seed: int, default=42
                随机种子
            n_jobs: int, default=4
                并行线程数，-1表示使用所有CPU，修改到4以适应大多数环境
            verbose: bool, default=False
                是否打印进度信息
            decreasing: bool, default=False
                变量处理顺序。False=按缺失数从少到多（官方默认）
        """
        self.categorical_vars = categorical_vars if categorical_vars else []
        self.continuous_vars = continuous_vars if continuous_vars else []
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.decreasing = decreasing

        # 存储最终模型
        self.models_ = {}
        # 存储OOB误差估计
        self.oob_error_ = {}
        # 存储迭代信息
        self.n_iter_ = 0

    def _compute_diff_continuous(self, X_new, X_old, missing_mask_cont):
        """
        计算连续变量的差异（论文公式）
        
        Δ_N = Σ(X_new - X_old)² / Σ(X_new²)
        
        只在原始缺失位置计算
        """
        if len(self.continuous_vars) == 0:
            return None
        
        diff_sum = 0.0
        new_sum_sq = 0.0
        
        for col in self.continuous_vars:
            if col not in missing_mask_cont.columns:
                continue
            mask = missing_mask_cont[col]
            if mask.sum() == 0:
                continue
            
            x_new = X_new.loc[mask, col].values.astype(float)
            x_old = X_old.loc[mask, col].values.astype(float)
            
            diff_sum += np.sum((x_new - x_old) ** 2)
            new_sum_sq += np.sum(x_new ** 2)
        
        if new_sum_sq == 0:
            return 0.0
        
        return diff_sum / new_sum_sq

    def _compute_diff_categorical(self, X_new, X_old, missing_mask_cat, num_na_cat):
        """
        计算分类变量的差异（论文公式）
        
        Δ_F = Σ(X_new ≠ X_old) / #NA
        
        只在原始缺失位置计算
        """
        if len(self.categorical_vars) == 0 or num_na_cat == 0:
            return None
        
        diff_count = 0
        
        for col in self.categorical_vars:
            if col not in missing_mask_cat.columns:
                continue
            mask = missing_mask_cat[col]
            if mask.sum() == 0:
                continue
            
            x_new = X_new.loc[mask, col].values
            x_old = X_old.loc[mask, col].values
            
            diff_count += np.sum(x_new != x_old)
        
        return diff_count / num_na_cat

    def impute(self, X_missing, record_sequence=False):
        """
        对混合型数据进行MissForest插补

        参数：
            X_missing: pandas.DataFrame
                带缺失的数据
            record_sequence: bool (R1 addition)
                记录每个 (iteration, column) 的已拟合随机森林与初始填充值，
                供 :meth:`transform` 在不相交的数据帧上重放。

        返回：
            X_imputed: pandas.DataFrame
                插补完成后的DataFrame
            self.models_: dict
                训练好的模型字典

        R1 note (P1 T1.4 / P0 finding B6)
        ---------------------------------
        The body of this method never referenced the ground-truth table, and the
        wrapper comment at R0 ``MissForest_v2.py:392-393`` asserted as much. That
        assertion is only half true: ``registry.py:152`` called
        ``set_categories_from_complete`` and handed in a frame whose categorical
        dtype already carried the ORACLE vocabulary, which line 177 below then
        adopts verbatim. In R1 the registry installs the observed vocabulary
        instead, so this file is unchanged in substance.
        """
        # 1. 复制数据
        df_missing = X_missing.copy().reset_index(drop=True)
        if record_sequence:
            self.sequence_ = []
            self.init_fill_ = {}
            self.category_mappings_ = {}
        
        # 2. 记录原始缺失位置
        original_missing_mask = X_missing.isna()
        
        # 3. 计算总变量数p，用于设置mtry
        p = len(df_missing.columns)
        mtry = max(1, int(np.floor(np.sqrt(p))))
        
        if self.verbose:
            print(f"[MissForest] 数据维度: {df_missing.shape[0]} 样本, {p} 变量")
            print(f"[MissForest] mtry = {mtry}")
        
        # 4. 处理分类列：转换为整数编码
        category_mappings = {}
        for col in self.categorical_vars:
            if col not in df_missing.columns:
                continue
            df_missing[col] = df_missing[col].astype("category")
            category_mappings[col] = list(df_missing[col].cat.categories)
            df_missing[col] = df_missing[col].cat.codes
            df_missing.loc[df_missing[col] == -1, col] = np.nan
        if record_sequence:
            self.category_mappings_ = {k: list(v) for k, v in category_mappings.items()}

        # 5. 初始填充（论文：mean for continuous, mode for categorical）
        for col in self.continuous_vars:
            if col not in df_missing.columns:
                raise ValueError(f"连续列 '{col}' 不存在于数据中")
            mean_val = df_missing[col].mean(skipna=True)
            if pd.isna(mean_val):
                mean_val = 0.0
            if record_sequence:
                self.init_fill_[col] = float(mean_val)
            df_missing[col] = df_missing[col].fillna(mean_val)

        for col in self.categorical_vars:
            if col not in df_missing.columns:
                raise ValueError(f"分类列 '{col}' 不存在于数据中")
            mode_series = df_missing[col].value_counts(dropna=True)
            if mode_series.shape[0] == 0:
                if record_sequence:
                    self.init_fill_[col] = 0.0
                df_missing[col] = df_missing[col].fillna(0)
            else:
                mode_code = mode_series.idxmax()
                if record_sequence:
                    self.init_fill_[col] = float(mode_code)
                df_missing[col] = df_missing[col].fillna(mode_code)

        # 6. 重新获取缺失mask（使用原始的）
        missing_mask = original_missing_mask.copy()
        
        # 7. 计算分类变量的总缺失数（用于停止准则）
        num_na_cat = 0
        for col in self.categorical_vars:
            if col in missing_mask.columns:
                num_na_cat += missing_mask[col].sum()
        
        # 8. 按缺失数量排序变量
        col_missing_count = original_missing_mask.sum()
        cols_with_missing = [c for c in df_missing.columns if col_missing_count[c] > 0]
        ordered_cols = sorted(cols_with_missing, 
                             key=lambda x: col_missing_count[x],
                             reverse=self.decreasing)
        
        if self.verbose:
            print(f"[MissForest] 有缺失的列数: {len(ordered_cols)}")
            for c in ordered_cols[:5]:  # 只显示前5个
                print(f"   {c}: {col_missing_count[c]} 个缺失")
            if len(ordered_cols) > 5:
                print(f"   ... 共 {len(ordered_cols)} 列")
        
        # 9. 初始化迭代
        df_current = df_missing.copy()
        df_previous = None
        
        # 用于停止准则的差异值
        prev_diff_cont = np.inf
        prev_diff_cat = np.inf
        
        # 10. 迭代
        for it in range(self.max_iter):
            if self.verbose:
                print(f"\n[MissForest] === 迭代 {it+1}/{self.max_iter} ===")
            
            # 保存上一次迭代的结果（用于停止准则和返回）
            df_previous = df_current.copy()
            
            # 遍历每个有缺失的列
            for col in ordered_cols:
                na_mask = missing_mask[col]
                
                if na_mask.sum() == 0:
                    continue
                
                # 准备训练数据
                other_cols = [c for c in df_current.columns if c != col]
                
                X_train = df_current.loc[~na_mask, other_cols].values
                y_train = df_current.loc[~na_mask, col].values
                X_pred = df_current.loc[na_mask, other_cols].values
                
                # 根据变量类型选择模型
                if col in self.continuous_vars:
                    # 回归：nodesize=5（官方默认）
                    model = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        max_features=mtry,           # 官方 mtry
                        min_samples_leaf=5,          # 官方 nodesize for regression
                        bootstrap=True,              # 官方 replace=TRUE
                        oob_score=True,              # 用于OOB误差估计
                        random_state=self.seed,
                        n_jobs=self.n_jobs
                    )
                else:
                    # 分类：nodesize=1（官方默认）
                    model = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_features=mtry,           # 官方 mtry
                        min_samples_leaf=1,          # 官方 nodesize for classification
                        bootstrap=True,              # 官方 replace=TRUE
                        oob_score=True,              # 用于OOB误差估计
                        random_state=self.seed,
                        n_jobs=self.n_jobs
                    )
                
                # 训练
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                
                # 预测并更新
                y_pred = model.predict(X_pred)
                df_current.loc[na_mask, col] = y_pred
                
                # 保存模型
                self.models_[col] = model

                # R1: record the fitted forest for transform() replay.
                if record_sequence:
                    self.sequence_.append({
                        "iter": it, "col": col,
                        "other_cols": list(other_cols), "model": model,
                    })

                if self.verbose and it == self.max_iter - 1:
                    # 最后一次迭代时显示OOB误差
                    if hasattr(model, 'oob_score_'):
                        self.oob_error_[col] = 1 - model.oob_score_
            
            # 计算停止准则的差异
            diff_cont = self._compute_diff_continuous(
                df_current, df_previous, 
                missing_mask[self.continuous_vars] if self.continuous_vars else pd.DataFrame()
            )
            diff_cat = self._compute_diff_categorical(
                df_current, df_previous,
                missing_mask[self.categorical_vars] if self.categorical_vars else pd.DataFrame(),
                num_na_cat
            )
            
            if self.verbose:
                if diff_cont is not None:
                    print(f"  Δ_N (连续): {diff_cont:.6f} (前次: {prev_diff_cont:.6f})")
                if diff_cat is not None:
                    print(f"  Δ_F (分类): {diff_cat:.6f} (前次: {prev_diff_cat:.6f})")
            
            # 检查停止准则γ（论文：当两种类型的差异都增加时停止）
            cont_increased = (diff_cont is not None and diff_cont > prev_diff_cont)
            cat_increased = (diff_cat is not None and diff_cat > prev_diff_cat)
            
            # 只有一种类型时，只检查该类型
            if diff_cont is None:
                stop_criterion = cat_increased
            elif diff_cat is None:
                stop_criterion = cont_increased
            else:
                # 两种类型都存在时，需要两者都增加才停止
                stop_criterion = cont_increased and cat_increased
            
            if stop_criterion and it > 0:
                if self.verbose:
                    print(f"\n[MissForest] 停止准则γ在迭代 {it+1} 触发，返回迭代 {it} 的结果")
                self.n_iter_ = it
                df_current = df_previous  # 返回前一次迭代的结果！
                break
            
            # 更新差异值
            if diff_cont is not None:
                prev_diff_cont = diff_cont
            if diff_cat is not None:
                prev_diff_cat = diff_cat
            
            self.n_iter_ = it + 1
        
        if self.verbose:
            print(f"\n[MissForest] 完成，共 {self.n_iter_} 次迭代")
        
        # 11. 将分类列的整数编码转换回原始类别
        df_imputed = df_current.copy()
        
        for col in self.categorical_vars:
            if col not in df_imputed.columns:
                continue
            
            # 分类器直接输出整数类别，不需要round
            df_imputed[col] = df_imputed[col].astype(int)
            
            original_categories = category_mappings[col]
            max_code = len(original_categories) - 1
            
            def code_to_cat(x):
                if x < 0 or x > max_code:
                    return np.nan
                return original_categories[int(x)]
            
            df_imputed[col] = df_imputed[col].apply(code_to_cat)
            df_imputed[col] = df_imputed[col].astype("category")

        if record_sequence:
            # Keep only the steps whose effect survived the gamma stopping rule.
            self.sequence_ = [s for s in self.sequence_ if s["iter"] < max(self.n_iter_, 1)]

        return df_imputed, self.models_

    # ------------------------------------------------------------------
    # R1 additions: fit / transform
    # ------------------------------------------------------------------
    def fit(self, X_train_missing, stats=None):
        """Fit the iterative-forest sequence on the training fold only."""
        self.stats_ = stats
        _ = self.impute(X_train_missing, record_sequence=True)
        return self

    def transform(self, X_new_missing):
        """Replay the fitted forests on a disjoint frame.

        Same contract as scikit-learn's ``IterativeImputer.transform``: no forest
        is refitted, the initial fill values come from the training fold, and the
        category codes use the training fold's vocabulary.
        """
        if getattr(self, "sequence_", None) is None:
            raise RuntimeError("MissForestImputer.transform called before fit")

        df = X_new_missing.copy().reset_index(drop=True)

        for col in self.categorical_vars:
            cats = self.category_mappings_.get(col, [])
            df[col] = pd.Categorical(df[col], categories=cats).codes.astype(float)
            df.loc[df[col] < 0, col] = np.nan

        for col in self.continuous_vars:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        mask_new = {c: df[c].isna().values.copy() for c in df.columns}

        for col, val in self.init_fill_.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        for step in self.sequence_:
            col = step["col"]
            m = mask_new.get(col)
            if m is None or not m.any():
                continue
            X_pred = df.loc[m, step["other_cols"]].values
            df.loc[m, col] = step["model"].predict(X_pred)

        df_imputed = df.copy()
        for col in self.categorical_vars:
            cats = self.category_mappings_.get(col, [])
            max_code = len(cats) - 1
            codes = df_imputed[col].astype(float).round().astype(int).clip(0, max_code)
            df_imputed[col] = pd.Series([cats[int(c)] for c in codes], index=df_imputed.index)
            df_imputed[col] = df_imputed[col].astype("category")

        return df_imputed, self.models_


# ============================================================
# 兼容性包装器（与baselines/registry.py接口兼容）
# ============================================================

class MissForestWrapper:
    """
    包装器类，提供与其他baseline一致的接口
    """
    def __init__(self,
                 categorical_vars=None,
                 continuous_vars=None,
                 seed=42,
                 n_estimators=100,
                 max_iter=10,
                 n_jobs=-1,
                 verbose=False,
                 **kwargs):
        
        self.categorical_vars = categorical_vars if categorical_vars else []
        self.continuous_vars = continuous_vars if continuous_vars else []
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        
    def impute(self, X_complete, X_missing):
        """执行插补（X_complete用于兼容接口，实际不使用）"""
        imputer = MissForestImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            n_estimators=self.n_estimators,
            max_iter=self.max_iter,
            seed=self.seed,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        
        X_imputed, _ = imputer.impute(X_missing)
        return X_imputed


# 保持向后兼容的别名
misforestImputer = MissForestImputer


if __name__ == "__main__":
    # 简单测试
    print("MissForest v2 测试")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n = 200
    
    # 连续变量
    x1 = np.random.randn(n) * 10 + 50
    x2 = np.random.randn(n) * 5 + 20
    x3 = x1 * 0.5 + x2 * 0.3 + np.random.randn(n) * 2  # 有相关性
    
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
    print(f"各列缺失数: {df_missing.isna().sum().to_dict()}")
    
    # 测试插补
    imputer = MissForestImputer(
        categorical_vars=['cat1', 'cat2'],
        continuous_vars=['x1', 'x2', 'x3'],
        n_estimators=50,  # 减少树数量加快测试
        max_iter=10,
        verbose=True,
    )
    
    df_imputed, models = imputer.impute(df_missing)
    
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
    
    print(f"\n迭代次数: {imputer.n_iter_}")