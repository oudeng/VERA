# GAIN_v5.py
# -*- coding: utf-8 -*-
"""
GAIN (Generative Adversarial Imputation Nets) - 优化版 v5
基于官方TensorFlow实现: https://github.com/jsyoon0823/GAIN

v5主要优化点（相比v4）：
1. [关键修复] 重构损失正确归一化：除以观测元素数量，而非总元素数量
2. [关键修复] Batch采样使用有放回采样（与官方一致）
3. [稳定性] 添加梯度裁剪防止梯度爆炸
4. [稳定性] 添加warmup学习率调度
5. [诊断] 添加详细的训练诊断信息（梯度范数、NaN检测等）
6. [准确性] 对抗损失也进行正确归一化
7. [架构] 可选添加LayerNorm提升稳定性
8. [分类变量] 改进softmax归一化处理

Reference:
    Yoon, J., Jordon, J., & Schaar, M. (2018). 
    GAIN: Missing Data Imputation using Generative Adversarial Nets. ICML 2018.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List, Dict, Tuple


class GAINImputer:
    """
    GAIN Imputer - 优化版
    
    主要改进：
    - 正确的损失函数归一化
    - 更稳定的训练过程
    - 详细的诊断输出
    """

    def __init__(self,
                 categorical_vars: Optional[List[str]] = None,
                 continuous_vars: Optional[List[str]] = None,
                 hidden_dim: int = 256,
                 batch_size: int = 128,
                 hint_rate: float = 0.9,
                 alpha: float = 100.0,
                 iterations: int = 10000,
                 learning_rate: float = 1e-3,
                 seed: int = 42,
                 use_gpu: bool = False,
                 print_loss_every: int = 1000,
                 # v5新增参数
                 grad_clip: float = 1.0,           # 梯度裁剪阈值
                 warmup_iters: int = 1000,         # warmup迭代数
                 use_layer_norm: bool = False,     # 是否使用LayerNorm
                 d_steps: int = 1,                 # 每轮D的训练步数
                 g_steps: int = 1,                 # 每轮G的训练步数
                 verbose: bool = True):            # 是否输出详细信息
        
        self.categorical_vars = categorical_vars if categorical_vars else []
        self.continuous_vars = continuous_vars if continuous_vars else []
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.seed = seed
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.print_loss_every = print_loss_every
        
        # v5新增
        self.grad_clip = grad_clip
        self.warmup_iters = warmup_iters
        self.use_layer_norm = use_layer_norm
        self.d_steps = d_steps
        self.g_steps = g_steps
        self.verbose = verbose

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        self.models_ = {}
        self.cat_mappings = {}
        self.training_history = []
        
    def _normalize_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Min-Max归一化到[0,1]范围（官方做法）
        """
        if fit:
            self.data_min = np.nanmin(data, axis=0)
            self.data_max = np.nanmax(data, axis=0)
            self.data_range = self.data_max - self.data_min
            # 防止除零
            self.data_range[self.data_range == 0] = 1.0
        
        normalized = (data - self.data_min) / self.data_range
        return normalized
    
    def _denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """反归一化"""
        return data * self.data_range + self.data_min

    class Generator(nn.Module):
        """
        Generator网络
        
        输入: [data, mask] concat -> dim*2
        输出: 生成的完整数据 -> dim (sigmoid)
        """
        def __init__(self, input_dim: int, hidden_dim: int, use_layer_norm: bool = False):
            super().__init__()
            
            if use_layer_norm:
                self.net = nn.Sequential(
                    nn.Linear(input_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                )
            else:
                self.net = nn.Sequential(
                    nn.Linear(input_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                )
            self._init_weights()
            
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
        def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            inputs = torch.cat([x, m], dim=1)
            return self.net(inputs)

    class Discriminator(nn.Module):
        """
        Discriminator网络
        
        输入: [data, hint] concat -> dim*2
        输出: 每个component是observed的概率 -> dim (sigmoid)
        """
        def __init__(self, input_dim: int, hidden_dim: int, use_layer_norm: bool = False):
            super().__init__()
            
            if use_layer_norm:
                self.net = nn.Sequential(
                    nn.Linear(input_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                )
            else:
                self.net = nn.Sequential(
                    nn.Linear(input_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                )
            self._init_weights()
            
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
        def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            inputs = torch.cat([x, h], dim=1)
            return self.net(inputs)

    def _prepare_data(self, X_complete, X_missing: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """数据预处理

        R1 (P1 T1.4): ``X_complete`` may now be ``None``. In that case the
        one-hot vocabulary is taken from the OBSERVED cells of ``X_missing``
        (the registry has already stamped the fitted vocabulary onto the
        categorical dtype), and no ground-truth matrix is built at all. When
        ``X_complete`` is a frame the R0 behavior is reproduced verbatim: the
        vocabulary came from the truth table (``:213-215`` below) and the
        returned ``all_data_comp`` was used to fit the min-max normaliser
        (``:374``).
        """
        deleaked = X_complete is None
        df_comp = None if deleaked else X_complete.reset_index(drop=True)
        df_mis = X_missing.reset_index(drop=True)

        if deleaked:
            return self._prepare_data_deleaked(df_mis)

        # 处理连续变量
        if self.continuous_vars:
            cont_data_comp = df_comp[self.continuous_vars].values.astype(float)
            cont_data_mis = df_mis[self.continuous_vars].values.astype(float)
            cont_mask = (~np.isnan(cont_data_mis)).astype(float)
        else:
            cont_data_comp = np.zeros((len(df_comp), 0))
            cont_data_mis = np.zeros((len(df_mis), 0))
            cont_mask = np.zeros((len(df_mis), 0))

        # 处理分类变量
        cat_data_comp_list = []
        cat_data_mis_list = []
        cat_mask_list = []
        
        for col in self.categorical_vars:
            df_comp[col] = df_comp[col].astype('category')
            categories = list(df_comp[col].cat.categories)
            self.cat_mappings[col] = categories
            num_cats = len(categories)
            
            # 完整数据one-hot
            cat_comp = np.zeros((len(df_comp), num_cats))
            codes_comp = df_comp[col].cat.codes.values
            cat_comp[np.arange(len(df_comp)), codes_comp] = 1.0
            cat_data_comp_list.append(cat_comp)
            
            # 缺失数据处理
            df_mis[col] = df_mis[col].astype('category')
            df_mis[col] = df_mis[col].cat.set_categories(categories)
            
            is_observed = ~df_mis[col].isna().values
            cat_mis = np.zeros((len(df_mis), num_cats))
            
            if is_observed.any():
                codes_mis = df_mis[col].cat.codes[is_observed].values
                cat_mis[is_observed, codes_mis] = 1.0
            
            cat_data_mis_list.append(cat_mis)
            
            cat_mask_col = np.zeros((len(df_mis), num_cats))
            cat_mask_col[is_observed, :] = 1.0
            cat_mask_list.append(cat_mask_col)

        # 合并数据
        if cat_data_comp_list:
            all_data_comp = np.hstack([cont_data_comp] + cat_data_comp_list)
            all_data_mis = np.hstack([cont_data_mis] + cat_data_mis_list)
            all_mask = np.hstack([cont_mask] + cat_mask_list)
        else:
            all_data_comp = cont_data_comp
            all_data_mis = cont_data_mis
            all_mask = cont_mask

        return all_data_comp, all_data_mis, all_mask

    def _prepare_data_deleaked(self, df_mis: pd.DataFrame) -> Tuple[None, np.ndarray, np.ndarray]:
        """Encode the incomplete table only. Same layout as :meth:`_prepare_data`."""
        df_mis = df_mis.copy()

        if self.continuous_vars:
            cont_data_mis = df_mis[self.continuous_vars].values.astype(float)
            cont_mask = (~np.isnan(cont_data_mis)).astype(float)
        else:
            cont_data_mis = np.zeros((len(df_mis), 0))
            cont_mask = np.zeros((len(df_mis), 0))

        cat_data_mis_list, cat_mask_list = [], []
        for col in self.categorical_vars:
            s = df_mis[col]
            if isinstance(s.dtype, pd.CategoricalDtype):
                categories = list(s.cat.categories)
            else:
                categories = sorted(pd.Series(s.dropna().unique()).tolist())
                s = pd.Categorical(s, categories=categories)
                df_mis[col] = s
            self.cat_mappings[col] = categories
            num_cats = len(categories)

            is_observed = ~df_mis[col].isna().values
            cat_mis = np.zeros((len(df_mis), num_cats))
            if is_observed.any():
                codes_mis = df_mis[col].cat.codes[is_observed].values
                cat_mis[is_observed, codes_mis] = 1.0
            cat_data_mis_list.append(cat_mis)

            cat_mask_col = np.zeros((len(df_mis), num_cats))
            cat_mask_col[is_observed, :] = 1.0
            cat_mask_list.append(cat_mask_col)

        if cat_data_mis_list:
            all_data_mis = np.hstack([cont_data_mis] + cat_data_mis_list)
            all_mask = np.hstack([cont_mask] + cat_mask_list)
        else:
            all_data_mis = cont_data_mis
            all_mask = cont_mask

        return None, all_data_mis, all_mask

    def _sample_batch(self, n_samples: int) -> np.ndarray:
        """
        随机采样batch索引 - 使用有放回采样（与官方一致）
        """
        if n_samples <= self.batch_size:
            return np.arange(n_samples)
        # v5修复：使用replace=True
        return np.random.choice(n_samples, self.batch_size, replace=True)
    
    def _sample_hint(self, mask: torch.Tensor, hint_rate: float) -> torch.Tensor:
        """
        生成hint向量（官方实现）
        
        H = M * B + 0.5 * (1 - B)
        其中 B ~ Bernoulli(hint_rate)
        """
        batch_size, dim = mask.shape
        b = (torch.rand(batch_size, dim, device=self.device) < hint_rate).float()
        hint = mask * b + 0.5 * (1 - b)
        return hint

    def _discriminator_loss(self, m: torch.Tensor, d_prob: torch.Tensor) -> torch.Tensor:
        """
        Discriminator损失函数 - 正确归一化版本
        
        官方公式：L_D = -E[M * log(D) + (1-M) * log(1-D)]
        
        关键修复：分别对观测位置和缺失位置计算损失，然后正确归一化
        """
        eps = 1e-8
        
        # 观测位置的损失（D应输出高值）
        obs_loss = -m * torch.log(d_prob + eps)
        
        # 缺失位置的损失（D应输出低值）
        mis_loss = -(1 - m) * torch.log(1 - d_prob + eps)
        
        # 正确归一化：分别按各自数量归一化
        num_obs = torch.sum(m) + eps
        num_mis = torch.sum(1 - m) + eps
        
        loss = torch.sum(obs_loss) / num_obs + torch.sum(mis_loss) / num_mis
        
        return loss

    def _generator_loss(self, x_true: torch.Tensor, m: torch.Tensor, 
                       g_sample: torch.Tensor, d_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generator损失函数 - 正确归一化版本
        
        官方公式：
        L_G = -E[(1-M) * log(D)] + alpha * MSE(M ⊙ X, M ⊙ G)
        
        关键修复：
        1. 对抗损失只在缺失位置计算，除以缺失元素数量
        2. 重构损失只在观测位置计算，除以观测元素数量
        """
        eps = 1e-8
        
        num_obs = torch.sum(m) + eps
        num_mis = torch.sum(1 - m) + eps
        
        # 对抗损失：让D把missing位置误判为observed
        # 只在缺失位置计算，除以缺失元素数量
        adversarial_loss = -torch.sum((1 - m) * torch.log(d_prob + eps)) / num_mis
        
        # 重构损失：在observed位置的MSE
        # 关键修复：除以观测元素数量，而非总元素数量
        squared_error = (x_true - g_sample) ** 2
        reconstruction_loss = torch.sum(m * squared_error) / num_obs
        
        # 总损失
        total_loss = adversarial_loss + self.alpha * reconstruction_loss
        
        return total_loss, adversarial_loss, reconstruction_loss

    def _get_lr_multiplier(self, iteration: int) -> float:
        """学习率warmup"""
        if iteration < self.warmup_iters:
            return (iteration + 1) / self.warmup_iters
        return 1.0

    def _compute_grad_norm(self, model: nn.Module) -> float:
        """计算模型梯度范数"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return np.sqrt(total_norm)

    def _check_nan(self, tensor: torch.Tensor, name: str) -> bool:
        """检查tensor中是否有NaN"""
        has_nan = torch.isnan(tensor).any().item()
        if has_nan and self.verbose:
            print(f"  警告: {name} 包含 NaN!")
        return has_nan

    def impute(self, X_complete, X_missing: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        主要插补方法

        R1 (P1 T1.4): pass ``X_complete=None`` for the de-leaked path. The two
        oracle touch points are (i) the one-hot vocabulary and (ii) the min-max
        normaliser, which R0 fitted on the ground-truth matrix at ``:374``.
        Everything downstream -- the generator/discriminator architecture, the
        hint mechanism, the loss normalization, the training schedule -- is
        untouched.
        """
        if X_complete is not None and X_complete.shape != X_missing.shape:
            raise ValueError("X_complete 和 X_missing 必须具有相同shape")

        print("=" * 70)
        print("GAIN v5 - 开始插补（优化版）")
        print("=" * 70)
        
        # 数据预处理
        print("\n[1/4] 数据预处理...")
        data_complete, data_missing, mask = self._prepare_data(X_complete, X_missing)
        
        n_samples, n_features = data_missing.shape
        cont_dim = len(self.continuous_vars)
        obs_rate = np.mean(mask)
        
        print(f"  样本数: {n_samples}")
        print(f"  特征数: {n_features} (连续: {cont_dim}, 分类one-hot: {n_features - cont_dim})")
        print(f"  观测比例: {obs_rate:.3f} (缺失比例: {1-obs_rate:.3f})")
        
        # Min-Max归一化
        # R0: the normaliser was fitted on the GROUND-TRUTH matrix (oracle leak).
        # R1: it is fitted on the incomplete matrix; _normalize_data already uses
        #     nanmin/nanmax (:93-94), so missing cells are ignored automatically.
        if data_complete is None:
            self._normalize_data(data_missing, fit=True)
        else:
            _ = self._normalize_data(data_complete, fit=True)
        data_missing_norm = self._normalize_data(data_missing, fit=False)
        
        # 对缺失位置填充随机值
        noise = np.random.uniform(0, 0.01, size=data_missing_norm.shape)
        data_filled = np.where(mask == 1, data_missing_norm, noise)
        
        # 检查数据
        if np.isnan(data_filled).any():
            print("  错误: data_filled 包含 NaN!")
            # 尝试修复
            data_filled = np.nan_to_num(data_filled, nan=0.5)
            print("  已将NaN替换为0.5")
        
        # 转换为tensor
        X = torch.tensor(data_filled, dtype=torch.float32, device=self.device)
        X_true = torch.tensor(data_filled, dtype=torch.float32, device=self.device)
        M = torch.tensor(mask, dtype=torch.float32, device=self.device)
        
        # 构建模型
        print("\n[2/4] 构建模型...")
        self.generator = self.Generator(n_features, self.hidden_dim, self.use_layer_norm).to(self.device)
        self.discriminator = self.Discriminator(n_features, self.hidden_dim, self.use_layer_norm).to(self.device)
        
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"  Generator参数量: {g_params:,}")
        print(f"  Discriminator参数量: {d_params:,}")
        
        # 优化器
        g_optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        
        print(f"\n[3/4] 训练参数:")
        print(f"  alpha: {self.alpha}")
        print(f"  hint_rate: {self.hint_rate}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  iterations: {self.iterations}")
        print(f"  learning_rate: {self.learning_rate}")
        print(f"  grad_clip: {self.grad_clip}")
        print(f"  warmup_iters: {self.warmup_iters}")
        print(f"  use_layer_norm: {self.use_layer_norm}")
        
        print(f"\n开始训练...")
        
        # 训练循环
        best_recon_loss = float('inf')
        nan_count = 0
        
        for iteration in range(self.iterations):
            # 学习率warmup
            lr_mult = self._get_lr_multiplier(iteration)
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = self.learning_rate * lr_mult
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = self.learning_rate * lr_mult
            
            # ========== 训练Discriminator ==========
            for _ in range(self.d_steps):
                idx = self._sample_batch(n_samples)
                X_batch = X[idx]
                X_true_batch = X_true[idx]
                M_batch = M[idx]
                
                H_batch = self._sample_hint(M_batch, self.hint_rate)
                
                d_optimizer.zero_grad()
                
                with torch.no_grad():
                    G_sample = self.generator(X_batch, M_batch)
                
                X_hat = M_batch * X_true_batch + (1 - M_batch) * G_sample
                D_prob = self.discriminator(X_hat.detach(), H_batch)
                
                D_loss = self._discriminator_loss(M_batch, D_prob)
                
                if torch.isnan(D_loss):
                    nan_count += 1
                    if nan_count > 100:
                        print("  错误: 训练不稳定，过多NaN损失")
                        break
                    continue
                
                D_loss.backward()
                
                # 梯度裁剪
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
                
                d_optimizer.step()
            
            # ========== 训练Generator ==========
            for _ in range(self.g_steps):
                idx = self._sample_batch(n_samples)
                X_batch = X[idx]
                X_true_batch = X_true[idx]
                M_batch = M[idx]
                
                H_batch = self._sample_hint(M_batch, self.hint_rate)
                
                g_optimizer.zero_grad()
                
                G_sample = self.generator(X_batch, M_batch)
                X_hat = M_batch * X_true_batch + (1 - M_batch) * G_sample
                D_prob = self.discriminator(X_hat, H_batch)
                
                G_loss, adv_loss, recon_loss = self._generator_loss(
                    X_true_batch, M_batch, G_sample, D_prob
                )
                
                if torch.isnan(G_loss):
                    nan_count += 1
                    if nan_count > 100:
                        print("  错误: 训练不稳定，过多NaN损失")
                        break
                    continue
                
                G_loss.backward()
                
                # 梯度裁剪
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip)
                
                g_optimizer.step()
            
            # 记录历史
            self.training_history.append({
                'iteration': iteration + 1,
                'D_loss': D_loss.item(),
                'G_loss': G_loss.item(),
                'adv_loss': adv_loss.item(),
                'recon_loss': recon_loss.item(),
            })
            
            # 跟踪最佳重构损失
            if recon_loss.item() < best_recon_loss:
                best_recon_loss = recon_loss.item()
            
            # 打印进度
            if (iteration + 1) % self.print_loss_every == 0:
                g_grad = self._compute_grad_norm(self.generator)
                d_grad = self._compute_grad_norm(self.discriminator)
                
                print(f"  Iter {iteration+1:5d}/{self.iterations}: "
                      f"D={D_loss.item():.4f}, G={G_loss.item():.4f} "
                      f"(adv={adv_loss.item():.4f}, recon={recon_loss.item():.6f}) "
                      f"grad_G={g_grad:.3f}, grad_D={d_grad:.3f}")

        # ========== 最终插补 ==========
        print(f"\n[4/4] 生成最终插补结果...")
        print(f"  最佳重构损失: {best_recon_loss:.6f}")
        
        self.generator.eval()
        with torch.no_grad():
            G_final = self.generator(X, M)
            X_imputed = M * X_true + (1 - M) * G_final
            imputed_data_norm = X_imputed.cpu().numpy()
        
        # 反归一化
        imputed_data = self._denormalize_data(imputed_data_norm)
        
        # 重构DataFrame
        result_df = self._reconstruct_dataframe(X_missing, imputed_data)
        
        self.models_['generator'] = self.generator
        self.models_['discriminator'] = self.discriminator
        
        print("=" * 70)
        print("GAIN v5 - 插补完成")
        print("=" * 70)
        
        return result_df, self.models_

    def _reconstruct_dataframe(self, X_missing: pd.DataFrame, imputed_data: np.ndarray) -> pd.DataFrame:
        """重构DataFrame"""
        df_result = X_missing.copy().reset_index(drop=True)
        
        # 连续变量
        for i, col in enumerate(self.continuous_vars):
            missing_mask = X_missing[col].isna()
            if missing_mask.any():
                imputed_values = imputed_data[missing_mask.values, i]
                df_result.loc[missing_mask, col] = imputed_values
                if self.verbose:
                    print(f"  连续变量 '{col}': 插补了 {missing_mask.sum()} 个缺失值")
                    print(f"    范围: [{imputed_values.min():.4f}, {imputed_values.max():.4f}]")
        
        # 分类变量
        start_idx = len(self.continuous_vars)
        for col in self.categorical_vars:
            categories = self.cat_mappings[col]
            num_cats = len(categories)
            end_idx = start_idx + num_cats
            
            missing_mask = X_missing[col].isna()
            if missing_mask.any():
                cat_probs = imputed_data[missing_mask.values, start_idx:end_idx]
                
                # 使用softmax归一化
                cat_probs = np.clip(cat_probs, 1e-8, 1.0)
                cat_probs = cat_probs / cat_probs.sum(axis=1, keepdims=True)
                
                predicted_indices = np.argmax(cat_probs, axis=1)
                predicted_labels = [categories[idx] for idx in predicted_indices]
                
                df_result.loc[missing_mask, col] = predicted_labels
                
                if self.verbose:
                    print(f"  分类变量 '{col}': 插补了 {missing_mask.sum()} 个缺失值")
                    unique, counts = np.unique(predicted_labels, return_counts=True)
                    dist_str = ", ".join([f"{u}:{c}" for u, c in zip(unique, counts)])
                    print(f"    分布: {dist_str}")
                    
                    # 显示置信度信息
                    max_probs = np.max(cat_probs, axis=1)
                    print(f"    置信度: mean={max_probs.mean():.3f}, min={max_probs.min():.3f}, max={max_probs.max():.3f}")
            
            start_idx = end_idx
        
        return df_result


# ============================================================
# 兼容性包装器（与baselines/registry.py接口兼容）
# ============================================================

class GAINWrapper:
    """
    包装器类，提供与其他baseline一致的接口
    """
    def __init__(self,
                 categorical_vars=None,
                 continuous_vars=None,
                 seed=42,
                 use_gpu=False,
                 # GAIN特有参数
                 hidden_dim=256,
                 batch_size=128,
                 hint_rate=0.9,
                 alpha=100.0,
                 iterations=10000,
                 learning_rate=1e-3,
                 print_loss_every=1000,
                 # v5新增参数
                 grad_clip=1.0,
                 warmup_iters=1000,
                 use_layer_norm=False,
                 verbose=True,
                 **kwargs):
        
        self.categorical_vars = categorical_vars if categorical_vars else []
        self.continuous_vars = continuous_vars if continuous_vars else []
        self.seed = seed
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.print_loss_every = print_loss_every
        self.grad_clip = grad_clip
        self.warmup_iters = warmup_iters
        self.use_layer_norm = use_layer_norm
        self.verbose = verbose
        
    def impute(self, X_complete, X_missing):
        """执行插补"""
        imputer = GAINImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            hint_rate=self.hint_rate,
            alpha=self.alpha,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            seed=self.seed,
            use_gpu=self.use_gpu,
            print_loss_every=self.print_loss_every,
            grad_clip=self.grad_clip,
            warmup_iters=self.warmup_iters,
            use_layer_norm=self.use_layer_norm,
            verbose=self.verbose,
        )
        
        X_imputed, _ = imputer.impute(X_complete, X_missing)
        return X_imputed


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    # 完整数据
    data = {
        'age': np.random.normal(50, 15, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'score': np.random.normal(75, 10, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
    }
    X_complete = pd.DataFrame(data)
    
    # 引入30%缺失
    X_missing = X_complete.copy()
    miss_rate = 0.3
    for col in X_missing.columns:
        mask = np.random.random(n_samples) < miss_rate
        X_missing.loc[mask, col] = np.nan
    
    print("测试GAIN v5...")
    print(f"缺失率: {X_missing.isna().mean().mean():.2%}")
    
    imputer = GAINImputer(
        categorical_vars=['gender'],
        continuous_vars=['age', 'income', 'score'],
        iterations=5000,
        print_loss_every=500,
    )
    
    X_imputed, _ = imputer.impute(X_complete, X_missing)
    
    # 计算RMSE
    for col in ['age', 'income', 'score']:
        missing_mask = X_missing[col].isna()
        true_vals = X_complete.loc[missing_mask, col]
        imputed_vals = X_imputed.loc[missing_mask, col]
        rmse = np.sqrt(np.mean((true_vals - imputed_vals) ** 2))
        nrmse = rmse / X_complete[col].std()
        print(f"{col}: RMSE={rmse:.4f}, NRMSE={nrmse:.4f}")
    
    # 分类变量准确率
    missing_mask = X_missing['gender'].isna()
    accuracy = (X_complete.loc[missing_mask, 'gender'] == X_imputed.loc[missing_mask, 'gender']).mean()
    print(f"gender: Accuracy={accuracy:.4f}")