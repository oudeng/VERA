# MIWAE_v3.py
# -*- coding: utf-8 -*-
"""
MIWAE (Missing Data Importance-Weighted Autoencoder) - 修正版
基于官方论文和实现: https://github.com/pamattei/miwae

主要修正点（相比v2）：
1. Decoder输出可学习的方差log_sigma（v2固定为1，严重错误）
2. 网络架构：tanh激活，无BatchNorm/Dropout（与论文一致）
3. K=20（训练时重要性采样数，论文UCI实验设置）
4. L=10000（插补时采样数，论文设置）
5. 增加训练迭代次数（论文用500,000 steps）
6. 插补使用自归一化重要性采样权重（v2用简单平均）
7. 关闭早停机制（VAE的ELBO波动大）
8. 修正ELBO计算中的多余项

Reference:
    Mattei, P.-A. & Frellsen, J. (2019).
    MIWAE: Deep Generative Modeling and Imputation of Incomplete Data Sets.
    ICML 2019. http://proceedings.mlr.press/v97/mattei19a.html
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MIWAEImputer:
    """
    MIWAE Imputer - 忠于原论文实现
    """

    def __init__(
        self,
        categorical_vars,
        continuous_vars,
        # 网络架构参数（论文Section 4.3）
        hidden_dims=[128, 128, 128],  # 3层，128 units
        latent_dim=10,                 # 论文UCI实验用10
        # 训练参数
        num_iw_samples=20,             # K=20（论文UCI实验）
        num_impute_samples=10000,      # L=10000（论文插补采样数）
        lr=1e-3,                       # 论文默认学习率
        batch_size=64,                 
        epochs=500,                    # 会根据数据量调整为约500k steps
        min_epochs=200,                # 最小训练轮数
        seed=42,
        use_gpu=False,
        print_loss_every=50,
        # 方差约束（论文Section 4.3：特征值>0.01）
        min_variance=0.01,
    ):
        self.categorical_vars = categorical_vars if categorical_vars else []
        self.continuous_vars = continuous_vars if continuous_vars else []
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.num_iw_samples = num_iw_samples
        self.num_impute_samples = num_impute_samples
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.min_epochs = min_epochs
        self.seed = seed
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.print_loss_every = print_loss_every
        self.min_variance = min_variance

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.encoder = None
        self.decoder = None
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.models_ = {}
        self.cat_mappings = {}
        self.cat_dims = []
        
        # 标准化参数
        self.cont_mean = None
        self.cont_std = None

    def _normalize_continuous(self, data, fit=True):
        """标准化连续变量（Z-score）"""
        if fit:
            self.cont_mean = np.nanmean(data, axis=0)
            self.cont_std = np.nanstd(data, axis=0)
            self.cont_std[self.cont_std == 0] = 1.0
        
        normalized = (data - self.cont_mean) / self.cont_std
        return normalized

    def _denormalize_continuous(self, data):
        """反标准化连续变量"""
        return data * self.cont_std + self.cont_mean

    def _prepare_data(self, X_missing_df):
        """数据预处理：转换为tensor格式"""
        df = X_missing_df.copy().reset_index(drop=True)
        n = df.shape[0]

        cont_list = []
        cont_mask_list = []
        cat_onehot_list = []
        cat_mask_list = []
        cont_idx = []
        cat_idx = []

        current_dim = 0

        # 1) 连续变量
        if self.continuous_vars:
            cont_data = df[self.continuous_vars].values.astype(float)
            cont_data_norm = self._normalize_continuous(cont_data, fit=True)
            
            for i, col in enumerate(self.continuous_vars):
                arr = cont_data_norm[:, i]
                mask_arr = (~np.isnan(arr)).astype(float)
                arr = np.nan_to_num(arr, nan=0.0)  # 零填充（论文推荐）

                cont_list.append(arr.reshape(n, 1))
                cont_mask_list.append(mask_arr.reshape(n, 1))
                cont_idx.append((current_dim, current_dim + 1))
                current_dim += 1

        # 2) 分类变量
        for col in self.categorical_vars:
            df[col] = df[col].astype('category')
            cats = list(df[col].cat.categories)
            self.cat_mappings[col] = cats
            num_classes = len(cats)
            self.cat_dims.append(num_classes)

            codes = df[col].cat.codes.values.astype(int)
            mask_arr = (codes != -1).astype(float)
            codes[codes == -1] = 0

            one_hot = np.zeros((n, num_classes), dtype=float)
            one_hot[np.arange(n), codes] = 1.0

            # 分类变量的mask：观测到的行所有类别位置都为1
            mask_one_hot = np.zeros((n, num_classes), dtype=float)
            for i in range(n):
                if mask_arr[i] == 1:
                    mask_one_hot[i, :] = 1.0

            cat_onehot_list.append(one_hot)
            cat_mask_list.append(mask_one_hot)
            cat_idx.append((current_dim, current_dim + num_classes))
            current_dim += num_classes

        # 合并
        if cont_list:
            cont_block = np.concatenate(cont_list, axis=1)
            cont_mask_block = np.concatenate(cont_mask_list, axis=1)
        else:
            cont_block = np.zeros((n, 0))
            cont_mask_block = np.zeros((n, 0))

        if cat_onehot_list:
            cat_block = np.concatenate(cat_onehot_list, axis=1)
            cat_mask_block = np.concatenate(cat_mask_list, axis=1)
        else:
            cat_block = np.zeros((n, 0))
            cat_mask_block = np.zeros((n, 0))

        data_np = np.concatenate([cont_block, cat_block], axis=1)
        mask_np = np.concatenate([cont_mask_block, cat_mask_block], axis=1)

        data_tensor = torch.tensor(data_np, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32, device=self.device)

        return data_tensor, mask_tensor, cont_idx, cat_idx

    class _Encoder(nn.Module):
        """
        Encoder网络 - 与论文一致
        
        论文架构：
        - 输入: [x * mask, mask] concat -> dim * 2
        - 3层hidden，128 units
        - tanh激活（论文明确指出）
        - 无BatchNorm，无Dropout
        - 输出: mu, logvar
        """
        def __init__(self, input_dim, hidden_dims, latent_dim):
            super().__init__()
            # 输入是 [data * mask, mask] 的concat
            dims = [input_dim * 2] + hidden_dims
            layers = []
            for i in range(len(hidden_dims)):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                layers.append(nn.Tanh())  # 论文用tanh
            self.encoder_net = nn.Sequential(*layers)
            
            self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
            
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x, mask):
            """
            Args:
                x: 数据（缺失位置已零填充）
                mask: 观测mask（1=observed, 0=missing）
            Returns:
                mu, logvar: 变分分布参数
            """
            # 零填充策略（论文Section 2.3推荐）
            x_masked = x * mask
            enc_input = torch.cat([x_masked, mask], dim=1)
            h = self.encoder_net(enc_input)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

    class _Decoder(nn.Module):
        """
        Decoder网络 - 与论文一致
        
        关键修正：输出可学习的方差log_sigma
        
        论文架构：
        - 输入: z (latent_dim)
        - 3层hidden，128 units
        - tanh激活
        - 输出: 
          - 连续变量: mu和log_sigma（可学习方差！）
          - 分类变量: logits
        """
        def __init__(self, latent_dim, hidden_dims, cont_dim, cat_dims, min_variance=0.01):
            super().__init__()
            self.cont_dim = cont_dim
            self.cat_dims = cat_dims
            self.min_variance = min_variance
            self.min_log_var = np.log(min_variance)
            
            dims = [latent_dim] + hidden_dims
            layers = []
            for i in range(len(hidden_dims)):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                layers.append(nn.Tanh())  # 论文用tanh
            self.decoder_net = nn.Sequential(*layers)
            
            # 连续变量：输出mu和log_sigma（关键修正！）
            self.fc_cont_mu = nn.Linear(hidden_dims[-1], cont_dim)
            self.fc_cont_logvar = nn.Linear(hidden_dims[-1], cont_dim)  # 可学习方差
            
            # 分类变量：输出logits
            self.fc_cat_logits = nn.ModuleList()
            for cd in cat_dims:
                self.fc_cat_logits.append(nn.Linear(hidden_dims[-1], cd))
            
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, z):
            """
            Returns:
                cont_mu: 连续变量均值
                cont_logvar: 连续变量对数方差（可学习！）
                cat_logits: 分类变量logits列表
            """
            h = self.decoder_net(z)
            
            cont_mu = self.fc_cont_mu(h)
            # 方差约束：论文Section 4.3要求特征值>0.01
            cont_logvar = self.fc_cont_logvar(h)
            cont_logvar = torch.clamp(cont_logvar, min=self.min_log_var)
            
            cat_logits = []
            for fc in self.fc_cat_logits:
                cat_logits.append(fc(h))
            
            return cont_mu, cont_logvar, cat_logits

    def _log_normal(self, x, mu, logvar):
        """
        计算高斯对数概率密度
        
        log N(x | mu, sigma^2) = -0.5 * (log(2π) + logvar + (x-mu)^2/var)
        """
        return -0.5 * (np.log(2 * np.pi) + logvar + (x - mu).pow(2) / torch.exp(logvar))

    def _compute_log_p_x_given_z(self, x, mask, dec_cont_mu, dec_cont_logvar, 
                                  dec_cat_logits, cont_idx, cat_idx):
        """
        计算 log p(x^o | z)
        
        只计算观测到的数据的对数似然
        """
        batch_size = x.shape[0]
        total_log_px = torch.zeros(batch_size, device=self.device)

        # 连续变量：高斯似然，使用可学习的方差
        for i, (start, end) in enumerate(cont_idx):
            x_i = x[:, start:end]
            mask_i = mask[:, start:end]
            mu_i = dec_cont_mu[:, i:i+1]
            logvar_i = dec_cont_logvar[:, i:i+1]  # 使用可学习的方差！
            
            log_pdf = self._log_normal(x_i, mu_i, logvar_i)
            total_log_px += (log_pdf * mask_i).sum(dim=1)

        # 分类变量：Categorical似然
        for j, (start, end) in enumerate(cat_idx):
            x_j = x[:, start:end]
            mask_j = mask[:, start:end]
            logits_j = dec_cat_logits[j]
            log_prob_j = F.log_softmax(logits_j, dim=1)
            # 只对观测到的行计算
            # mask_j的每一行要么全1要么全0
            row_mask = mask_j[:, 0:1]  # 取第一列作为行mask
            total_log_px += (log_prob_j * x_j).sum(dim=1) * row_mask.squeeze()

        return total_log_px

    def _compute_importance_weights(self, x, mask, mu_z, logvar_z, z_samples,
                                     dec_cont_mu, dec_cont_logvar, dec_cat_logits,
                                     cont_idx, cat_idx):
        """
        计算重要性权重 w_k = p(x^o|z_k) * p(z_k) / q(z_k|x^o)
        
        返回归一化后的log权重
        """
        batch_size, K, d = z_samples.shape
        
        # 展平处理
        z_flat = z_samples.view(batch_size * K, d)
        x_rep = x.unsqueeze(1).repeat(1, K, 1).view(batch_size * K, -1)
        mask_rep = mask.unsqueeze(1).repeat(1, K, 1).view(batch_size * K, -1)
        
        # log p(x^o | z)
        log_px = self._compute_log_p_x_given_z(
            x_rep, mask_rep, dec_cont_mu, dec_cont_logvar, dec_cat_logits,
            cont_idx, cat_idx
        ).view(batch_size, K)
        
        # log p(z) - 标准正态先验
        log_pz = -0.5 * (z_samples.pow(2) + np.log(2 * np.pi)).sum(dim=2)
        
        # log q(z | x^o)
        mu_z_rep = mu_z.unsqueeze(1).repeat(1, K, 1)
        logvar_z_rep = logvar_z.unsqueeze(1).repeat(1, K, 1)
        log_qz = -0.5 * (logvar_z_rep + (z_samples - mu_z_rep).pow(2) / torch.exp(logvar_z_rep) + np.log(2 * np.pi)).sum(dim=2)
        
        # log w = log p(x^o|z) + log p(z) - log q(z|x^o)
        log_w = log_px + log_pz - log_qz
        
        return log_w

    def impute(self, X_incomplete, X_missing):
        """
        主要插补方法

        Args:
            X_incomplete: 完整数据（用于获取schema，实际不使用其值）
            X_missing: 带缺失的数据

        Returns:
            X_imputed: 插补后的DataFrame
            models_: 训练好的模型字典

        R1 note (P1 T1.4 / P0 finding B6)
        ---------------------------------
        The body genuinely never reads ``X_incomplete``'s values -- only its
        shape. The leak was one level up: ``registry.py:250`` ran
        ``set_categories_from_complete`` first, so ``X_missing`` arrived with the
        ORACLE category vocabulary baked into its categorical dtype, which
        ``_prepare_data`` then adopts. Continuous standardization was already
        observed-only (``_normalize_continuous(..., fit=True)`` on ``X_missing``).
        In R1 the registry stamps the observed vocabulary instead and passes
        ``X_incomplete=None``.
        """
        if X_incomplete is not None and X_incomplete.shape != X_missing.shape:
            raise ValueError("X_incomplete 和 X_missing 的 shape 必须一致。")

        print("=" * 60)
        print("MIWAE v3 - 开始训练")
        print("=" * 60)

        # 数据预处理
        data_tensor, mask_tensor, cont_idx, cat_idx = self._prepare_data(X_missing)
        n, D_input = data_tensor.shape
        cont_dim = len(self.continuous_vars)
        cat_dims = self.cat_dims

        print(f"数据维度: {n} 样本, {D_input} 特征")
        print(f"连续变量: {cont_dim}, 分类变量: {len(cat_dims)}")
        print(f"观测比例: {mask_tensor.mean().item():.3f}")

        # 构建模型
        self.encoder = self._Encoder(
            input_dim=D_input,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim
        ).to(self.device)
        
        self.decoder = self._Decoder(
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims[::-1],  # 镜像架构
            cont_dim=cont_dim,
            cat_dims=cat_dims,
            min_variance=self.min_variance
        ).to(self.device)

        # 优化器（论文无weight decay）
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr
        )

        # 数据加载
        dataset = TensorDataset(data_tensor, mask_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 计算实际训练步数
        steps_per_epoch = len(loader)
        total_steps = self.epochs * steps_per_epoch
        
        print(f"\n训练参数:")
        print(f"  K (importance samples): {self.num_iw_samples}")
        print(f"  latent_dim: {self.latent_dim}")
        print(f"  hidden_dims: {self.hidden_dims}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  epochs: {self.epochs}")
        print(f"  总步数: {total_steps}")
        print(f"  learning_rate: {self.lr}")
        print(f"\n开始训练...")

        # 训练循环（无早停，固定epochs）
        for epoch in range(self.epochs):
            self.encoder.train()
            self.decoder.train()
            total_loss = 0.0
            
            for xb, mb in loader:
                batch_size = xb.shape[0]
                
                # Encoder
                mu_z, logvar_z = self.encoder(xb, mb)

                # 重参数化采样 K 个 z
                mu_z_rep = mu_z.unsqueeze(1).repeat(1, self.num_iw_samples, 1)
                logvar_z_rep = logvar_z.unsqueeze(1).repeat(1, self.num_iw_samples, 1)
                eps = torch.randn_like(mu_z_rep, device=self.device)
                z_samples = mu_z_rep + eps * torch.exp(0.5 * logvar_z_rep)

                # Decoder
                z_flat = z_samples.view(batch_size * self.num_iw_samples, self.latent_dim)
                dec_cont_mu, dec_cont_logvar, dec_cat_logits = self.decoder(z_flat)

                # 计算MIWAE bound (Equation 4)
                log_w = self._compute_importance_weights(
                    xb, mb, mu_z, logvar_z, z_samples,
                    dec_cont_mu, dec_cont_logvar, dec_cat_logits,
                    cont_idx, cat_idx
                )
                
                # Log-sum-exp trick for numerical stability
                # L_K = E[log(1/K * sum_k w_k)] = E[log(sum_k w_k) - log(K)]
                max_log_w, _ = torch.max(log_w, dim=1, keepdim=True)
                log_sum_w = max_log_w.squeeze() + torch.log(
                    torch.exp(log_w - max_log_w).sum(dim=1) + 1e-10
                )
                # MIWAE bound: log(1/K * sum w_k) = log(sum w_k) - log(K)
                miwae_bound = log_sum_w - np.log(self.num_iw_samples)
                
                # 最大化MIWAE bound = 最小化负MIWAE bound
                loss = -miwae_bound.mean()
                
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                    max_norm=5.0
                )
                
                optimizer.step()
                total_loss += loss.item() * batch_size

            avg_loss = total_loss / n
            
            # 打印进度
            if (epoch + 1) % self.print_loss_every == 0:
                print(f"  Epoch {epoch+1:4d}/{self.epochs}: neg_MIWAE_bound = {avg_loss:.4f}")

        self.models_['encoder'] = self.encoder
        self.models_['decoder'] = self.decoder

        # ========== 插补阶段 ==========
        print(f"\n开始插补 (L={self.num_impute_samples} samples)...")
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # Encoder
            mu_z, logvar_z = self.encoder(data_tensor, mask_tensor)
            
            # 采样 L 个 z 用于插补
            L = self.num_impute_samples
            mu_z_rep = mu_z.unsqueeze(1).repeat(1, L, 1)
            logvar_z_rep = logvar_z.unsqueeze(1).repeat(1, L, 1)
            eps = torch.randn((n, L, self.latent_dim), device=self.device)
            z_samples = mu_z_rep + eps * torch.exp(0.5 * logvar_z_rep)

            # Decoder
            z_flat = z_samples.view(n * L, self.latent_dim)
            dec_cont_mu, dec_cont_logvar, dec_cat_logits = self.decoder(z_flat)
            
            # 重塑
            dec_cont_mu = dec_cont_mu.view(n, L, cont_dim)
            dec_cont_logvar = dec_cont_logvar.view(n, L, cont_dim)
            dec_cat_logits_reshaped = []
            for j, cd in enumerate(cat_dims):
                dec_cat_logits_reshaped.append(dec_cat_logits[j].view(n, L, cd))

            # 计算重要性权重（关键修正！）
            log_w = self._compute_importance_weights(
                data_tensor, mask_tensor, mu_z, logvar_z, z_samples,
                dec_cont_mu.view(n * L, cont_dim),
                dec_cont_logvar.view(n * L, cont_dim),
                [lg.view(n * L, -1) for lg in dec_cat_logits_reshaped],
                cont_idx, cat_idx
            )
            
            # 归一化权重 (self-normalized importance sampling)
            max_log_w, _ = torch.max(log_w, dim=1, keepdim=True)
            w = torch.exp(log_w - max_log_w)
            w_normalized = w / (w.sum(dim=1, keepdim=True) + 1e-10)  # (n, L)

            # 连续变量：加权平均 E[x^m|x^o] ≈ sum_l w_l * mu_l
            # 使用条件期望 E[x|z] = mu_theta(z)（论文Equation 11）
            w_expanded = w_normalized.unsqueeze(2)  # (n, L, 1)
            imputed_cont_norm = (dec_cont_mu * w_expanded).sum(dim=1)  # (n, cont_dim)
            imputed_cont = self._denormalize_continuous(imputed_cont_norm.cpu().numpy())

            # 分类变量：加权平均概率后取argmax
            imputed_cat = []
            for j, cd in enumerate(cat_dims):
                logits_j = dec_cat_logits_reshaped[j]  # (n, L, num_classes)
                probs_j = F.softmax(logits_j, dim=2)   # (n, L, num_classes)
                # 加权平均概率
                weighted_probs = (probs_j * w_expanded).sum(dim=1)  # (n, num_classes)
                # 选择最大概率的类别
                preds = weighted_probs.argmax(dim=1).cpu().numpy()
                imputed_cat.append(preds)

        # 填充结果
        df_imp = X_missing.copy().reset_index(drop=True)

        # 连续变量插补
        for i, col in enumerate(self.continuous_vars):
            missing_mask = X_missing[col].isna()
            if missing_mask.sum() > 0:
                imputed_values = imputed_cont[:, i]
                df_imp.loc[missing_mask, col] = imputed_values[missing_mask.values]
                print(f"  连续变量 '{col}': 插补了 {missing_mask.sum()} 个缺失值")

        # 分类变量插补
        for j, col in enumerate(self.categorical_vars):
            missing_mask = X_missing[col].isna()
            if missing_mask.sum() > 0:
                preds = imputed_cat[j]
                cats = self.cat_mappings[col]
                preds = np.clip(preds, 0, len(cats) - 1)
                cat_labels = [cats[idx] for idx in preds]
                
                df_imp[col] = df_imp[col].astype('object')
                for idx in range(len(missing_mask)):
                    if missing_mask.iloc[idx]:
                        df_imp.loc[idx, col] = cat_labels[idx]
                
                print(f"  分类变量 '{col}': 插补了 {missing_mask.sum()} 个缺失值")
                
                # 显示分布
                imputed_labels = [cat_labels[i] for i in range(len(cat_labels)) if missing_mask.iloc[i]]
                unique_preds, counts = np.unique(imputed_labels, return_counts=True)
                print(f"    分布: {dict(zip(unique_preds, counts))}")

        print("=" * 60)
        print("MIWAE v3 - 插补完成")
        print("=" * 60)

        return df_imp, self.models_


# ============================================================
# 兼容性包装器（与baselines/registry.py接口兼容）
# ============================================================

class MIWAEWrapper:
    """
    包装器类，提供与其他baseline一致的接口
    """
    def __init__(self,
                 categorical_vars=None,
                 continuous_vars=None,
                 seed=42,
                 use_gpu=False,
                 # MIWAE特有参数
                 hidden_dims=[128, 128, 128],
                 latent_dim=10,
                 num_iw_samples=20,
                 num_impute_samples=10000,
                 lr=1e-3,
                 batch_size=64,
                 epochs=500,
                 print_loss_every=50,
                 **kwargs):
        
        self.categorical_vars = categorical_vars if categorical_vars else []
        self.continuous_vars = continuous_vars if continuous_vars else []
        self.seed = seed
        self.use_gpu = use_gpu
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.num_iw_samples = num_iw_samples
        self.num_impute_samples = num_impute_samples
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.print_loss_every = print_loss_every
        
    def impute(self, X_complete, X_missing):
        """执行插补"""
        imputer = MIWAEImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            num_iw_samples=self.num_iw_samples,
            num_impute_samples=self.num_impute_samples,
            lr=self.lr,
            batch_size=self.batch_size,
            epochs=self.epochs,
            seed=self.seed,
            use_gpu=self.use_gpu,
            print_loss_every=self.print_loss_every,
        )
        
        X_imputed, _ = imputer.impute(X_complete, X_missing)
        return X_imputed


if __name__ == "__main__":
    # 简单测试
    print("MIWAE v3 测试")
    
    # 创建测试数据
    np.random.seed(42)
    n = 200
    
    # 连续变量
    x1 = np.random.randn(n) * 10 + 50
    x2 = np.random.randn(n) * 5 + 20
    
    # 分类变量
    cat1 = np.random.choice(['A', 'B', 'C'], n)
    
    df_complete = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'cat1': cat1
    })
    
    # 引入缺失
    df_missing = df_complete.copy()
    miss_idx_x1 = np.random.choice(n, size=int(n*0.2), replace=False)
    miss_idx_x2 = np.random.choice(n, size=int(n*0.2), replace=False)
    miss_idx_cat = np.random.choice(n, size=int(n*0.2), replace=False)
    
    df_missing.loc[miss_idx_x1, 'x1'] = np.nan
    df_missing.loc[miss_idx_x2, 'x2'] = np.nan
    df_missing.loc[miss_idx_cat, 'cat1'] = np.nan
    
    print(f"缺失比例: x1={len(miss_idx_x1)/n:.1%}, x2={len(miss_idx_x2)/n:.1%}, cat1={len(miss_idx_cat)/n:.1%}")
    
    # 测试插补（减少参数加快测试）
    imputer = MIWAEImputer(
        categorical_vars=['cat1'],
        continuous_vars=['x1', 'x2'],
        hidden_dims=[64, 64],
        latent_dim=5,
        num_iw_samples=10,
        num_impute_samples=1000,
        epochs=100,
        print_loss_every=20,
    )
    
    df_imputed, _ = imputer.impute(df_complete, df_missing)
    
    # 计算RMSE
    rmse_x1 = np.sqrt(np.mean((df_imputed.loc[miss_idx_x1, 'x1'].values - 
                                df_complete.loc[miss_idx_x1, 'x1'].values) ** 2))
    rmse_x2 = np.sqrt(np.mean((df_imputed.loc[miss_idx_x2, 'x2'].values - 
                                df_complete.loc[miss_idx_x2, 'x2'].values) ** 2))
    
    print(f"\nRMSE: x1={rmse_x1:.4f}, x2={rmse_x2:.4f}")
    
    # 计算分类准确率
    acc_cat = np.mean(df_imputed.loc[miss_idx_cat, 'cat1'].values == 
                      df_complete.loc[miss_idx_cat, 'cat1'].values)
    print(f"分类准确率: cat1={acc_cat:.2%}")