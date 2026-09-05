# Adjudicated configuration decisions

Every adjudicated configuration decision is registered here with its rationale, the affected configuration keys, and -- where one exists -- an executable assertion in `tests/test_adjudications.py`. The register exists because of a real declared-value != implemented-value incident: a device decision was accepted in writing but never landed in the scheduling config, and nothing errored. **A decision is in force only once a test asserts it.**

---

| 编号 | 日期 | 裁定 | 理由 | 受影响的键 / 位置 | 断言 |
|---|---|---|---|---|---|
| **A-1** | P2c | 全部神经方法**关闭早停**，跑满代码中已配置的 **200 轮** | 原早停规则的停止信号是逐批随机掩码位置上算出的**训练**损失（噪声大），且轮数上限与 `CosineAnnealingLR` 的 `T_max` 绑定，导致"提高预算反而更早停止"（B76）。200 不是我们选的，是代码里本来就有的值——因此不可能被指为事后调参 | `configs/training_protocol.yaml`：`protocol.disable_early_stopping`、`protocol.epochs` | `test_a1_training_protocol` |
| **A-2** | P2e §3.1 | **SNI 的规范设备 = CPU** | 四条理由，按重要性：(i) 让 ESM 的硬件声明变成真的而非需更正的错误（B2/B83——在 CUDA 上，论文声称的 CPU 复现不出 MIMIC 的 R² 符号）；(ii) 更快，0.77–0.95× 于 CUDA；(iii) 更确定（无 TF32、无 cuDNN 算法选择）；(iv) **让 SNI 的数字变差**（MIMIC +0.089 → −0.038），说明设备不是按数字挑的 | `configs/scheduling.yaml`：`method_placement.SNI` | `test_a2_sni_on_cpu` |
| **A-3** | P2f T2f.3 | **BLAS 线程数固定为 2 并验证** | 线程数会改变结果（B84：R0 表上 Macro-F1 漂移 0.052 = Table S3 头条效应的 22%）。必须在 numpy/torch **导入之前**设定，并断言 `torch.get_num_threads()` 等于期望值——env var 是请求，不是事实（B48 的教训） | `experiments/run_grid.py` 顶部；`SNI_NUM_THREADS` 可覆盖 | `test_a3_threads_pinned_before_torch` |
| **A-4** | P2b / T2.0(c) | **GPU 并发上限 = 1** | 两个并发 GPU 作业实测 **172×** 劣化（B81）。并发 2 时两个各需 107.9 s 的作业在 1079 s 内一个都没完成。机制是 CUDA 上下文时间片切换，而非 CPU 超订（全程负载 2.06/32 核） | `configs/scheduling.yaml`：`policy.gpu_queue.max_concurrent` | `test_a4_gpu_serial` |
| **A-5** | P2c / P2f | **CPU 并发上限 = 12** | T2c.1 实测：KNN×eICU 每格 101.8 s(3 路) → 39.1 s(12 路)。**上限由 HyperImpute 决定而非 KNN**——KNN 到 16 路仍在改善，HyperImpute 在 16 路 **0/16 完成** | `configs/scheduling.yaml`：`policy.cpu_queue.max_concurrent` | `test_a5_cpu_workers` |
| **A-6** | P2b 决策 3 | **出厂掩码 = 打乱版**（`clinical_v1_shuffled`） | 配行空间指纹断言，防止打乱表与未打乱掩码错配——这种错配**不会报任何错** | `data/masks/clinical_v1_shuffled/` | `test_a6_shuffled_masks_are_factory` |
| **A-7** | P2 / P2b | **移除 target 列与零方差列** | target 作为特征造成泄漏；零方差列使指标退化 | `configs/datasets.yaml` | `test_a7_no_target_as_feature` |
| **A-8** | P2 §4.4 | **MIMIC 换表**（2052×8 → 2849×16），8/06 回退门**判定不回退** | 旧表含确定性派生列与 target 泄漏。六项判据全部通过：无零方差列、无确定性派生列、target 不在特征集、连续列在生理范围、类别列每类 ≥20 样本、九方法均能成功运行且量级合理（62 次运行零崩溃） | `configs/datasets.yaml`(record in the private development history)| `test_a8_mimic_table_shape` |
| **A-9** | P2e §5 | **CDC2022 的 n = 1000**（d=41 不变） | n=3000 时该格的成本外推区间跨 **7.7 倍**，且此前三个成本模型全部低估实测下界。**R2-4 要的是表宽（d）不是样本量（n）**，所以降 n 不伤科学论点 | `experiments/budget_panel.py`：`ROWS` | `test_a9_cdc2022_rows` |
| **A-10** | P2b 决策 1 | **聚合规则 = 中位数为主 + 发散率列** | 对所有方法一视同仁地施加，不为某个方法特设；发散率列对每个方法都记录（含 MeanMode 这种按构造发散率≈1 的参照方法，已标注） | `stats/aggregate_grid.py` | — |
| **A-11** | P2c §1.2 | **B77**：ESM 的 "100 epochs" 更正为 **200** | `registry.py:300-309` 的 dataclass 默认值是 200，且 R0 的 `budget_params` 对该方法为空。属 B1 同类（声明值 ≠ 实现值） | ESM §S3.4 `:234` | — |
| **A-12** | P2f §2 裁定 2 | **不做 dropout 归零对照**，替换为数值扰动 ρ | B84 之后问题的形状变了：统一现象是"SNI 拟合出的模型对本不该影响它的数值扰动敏感"，而非"dropout 干的"。ESM 需要写的是**现象的量级与后果**，不是 PyTorch 内部哪一行 | — | — |


---

## A-13 正字法：美式为准（P7-A §1，第一作者 2026-09-04）

| 项 | 内容 |
|---|---|
| **裁定** | 全文（main + ESM + 图注 + 生成器字符串）统一 **美式拼写**；美式为 canonical，英式变体入扫描族 |
| **取代** | 第七次内审 §13.3「统一为 -ise/-yse 一族」，及其在 **P5R-R 裁定 3** 的重申。两者自本日起失效 |
| **登记处** | `docs/terminology_registry.json` → `orthography`（双向 map，139 组）；豁免见 `scan_exemptions.orthography` |
| **断言** | `reporting/facts_gate.py --selftest`（六条正字法夹具）＋ `compile_gate` 编译前/后双层拦截。**裁定不是写下来就生效的，是被断言之后才生效的** |
| **落地** | 手稿 179 处、`experiments/` 50 处、`configs/missingness.yaml` 2 处；三层归零 |

### §15.6 逐字引文：一字之差换了位置，没有消失

第七次内审 §15.6 的原文本身**英美混拼**：

> ...consistent with an influence of the **regularizer** and with increased **optimisation** variability...

`regularizer` 是美式，`optimisation` 是英式，同一句里。**任何单一正字法都无法逐字复现这句话**——这不是本轮造成的，是原文如此。

- **IR7 §13.3（英式）**：`regularizer` → `regulariser`，与原文差一字，当时**记在回执候裁**（RESPONSE_TO_IR7 §13.3）。
- **P7-A §1（美式）**：`regulariser` → `regularizer`，**恰好回到原文拼写**——指令预见到了这一点；同时 `optimisation` → `optimization`，**新产生**与原文的一字之差。

**净结果**：差异数不变（仍为 1），位置从 `regularizer` 移到 `optimisation`。IR7 的候裁项因此**关闭**——不是因为问题解决了，而是因为它换了一个词，且换到了更无害的一侧（`optimisation` 是通用词，`regularizer` 是该句的技术主语）。

正文承载的是**论断**，不是拼写；两版的科学内容逐字相同。此行即 §1.1 要求的取代记录。

---
