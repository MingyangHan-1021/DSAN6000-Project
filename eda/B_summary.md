# B 部分工作小结 — 文本的语言特征与可区分性（Wenhao Zhou）

> 项目：Scalable Fraud Detection in Banking Chatbot Conversations
> 范围：EDA 第一部分（文本侧），B1–B3 三张图
> 分支：`wenhao` ｜ 最近提交：添加 EDA part B

---

## 一、分工定位

EDA 按数据来源切成两半，四人的交付物互不重叠：

| 成员 | 部分 | 图 | 使用数据 |
| --- | --- | --- | --- |
| Yuhua Pang | 文本的业务结构与语义（第一部分） | A1–A3 | `banking77_clean`、`dim_intent` |
| **Wenhao Zhou（本小结）** | **文本的语言特征与可区分性（第一部分）** | **B1–B3** | `banking77_clean`、`dim_intent` |
| Mingyang Han | 模拟数据的总体概况与时间行为（第二部分） | C1–C4 | `messages`、`users` |
| Zhaoyang Dong | 模拟数据的序列与地理行为（第二部分） | D1–D3 | `messages`、`users`、`banking77_clean`、`dim_intent` |

- 我与 Yuhua 共用同样的两份文本数据，但问题不同：A 回答"高风险的意图和普通咨询在**语义**上能不能分开"（统计、旭日图、UMAP 语义地图），B 回答"**哪些意图会被混淆、各类欺诈的客户在用什么词**"（混淆矩阵、区分性词汇、紧急度）。
- 我不触碰模拟消息流，因此与 C/D 完全没有交叉，也不会与他们的产出口径冲突。
- 唯一共用的上游是 `dim_intent` 的领域标注（77 个意图的 `journey_stage` / `fraud_type` / `risk_level`，22 个意图属欺诈相关、6 类欺诈），这是全组统一输入，我只读不改。

---

## 二、交付物清单

| 文件 | 内容 |
| --- | --- |
| `eda/eda_B.ipynb` | 15 个单元格（7 代码），Restart & Run All 可从零重现三张图 |
| `figures/B_confusion_heatmap.png` | B1 意图混淆热力图 + 欺诈漏检分解（四面板） |
| `figures/B_keywords_by_fraud_type.png` | B2 各欺诈类型区分性关键词 + 关键词规则检验 |
| `figures/B_urgency_by_risk.png` | B3 紧急程度分析（四面板） |
| `eda/interpretations.md` | 三张图的英文解读（含银行业视角），可直接用于报告/网站 |
| `eda/_common.py`、`b1_confusion.py`、`b2_keywords.py`、`b3_urgency.py` | 分析代码（notebook 单元格即由这些脚本生成） |
| `eda/output/`（13 个文件） | 全部明细：混淆对、漏检分解、关键词、规则评估、紧急度比例、敏感性等 |

**数据来源**：本地用 `src/data/prepare_banking77.py` 从 HuggingFace `PolyAI/banking77` 生成
`banking77_clean`（13,083 条，train 10,003 / test 3,080）与 `dim_intent`（77 行，22 个
`fraud_relevant`），与仓库脚本的输出规格逐项一致（行数、拆分、意图名顺序、风险分布均校验通过）。

---

## 三、核心结论

### B1　哪些意图会被混淆，欺诈信号从哪里漏掉

TF-IDF（1–2 gram）+ 多分类逻辑回归，训练集训练、测试集评估：**accuracy 0.891，macro-F1 0.891**。

- 把 77 个意图按 `journey_stage` 排序后，误判主要落在**同一旅程的阶段块内**，跨阶段错误明显更少。
- Top-20 混淆对中 **15 对无害、5 对危险**（危险 = 欺诈意图被分流到非欺诈或不同欺诈队列）。
- 测试集中意图属欺诈相关的消息共 **880 条**，其中 **56 条（6.4%，95% CI 4.9%–8.2%）被判进普通客服队列**，另 5 条（0.6%）进了错误欺诈团队——即约每 15 条就有 1 条根本进不了欺诈流程。
- 分类看：`virtual_card_abuse` 漏 12.5% 最严重，`card_compromise` 8.1%、`app_fraud_money_mule` 7.5%、`refund_chargeback_abuse` 5.0%、`account_takeover` 4.5%，`identity_kyc_bypass` 仅 0.8%。
- 最典型的文本无法区分案例：`pin_blocked`（账户盗用，high）与 `get_physical_card`（普通卡片物流），"How do I get my PIN unlocked?" 与 "I need my PIN" 几乎同构。

### B2　各类欺诈的客户在用什么词，这些词够不够用

对数几率比（Monroe 等 "Fightin' Words"，带信息性 Dirichlet 先验），每类欺诈对比**训练集其余消息**，unigram + bigram 并按 token 贪心去重。

- account_takeover：`pin`、`change`、`details`、`phone`、`use app`
- app_fraud_money_mule：`transfer`、`transaction`、`cancel`、`wrong account`、`beneficiary`
- card_compromise：`recognise`、`withdrawal`、`stolen`、`didn't make`、`wallet`
- refund_chargeback_abuse：`refund`、`item`、`bought`、`seller`、`reverted`
- identity_kyc_bypass：`verify`、`verification`、`use account`、`identity`
- virtual_card_abuse：`virtual`、`cards`、`restrictions`、`limits`

这些词几乎全是客户自己账户旅程中的产品名词和动作动词，而非威胁性语言 → 应按欺诈类型建分层词表，而不是一个通用"欺诈词典"。

把每类前 10 个词当作独立检测器（词表只用训练集选，测试集评估）：**总体召回 77%、精确率 29%**。
标出 2,355 条测试消息，只有 678 条为真——相当于为抓 678 个案子要审约 1,700 个正常客户。

| 类型 | 召回 | 精确率 |
| --- | --- | --- |
| virtual_card_abuse | 92% | 39% |
| identity_kyc_bypass | 84% | 63% |
| account_takeover | 80% | 44% |
| card_compromise | 76% | **14%** |
| app_fraud_money_mule | 72% | 20% |
| refund_chargeback_abuse | 57% | 54% |

### B3　"欺诈语气急"成立吗

按文档示例自建三组词表（时间压力 / 情绪求助 / 事件威胁），13,083 条消息逐条打标。

| 词类 | low | medium | high |
| --- | --- | --- | --- |
| 时间压力（immediately / urgent） | 4.1% | 6.1% | 4.5% |
| 情绪求助（help / worried） | 6.0% | 11.0% | 10.7% |
| 事件威胁（someone / stolen） | 0.8% | 0.7% | **18.3%** |
| 任一 | 9.9% | 15.1% | 27.0% |

- 时间压力词基本无区分力；风险梯度全部来自事件词，且几乎集中在 `card_compromise`（39.2%）。
- 两类欺诈比普通客户**更平静**：`virtual_card_abuse` 5.8%、`identity_kyc_bypass` 4.3%，低于普通消息的 9.9%。
- 控制句长后 high vs low 的 OR 由 3.35 升到 **4.80**（高风险消息并不更长），而 medium 由 1.61 降到 1.24（中风险信号主要是"话多"）。
- 意图层面差异极大：高风险组内 `compromised_card` 85.7%、`lost_or_stolen_card` 73.0%，而 `why_verify_identity` 仅 1.9%、`edit_personal_details` 3.1%；低风险中位数 7.8%，其中 `terminate_account` 达 35.8%，比 12 个高风险意图里的 9 个都高。
- 词表敏感性：brief 3.3/6.9/20.9、current 9.9/15.1/27.0、broad 16.7/22.6/38.7，"high 高于 low"在所有版本成立；仅剩强紧急词时塌缩为 0.7/2.8/1.7。

---

## 四、方法与可复现性

- **严格 train/test 分离**：B1 训练与评估分开；B2 的词表与词频**只在训练集**上估计；B3 词表先验固定（直接取文档给的例子），不依赖数据调参。
- **统计处理**：比例一律给 Wilson 95% 区间；Top-20 混淆对做 500 次 bootstrap 稳定性检验（仅前 6 对 ≥80% 稳定，尾部 22%–28%，因此不作强解读）；风险等级差异做卡方检验；紧急度结论做句长控制（logistic 回归）与词表敏感性两类稳健性检查。
- **运行**：`eda/` 目录下打开 notebook，Restart & Run All；依赖见仓库 `requirements.txt`，另需 `scikit-learn`、`matplotlib`、`seaborn`、`scipy`。

---

## 五、局限

1. 文本侧只有 13,083 条、77 个意图（每意图约 170 条），Top-20 混淆对中尾部计数仅 3–9 条，名次接近并列。
2. `risk_level` 与 `fraud_type` 是组内 `config/intent_taxonomy.yaml` 的领域标注，不是银行给出的真值；B3 已表明它作为欺诈代理是粗粒度的。
3. 本部分完全不涉及模拟消息流，因此不能声称任何行为层面的可区分性。

---

## 六、对下游（ML 部分）的接口

- B1 的 **6.4% 漏检率** 与 B2 的 **29% 规则精确率** 是"文本单独不够用"的量化证据。
- B2 的词表本质是**意图层面**的词，靠它做欺诈检测等于重学 77 类意图分类器。
- B3 表明**语气特征只对卡片盗刷有效**，身份验证 / KYC / 虚拟卡类必须靠行为特征。
- 因此 ML 部分"纯文本 vs 文本 + 行为"的对比，预期收益主要来自后三类欺诈，可直接引用以上三组数字作为立论。
