# 古代玻璃制品成分分析与鉴别

2022年高教社杯全国大学生数学建模竞赛 C 题。

## 项目结构

```
MathModel/
│
├── data/                           # 原始数据 (CSV + 描述)
│   ├── 表单1_文物信息.csv           # 58件文物: 编号、纹饰、类型、颜色、风化状态
│   ├── 表单2_化学成分.csv           # 69条采样: 14种氧化物百分比
│   ├── 表单3_未知类别.csv           # 8个待分类样品 (A1-A8)
│   ├── 问题描述.md                  # 完整题目原文 + 附件说明
│   └── 数据说明.md                  # 字段含义、采样点命名规则、分析要点
│
├── src/                            # 建模代码 (7个独立脚本)
│   ├── problem1a.py                # 风化 × 类型/纹饰/颜色的关联分析
│   ├── problem1b.py                # 风化前后化学成分统计规律
│   ├── problem1c.py                # 差值法预测风化前成分
│   ├── problem2a.py                # 高钾/铅钡分类规则 (决策树 + LDA)
│   ├── problem2b.py                # 亚类划分 (PCA + K-means)
│   ├── problem3.py                 # A1-A8 未知样品分类 + 敏感性
│   ├── problem4.py                 # 成分关联关系 + 类型间差异
│   ├── validate.py                 # 集成验证 (43条断言, 独立于建模代码)
│   └── test_critical.py            # 关键函数单元测试 (38条)
│
├── analysis/                       # 产出 (文档 + 图表 + 结果)
│   ├── problem1.md                 # 问题1 完整分析报告
│   ├── problem2.md                 # 问题2 完整分析报告
│   ├── problem3.md                 # 问题3 完整分析报告
│   ├── problem4.md                 # 问题4 完整分析报告
│   ├── modeling_principles.md      # 建模原理教程 (费曼对话体)
│   ├── fig*.png                    # 11张图表
│   ├── problem*_results.json       # 统计检验结果
│   └── problem*_predictions.csv    # 预测结果表
│
├── pyproject.toml                  # 依赖声明 (uv 管理)
├── uv.lock                         # 锁定依赖版本
├── CLAUDE.md                       # Claude Code 项目指南
└── README.md                       # 本文件
```

## 快速开始

```powershell
# 安装依赖
uv sync

# 跑全部建模
uv run python src/problem1a.py
uv run python src/problem1b.py
uv run python src/problem1c.py
uv run python src/problem2a.py
uv run python src/problem2b.py
uv run python src/problem3.py
uv run python src/problem4.py

# 跑验证
uv run python src/validate.py        # 43条集成验证
uv run python src/test_critical.py   # 38条关键函数单元测试
```

## 四道题核心结论

| 问题 | 核心发现 |
|------|---------|
| **1. 风化分析** | 类型是风化唯一显著因素 (p=0.011)；高钾淋溶模式 vs 铅钡交换模式；差值法预测 MAE<1% |
| **2. 分类与亚类** | BaO>3.14% 完美区分两类 (准确率92%)；高钾4亚类、铅钡3亚类，本质反映风化梯度 |
| **3. 未知样品鉴别** | A1/A2/A5/A6/A7→高钾；A3/A4/A8→铅钡；A2为异常纯铅玻璃 |
| **4. 关联分析** | 18对氧化物关联显著不同；高钾关联被风化重塑，铅钡保留原始矿物信号 |

## 方法索引

| 方法 | 用途 | 出现位置 |
|------|------|---------|
| 卡方检验 / Fisher 精确检验 | 分类变量关联 | 1a |
| Mann-Whitney U + Bonferroni | 两组比较 (多重检验校正) | 1b |
| 差值法 | 配对数据预测 | 1c |
| 决策树 / LDA | 分类规则提取 | 2a |
| ROC-AUC, 交叉验证 | 分类器评估 | 2a |
| PCA + K-means + 轮廓系数 | 亚类发现 | 2b |
| Bootstrap / 扰动法 | 聚类敏感性 | 2b |
| 多分类器投票 | 未知样品共识分类 | 3 |
| CLR 变换 + Fisher z | 成分数据关联分析 | 4 |

## 测试策略

三层防护：

1. **单元测试** (`test_critical.py`) — 测函数逻辑: ID解析、CLR变换、差值预测、BaO分类
2. **集成验证** (`validate.py`) — 独立重算关键断言，不依赖建模代码
3. **代码 review** — 每步完成后人眼检查

```powershell
uv run python src/test_critical.py   # 38 assertions
uv run python src/validate.py        # 43 assertions
```

## 阅读指南

- **新手入门**: 先读 `analysis/modeling_principles.md`，费曼对话体讲每个方法
- **理解数据**: `data/数据说明.md` + `data/问题描述.md`
- **看结果**: `analysis/problem1.md` → `problem2.md` → `problem3.md` → `problem4.md`
- **看代码**: `src/` 下按问题编号读，每个脚本 ~200 行、自包含
