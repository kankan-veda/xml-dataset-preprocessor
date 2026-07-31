---
name: xml-dataset-preprocessor
description: "批量预处理图像数据集中的 XML 标注文件（PASCAL VOC 格式）：转换为 YOLO 格式 TXT，图文匹配清理冗余，标签统计与极端样本过滤（支持交互确认和中间 YAML 文件两种模式），按 7:2:1 比例智能划分训练/验证/测试集并全面校验。当用户提到 XML 标注转 YOLO 格式、数据集划分、训练/验证/测试集拆分、PASCAL VOC 标注转换、批量图像预处理，或描述此类多步骤数据准备流程时使用。"
---

# XML Dataset Preprocessor

五步流水线：XML 转 YOLO TXT → 图文匹配清理 → 标签统计并生成 YAML 配置 → 执行过滤 → 7:2:1 智能划分 → 最终校验。每脚本独立可执行，通过参数传递路径。

## 目录约定

```
project/
├── images/          # 原始图片（.jpg/.png/.jpeg）
├── xmls/            # 原始 XML 标注
├── labels/          # 转换后的 YOLO TXT（第一步产出）
├── cleaned/         # 清洗后的 labels（第三步产出）
├── dataset/
│   ├── train/images/ & train/labels/
│   ├── val/images/   & val/labels/
│   └── test/images/  & test/labels/
└── logs/            # 校验日志
```

## 工作流程

顺序执行，每步一个脚本。

---

### 第一步：XML → YOLO TXT

`python scripts/xml_to_yolo.py --xml-dir <xmls> --img-dir <images> --output-dir <labels> --classes-output <classes.txt>`

**核心逻辑：**

第一次扫描全部 XML，收集所有 `<name>` 标签，生成类别名 → ID 的映射文件 `classes.txt`。

第二次逐文件解析：

```
对每个 XML 文件：
  1. 读 <size><width>/<height>
  2. 如果有且非零 → 直接用，不进 PIL（纯文本解析，很快）
  3. 如果缺失或为 0 → 用 PIL 打开对应图片文件获取实际宽高
  4. 图片也不存在 → 记入 logs/problematic.txt，跳过该样本
  5. 解析每个 <object><bndbox> 中的 xmin/ymin/xmax/ymax
  6. 归一化：
     x_center = (xmin + xmax) / 2 / width
     y_center = (ymin + ymax) / 2 / height
     w = (xmax - xmin) / width
     h = (ymax - ymin) / height
  7. 写入 TXT：每行 class_id x_center y_center width height（6 位小数）
```

**校验：** 输出后校验每行是否 5 列、坐标是否在 [0,1] 内、无 NaN/Inf。校验结果写入 `logs/conversion_check.txt`。

**设计理由：** 尺寸完整的 XML 占绝大多数，直接解析文本快而不碰 I/O。只有极少数缺尺寸的样本需要 PIL 回退，补了也不成瓶颈。跳过图片都不存在的样本，因为用默认尺寸会导致归一化坐标超出 [0,1]，YOLO 训练直接崩。

---

### 第二步：图文匹配与清理

`python scripts/match_and_clean.py --labels-dir <labels> --img-dir <images>`

```
遍历 labels/ 下所有 .txt：
  找 images/ 下同名图片（支持 .jpg/.jpeg/.png，不区分大小写）
  找不到 → 删除该 .txt，写入 logs/deletion_log.txt
反查：有图片无 txt → 打印提示，不操作
```

**设计理由：** YOLO 训练时标签文件存在但图片不存在，会抛 MissingImageException 卡住训练。删除孤儿标签而不删除孤儿图片，因为图片可以作为背景负样本。

---

### 第三步：重复数据去重

`python scripts/deduplicate_dataset.py --img-dir <images> --labels-dir <labels> --output-dir <duplicates>`

```
遍历 images/ 下所有图片：
  按文件内容计算 SHA-256
  内容完全相同的图片归为一组（复制改名也算重复）
  每组保留文件名排序第一的 png + 同名 txt
  其余 png + 同名 txt 移动到 duplicates/（不删除，方便复查）
  处理结果写入 logs/dedup_log.txt
同组图片的 txt 内容不一致 → 打印 WARNING，保留第一份，其余照常移走
```

**设计理由：** 同一张图片如果以不同文件名重复出现在数据集中，划分时可能同时进入 train 和 test，造成数据泄露、指标虚高。用内容哈希而不是文件名判断，能发现复制改名的重复图片。移动而非删除，避免误判后无法恢复；确认无误后可手动清理 `duplicates/`。先加 `--dry-run` 预览，确认后再正式运行。


### 第四步：统计、配置、过滤（拆为 A/B 两个子步骤）

此步骤支持两种运行路径，选择权在用户：

| 场景 | 谁改 YAML | 流程 |
|------|-----------|------|
| **Codex 对话模式**（默认） | Codex 编辑 YAML | Codex 读取终端输出展示给用户 → 用户口头确认 → Codex 写入 filter 字段 → 自动调 apply_filter.py |
| **中间文件模式** | 用户手动编辑 | 用户打开 YAML 改 filter 字段 → 保存 → 自己跑 apply_filter.py |

#### 第四步A：统计并生成配置文件

`python scripts/stats_and_filter.py --labels-dir <labels> --img-dir <images> --output <filter_plan.yaml>`

**终端输出：** 总样本数、总目标数、每类别频次、目标数分布（直方图）、零目标和单目标样本列表。

**生成的 `filter_plan.yaml` 结构：**

```yaml
summary:
  total_samples: 1245
  total_objects: 3872
  zero_object_samples: 23
  single_object_samples: 156
  zero_object_files: [image_007.txt, image_089.txt, ...]
  single_object_files: [image_012.txt, ...]

class_distribution:
  person: 2103
  car: 1042
  bicycle: 432
  dog: 295

object_count_distribution:
  "0": 23
  "1": 156
  "2-3": 478
  "4-7": 412
  "8+": 176

# → 由用户或 Codex 填写以下字段 ←
filter:
  remove_zero_object: false
  remove_single_object: false
  remove_below_threshold: 0
  also_remove_images: true
```

**设计理由：** filter 段全设为 false，避免脚本替用户做决定。任何过滤行为都必须是主动选择的结果。YAML 文件持久化了过滤决策，可以回头复查。

#### 第四步B：执行过滤

`python scripts/apply_filter.py --plan <filter_plan.yaml> --labels-dir <labels> --img-dir <images> --output-dir <cleaned>`

**逻辑：**
1. 加载 YAML，读取 `filter:` 段
2. 如果所有规则都是 false → 打印提示，直接将全部文件复制到 `cleaned/`
3. 否则按规则删除零目标 / 单目标 / 低于阈值的 TXT（及对应图片）
4. 剩余文件复制到 `cleaned/`
5. 写入 `logs/filter_execution.log`

---

### 第五步：7:2:1 智能划分

`python scripts/smart_split.py --labels-dir <cleaned> --img-dir <img> --output-dir <dataset> --classes-file <classes.txt> [--seed 42]`

**算法设计：**
1. 统计每类别的全局目标数
2. 按每个样本的目标个数从小到大排序（目标少的优先分配，减少损失）
3. 对每类别单独分配：10% 到测试集、20% 到验证集、70% 到训练集
4. 合并冲突（多类别样本被分到不同集合时，按"稀有种类的样本优先去该种类最缺的子集"解决）
5. 迭代优化（最多 100 轮）：交换样本，最小化实际比例与 7:2:1 的偏差
6. 保证每类别在三个子集中至少出现 1 次（总量不足 3 的类别全放训练集）
7. 复制到 `dataset/{train,val,test}/{images,labels}/`
8. 输出划分报告：各子集数量和比例、各类别分布对比

**参数 `--seed`：** 确保可复现。不传则用时间戳种子。

**使用复制而非移动：** 原始数据不动，方便重来或对比。

## 输出报告说明

划分完成后会打印两级报表：

**样本级分布**：train/val/test 各有多少张图片，以及实际比例与 7:2:1 的偏差。

**类别级分布**：每个类别在三个子集中的目标数：
- `(占类)` 列：该类的 7:2:1 分布。理想值是 `70% / 20% / 10%`，越接近越好
- `(占集)` 列：该类占所在子集的比例。训练集、验证集、测试集三个子集中各类比例应相近（否则说明划分有偏）


---

### 第六步：最终校验

`python scripts/final_validate.py --dataset-dir <dataset> --log-dir <logs>`

**检查项目：**
- 每个子集内图片数 == TXT 数
- 三个子集 TXT 总数 == cleaned/ 中 TXT 总数
- 无空 TXT（0 行或只有空白）
- 每行 5 列、class_id 在 classes.txt 范围内、坐标在 [0,1]
- PIL 确认每张图片可打开（不损坏）
- 生成 `logs/validation_report.txt`

**严重度分级：**

| 级别 | 条件 | 输出 |
|------|------|------|
| OK | 全部通过 | 报告最后一行写 PASS |
| WARNING | 少量差异（<5 个） | 列出差异，建议复查 |
| ERROR | 明显差异（>=5 个） | 高亮标记，询问是否继续 |
| CRITICAL | 总数不匹配、关键文件缺失 | 建议停止使用此数据集 |

---

## 依赖

```bash
pip install Pillow lxml pyyaml
```

## 重要提醒

- 第一步中 PIL 只在 XML 缺尺寸时才被调用，不会影响大多数样本的性能
- 第三步的 filter_plan.yaml 是核心枢纽：Codex 对话模式中由 Codex 编辑，中间文件模式中由用户手动编辑——脚本本身不需要知道"我是什么模式"
- 所有路径参数化，不硬编码
- 每个脚本均支持 `--help` 查看参数
