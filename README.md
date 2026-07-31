# XML Dataset Preprocessor

一套完整的图像数据集预处理流水线：将 PASCAL VOC 格式的 XML 标注批量转换为 YOLO 格式 TXT，执行图文匹配清洗、标签统计与极端样本过滤，按 7:2:1 比例智能划分训练/验证/测试集，并做最终校验。

## 安装

```bash
pip install Pillow lxml pyyaml
git clone https://github.com/kankan-veda/xml-dataset-preprocessor.git  ~/.codex/skills/xml-dataset-preprocessor
```

## 工作流程

| 步骤 | 脚本 | 功能 |
|------|------|------|
| 第一步 | xml_to_yolo.py | XML 转 YOLO TXT（XML 尺寸优先，缺失时 PIL 回退） |
| 第二步 | match_and_clean.py | 删除无对应图片的孤 TXT 文件 |
| 第三步 | deduplicate_dataset.py | 检测重复图片，只保留一组 png+txt，其余移走 |
| 第三步A | stats_and_filter.py | 统计标签分布，生成 filter_plan.yaml |
| 第三步B | apply_filter.py | 按 YAML 配置执行过滤 |
| 第四步 | smart_split.py | 分层采样 + 迭代优化，7:2:1 划分 |
| 第五步 | final_validate.py | 全面校验（数量、格式、图片完整性） |

## 使用方式

### 作为 Codex Skill 使用

安装后告诉 Codex "帮我把这份数据集的 XML 标注转成 YOLO 格式，按 7:2:1 划分"，它会自动调用此 Skill 完成全流程。

### 作为独立脚本使用

```bash
python scripts/xml_to_yolo.py --xml-dir xmls/ --img-dir images/ --output-dir labels/ --classes-output classes.txt
python scripts/match_and_clean.py --labels-dir labels/ --img-dir images/
python scripts/deduplicate_dataset.py --img-dir labels/ --labels-dir images/ --output-dir duplicates/
python scripts/stats_and_filter.py --labels-dir labels/ --img-dir images/ --output filter_plan.yaml
# 编辑 filter_plan.yaml 后：
python scripts/apply_filter.py --plan filter_plan.yaml --labels-dir labels/ --img-dir images/ --output-dir cleaned/
python scripts/smart_split.py --labels-dir cleaned/ --img-dir images/ --output-dir dataset/ --classes-file classes.txt
python scripts/final_validate.py --dataset-dir dataset/ --log-dir logs/
```

## 依赖

```bash
pip install Pillow lxml pyyaml
```

## 许可

MIT
