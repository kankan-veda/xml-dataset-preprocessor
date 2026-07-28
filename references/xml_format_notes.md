# XML 标注格式差异说明

本文件列出各标注工具生成的 PASCAL VOC XML 的常见差异，在第一步转换遇到格式不兼容时参考。

---

## LabelImg（最常用，推荐）

- 根标签：`<annotation>`
- `<size>` 一定存在，`<width>` 和 `<height>` 为整数
- `<bndbox>` 下为 `xmin` / `ymin` / `xmax` / `ymax`，整数或小数
- 图片路径在 `<path>` 标签中，不一定与 `--img-dir` 一致
- 无特殊处理，直接按标准解析即可

## LabelMe

- 根标签：`<annotation>`
- 可能有 `<polygon>` 而非 `<bndbox>`
  - 处理方式：取多边形所有顶点坐标的 min/max 转为外接矩形
  - `xmin = min(pt[0] for pt in points)`
  - `ymin = min(pt[1] for pt in points)`
  - `xmax = max(pt[0] for pt in points)`
  - `ymax = max(pt[1] for pt in points)`
- `<size>` 结构同上，无特殊差异

## CVAT

- **根标签不同：** `<annotations>` 而非 `<annotation>`
  - 检测方法：根标签名
- **图片标签格式：** `<image id="0" name="xxx.jpg" width="1920" height="1080">`，而非 `<annotation><size>` 结构
  - 处理方式：从 `<image>` 标签属性读取 `width` 和 `height`
- `<bndbox>` 结构同上，但属性名可能不同（如 `xtl` / `ytl` / `xbr` / `ybr` 而非 `xmin` / `ymin` / `xmax` / `ymax`）
  - 处理方式：将 `xtl` `ytl` `xbr` `ybr` 映射为 `xmin` `ymin` `xmax` `ymax`

## 自家工具 / 未知来源

常见异常及处理建议：

| 异常 | 表现 | 建议 |
|------|------|------|
| 尺寸写 0 | `<width>0</width>` | 触发 PIL 回退读图片尺寸 |
| 坐标未归一化 | 坐标值远大于宽高 | 检查原始坐标单位，先除以图片尺寸再归一化 |
| 坐标小数位数过多 | 如 14 位小数 | 在归一化后再做舍入，保持 6 位小数输出 |
| 缺少 `<size>` 标签 | `<annotation>` 下无 `<size>` | 触发 PIL 回退 |
| 缺少 `<object>` | XML 虽存在但无标注对象 | 输出空 TXT，第三步处理 |
| 非标准标签名 | `<box>` 代替 `<bndbox>` | 在 xml_to_yolo.py 中增加 fallback 解析路径 |

## 格式检测速查表

执行 python 快速判断格式：

```python
# 读取 XML 根标签，判断格式
import xml.etree.ElementTree as ET
tree = ET.parse("sample.xml")
root = tree.getroot()
tag = root.tag
if tag == "annotations":
    print("CVAT 格式")
elif tag == "annotation":
    # 检查是否有 <image> 子标签
    if root.find("image") is not None:
        print("CVAT 单图格式")
    else:
        print("标准 VOC / LabelImg 格式")
else:
    print(f"未知格式: {tag}")