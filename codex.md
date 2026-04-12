# codex 进展文档

## 项目目标
- 使用 YOLO-Master 在 GYU-DET 数据集上训练目标检测模型。

## 当前阶段（2026-03-30）
- 已完成仓库初步检查并建立进展文档。
- 当前训练入口文件：`train.py`
- 当前数据集配置文件：`GZ-DET.yaml`

## 已确认配置
1. 模型与训练脚本（`train.py`）
- 使用模型权重：`YOLO-Master-EsMoE-M.pt`
- 训练数据配置：`GZ-DET.yaml`
- 训练轮数：`epochs=300`
- 启用预训练：`pretrained=True`

2. 数据集配置（`GZ-DET.yaml`）
- `train`: `/root/autodl-fs/GYU-DET/train`
- `val`: `/root/autodl-fs/GYU-DET/valid`
- 类别总数：6
- 类别名称：`Crack`, `Breakage`, `Comb`, `Hole`, `Reinforcement`, `Seepage`

## 当前风险与注意事项
- `GZ-DET.yaml` 中使用的是绝对路径（`/root/autodl-fs/...`），这通常对应 Linux 训练环境；若在本机 Windows 直接运行，需要改为可访问路径。
- `train.py` 未显式设置 `imgsz`、`batch`、`device`、`workers`、`project/name`、`seed` 等参数，后续若要复现实验建议固定。

## 桥梁病害优化思路（2026-03-30）
### 1. 网络结构优化方向
- 当前 `YOLO-Master-M` 检测头为 `P3/P4/P5`，对桥梁病害中的小目标并不占优；优先考虑增加 `P2` 检测头，提升对细小裂缝、孔洞、剥落边缘等目标的感受能力。
- 对于细长裂缝这类极端长宽比目标，普通水平框会引入大量背景，建议优先评估两条路线：
- 路线 A：继续做水平框检测，但在 `P2/P3` 侧增加更强的浅层特征融合与高分辨率输入。
- 路线 B：如果允许重新标注，迁移到 `OBB` 或分割任务；仓库中已有 `yolo-master-obb-m.yaml`，对细长裂缝通常更匹配。
- 在裂缝检测分支中可考虑加入方向敏感模块，例如非对称卷积（`1xk`、`kx1`）或条带池化，以增强对横向、纵向、斜向细线结构的响应。
- 颈部网络建议强调浅层细节传递，而不是继续单纯加深高层语义；桥梁病害尤其依赖边缘、纹理、细线信息。

### 2. 数据增强与处理方向
- 对桥下光照不均，建议采用“轻量预处理 + 在线增强”组合，而不是只靠训练时颜色扰动。
- 轻量预处理建议优先评估：
- CLAHE（局部对比度增强），改善阴影区域细节。
- Gamma/亮度归一化，降低同一病害在不同光照下的分布差异。
- Retinex 或同类低照度增强，仅建议离线生成一版对照数据做消融，不宜一开始全量替换原图。
- 对运动模糊和成像退化，建议加入有控制的退化增强：
- `MotionBlur`、`Blur`、`MedianBlur`
- `GaussNoise`
- 轻度 `Sharpen`
- 裂缝类细目标不适合过强的几何形变，`mosaic`、大尺度 `scale`、强 `perspective` 需要保守，否则容易破坏真实细线形态。
- 小目标与少样本类别建议增加目标感知裁剪、局部放大、按类别均衡采样，而不是只提高全局增强强度。

### 3. 结合当前仓库的可落地点
- 结构侧：
- 可基于 `ultralytics/cfg/models/master/v0/det/yolo-master-m.yaml` 派生一个带 `P2` 的桥梁病害专用模型配置。
- 若要验证极端长宽比目标，可直接评估 `ultralytics/cfg/models/master/v0_1/obb/yolo-master-obb-m.yaml` 路线。
- 增强侧：
- 仓库增强流水线位于 `ultralytics/data/augment.py`，已包含 `Albumentations`、`CLAHE`、`Blur` 等入口，适合插入桥梁场景定制增强。
- 推理侧：
- 仓库已有 `Sparse SAHI` 相关脚本和参数，适合高分辨率图像中的小目标检测，可作为结构优化之外的强基线。

### 4. 建议的优化优先级
1. 先做不改标注的强基线：提高 `imgsz`，增加 `P2` 检测头，配合适度光照/模糊增强。
2. 再做推理增强：验证 `Sparse SAHI` 对小裂缝、小孔洞的召回提升。
3. 如果裂缝类目标仍明显漏检，再评估 `OBB` 或分割标注路线；这是处理中高长宽比目标更根本的方法。
4. 最后再考虑更复杂的结构模块，如方向卷积、注意力或专用裂缝分支，避免一开始改动过大导致无法判断收益来源。

### 5. 当前判断
- 对 GYU-DET 这类桥梁病害数据，最值得优先投入的通常不是“盲目加大主干网络”，而是：
- 高分辨率输入 / 切片推理
- `P2` 小目标检测头
- 针对低照度与模糊的定制增强
- 对细长裂缝采用更合适的标注与建模方式（优先 OBB/分割）

## 本轮已落地（P2 + 高分辨率 + 切片推理）
1. 新建并覆盖桥梁病害专用 `P2` 检测模型配置：
- `ultralytics/cfg/models/master/v0_1/det/yolo-master-m.yaml`
- 检测头从 `P3/P4/P5` 调整为 `P2/P3/P4/P5`

2. 训练脚本已切到新模型并提升分辨率：
- `train.py` 使用 `MODEL_CFG=ultralytics/cfg/models/master/v0_1/det/yolo-master-m.yaml`
- `TRAIN_IMGSZ=960`（可切换为 `1280`）
- 预训练权重改为显式字符串加载：`pretrained="YOLO-Master-EsMoE-M.pt"`
- 实验输出目录：`runs/gyu_det/p2_imgsz{TRAIN_IMGSZ}`

3. 新增切片推理脚本：
- `predict_sahi.py` 默认启用 `sparse_sahi=True`
- 推理分辨率 `PREDICT_IMGSZ=1280`
- 切片参数：`SLICE_SIZE=960`、`OVERLAP_RATIO=0.2`

4. 当前验证状态：
- 已完成文件级检查。
- 模型实例化运行受环境依赖阻断：当前环境缺少 `cv2`（`ModuleNotFoundError: No module named 'cv2'`）。

## 训练衔接说明（2026-03-30）
- 可以使用官方 COCO 预训练权重继续训练 GYU-DET（迁移学习）。
- 当前我们已改为 `P2` 四头结构，与官方常见 `P3/P4/P5` 三头结构不完全一致，因此是“部分权重加载 + 新增层随机初始化”，不是严格意义的断点续训。
- 若要“严格续训（resume）”，需使用完全相同结构并加载你自己上一次训练生成的 `last.pt`。

## SAM分割路线可行性（2026-03-30）
- 可以使用 `SAM`，并且该仓库已内置相关能力，不需要额外改框架主代码。
- 可用入口：
- `ultralytics/data/annotator.py` 的 `auto_annotate()`：用检测模型框出目标，再用 `SAM` 生成分割轮廓并导出 YOLO 分割标签。
- `ultralytics/data/converter.py` 的 `yolo_bbox2segment()`：将现有检测数据批量转换为分割标签。
- 分割训练模型已提供：
- `ultralytics/cfg/models/master/v0_1/seg/yolo-master-seg-m.yaml`（以及 n/s/l/x 变体）。

### 推荐落地流程
1. 先用你已训练好的检测权重 + `SAM` 自动生成初始 mask 标签。
2. 对裂缝、孔洞等关键类别进行人工抽检和修正。
3. 使用 `yolo-master-seg-m.yaml` 在分割数据上训练，输出实例 mask。
4. 保留检测模型作为候选分支，与分割模型对比速度与精度后再定最终部署方案。

## 本轮已落地（SAM自动标注 + 分割训练）
1. 新增 SAM 自动标注脚本：
- `auto_annotate_sam.py`
- 读取 `GZ-DET.yaml` 的 `train/val` 路径，自动定位图片目录和标签目录。
- 调用 `auto_annotate()`（检测模型 + SAM）生成 YOLO 分割标签。
- 默认行为：先备份原检测标签，再清空旧标签并写入新的分割标签。

2. 新增分割训练脚本：
- `train_seg.py`
- 使用 `ultralytics/cfg/models/master/v0_1/seg/yolo-master-seg-m.yaml`
- 默认 `imgsz=960`，可切换 `1280`
- 若存在 `runs/gyu_det/p2_imgsz960/weights/best.pt`，会优先用于初始化权重。

3. 新增分割数据配置：
- `GZ-DET-seg.yaml`
- 与检测数据同路径同类别定义，用于分割训练入口。

4. 校验结果：
- `python -m py_compile auto_annotate_sam.py train_seg.py` 通过（语法级校验通过）。

## 显存不足处理（2026-03-30）
- 问题：`TRAIN_IMGSZ=960` 在当前显存条件下训练失败。
- 已调整为低显存稳跑配置：
- `train.py`：`TRAIN_IMGSZ=640`、`batch=-1`（AutoBatch），实验名改为 `p2_imgsz640_autobatch`。
- `train_seg.py`：`TRAIN_IMGSZ=640`、`batch=-1`（AutoBatch），实验名改为 `seg_m_imgsz640_autobatch`。
- 分割初始化权重改为候选自动匹配：
- 优先读取 `runs/gyu_det/p2_imgsz640_autobatch/weights/best.pt`
- 若不存在则回退 `runs/gyu_det/p2_imgsz960/weights/best.pt`

## moe_loss为0问题定位（2026-03-30）
- 现象：训练日志中 `moe_loss` 持续为 `0`。
- 根因：`OptimizedMOEImproved` 在 forward 中已计算并写入 `MOE_LOSS_REGISTRY`，但类本身缺少 `aux_loss` 属性，导致 `ultralytics/utils/loss.py` 的聚合逻辑无法读取该模块的 MoE 辅助损失。
- 修复：在 `ultralytics/nn/modules/moe/modules.py` 为 `OptimizedMOEImproved` 添加 `aux_loss` property（从 `MOE_LOSS_REGISTRY` 读取，默认返回同设备 0 张量）。
- 校验：`python -m py_compile ultralytics/nn/modules/moe/modules.py ultralytics/utils/loss.py` 通过。

## 路线文档产出（2026-03-31）
- 已新增 `path.md`，对桥梁病害项目整体技术路线进行统一梳理。
- 内容覆盖：检测主线、分割主线、MoE稳定性修复、数据增强策略、实验推进顺序与最终交付形态。

## 本轮已落地（桥下光照与运动模糊增强，2026-03-31）
1. 检测训练脚本增强（`train.py`）
- 增加桥梁场景定制在线增强：`MotionBlur/Blur/MedianBlur`、`CLAHE`、`RandomBrightnessContrast`、`RandomGamma`、`GaussNoise/ImageCompression`、`Sharpen`。
- 增强策略改为“几何保守 + 光照/退化增强优先”：
- 几何参数收敛：`degrees=0.0`、`translate=0.06`、`scale=0.25`、`perspective=0.0`。
- 颜色/亮度增强：`hsv_h=0.012`、`hsv_s=0.55`、`hsv_v=0.35`。
- 混合增强：`mosaic=0.80`、`mixup=0.05`、`close_mosaic=15`。
- 训练实验名更新为：`p2_imgsz640_autobatch_bridge_aug`。
- 数据集配置自动切换：若存在 `GZ-DET-enhanced.yaml`，`train.py` 将优先使用增强后数据；否则回退 `GZ-DET.yaml`。
- 兼容性处理：若环境未安装 `albumentations`，会自动回退到默认增强并给出警告，不阻断训练。

2. 新增离线预处理脚本（`scripts/preprocess_bridge_dataset.py`）
- 目标：在训练前先做图像质量统一，缓解桥下光照不均和部分运动模糊样本问题。
- 处理流程：
- `CLAHE(L通道)`：提升阴影细节与局部对比度。
- `自适应Gamma`：按图像平均亮度自动校正曝光分布。
- `轻度去模糊`：基于 `Laplacian` 清晰度阈值，仅对低清晰度图像触发反锐化（Unsharp Mask）。
- 支持从 `GZ-DET.yaml` 读取 `train/val/test`，输出增强后的数据目录与新的 YAML（默认 `GZ-DET-enhanced.yaml`）。
- 标签处理：保持标注不变并复制到输出目录，保证可直接用于 YOLO 训练。

3. 语法校验
- 已执行：`python -m py_compile train.py scripts/preprocess_bridge_dataset.py`
- 结果：通过。

4. 可用性修正
- `scripts/preprocess_bridge_dataset.py` 已改为 `cv2` 按需导入：
- `--help` 可在未安装 `opencv-python` 时正常查看。
- 真正执行预处理时若缺少 `cv2`，会提示：`pip install opencv-python`。

## 下一小阶段计划
1. 重新启动训练并观察 `moe_loss` 是否变为非零（前几个 iteration 即可确认）。
2. 跑通 `imgsz=640 + AutoBatch` 检测基线后，再执行 SAM 自动标注与分割训练。
3. 记录 `moe_loss`、`box/cls/dfl` 曲线，评估 MoE 正则是否稳定。
4. 若显存仍有余量，再逐步提升到 `imgsz=768` 做二次对比。

## 变更记录
- 2026-03-30：初始化 `codex.md`，记录项目目标、当前配置、风险和下一步计划。
- 2026-03-30：补充桥梁病害优化思路，明确结构优化、数据增强和后续实验优先级。
- 2026-03-30：落地 `P2` 专用检测模型、提升训练分辨率配置，并新增 `Sparse SAHI` 切片推理脚本。
- 2026-03-30：确认 `SAM` 分割路线可行，补充分割自动标注与训练的落地流程。
- 2026-03-30：新增 `auto_annotate_sam.py`、`train_seg.py`、`GZ-DET-seg.yaml`，形成检测到分割的可执行闭环。
- 2026-03-30：针对显存不足将检测与分割训练下调至 `imgsz=640` 并启用 `AutoBatch`。
- 2026-03-30：修复 `OptimizedMOEImproved` 缺少 `aux_loss` 属性导致 `moe_loss` 始终为 0 的问题。
- 2026-03-31：新增 `path.md`，形成项目整体技术路线文档。
- 2026-03-31：落地桥梁场景数据增强与预处理，新增 `train.py` 定制增强和 `scripts/preprocess_bridge_dataset.py` 离线处理脚本。
- 2026-03-31：修正预处理脚本运行体验，改为 `cv2` 按需导入并增加缺依赖提示。
- 2026-03-31：`train.py` 新增增强数据 YAML 自动优先切换逻辑（增强版存在则自动使用）。
- 2026-04-09：`auto_annotate_sam.py` 改为直接使用训练集现有 YOLO 检测框标签做 SAM 分割，不再依赖新训练检测模型推理框。
- 2026-04-11：修复验证集分割未生成问题：移除 `yolo_bbox2segment()` 的提前退出路径，改为逐图像强制按现有检测框执行 SAM 分割，并兼容 `val/valid` 键名。

## 本轮已落地（使用已有YOLO框标签进行SAM分割，2026-04-09）
1. `auto_annotate_sam.py` 逻辑已由“`det_model + SAM` 自动检测分割”切换为“读取现有检测标签 + SAM 分割”：
- 删除 `DET_MODEL`、`conf/iou/imgsz/max_det` 等检测推理参数。
- 新增调用 `ultralytics.data.converter.yolo_bbox2segment()`，直接基于标签框生成分割标签。

2. 标签安全替换流程：
- 先备份原检测标签到 `labels_bbox_backup`（若不存在则创建）。
- SAM 结果先写入临时目录 `labels_sam_tmp`，避免覆盖读取中的原标签。
- 生成成功后清空旧检测标签，并用分割标签替换；默认行为由 `REPLACE_LABELS_WITH_SEG=True` 控制。

3. 校验：
- 已执行 `python -m py_compile auto_annotate_sam.py`，语法通过。

## 本轮已落地（修复valid验证集未分割，2026-04-11）
1. `auto_annotate_sam.py` 核心转换逻辑调整：
- 不再直接调用 `yolo_bbox2segment()`（该路径在检测到分割标签时会整分支提前返回）。
- 新增 `force_bbox_to_segment()`：逐图像读取 YOLO 检测框，调用 `SAM` 生成分割并写入标签，避免 `valid` 被跳过。

2. 数据集键名兼容：
- 新增 `SPLIT_ALIASES`，验证集同时支持 `val` 和 `valid`（优先读取实际存在的键）。

3. 输出与替换流程保持安全：
- 仍先写入临时目录，再整体替换原标签，保留检测标签备份机制。

4. 校验：
- 已执行 `python -m py_compile auto_annotate_sam.py`，通过。

## 本轮已落地（test集评估入口，2026-04-12）
1. 数据集 YAML 增加 `test` 键：
- `GZ-DET.yaml` 新增 `test: /root/autodl-fs/GYU-DET/test`
- `GZ-DET-seg.yaml` 新增 `test: /root/autodl-fs/GYU-DET/test`

2. 新增统一评估脚本：
- `scripts/eval_test_set.py`
- 支持一次命令同时评估检测模型与分割模型（`split=test`）。
- 自动校验 YAML 是否定义目标 split，避免误跑到 `val`。
- 导出统一汇总文件：`runs/gyu_eval/<name>/metrics_summary.json`
- 汇总项包含：模型路径、数据 YAML、run 保存目录、核心 metrics 字典和速度统计。

3. 校验：
- 已执行 `python -m py_compile scripts/eval_test_set.py`，通过。
- 兼容性修正：`scripts/eval_test_set.py` 改为 `ultralytics` 按需导入，并用 `PyYAML` 读取数据配置，确保在未安装 `cv2` 时也可正常执行 `--help`。

## 下一小阶段计划（2026-04-12）
1. 扩展 `app.py`：新增像素-物理系数输入与单位输入。
2. 在分割结果上实现病害长度与最大宽度估计算法（优先骨架+距离变换，带回退策略）。
3. 将测量结果写入前端表格并在摘要区展示，形成“上传一张图即完成检测+分割+尺寸估算”闭环。

## 变更记录（增量）
- 2026-04-12：`GZ-DET.yaml`、`GZ-DET-seg.yaml` 补充 `test` split 路径。
- 2026-04-12：新增 `scripts/eval_test_set.py`，用于检测/分割 test 统一评估与 JSON 汇总导出。

## 本轮已落地（前端测量能力，2026-04-12）
1. 前端参数扩展（`app.py`）：
- 新增像素到物理量换算输入：`Physical per Pixel`（例如 `0.2 mm/px`）。
- 新增单位名称输入：`Unit Name`（如 `mm`、`cm`）。

2. 分割尺寸估计算法（`app.py`）：
- 新增 `_measure_mask_geometry()`，对每个实例 mask 估算：
  - 面积：`Mask Area(px^2)`
  - 长度：`Length(px)`
  - 最大宽度：`Max Width(px)`
- 核心方法：
  - 形态学骨架提取（无额外依赖）
  - 8 邻域骨架边长累计估算长度
  - 距离变换估算局部最大厚度（`2 * max(distance)`）作为最大宽度
  - 加入 `minAreaRect` 回退，提升异常形状下鲁棒性

3. 前端结果展示：
- 表格新增列：
  - `Mask Area(px^2)`、`Length(px)`、`Max Width(px)`
  - `Length(<unit>)`、`Max Width(<unit>)`
- 摘要区增加测量统计（可显示平均长度和平均最大宽度，按单位输出）。

4. 校验：
- 已执行 `python -m py_compile app.py`，通过。
- 已执行 `python scripts/eval_test_set.py --help`，通过（参数可正常展示）。

## 本轮已落地（test分割mask生成支持，2026-04-12）
1. `auto_annotate_sam.py` 扩展 test split：
- `SPLIT_ALIASES` 新增 `"test": ("test",)`。
- 新增 `TARGET_SPLITS = ("train", "val", "test")` 并用于主循环。

2. 效果：
- 可直接按已有 YOLO 检测框标签为 `test` 生成 SAM 分割标签，便于后续 `split=test` 的分割评估。

3. 校验：
- 已执行 `python -m py_compile auto_annotate_sam.py`，通过。
- 已执行 `python -m py_compile app.py scripts/eval_test_set.py auto_annotate_sam.py train.py train_seg.py`，通过。

## 下一小阶段计划（2026-04-12，更新）
1. 给出 `test` 全流程命令模板（检测评估、test mask 生成、分割评估、前端启动）。
2. 如需，再按你的实际权重路径代入并固化成一键脚本。

## 本轮已落地（独立中文前端，仅det/seg，2026-04-12）
1. 新增独立前端文件（不使用原 `app.py`）：
- `app_cn_det_seg.py`

2. 功能范围（按需求仅保留两类任务）：
- 任务仅有 `det` 和 `seg`。
- 中文界面：参数区、图像结果页、实例结果明细页。
- 支持模型扫描、下拉选择、自定义路径加载、路径校验。

3. 推理与测量能力：
- 单图上传后可做检测/分割推理并可视化。
- 分割实例支持尺寸估算：
  - 掩膜面积（px²）
  - 病害长度（px）
  - 最大宽度（px）
- 支持像素-物理系数换算：
  - `长度(物理)`、`最大宽度(物理)`、`单位`

4. 校验：
- 已执行 `python -m py_compile app_cn_det_seg.py`，通过。

## 变更记录（增量）
- 2026-04-12：新增 `app_cn_det_seg.py`，构建不依赖原 `app.py` 的中文前端，且任务限制为 `det/seg`。

## 本轮修正（代码英文命名，界面中文，2026-04-12）
1. 按要求重写 `app_cn_det_seg.py`：
- Python 代码中的类名、函数名、变量名全部改为英文命名。
- 界面展示文本保持中文。

2. 功能保持不变：
- 任务仅 `det/seg`。
- 图片上传推理、模型选择/路径校验、检测结果表格输出。
- 分割掩膜尺寸估算（长度、最大宽度）与像素-物理换算。

3. 校验：
- 已执行 `python -m py_compile app_cn_det_seg.py`，通过。
