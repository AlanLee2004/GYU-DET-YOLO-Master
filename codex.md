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
