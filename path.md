# GYU-DET 桥梁病害项目整体技术路线

## 1. 项目目标
- 面向桥梁病害场景（裂缝、破损、孔洞、钢筋外露、渗水等）构建可落地的检测与分割方案。
- 第一阶段优先保证小目标与细长目标召回；第二阶段输出高质量实例 mask。

## 2. 总体方案
- 主线采用两阶段路线：
1. `检测主线`：YOLO-Master（MoE）+ P2 小目标检测头 + 切片推理。
2. `分割主线`：基于检测结果结合 SAM 自动标注生成 mask，再训练 YOLO-Master-Seg 输出实例分割。

## 3. 检测主线（当前主战路线）

### 3.1 模型结构
- 使用 `ultralytics/cfg/models/master/v0_1/det/yolo-master-m.yaml`（已改为 `P2/P3/P4/P5` 四检测头）。
- 相比原 `P3/P4/P5`，新增 `P2` 强化细小裂缝、细小病害目标检测能力。

### 3.2 训练策略
- 训练入口：`train.py`
- 核心配置：
- `imgsz=640`（低显存稳跑基线，后续可升 `768/960/1280`）
- `batch=-1`（AutoBatch 自动匹配显存）
- `pretrained=YOLO-Master-EsMoE-M.pt`（迁移学习）

### 3.3 推理策略
- 推理入口：`predict_sahi.py`
- 采用 `Sparse SAHI` 切片推理应对高分辨率图像中的小病害漏检问题。
- 建议作为精度增强推理分支，按场景权衡速度与召回。

## 4. 分割主线（检测到分割闭环）

### 4.1 自动标注
- 脚本：`auto_annotate_sam.py`
- 方法：检测框引导 + SAM 生成轮廓，导出 YOLO 分割标签。
- 安全机制：自动备份原检测标签，再写入分割标签。

### 4.2 分割训练
- 模型配置：`ultralytics/cfg/models/master/v0_1/seg/yolo-master-seg-m.yaml`
- 数据配置：`GZ-DET-seg.yaml`
- 训练脚本：`train_seg.py`
- 默认低显存配置：`imgsz=640 + batch=-1`。

## 5. MoE 训练稳定性路线
- 发现并修复 `moe_loss` 长期为 0 的代码问题：
- 根因：`OptimizedMOEImproved` 缺少 `aux_loss` 属性，导致全局损失聚合无法读取 MoE 辅助损失。
- 修复：在 `ultralytics/nn/modules/moe/modules.py` 中补充 `aux_loss` property。

## 6. 数据与增强路线
- 重点围绕桥下复杂光照和退化成像：
- 光照：CLAHE / Gamma 等轻量预处理 + 颜色扰动增强。
- 模糊：MotionBlur / Blur / Noise 等可控退化增强。
- 约束：对细长裂缝避免过强几何增强，防止目标形态失真。

## 7. 实验推进顺序（推荐）
1. 跑通检测基线：`P2 + imgsz640 + AutoBatch`。
2. 验证 `moe_loss` 非零并观察检测损失曲线。
3. 启用 `Sparse SAHI` 做推理增强对比。
4. 执行 SAM 自动标注并人工抽检关键类别。
5. 训练分割模型并与检测主线做精度/速度对比。
6. 在显存允许时逐步升分辨率（`768 -> 960 -> 1280`）。

## 8. 交付形态
- 检测模型：用于快速病害检出与定位。
- 分割模型：用于病害轮廓精细化与面积/形态评估。
- 双模型协同：在线快速检测 + 离线高精度分割。
