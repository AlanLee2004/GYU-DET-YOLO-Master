import gc
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")


class UIConfig:
    DEFAULT_MODELS = {
        "det": "yolov8n.pt",
        "seg": "yolov8n-seg.pt",
    }
    THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")


class ModelManager:
    def __init__(self, ckpt_root: Path):
        self.ckpt_root = ckpt_root
        self.current_model: Optional[YOLO] = None
        self.current_model_path: str = ""
        self.current_task: str = "det"

    def scan_checkpoints(self) -> Dict[str, List[str]]:
        model_map = {"det": [], "seg": []}
        if not self.ckpt_root.exists():
            return model_map

        for p in self.ckpt_root.rglob("*.pt"):
            if p.is_dir():
                continue
            path_str = str(p.resolve())
            filename = p.name.lower()
            parent = p.parent.name.lower()
            if "seg" in filename or "seg" in parent:
                model_map["seg"].append(path_str)
            else:
                model_map["det"].append(path_str)

        for task in model_map:
            model_map[task] = sorted(list(set(model_map[task])))
        return model_map

    def unload_model(self) -> None:
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_model(self, model_path: str, task: str) -> YOLO:
        target_path = (model_path or "").strip()
        if not target_path or not os.path.exists(target_path):
            target_path = UIConfig.DEFAULT_MODELS[task]
        elif os.path.isdir(target_path):
            candidates = [
                os.path.join(target_path, "weights", "best.pt"),
                os.path.join(target_path, "weights", "last.pt"),
                os.path.join(target_path, "best.pt"),
                os.path.join(target_path, "last.pt"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    target_path = c
                    break

        if self.current_model is not None and self.current_model_path == target_path:
            return self.current_model

        self.unload_model()
        model = YOLO(target_path)
        self.current_model = model
        self.current_model_path = target_path
        self.current_task = task
        return model

    def current_device(self) -> str:
        try:
            if self.current_model is not None:
                return str(next(self.current_model.model.parameters()).device)
        except Exception:
            pass
        return "unknown"


class BridgeDamageUI:
    def __init__(self, ckpt_root: str):
        self.ckpt_root = Path(ckpt_root)
        self.model_manager = ModelManager(self.ckpt_root)
        self.model_map = self.model_manager.scan_checkpoints()

    @staticmethod
    def morphological_skeletonize(binary_mask: np.ndarray) -> np.ndarray:
        img = (binary_mask > 0).astype(np.uint8) * 255
        skeleton = np.zeros_like(img)
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, cross)
            temp = cv2.subtract(img, opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = cv2.erode(img, cross)
            if cv2.countNonZero(img) == 0:
                break
        return skeleton

    @staticmethod
    def skeleton_length_px(skeleton: np.ndarray) -> float:
        ys, xs = np.where(skeleton > 0)
        if len(ys) == 0:
            return 0.0

        points = {(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())}
        neighbors = [
            (-1, -1, np.sqrt(2.0)),
            (-1, 0, 1.0),
            (-1, 1, np.sqrt(2.0)),
            (0, 1, 1.0),
            (1, 1, np.sqrt(2.0)),
            (1, 0, 1.0),
            (1, -1, np.sqrt(2.0)),
            (0, -1, 1.0),
        ]
        length = 0.0
        for y, x in points:
            for dy, dx, weight in neighbors:
                ny, nx = y + dy, x + dx
                if (ny, nx) in points and (ny > y or (ny == y and nx > x)):
                    length += float(weight)
        return length

    def measure_mask_geometry(self, mask_data: Any) -> Dict[str, float]:
        if isinstance(mask_data, torch.Tensor):
            mask = mask_data.detach().cpu().numpy()
        else:
            mask = np.asarray(mask_data)
        if mask.ndim == 3:
            mask = mask[0]

        binary = (mask > 0.5).astype(np.uint8) * 255
        area_px2 = float(cv2.countNonZero(binary))
        if area_px2 <= 0:
            return {"area_px2": 0.0, "length_px": 0.0, "max_width_px": 0.0}

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"area_px2": area_px2, "length_px": 0.0, "max_width_px": 0.0}

        main_contour = max(contours, key=cv2.contourArea)
        min_rect = cv2.minAreaRect(main_contour)
        rect_w, rect_h = float(min_rect[1][0]), float(min_rect[1][1])
        rect_length = max(rect_w, rect_h)
        rect_width = min(rect_w, rect_h)

        dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        skeleton = self.morphological_skeletonize(binary)
        centerline_length = self.skeleton_length_px(skeleton)
        if centerline_length > 1e-6:
            length_px = max(centerline_length, rect_length)
        else:
            length_px = rect_length

        if cv2.countNonZero(skeleton) > 0:
            max_width_px = float(dist_map[skeleton > 0].max() * 2.0)
        else:
            max_width_px = float(dist_map.max() * 2.0)
        max_width_px = max(max_width_px, rect_width)

        return {
            "area_px2": area_px2,
            "length_px": float(length_px),
            "max_width_px": float(max_width_px),
        }

    @staticmethod
    def safe_round(value: Optional[float], digits: int = 3) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return round(float(value), digits)

    def run_inference(
        self,
        task: str,
        image: np.ndarray,
        model_dropdown: str,
        custom_model_path: str,
        conf: float,
        iou: float,
        device: str,
        max_det: float,
        line_width: float,
        force_cpu: bool,
        retina_masks: bool,
        physical_per_pixel: float,
        unit_name: str,
    ):
        if image is None:
            return None, None, "⚠️ 请先上传图片。"

        device_opt = "cpu" if force_cpu else (device if device else "")
        line_width_opt = int(line_width) if line_width > 0 else None
        max_det_opt = int(max_det)
        scale = float(physical_per_pixel) if physical_per_pixel and physical_per_pixel > 0 else None
        unit = (unit_name or "").strip() or "mm"

        options = {}
        if retina_masks:
            options["retina_masks"] = True
        if task == "seg" and "retina_masks" not in options:
            options["retina_masks"] = True

        model_path = (custom_model_path or "").strip() or (model_dropdown or "").strip()
        try:
            model = self.model_manager.load_model(model_path, task)
        except Exception as e:
            return image, None, f"❌ 加载模型失败：{e}"

        try:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results = model(
                image_bgr,
                conf=conf,
                iou=iou,
                device=device_opt,
                max_det=max_det_opt,
                line_width=line_width_opt,
                **options,
            )
        except Exception as e:
            return image, None, f"❌ 推理失败：{e}"

        res = results[0]
        output_img = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)

        headers = [
            "类别ID",
            "类别名",
            "置信度",
            "x1",
            "y1",
            "x2",
            "y2",
            "掩膜面积(px²)",
            "长度(px)",
            "最大宽度(px)",
            "长度(物理)",
            "最大宽度(物理)",
            "单位",
        ]
        rows: List[Dict[str, Any]] = []

        if res.boxes is not None:
            for i, box in enumerate(res.boxes):
                try:
                    cls_id = int(box.cls[0]) if box.cls.numel() > 0 else 0
                    cls_name = model.names[cls_id]
                    conf_value = float(box.conf[0]) if box.conf.numel() > 0 else 0.0
                    xyxy = box.xyxy[0].tolist()

                    area_px2 = None
                    length_px = None
                    width_px = None
                    length_phy = None
                    width_phy = None

                    if res.masks is not None and getattr(res.masks, "data", None) is not None and i < len(res.masks.data):
                        geo = self.measure_mask_geometry(res.masks.data[i])
                        area_px2 = geo["area_px2"]
                        length_px = geo["length_px"]
                        width_px = geo["max_width_px"]
                        if scale is not None:
                            length_phy = length_px * scale
                            width_phy = width_px * scale

                    rows.append(
                        {
                            "类别ID": cls_id,
                            "类别名": cls_name,
                            "置信度": round(conf_value, 3),
                            "x1": round(xyxy[0], 1),
                            "y1": round(xyxy[1], 1),
                            "x2": round(xyxy[2], 1),
                            "y2": round(xyxy[3], 1),
                            "掩膜面积(px²)": self.safe_round(area_px2, 1),
                            "长度(px)": self.safe_round(length_px, 2),
                            "最大宽度(px)": self.safe_round(width_px, 2),
                            "长度(物理)": self.safe_round(length_phy, 3),
                            "最大宽度(物理)": self.safe_round(width_phy, 3),
                            "单位": unit if length_phy is not None else "",
                        }
                    )
                except Exception:
                    continue

        df = pd.DataFrame(rows, columns=headers)
        measured_rows = [x for x in rows if x.get("长度(px)") is not None]
        inference_ms = res.speed.get("inference", 0.0)
        runtime_device = self.model_manager.current_device()

        measure_text = ""
        if measured_rows:
            mean_len_px = float(np.mean([x["长度(px)"] for x in measured_rows]))
            mean_w_px = float(np.mean([x["最大宽度(px)"] for x in measured_rows]))
            if scale is not None:
                measure_text = (
                    f"- **测量实例数**：{len(measured_rows)}，平均长度 `{mean_len_px * scale:.3f} {unit}`，"
                    f"平均最大宽度 `{mean_w_px * scale:.3f} {unit}`\n"
                )
            else:
                measure_text = (
                    f"- **测量实例数**：{len(measured_rows)}，平均长度 `{mean_len_px:.2f} px`，"
                    f"平均最大宽度 `{mean_w_px:.2f} px`\n"
                )

        summary = (
            f"### ✅ 推理完成\n"
            f"- **任务**：`{task}`\n"
            f"- **模型**：`{Path(self.model_manager.current_model_path).name}`\n"
            f"- **推理时间**：`{inference_ms:.1f} ms`\n"
            f"- **目标数量**：{len(rows)}\n"
            f"- **设备**：`{runtime_device}`\n"
            f"{measure_text}"
        )
        return output_img, df, summary

    def refresh_model_choices(self, task: str):
        self.model_map = self.model_manager.scan_checkpoints()
        choices = self.model_map.get(task, [])
        if not choices:
            choices = [UIConfig.DEFAULT_MODELS[task]]
        return gr.update(choices=choices, value=choices[0])

    def validate_model_path(self, task: str, path_text: str) -> str:
        path_text = (path_text or "").strip()
        if not path_text:
            return "⚠️ 请输入模型路径。"
        path = Path(path_text)
        if not path.exists():
            return f"❌ 路径不存在：`{path_text}`"

        if path.is_dir():
            candidates = [
                path / "weights" / "best.pt",
                path / "weights" / "last.pt",
                path / "best.pt",
                path / "last.pt",
            ]
            matched = [x for x in candidates if x.exists()]
            if not matched:
                return f"❌ 目录内未找到模型文件：`{path_text}`"
            path = matched[0]

        try:
            model = YOLO(str(path))
            num_classes = len(model.names)
            return (
                f"### ✅ 路径可用\n"
                f"- **任务类型**：`{model.task}`（当前任务：`{task}`）\n"
                f"- **类别数**：{num_classes}\n"
                f"- **权重文件**：`{path}`"
            )
        except Exception as e:
            return f"❌ 模型无效：{e}"

    def launch(self):
        with gr.Blocks(title="桥梁病害检测与分割前端（中文）", theme=UIConfig.THEME) as app:
            gr.Markdown("# 桥梁病害检测与分割前端（中文）")

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### 参数设置")
                    task_radio = gr.Radio(choices=["det", "seg"], value="det", label="任务类型")

                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=self.model_map["det"] if self.model_map["det"] else [UIConfig.DEFAULT_MODELS["det"]],
                            value=self.model_map["det"][0] if self.model_map["det"] else UIConfig.DEFAULT_MODELS["det"],
                            label="模型权重",
                            interactive=True,
                            scale=5,
                        )
                        refresh_btn = gr.Button("刷新", scale=1, min_width=10, size="sm")

                    custom_model_path = gr.Textbox(
                        value="",
                        label="自定义模型路径（文件或目录）",
                        placeholder="./runs/xxx/weights/best.pt",
                        interactive=True,
                    )
                    validate_btn = gr.Button("检查路径", size="sm")

                    with gr.Accordion("高级参数", open=True):
                        conf_slider = gr.Slider(0, 1, 0.25, step=0.01, label="置信度阈值")
                        iou_slider = gr.Slider(0, 1, 0.7, step=0.01, label="IoU 阈值")

                        with gr.Row():
                            max_det_num = gr.Number(300, label="最多目标数", precision=0)
                            line_width_num = gr.Number(0, label="线宽（0为自动）", precision=0)

                        with gr.Row():
                            device_txt = gr.Textbox("0", label="设备ID（如 0 或 cpu）")
                            force_cpu_chk = gr.Checkbox(False, label="强制CPU")

                        retina_masks_chk = gr.Checkbox(True, label="高分辨率掩膜（分割建议开启）")

                    gr.Markdown("### 尺寸换算参数")
                    with gr.Row():
                        physical_per_pixel_num = gr.Number(
                            value=1.0,
                            label="每像素物理量",
                            precision=6,
                            info="例如 0.2 表示 1 px = 0.2 mm",
                        )
                        unit_name_txt = gr.Textbox(value="mm", label="单位名")

                    run_btn = gr.Button("开始推理", variant="primary", size="lg")

                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("图像结果"):
                            with gr.Row():
                                input_img = gr.Image(type="numpy", label="输入图片", height=500)
                                output_img = gr.Image(type="numpy", label="输出结果", height=500, interactive=False)
                            info_md = gr.Markdown(value="等待输入...")

                        with gr.TabItem("检测/测量数据"):
                            output_df = gr.Dataframe(
                                headers=[
                                    "类别ID",
                                    "类别名",
                                    "置信度",
                                    "x1",
                                    "y1",
                                    "x2",
                                    "y2",
                                    "掩膜面积(px²)",
                                    "长度(px)",
                                    "最大宽度(px)",
                                    "长度(物理)",
                                    "最大宽度(物理)",
                                    "单位",
                                ],
                                label="实例结果明细",
                            )

            task_radio.change(fn=self.refresh_model_choices, inputs=task_radio, outputs=model_dropdown)
            refresh_btn.click(fn=self.refresh_model_choices, inputs=task_radio, outputs=model_dropdown)
            validate_btn.click(fn=self.validate_model_path, inputs=[task_radio, custom_model_path], outputs=info_md)

            run_btn.click(
                fn=self.run_inference,
                inputs=[
                    task_radio,
                    input_img,
                    model_dropdown,
                    custom_model_path,
                    conf_slider,
                    iou_slider,
                    device_txt,
                    max_det_num,
                    line_width_num,
                    force_cpu_chk,
                    retina_masks_chk,
                    physical_per_pixel_num,
                    unit_name_txt,
                ],
                outputs=[output_img, output_df, info_md],
            )

        app.launch(share=False, inbrowser=True)


if __name__ == "__main__":
    ckpts_dir = Path(__file__).parent / "ckpts"
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    ui = BridgeDamageUI(str(ckpts_dir))
    ui.launch()
