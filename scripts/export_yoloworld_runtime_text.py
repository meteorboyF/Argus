#!/usr/bin/env python3
"""Export YOLO-World with a runtime, single-label CLIP embedding input.

Run on the Jetson. Keeping one class fixes the output shape for TensorRT while
the input embedding changes for every requested object name.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLOWorld


class RuntimeTextYOLOWorld(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        return self.model.predict(images, txt_feats=text_embeddings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="/opt/argus/models/yolov8s-worldv2.pt")
    parser.add_argument("--output", default="/opt/argus/exports/yoloworld_runtime_text_640.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLOWorld(args.weights).model.eval().cpu()
    head = model.model[-1]
    head.nc = 1
    head.export = True
    head.format = "onnx"
    wrapper = RuntimeTextYOLOWorld(model).eval()
    images = torch.zeros(1, 3, args.imgsz, args.imgsz)
    text = torch.randn(1, 1, 512)
    text /= text.norm(dim=-1, keepdim=True)
    torch.onnx.export(
        wrapper, (images, text), output,
        input_names=["images", "text_embeddings"], output_names=["output0"],
        opset_version=16, do_constant_folding=True,
    )
    print(output)


if __name__ == "__main__":
    main()
