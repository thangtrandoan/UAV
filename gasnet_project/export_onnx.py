import os
import argparse

import torch

from bcgg import infer_gate_config
from train import GASNet
from utils import normalize_state_dict


def _load_state_dict(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return normalize_state_dict(checkpoint[key])
    return normalize_state_dict(checkpoint)


def _infer_num_classes(state_dict: dict, default: int = 6302) -> int:
    weight = state_dict.get("classifier_global.weight")
    return int(weight.shape[0]) if weight is not None else default


def main():
    parser = argparse.ArgumentParser(description="Export GASNet to ONNX (TensorRT-ready)")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--checkpoint", default=os.path.join(base_dir, "test", "gasnet.best.pth"))
    parser.add_argument("--output", default=os.path.join(base_dir, "test", "gasnet.onnx"))
    parser.add_argument(
        "--backbone",
        default="resnet50_ibn",
        choices=("resnet50", "resnet50_ibn", "swin_t", "dinov3_convnext"),
    )
    parser.add_argument("--num-classes", type=int, default=None, help="Override; inferred from checkpoint by default")
    parser.add_argument(
        "--deploy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fold BN/RepDW before export (default: on)",
    )
    args = parser.parse_args()

    onnx_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    print("1. Đang khởi tạo mô hình GASNet...")
    state_dict = None
    if os.path.exists(args.checkpoint):
        state_dict = _load_state_dict(args.checkpoint)
    else:
        print(f"[CẢNH BÁO] Không tìm thấy {args.checkpoint}. Dùng trọng số ngẫu nhiên!")

    gate_config = infer_gate_config(state_dict or {})
    num_classes = args.num_classes or _infer_num_classes(state_dict or {})
    print(f"   gate={gate_config['gate']}, num_classes={num_classes}")

    model = GASNet(
        num_classes=num_classes,
        use_pretrained=False,
        backbone=args.backbone,
        use_gem=True,
        **gate_config,
    )

    if state_dict is not None:
        print(f"2. Nạp trọng số từ: {args.checkpoint}")
        model_state = model.state_dict()
        filtered = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model.load_state_dict(filtered, strict=False)
        print(f"=> Nạp {len(filtered)}/{len(model_state)} tensor thành công!")

    model.eval()
    if args.deploy:
        model = model.to_deploy()
        print("=> Đã fold BN/RepDW (deploy graph)")

    # Tạo dummy_input với đúng shape mong đợi (Batch=1, Channels=3, H=224, W=224)
    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")

    # Chạy thử để xác định số lượng output thực tế của model
    print("3. Chạy thử model để xác định cấu trúc output...")
    with torch.no_grad():
        test_out = model(dummy_input)

    # Xác định output_names dựa trên cấu trúc thực tế
    if isinstance(test_out, (tuple, list)):
        output_names = [f"output_{i}" for i in range(len(test_out))]
        print(f"   Model có {len(test_out)} outputs:")
        for i, o in enumerate(test_out):
            print(f"   - output_{i}: shape = {list(o.shape)}")
    else:
        output_names = ["output"]
        print(f"   Model có 1 output: shape = {list(test_out.shape)}")

    # Export sang ONNX
    print("4. Bắt đầu quá trình Export sang ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,          # Opset 13 tương thích tốt với TensorRT trên Jetson
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names  # Tự động đặt tên cho tất cả các output
    )

    print(f"\n[Hoàn tất] Đã lưu file ONNX tại: {onnx_path}")
    print(f"   Output nodes: {output_names}")
    print("Bạn có thể dùng file này để build TensorRT Engine bằng lệnh:")
    print(f"   /usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --saveEngine=gasnet_fp16.engine --fp16 --workspace=4096")


if __name__ == "__main__":
    main()
