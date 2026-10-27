import os
import argparse

import torch

from bcgg import infer_gate_config
from train import GASNet, infer_backbone_name
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
        default=None,
        choices=("resnet50", "resnet50_ibn", "swin_t", "dinov3_convnext"),
        help="Backbone; default auto-detects from the checkpoint",
    )
    parser.add_argument("--num-classes", type=int, default=None, help="Override; inferred from checkpoint by default")
    parser.add_argument(
        "--deploy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fold BN/RepDW before export (default: on)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset (13 is the safest for TensorRT 8.x on Jetson; try 17 for native LayerNorm)",
    )
    parser.add_argument(
        "--use-gem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Model was trained with GeM pooling (default: on; must match training)",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare ONNX (onnxruntime) against PyTorch before finishing (default: on)",
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
    backbone = args.backbone or infer_backbone_name(state_dict or {}, default="resnet50_ibn")
    num_classes = args.num_classes or _infer_num_classes(state_dict or {})
    print(f"   backbone={backbone}, gate={gate_config['gate']}, num_classes={num_classes}")

    model = GASNet(
        num_classes=num_classes,
        use_pretrained=False,
        backbone=backbone,
        use_gem=args.use_gem,
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
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names  # Tự động đặt tên cho tất cả các output
    )

    # Kiểm tra ONNX vs PyTorch nếu có onnxruntime (bỏ qua nếu chưa cài)
    if args.verify:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            print("5. [Bỏ qua verify] Chưa cài onnxruntime (pip install onnxruntime).")
        else:
            print("5. Verify ONNX vs PyTorch...")
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            expected = test_out if isinstance(test_out, (tuple, list)) else (test_out,)
            got = session.run(None, {"input": dummy_input.numpy()})
            worst = 0.0
            for i, (ref, out) in enumerate(zip(expected, got)):
                diff = float(np.abs(ref.detach().numpy() - out).max())
                worst = max(worst, diff)
                print(f"   output_{i}: max|torch-onnx| = {diff:.3e}")
            print("   VERIFY OK" if worst < 1e-3 else "   [WARN] sai khác lớn, kiểm tra lại export")

    engine_name = f"{os.path.splitext(os.path.basename(onnx_path))[0]}_fp16.engine"
    print(f"\n[Hoàn tất] Đã lưu file ONNX tại: {onnx_path}")
    print(f"   Output nodes: {output_names}")
    print("Bước tiếp theo — build TensorRT engine TRÊN THIẾT BỊ ĐÍCH (Jetson):")
    print(f"   /usr/src/tensorrt/bin/trtexec --onnx={os.path.basename(onnx_path)} "
          f"--saveEngine={engine_name} --fp16 --memPoolSize=workspace:4096")
    print("   (TensorRT 8.x: dùng --workspace=4096 thay cho --memPoolSize=workspace:4096)")


if __name__ == "__main__":
    main()
