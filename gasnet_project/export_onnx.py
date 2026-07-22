import os
import torch
from train import GASNet

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "test", "gasnet.best.pth")
    onnx_path = os.path.join(base_dir, "test", "gasnet.onnx")
    
    print("1. Đang khởi tạo mô hình GASNet...")
    model = GASNet(
        num_classes=6302, 
        use_pretrained=False, 
        backbone="resnet50_ibn", 
        use_gem=True
    )
    
    if os.path.exists(model_path):
        print(f"2. Nạp trọng số từ: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        print("=> Nạp trọng số thành công!")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy {model_path}. Dùng trọng số ngẫu nhiên!")
        
    model.eval()
    
    # Tạo dummy_input với đúng shape mong đợi (Batch=1, Channels=3, H=224, W=224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
    
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
