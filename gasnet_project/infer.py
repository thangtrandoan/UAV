import os
import time
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Import trực tiếp class GASNet từ file train.py có sẵn trong thư mục
from train import GASNet

def count_parameters(model):
    """Đếm tổng số lượng tham số có thể huấn luyện của mô hình"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class InferDataset(Dataset):
    def __init__(self, img_dir, img_names, transform):
        self.img_dir = img_dir
        self.img_names = img_names
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = self.transform(img)
            return tensor, img_name
        except Exception as e:
            print(f"Lỗi khi nạp ảnh {img_name}: {e}")
            return torch.zeros(3, 224, 224), img_name

def main():
    parser = argparse.ArgumentParser(description="GASNet Inference Batching with AMP and TTA")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch Size (tuỳ chỉnh, mặc định 1)")
    parser.add_argument("--amp", action="store_true", help="Bật Mixed Precision (FP16) để ép tốc độ x2 (Chỉ có tác dụng với GPU)")
    parser.add_argument("--tta-flip", action="store_true", help="Bật Lật ảnh ngang (TTA). Dự kiến làm tốc độ chậm đi 1 nửa")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_img_dir = os.path.join(base_dir, "test", "test_image")
    output_dir = os.path.join(base_dir, "test", "output")
    
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Đang khởi tạo mô hình GASNet với cấu hình từ evaluate_vrai...")
    model = GASNet(
        num_classes=6302, 
        use_pretrained=False, 
        backbone="resnet50_ibn", 
        use_gem=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True # Tối ưu hoá tốc độ convolution của cuDNN
        
    model_path = os.path.join(base_dir, "test", "gasnet.best.pth")
    if os.path.exists(model_path):
        print(f"Đang nạp trọng số từ file: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
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
        print("=> Nạp trọng số THÀNH CÔNG!")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy file: {model_path}")
        print("=> Đang dùng trọng số Khởi tạo ngẫu nhiên (Random Init)!")
    
    num_params = count_parameters(model)
    print(f"Tổng số tham số: {num_params:,}")
    
    model.eval()
    model = model.to(device)
    
    test_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    report_lines = []
    report_lines.append("=== BÁO CÁO KẾT QUẢ INFERENCE GASNET ===")
    report_lines.append(f"- Tổng số tham số mô hình: {num_params:,}")
    report_lines.append(f"- Thiết bị tính toán: {device}")
    report_lines.append(f"- Batch Size: {args.batch_size}")
    report_lines.append(f"- Kỹ thuật tăng tốc AMP (FP16): {'BẬT' if args.amp else 'TẮT'}")
    report_lines.append(f"- Kỹ thuật TTA-Flip: {'BẬT' if args.tta_flip else 'TẮT'}")
    report_lines.append("-" * 40)
    
    if not img_files:
        msg = f"Chưa có file ảnh nào trong thư mục {test_img_dir}."
        print(msg)
        report_lines.append(msg)
    else:
        print(f"Tìm thấy {len(img_files)} ảnh. Bắt đầu xử lý...")
        
        dataset = InferDataset(test_img_dir, img_files, test_tfms)
        num_workers = 4 if args.batch_size > 1 else 0
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=(device.type == 'cuda')
        )

        if device.type == 'cuda':
            print("Đang chạy warmup model...")
            dummy_input = torch.zeros(1, 3, 224, 224).to(device)
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if args.amp else torch.autocast(device_type='cpu', enabled=False)
            with torch.no_grad(), amp_ctx:
                for _ in range(3): # Warmup vài vòng để AMP khởi tạo cache đầy đủ
                    _ = model(dummy_input)
            torch.cuda.synchronize()

        total_infer_time = 0.0
        
        for batch_idx, (batch_tensors, batch_names) in enumerate(dataloader):
            batch_tensors = batch_tensors.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                # Xử lý ngữ cảnh AMP (Mixed Precision)
                amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if (args.amp and device.type == 'cuda') else torch.autocast(device_type='cpu', enabled=False)
                
                with amp_ctx:
                    outputs = model(batch_tensors)
                    features = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    
                    # Nếu bật TTA-Flip
                    if args.tta_flip:
                        # Lật ảnh theo chiều ngang (dim = 3 là Width)
                        flip_tensors = torch.flip(batch_tensors, dims=[3])
                        outputs_flip = model(flip_tensors)
                        features_flip = outputs_flip[0] if isinstance(outputs_flip, (tuple, list)) else outputs_flip
                        
                        # Cộng trung bình 2 vector lại
                        features = (features + features_flip) / 2.0
                    
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_time_ms = (end_time - start_time) * 1000
            total_infer_time += batch_time_ms
            print(f"[Batch {batch_idx+1}] Xử lý {len(batch_names)} ảnh trong {batch_time_ms:.2f} ms")
            
            for i in range(len(batch_names)):
                img_name = batch_names[i]
                feat_vector = features[i]
                
                feat_shape = list(feat_vector.shape)
                res_str = f"Ảnh: {img_name:20s} | Kích thước Vector: {feat_shape} | (Đã bỏ qua lưu file)"
                report_lines.append(res_str)
                
        avg_time_per_image = total_infer_time / len(img_files)
        avg_str = f"\n=> TỐC ĐỘ TRUNG BÌNH: {avg_time_per_image:.2f} ms/ảnh"
        print(avg_str)
        report_lines.append(avg_str)
        
    report_path = os.path.join(output_dir, f"report_batch{args.batch_size}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\n[Hoàn tất] Kết quả lưu tại: {report_path}")

if __name__ == "__main__":
    main()
