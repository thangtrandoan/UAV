import os
import time
import argparse
import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Khởi tạo TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# =============================================
# Preprocessing thuần numpy (không cần PyTorch)
# =============================================
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

def preprocess(img_path):
    """Đọc ảnh, resize, center-crop, normalize → numpy (1, 3, 224, 224) float32"""
    img = Image.open(img_path).convert("RGB")
    
    # Resize về 256x256
    img = img.resize((256, 256), Image.BILINEAR)
    
    # Center crop 224x224
    left = (256 - 224) // 2   # = 16
    top  = (256 - 224) // 2   # = 16
    img = img.crop((left, top, left + 224, top + 224))
    
    # Chuyển sang numpy float32, scale về [0, 1], rồi chuẩn hoá
    arr = np.array(img, dtype=np.float32) / 255.0   # (224, 224, 3)
    arr = arr.transpose(2, 0, 1)                     # (3, 224, 224) — CHW
    arr = (arr - MEAN) / STD                          # Normalize
    
    return np.expand_dims(arr, axis=0)                # (1, 3, 224, 224)


# =============================================
# TensorRT Model Wrapper
# =============================================
class TRTModel:
    def __init__(self, engine_path):
        print(f"Đang nạp file TensorRT Engine từ {engine_path}...")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Cấp phát bộ nhớ cho Input và Output
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        num_bindings = self.engine.num_bindings
        print(f"Engine có {num_bindings} binding(s):")
        
        for i in range(num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            is_input = self.engine.binding_is_input(i)
            
            # Tính kích thước bộ nhớ cần cấp
            size = 1
            for s in shape:
                size *= abs(s)  # abs() phòng trường hợp dimension = -1
            
            # Cấp phát bộ nhớ RAM (Host) và VRAM (Device)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            binding_info = {
                'host': host_mem,
                'device': device_mem,
                'shape': tuple(shape),
                'name': name,
                'dtype': dtype
            }
            
            if is_input:
                self.inputs.append(binding_info)
                print(f"  [INPUT]  {name}: shape={shape}, dtype={dtype}")
            else:
                self.outputs.append(binding_info)
                print(f"  [OUTPUT] {name}: shape={shape}, dtype={dtype}")

    def infer(self, input_np):
        """Chạy inference với input numpy array, trả về list các output numpy arrays."""
        # Copy input vào host memory
        np.copyto(self.inputs[0]['host'], input_np.ravel())
        
        # Chuyển dữ liệu từ RAM sang VRAM (CPU -> GPU)
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        # Chạy Inference trên GPU
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Chuyển dữ liệu kết quả từ VRAM về RAM (GPU -> CPU)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            
        # Đồng bộ hoá
        self.stream.synchronize()
        
        # Trả về tất cả các output
        results = []
        for out in self.outputs:
            result = np.copy(out['host'])  # Copy ra để an toàn
            result = result.reshape(out['shape'])
            results.append(result)
        
        return results


# =============================================
# Main
# =============================================
def main():
    parser = argparse.ArgumentParser(description="GASNet TensorRT Inference")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_img_dir = os.path.join(base_dir, "test", "test_image")
    output_dir = os.path.join(base_dir, "test", "output")
    engine_path = os.path.join(base_dir, "test", "gasnet_fp16.engine")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(engine_path):
        print(f"[LỖI] Không tìm thấy file: {engine_path}")
        print("Bạn cần build Engine trước bằng lệnh:")
        print(f"  /usr/src/tensorrt/bin/trtexec --onnx=gasnet.onnx --saveEngine=gasnet_fp16.engine --fp16 --workspace=4096")
        return

    # Nạp TensorRT Engine
    model = TRTModel(engine_path)
    
    # Lấy danh sách ảnh
    img_files = sorted([
        f for f in os.listdir(test_img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if not img_files:
        print(f"Chưa có file ảnh nào trong thư mục: {test_img_dir}")
        return
    
    print(f"\nBắt đầu TensorRT Inference cho {len(img_files)} ảnh...")
    
    # Warmup: Chạy 10 vòng để GPU ổn định tần số và cache
    print("Đang chạy warmup TensorRT (10 vòng)...")
    dummy_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
    for _ in range(10):
        model.infer(dummy_input)
        
    # Inference thật
    total_time = 0.0
    report_lines = []
    report_lines.append("=== BÁO CÁO KẾT QUẢ INFERENCE GASNET (TensorRT FP16) ===")
    report_lines.append(f"- Số ảnh: {len(img_files)}")
    report_lines.append(f"- Engine: {engine_path}")
    report_lines.append("-" * 50)
    
    for idx, img_name in enumerate(img_files):
        img_path = os.path.join(test_img_dir, img_name)
        
        # Preprocess ảnh (thuần numpy, không cần PyTorch)
        input_np = preprocess(img_path)
        
        # Đo thời gian inference (không tính preprocess)
        start_time = time.perf_counter()
        outputs = model.infer(input_np)
        end_time = time.perf_counter()
        
        infer_ms = (end_time - start_time) * 1000
        total_time += infer_ms
        
        # Lấy output đầu tiên (global feature vector)
        feat = outputs[0]
        feat_shape = list(feat.shape)
        
        line = f"Ảnh: {img_name:20s} | Vector: {str(feat_shape):15s} | TRT: {infer_ms:.2f} ms"
        print(line)
        report_lines.append(line)
        
    avg_time = total_time / len(img_files)
    summary = f"\n=> TỐC ĐỘ TRUNG BÌNH TENSORRT FP16: {avg_time:.2f} ms/ảnh"
    print(summary)
    report_lines.append(summary)
    
    # Lưu report
    report_path = os.path.join(output_dir, "report_tensorrt_fp16.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[Hoàn tất] Kết quả lưu tại: {report_path}")

if __name__ == "__main__":
    main()
