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
    """Đọc ảnh, resize thẳng 224x224, normalize → numpy (1, 3, 224, 224) float32"""
    img = Image.open(img_path).convert("RGB")

    # Resize thẳng về 224x224 (không center-crop)
    img = img.resize((224, 224), Image.BILINEAR)
    
    # Chuyển sang numpy float32, scale về [0, 1], rồi chuẩn hoá
    arr = np.array(img, dtype=np.float32) / 255.0   # (224, 224, 3)
    arr = arr.transpose(2, 0, 1)                     # (3, 224, 224) — CHW
    arr = (arr - MEAN) / STD                          # Normalize
    
    return np.expand_dims(arr, axis=0)                # (1, 3, 224, 224)


# =============================================
# TensorRT Model Wrapper (hỗ trợ TRT 8.x và 10.x)
# =============================================
TRT_MAJOR = int(trt.__version__.split(".")[0])


class TRTModel:
    def __init__(self, engine_path):
        print(f"Đang nạp TensorRT Engine: {engine_path} (TensorRT {trt.__version__})...")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                "Không deserialize được engine. Engine phải được build trên đúng máy này "
                "và đúng phiên bản TensorRT."
            )

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs = []
        self.outputs = []
        self.bindings = []  # chỉ dùng cho API cũ (TRT < 10)

        if TRT_MAJOR >= 10:
            self._init_trt10()
        else:
            self._init_legacy()

    def _init_trt10(self):
        print(f"Engine có {self.engine.num_io_tensors} tensor I/O:")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            self._add_tensor(name, shape, dtype, is_input)

    def _init_legacy(self):
        print(f"Engine có {self.engine.num_bindings} binding(s):")
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            info = self._add_tensor(name, shape, dtype, is_input)
            self.bindings.append(int(info["device"]))

    def _add_tensor(self, name, shape, dtype, is_input):
        size = int(np.prod(shape)) if len(shape) else 1
        host = cuda.pagelocked_empty(size, dtype)
        device = cuda.mem_alloc(host.nbytes)
        info = {"name": name, "host": host, "device": device, "shape": shape, "dtype": dtype}
        (self.inputs if is_input else self.outputs).append(info)
        tag = "INPUT " if is_input else "OUTPUT"
        print(f"  [{tag}] {name}: shape={list(shape)}, dtype={np.dtype(dtype).name}")
        return info

    def infer(self, input_np):
        """Chạy inference với input numpy (1,3,224,224), trả về list output numpy."""
        np.copyto(self.inputs[0]["host"], input_np.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        if TRT_MAJOR >= 10:
            for tensor in self.inputs + self.outputs:
                self.context.set_tensor_address(tensor["name"], int(tensor["device"]))
            ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async thất bại.")

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        return [np.copy(out["host"]).reshape(out["shape"]) for out in self.outputs]


# =============================================
# Main
# =============================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="GASNet TensorRT Inference")
    parser.add_argument("--engine", default=os.path.join(base_dir, "test", "gasnet_fp16.engine"),
                        help="Đường dẫn file .engine đã build")
    parser.add_argument("--image-dir", default=os.path.join(base_dir, "test", "test_image"),
                        help="Thư mục ảnh đầu vào")
    parser.add_argument("--output-dir", default=os.path.join(base_dir, "test", "output"),
                        help="Thư mục lưu report")
    parser.add_argument("--warmup", type=int, default=10, help="Số vòng warmup")
    args = parser.parse_args()

    test_img_dir = args.image_dir
    output_dir = args.output_dir
    engine_path = args.engine
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(engine_path):
        engine_name = os.path.basename(engine_path)
        print(f"[LỖI] Không tìm thấy file: {engine_path}")
        print("Bạn cần build Engine trước bằng lệnh (chạy trên chính thiết bị này):")
        pool = "--workspace=4096" if TRT_MAJOR < 9 else "--memPoolSize=workspace:4096"
        print(f"  /usr/src/tensorrt/bin/trtexec --onnx=gasnet.onnx --saveEngine={engine_name} --fp16 {pool}")
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
    
    # Warmup: Chạy vài vòng để GPU ổn định tần số và cache
    print(f"Đang chạy warmup TensorRT ({args.warmup} vòng)...")
    dummy_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
    for _ in range(max(0, args.warmup)):
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
