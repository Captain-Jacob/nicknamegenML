
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch compiled CUDA:", torch.version.cuda)
print("Runtime GPU CUDA:", torch.cuda.get_device_capability())
print("GPU name:", torch.cuda.get_device_name() if torch.cuda.is_available() else "No GPU")

#basit test için ,for basic test