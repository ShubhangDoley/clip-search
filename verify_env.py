import sys
import torch

def main():
    print("=" * 50)
    print("      SEMANTIC IMAGE SEARCH - ENV VERIFICATION")
    print("=" * 50)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"GPU Model: {gpu_name}")
        print(f"Total VRAM: {vram_mb:.0f} MB")
        
        # Test small tensor operation on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        print("GPU Compute Test (1000x1000 matmul): SUCCESS")
    else:
        print("WARNING: CUDA is NOT available! GPU acceleration disabled.")
        
    print("-" * 50)
    
    # Check optional packages
    packages = ["open_clip", "transformers", "faiss", "PIL", "numpy", "kaggle"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "Installed")
            print(f"  [x] {pkg:15s}: {version}")
        except ImportError:
            print(f"  [ ] {pkg:15s}: NOT INSTALLED")
            
    print("=" * 50)

if __name__ == "__main__":
    main()
