import torch
from turboquant import TurboQuantMSELayer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Allocating massive baseline tensor...")
    dummy_kv_cache = torch.randn(50000, 1536, device=device) 
    
    baseline_mb = dummy_kv_cache.element_size() * dummy_kv_cache.nelement() / (1024 ** 2)
    print(f"Baseline FP32 KV Cache Memory : {baseline_mb:.2f} MB")
    
    layer = TurboQuantMSELayer(d=1536, b=4).to(device)
    
    print("\nCompressing with TurboQuant...")
    packed_cache = layer.encode_inference(dummy_kv_cache)
    
    quantized_mb = packed_cache.element_size() * packed_cache.nelement() / (1024 ** 2)
    print(f"TurboQuant 4-bit KV Cache Memory: {quantized_mb:.2f} MB")
    
    print("\n========================================")
    print(f"MEMORY SAVED: {baseline_mb - quantized_mb:.2f} MB")
    print(f"COMPRESSION RATIO: {baseline_mb / quantized_mb:.1f}x")
    print("========================================")

if __name__ == "__main__":
    main()
