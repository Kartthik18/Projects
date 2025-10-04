import gc
import torch
import weakref

def cleanup_gpu_memory(obj=None, verbose=False):
    """Clean up CUDA memory (use after training/inference)."""
    if not torch.cuda.is_available():
        if verbose:
            print("[INFO] CUDA not available.")
        return

    def stats():
        return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

    torch.cuda.synchronize()
    if verbose:
        alloc, reserv = stats()
        print(f"[Before] Alloc: {alloc/1024**2:.2f} MB | Reserv: {reserv/1024**2:.2f} MB")

    if obj is not None:
        ref = weakref.ref(obj)
        del obj
        gc.collect()
        if ref() is not None and verbose:
            print("[WARN] Object not fully released.")

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    if verbose:
        alloc, reserv = stats()
        print(f"[After]  Alloc: {alloc/1024**2:.2f} MB | Reserv: {reserv/1024**2:.2f} MB")
