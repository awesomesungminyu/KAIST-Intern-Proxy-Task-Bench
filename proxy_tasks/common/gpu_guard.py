import os, sys, torch

def require_cuda(tag=""):
    """GPU를 못 잡으면 CPU로 떨어지지 않고 즉시 종료한다."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None and vis.strip() == "":
        sys.exit(f"[GPU-GUARD]{tag} FAIL: CUDA_VISIBLE_DEVICES 비어 있음")
    if not torch.cuda.is_available():
        sys.exit(f"[GPU-GUARD]{tag} FAIL: torch.cuda.is_available()==False")
    try:
        torch.zeros(1).cuda(); torch.cuda.synchronize()
    except Exception as e:
        sys.exit(f"[GPU-GUARD]{tag} FAIL: CUDA init 실패: {e}")
    cap = torch.cuda.get_device_capability(0)
    print(f"[GPU-GUARD]{tag} OK {torch.cuda.get_device_name(0)} "
          f"sm_{cap[0]}{cap[1]} pid={os.getpid()}", flush=True)
    return "cuda"
