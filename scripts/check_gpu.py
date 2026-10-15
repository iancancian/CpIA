import json
import platform
import subprocess
from datetime import datetime, timezone

import torch


def main() -> None:
    print("=== Diagnostico GPU ===")
    print(f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}")
    print(f"python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_disponivel: {torch.cuda.is_available()}")
    print(f"torch_cuda: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"memoria_total_gb: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")
        print(f"capability: {torch.cuda.get_device_capability(0)}")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        print(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"nvidia-smi indisponivel: {error}")
    print(json.dumps({"cuda": torch.cuda.is_available()}, ensure_ascii=True))


if __name__ == "__main__":
    main()
