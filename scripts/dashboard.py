"""Start the operator dashboard.

    python scripts/dashboard.py --source 0 --model models/pcb1.pt          # live camera
    python scripts/dashboard.py --source data/visa/pcb1/test --model models/pcb1.pt --max-fps 2 --loop
    python scripts/dashboard.py --source 0                                 # no model yet: use "Teach new board"

Then open http://<device-ip>:8000 (or http://localhost:8000 on the device).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_assist.config import load_config
from qc_assist.dashboard import create_app
from qc_assist.inspector import Inspector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="camera index or image folder")
    ap.add_argument("--model")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx", "hailo"],
                    help="where the CNN runs: PyTorch or ONNX Runtime on the CPU, or the Hailo NPU")
    ap.add_argument("--config")
    ap.add_argument("--max-fps", type=float, help="slow down folder replay so people can watch")
    ap.add_argument("--loop", action="store_true", help="replay the folder forever")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to view from another device on the LAN")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    inspector = Inspector(load_config(args.config), args.source, args.model, args.max_fps, args.loop, args.backend)
    inspector.start()
    create_app(inspector).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
