"""QC-Assist application: `python -m qc_assist`, then open http://127.0.0.1:8000.

Defaults come from the `app:` section of configs/default.yaml; any of them can be overridden:
    python -m qc_assist --source http://127.0.0.1:8080/shot.jpg     # phone camera (IP Webcam app)
    python -m qc_assist --source 0 --model models/pcb4.pt --backend torch
"""

import argparse
from pathlib import Path

from . import PROJECT_ROOT
from .config import load_config
from .dashboard import create_app
from .inspector import Inspector


def main():
    ap = argparse.ArgumentParser(prog="python -m qc_assist", description="QC-Assist operator application")
    ap.add_argument("--config", help="default: configs/default.yaml")
    ap.add_argument("--source", help="camera index, image folder, or snapshot/stream URL")
    ap.add_argument("--model", help="trained product model (.pt); omit the file to start in teach mode")
    ap.add_argument("--backend", choices=["torch", "onnx", "hailo"])
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--max-fps", type=float, help="limit the inspection rate (e.g. for an image folder)")
    ap.add_argument("--loop", action="store_true", help="replay an image folder forever")
    args = ap.parse_args()

    cfg = load_config(args.config)
    app = {**{"source": "0", "model": None, "backend": "torch", "host": "127.0.0.1", "port": 8000},
           **(cfg.get("app") or {})}
    for k in ("source", "model", "backend", "host", "port"):
        if getattr(args, k) is not None:
            app[k] = getattr(args, k)

    model = Path(app["model"]) if app["model"] else None
    if model is not None and not model.is_absolute():
        model = PROJECT_ROOT / model
    if model is not None and not model.exists():
        print(f"no model at {model}: starting in teach mode")
        model = None
    backend = app["backend"]
    if model is not None and backend == "onnx" and not Path(str(model.with_suffix("")) + "_embedder.onnx").exists():
        backend = "torch"  # e.g. a taught model: no ONNX export yet
    if model is None:
        backend = "torch"

    inspector = Inspector(cfg, str(app["source"]), str(model) if model else None, args.max_fps, args.loop, backend)
    inspector.start()
    print(f"QC-Assist: {app['source']} | model {model.name if model else 'none (teach mode)'} | {backend} | "
          f"Llama {'ExecuTorch' if inspector.advisor.available else 'not configured'}")
    print(f"open http://{'127.0.0.1' if app['host'] == '0.0.0.0' else app['host']}:{app['port']}")
    create_app(inspector).run(host=app["host"], port=int(app["port"]), threaded=True)


if __name__ == "__main__":
    main()
