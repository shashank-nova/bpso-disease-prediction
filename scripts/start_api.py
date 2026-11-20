"""Train the demo inference artifact, then start the API server."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    model_path = Path(os.environ.get("BPSO_MODEL", "/tmp/disease_model.joblib"))
    model_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/train_model.py",
            "--config",
            "configs/reduced.yaml",
            "--dataset",
            "breast_cancer",
            "--output",
            str(model_path),
        ],
        check=True,
    )

    os.environ["BPSO_MODEL"] = str(model_path)
    uvicorn.run(
        "bpso_fs.api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
    )


if __name__ == "__main__":
    main()
