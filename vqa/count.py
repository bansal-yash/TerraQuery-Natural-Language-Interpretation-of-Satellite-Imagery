#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
from PIL import Image

def count_pixels(mask_path: Path) -> int:
    arr = np.array(Image.open(mask_path).convert("L"))
    return int(np.count_nonzero(arr))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <mask_path>")
    mask_file = Path(sys.argv[1])
    if not mask_file.exists():
        sys.exit(f"Mask not found: {mask_file}")
    total = count_pixels(mask_file)
    print(f"{mask_file}: {total} foreground pixels")