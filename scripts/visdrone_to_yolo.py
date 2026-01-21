import os
from pathlib import Path

# =========================
# 路径配置（按你当前机器）
# =========================
ROOT = Path("/root/workspace/data/datasets")

VISDRONE_ROOT = ROOT / "VisDrone"
OUTPUT_ROOT = ROOT / "VisDrone_YOLO"

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
}

# =========================
# VisDrone 类别（0-based）
# =========================
# VisDrone label format:
# x, y, w, h, score, class_id, truncation, occlusion
# class_id ∈ [1, 10]
NUM_CLASSES = 10


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def convert_one_txt(
    ann_path: Path,
    img_w: int,
    img_h: int,
    save_path: Path,
):
    """
    Convert single VisDrone annotation txt to YOLO txt
    """
    lines_out = []

    with ann_path.open("r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue

            x, y, w, h = map(float, parts[:4])
            class_id = int(parts[5])

            # VisDrone: ignore region or invalid label
            if class_id <= 0 or class_id > NUM_CLASSES:
                continue

            # convert to YOLO (0-based)
            cls = class_id - 1

            # bbox -> center format
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            bw = w / img_w
            bh = h / img_h

            # filter invalid boxes
            if bw <= 0 or bh <= 0:
                continue

            lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    with save_path.open("w") as f:
        f.write("\n".join(lines_out))


def get_image_size(img_path: Path):
    """
    Read image size without cv2 / PIL
    (JPEG & PNG supported)
    """
    with img_path.open("rb") as f:
        data = f.read(24)

    if data.startswith(b"\211PNG\r\n\032\n"):
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        return w, h

    if data[0:2] == b"\xff\xd8":  # JPEG
        f = img_path.open("rb")
        f.read(2)
        while True:
            byte = f.read(1)
            if not byte:
                break
            if byte == b"\xff":
                marker = f.read(1)
                if marker in [b"\xc0", b"\xc2"]:
                    f.read(3)
                    h = int.from_bytes(f.read(2), "big")
                    w = int.from_bytes(f.read(2), "big")
                    return w, h
                else:
                    size = int.from_bytes(f.read(2), "big")
                    f.read(size - 2)
        f.close()

    raise RuntimeError(f"Cannot read image size: {img_path}")


def convert_split(split: str, folder_name: str):
    print(f"\n=== Converting {split} ===")

    img_dir = VISDRONE_ROOT / folder_name / "images"
    ann_dir = VISDRONE_ROOT / folder_name / "annotations"

    out_img_dir = OUTPUT_ROOT / split / "images"
    out_lbl_dir = OUTPUT_ROOT / split / "labels"

    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    images = sorted(img_dir.glob("*.jpg"))
    print(f"Found {len(images)} images")

    for img_path in images:
        ann_path = ann_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue

        w, h = get_image_size(img_path)

        save_txt = out_lbl_dir / ann_path.name
        convert_one_txt(ann_path, w, h, save_txt)

        # copy image (软链接，省空间)
        target_img = out_img_dir / img_path.name
        if not target_img.exists():
            os.symlink(img_path, target_img)

    print(f"{split} done.")


def main():
    ensure_dir(OUTPUT_ROOT)

    for split, folder in SPLITS.items():
        convert_split(split, folder)

    print("\nAll done. YOLO dataset is ready.")


if __name__ == "__main__":
    main()