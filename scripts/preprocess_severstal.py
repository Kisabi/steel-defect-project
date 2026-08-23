"""
Препроцессинг датасета Severstal: Steel Defect Detection.

Что делает:
1. Читает train.csv (ImageId, ClassId, EncodedPixels)
2. Декодирует RLE-маски и объединяет их в единую multi-class маску на изображение
   (значения пикселей: 0 = фон, 1-4 = класс дефекта)
3. Сохраняет маски как PNG в data/processed/masks/
4. Делает train/val/test split на уровне ImageId (не строк!), чтобы одно изображение
   целиком попадало в один сплит, и сохраняет индексы в data/splits/*.csv

Запуск (внутри Docker-контейнера с активным venv):
    python scripts/preprocess_severstal.py \
        --raw-dir data/raw/severstal-steel-defect-detection \
        --output-dir data/processed \
        --splits-dir data/splits

Ожидаемая структура raw-dir:
    train.csv
    train_images/*.jpg
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_HEIGHT = 256
IMG_WIDTH = 1600
NUM_CLASSES = 4  # классы дефектов 1..4, 0 зарезервирован под фон


def rle_decode(rle_string: str, shape=(IMG_HEIGHT, IMG_WIDTH)) -> np.ndarray:
    """Декодирует одну RLE-строку в бинарную маску формы `shape`.

    Severstal хранит RLE в column-major (Fortran) порядке — это важно,
    иначе маска получится "рассыпанной" по горизонтали.
    """
    if not isinstance(rle_string, str) or rle_string.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_string.strip().split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # RLE в датасете 1-indexed
    ends = starts + lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1

    return mask.reshape(shape, order="F")


def build_multiclass_mask(rows: pd.DataFrame) -> np.ndarray:
    """Объединяет все RLE-строки одного ImageId в единую маску 0..4.

    При пересечении классов (редко, но бывает) побеждает класс с бОльшим
    номером — это осознанное упрощение, для диплома стоит явно упомянуть
    как ограничение методологии.
    """
    combined = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    for _, row in rows.iterrows():
        class_id = int(row["ClassId"])
        binary_mask = rle_decode(row["EncodedPixels"], (IMG_HEIGHT, IMG_WIDTH))
        combined[binary_mask == 1] = class_id
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--splits-dir", type=str, required=True)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    splits_dir = Path(args.splits_dir)

    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "train.csv"
    print(f"Читаю {csv_path} ...")
    df = pd.read_csv(csv_path)

    expected_cols = {"ImageId", "ClassId", "EncodedPixels"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"Ожидались колонки {expected_cols}, а в файле: {list(df.columns)}. "
            "Проверь формат train.csv — возможно, это старая версия "
            "(ImageId_ClassId, EncodedPixels), тогда нужно сначала разбить ImageId_ClassId."
        )

    # только строки с непустой RLE (иначе дефекта на этом классе нет)
    df_with_defects = df.dropna(subset=["EncodedPixels"])

    image_ids = df["ImageId"].unique()
    print(f"Всего уникальных изображений: {len(image_ids)}")

    ids_with_defects = df_with_defects["ImageId"].unique()
    print(f"Изображений с хотя бы одним дефектом: {len(ids_with_defects)}")

    records = []
    for image_id in image_ids:
        rows = df_with_defects[df_with_defects["ImageId"] == image_id]

        if len(rows) > 0:
            mask = build_multiclass_mask(rows)
        else:
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

        mask_filename = Path(image_id).stem + ".png"
        Image.fromarray(mask, mode="L").save(masks_dir / mask_filename)

        records.append(
            {
                "image_id": image_id,
                "image_path": f"train_images/{image_id}",
                "mask_path": f"masks/{mask_filename}",
                "has_defect": int(len(rows) > 0),
            }
        )

    manifest = pd.DataFrame(records)
    print(f"Обработано изображений: {len(manifest)}")

    # split на уровне изображений, стратифицированный по наличию дефекта,
    # чтобы train/val/test имели сопоставимую долю "пустых" изображений
    train_val, test = train_test_split(
        manifest,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=manifest["has_defect"],
    )
    relative_val_size = args.val_size / (1 - args.test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=args.random_state,
        stratify=train_val["has_defect"],
    )

    train.to_csv(splits_dir / "train.csv", index=False)
    val.to_csv(splits_dir / "val.csv", index=False)
    test.to_csv(splits_dir / "test.csv", index=False)

    print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")
    print(f"Маски сохранены в {masks_dir}")
    print(f"Сплиты сохранены в {splits_dir}")


if __name__ == "__main__":
    main()