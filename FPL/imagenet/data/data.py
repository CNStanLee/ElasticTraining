import os
import sys
import subprocess
import io
import importlib.util
import tensorflow as tf

# ----------------------------------------------------------------------------
# Global HuggingFace cache location (before ANY datasets import!)
# ----------------------------------------------------------------------------

HF_CACHE_ROOT = "/mnt/data6/hf_cache"
os.makedirs(HF_CACHE_ROOT, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_CACHE_ROOT)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_CACHE_ROOT, "datasets"))


# ---------------------------------------------------------------------------
# Auto-download helpers (HuggingFace datasets)
# ---------------------------------------------------------------------------

def _pip_install(*packages: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages]
    )


def _ensure_hf_libs() -> None:
    """Install huggingface_hub and datasets if not present (without importing)."""
    missing = []
    for pkg in ["datasets", "huggingface_hub"]:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    if missing:
        print(f"[data] Installing: {' '.join(missing)} ...")
        _pip_install(*missing)


def _check_hf_token() -> str:
    """
    Return a HuggingFace token from env or cached login.
    Exit with instructions if none is found.
    """
    # 1. Environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token

    # 2. Cached token from `huggingface-cli login`
    cache_token = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(cache_token):
        with open(cache_token) as f:
            token = f.read().strip()
        if token:
            return token

    print(
        "\n[data] HuggingFace token not found.\n"
        "  One-time setup (30 seconds):\n"
        "  1. Go to https://huggingface.co/settings/tokens\n"
        "  2. Create a token (Read access is enough)\n"
        "  3. Accept the dataset terms at:\n"
        "     https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
        "  4. Run ONE of the following:\n"
        "       export HF_TOKEN=hf_xxxxxxxxxxxx   # temporary\n"
        "       huggingface-cli login              # permanent cached login\n"
        "  Then re-run the training script.\n"
    )
    sys.exit(1)


def download_imagenet(dest_dir: str) -> None:
    """
    Download ImageNet-1k from HuggingFace (ILSVRC/imagenet-1k) and
    organise it into the standard folder layout::

        dest_dir/
            train/<synset_id>/<img>.JPEG
            val/<synset_id>/<img>.JPEG

    Requires a HuggingFace account + token with access to
    https://huggingface.co/datasets/ILSVRC/imagenet-1k
    """
    _ensure_hf_libs()
    token = _check_hf_token()

    # At this point HF_HOME / HF_DATASETS_CACHE are already set globally above.
    from datasets import load_dataset  # type: ignore

    os.makedirs(dest_dir, exist_ok=True)

    for split, out_subdir in [("train", "train"), ("validation", "val")]:
        out_dir = os.path.join(dest_dir, out_subdir)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            print(f"[data] {out_subdir}/ already exists, skipping download.")
            continue

        print(f"[data] Downloading ImageNet-1k {split} split from HuggingFace ...")
        ds = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            token=token,
            trust_remote_code=True,
        )

        # Class names are the synset IDs (e.g. 'n01440764')
        class_names = ds.features["label"].names

        print(f"[data] Saving {len(ds):,} images to {out_dir} ...")
        os.makedirs(out_dir, exist_ok=True)

        for i, example in enumerate(ds):
            synset = class_names[example["label"]]
            class_dir = os.path.join(out_dir, synset)
            os.makedirs(class_dir, exist_ok=True)

            img = example["image"]  # PIL Image
            img_path = os.path.join(class_dir, f"{i:08d}.JPEG")
            img.save(img_path, format="JPEG", quality=95)

            if (i + 1) % 10000 == 0:
                print(f"[data]   {i+1:,} / {len(ds):,} images saved")

        print(f"[data] {out_subdir}/ done.")

    print(f"[data] ImageNet ready at: {dest_dir}")


# ---------------------------------------------------------------------------
# Path resolver
# ---------------------------------------------------------------------------

def resolve_data_dir(requested: str) -> str:
    """
    Return a valid ImageNet root directory (with ``train/`` and ``val/``).

    If the path does not exist or is incomplete, ImageNet is automatically
    downloaded from HuggingFace into ``requested``.
    """

    def _valid(p):
        return (
            os.path.isdir(p)
            and os.path.isdir(os.path.join(p, "train"))
            and os.path.isdir(os.path.join(p, "val"))
        )

    requested = os.path.abspath(requested)

    if _valid(requested):
        print(f"[data] ImageNet found at: {requested}")
        return requested

    print(f'[data] ImageNet not found at "{requested}". Starting auto-download...')
    download_imagenet(requested)

    if not _valid(requested):
        print("[data] ERROR: Download completed but train/ or val/ still missing.")
        sys.exit(1)

    return requested


# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------

_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def _normalize(image, label):
    """Scale to [0,1] then apply ImageNet channel normalisation."""
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - _MEAN) / _STD
    return image, label


def _augment_train(image, label):
    """Random resized crop + horizontal flip for training."""
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_crop(image, [224, 224, 3])
    image = tf.image.random_flip_left_right(image)
    return image, label


def _preprocess_val(image, label):
    """Resize to 256 then centre-crop to 224 for validation."""
    image = tf.image.resize(image, [256, 256])
    image = tf.image.crop_to_bounding_box(image, 16, 16, 224, 224)
    return image, label


def get_imagenet_dataloaders(
    data_dir,
    batch_size=256,
    num_workers=8,
    pin_memory=True,  # kept for API compatibility
):
    """
    Build ImageNet train and validation tf.data pipelines.

    Args:
        data_dir (str): Root ImageNet directory with 'train/' and 'val/' sub-folders.
        batch_size (int): Batch size.
        num_workers (int): Number of parallel map workers (AUTOTUNE if <= 0).
        pin_memory (bool): Unused, kept for API compatibility.

    Returns:
        train_ds (tf.data.Dataset), val_ds (tf.data.Dataset), num_classes (int)
    """
    num_parallel = tf.data.AUTOTUNE if num_workers <= 0 else num_workers

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    class_names = sorted(
        [
            d
            for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ]
    )
    num_classes = len(class_names)

    def _make_dataset(directory, is_train):
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=(256, 256),
            batch_size=None,
            shuffle=is_train,
            seed=42,
        )
        if is_train:
            ds = ds.map(_augment_train, num_parallel_calls=num_parallel)
        else:
            ds = ds.map(_preprocess_val, num_parallel_calls=num_parallel)
        ds = ds.map(_normalize, num_parallel_calls=num_parallel)
        ds = ds.batch(batch_size, drop_remainder=is_train)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _make_dataset(train_dir, is_train=True)
    val_ds = _make_dataset(val_dir, is_train=False)

    return train_ds, val_ds, num_classes