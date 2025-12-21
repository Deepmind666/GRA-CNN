"""
ImageNet Subset Downloader
Downloads ImageNet validation set and creates a 100-class subset for experiments.

Options:
1. Download from HuggingFace (recommended, faster)
2. Download from ILSVRC2012 directly (requires registration)
3. Use torchvision.datasets.ImageNet (if already have data)
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

def download_imagenet_from_huggingface(save_dir, num_classes=100):
    """
    Download ImageNet subset using HuggingFace datasets
    This downloads a smaller subset suitable for experiments
    """
    try:
        from datasets import load_dataset
        import torchvision.transforms as transforms
        from PIL import Image
        from tqdm import tqdm
    except ImportError:
        print("Installing required packages...")
        os.system(f"{sys.executable} -m pip install datasets pillow tqdm")
        from datasets import load_dataset
        import torchvision.transforms as transforms
        from PIL import Image
        from tqdm import tqdm
    
    print(f"Downloading ImageNet-1K from HuggingFace...")
    print("This may take a while (full dataset is ~150GB)...")
    
    # Load validation split only for initial experiments
    dataset = load_dataset(
        "imagenet-1k", 
        split="validation",
        trust_remote_code=True,
        token=True  # Requires HuggingFace login
    )
    
    # Create directory structure
    val_dir = os.path.join(save_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    
    # Get unique labels
    labels = list(set(dataset["label"]))[:num_classes]
    
    # Create class directories and save images
    for item in tqdm(dataset, desc="Saving images"):
        if item["label"] in labels:
            class_dir = os.path.join(val_dir, f"class_{item['label']:04d}")
            os.makedirs(class_dir, exist_ok=True)
            
            img = item["image"]
            img_path = os.path.join(class_dir, f"{len(os.listdir(class_dir)):05d}.JPEG")
            img.save(img_path)
    
    print(f"Dataset saved to {save_dir}")
    return save_dir


def download_tiny_imagenet(save_dir):
    """
    Download Tiny-ImageNet-200 as alternative
    Smaller (500MB) but still useful for experiments
    """
    import urllib.request
    import zipfile
    from tqdm import tqdm
    
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(save_dir, "tiny-imagenet-200.zip")
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("Downloading Tiny-ImageNet-200...")
    
    # Download with progress bar
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    
    os.remove(zip_path)
    print(f"Tiny-ImageNet saved to {os.path.join(save_dir, 'tiny-imagenet-200')}")
    
    return os.path.join(save_dir, "tiny-imagenet-200")


def download_imagenet_100_kaggle(save_dir):
    """
    Download ImageNet-100 subset from Kaggle
    Requires Kaggle API credentials
    """
    try:
        import kaggle
        print("Downloading ImageNet-100 from Kaggle...")
        kaggle.api.dataset_download_files(
            "ambityga/imagenet100", 
            path=save_dir, 
            unzip=True
        )
        print(f"ImageNet-100 saved to {save_dir}")
        return save_dir
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Please ensure Kaggle API credentials are set up")
        return None


def create_imagenet_subset_from_existing(source_dir, target_dir, num_classes=100):
    """
    Create ImageNet-100 subset from existing full ImageNet
    """
    from tqdm import tqdm
    
    for split in ['train', 'val']:
        source_split = os.path.join(source_dir, split)
        target_split = os.path.join(target_dir, split)
        
        if not os.path.exists(source_split):
            print(f"Warning: {source_split} not found")
            continue
            
        classes = sorted(os.listdir(source_split))[:num_classes]
        
        for cls in tqdm(classes, desc=f"Copying {split}"):
            src = os.path.join(source_split, cls)
            dst = os.path.join(target_split, cls)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
    
    print(f"ImageNet-{num_classes} created at {target_dir}")
    return target_dir


def main():
    parser = argparse.ArgumentParser(description="Download ImageNet dataset")
    parser.add_argument("--method", type=str, default="tiny", 
                       choices=["huggingface", "tiny", "kaggle", "subset"],
                       help="Download method")
    parser.add_argument("--save-dir", type=str, default="./data",
                       help="Directory to save dataset")
    parser.add_argument("--source-dir", type=str, default=None,
                       help="Source ImageNet directory (for subset method)")
    parser.add_argument("--num-classes", type=int, default=100,
                       help="Number of classes for subset")
    args = parser.parse_args()
    
    if args.method == "huggingface":
        download_imagenet_from_huggingface(args.save_dir, args.num_classes)
    elif args.method == "tiny":
        download_tiny_imagenet(args.save_dir)
    elif args.method == "kaggle":
        download_imagenet_100_kaggle(args.save_dir)
    elif args.method == "subset":
        if args.source_dir is None:
            print("Error: --source-dir required for subset method")
            return
        create_imagenet_subset_from_existing(
            args.source_dir, 
            os.path.join(args.save_dir, f"imagenet{args.num_classes}"),
            args.num_classes
        )


if __name__ == "__main__":
    main()
