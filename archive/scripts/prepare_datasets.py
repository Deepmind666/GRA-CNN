import os
import requests
import zipfile
import torchvision.datasets as datasets
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def prepare_cifar100(root='./data'):
    print("Checking/Downloading CIFAR-100...")
    if not os.path.exists(root):
        os.makedirs(root)
    # Torchvision handles the download logic check
    datasets.CIFAR100(root=root, train=True, download=True)
    datasets.CIFAR100(root=root, train=False, download=True)
    print("CIFAR-100 ready.")

def prepare_tinyimagenet(root='./data'):
    print("Checking/Downloading Tiny-ImageNet...")
    if not os.path.exists(root):
        os.makedirs(root)
    
    dataset_path = os.path.join(root, 'tiny-imagenet-200')
    zip_path = os.path.join(root, 'tiny-imagenet-200.zip')
    
    if os.path.exists(dataset_path):
        print("Tiny-ImageNet already extracted.")
        return

    if not os.path.exists(zip_path):
        print("Downloading Tiny-ImageNet zip...")
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        download_file(url, zip_path)
    
    print("Extracting Tiny-ImageNet...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    
    # Tiny-ImageNet val folder structure needs adjustment for ImageFolder
    # Current: val/images/*.JPEG, val/val_annotations.txt
    # Need: val/class_id/*.JPEG
    
    val_dir = os.path.join(dataset_path, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_annot_file = os.path.join(val_dir, 'val_annotations.txt')
    
    if os.path.exists(val_img_dir):
        print("Restructuring validation set...")
        with open(val_annot_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_id = parts[1]
                
                class_dir = os.path.join(val_dir, class_id)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                
                src = os.path.join(val_img_dir, img_name)
                dst = os.path.join(class_dir, img_name)
                if os.path.exists(src):
                    os.rename(src, dst)
        
        # Remove empty images folder and txt
        try:
            os.rmdir(val_img_dir)
        except:
            pass
            
    print("Tiny-ImageNet ready.")

if __name__ == "__main__":
    prepare_cifar100(root='C:/GRA-CNN/project/data')
    prepare_tinyimagenet(root='C:/GRA-CNN/project/data')
