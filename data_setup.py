"""
contains functionality for creating pytorch dataloaders for image classification data
"""
import os 
import torch
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 
from pathlib import Path 
import requests
import zipfile

NUM_WORKERS = os.cpu_count()

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path

def create_dataloaders(
    train_dir: str,  
    test_dir: str, 
    train_transform: transforms.Compose, 
    test_transform: transforms.Compose,
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
    """
    creates training and testing DataLoaders. 

    Takes in a training directory and testing directory path and turns them
    into pytorch datasets and then into pytorch dataloaders. 

    Args:
        train_dir: path to training directory. 
        test_dir: path to testing directory 
        transform: torchvision transforms to perform on training and testing data. 
        batch_size: number of samples per batch in each of the dataloaders. 
        num_workers: an integer for number of workers per dataloader.

    returns: 
        A tuple of (train_dataloader, test_dataloader, class_names).
        where class_names is a list of the target classes. 

        Example usage: 
            train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir, 
                                                                                test_dir=path/to/test_dir, 
                                                                                transform=some_transform,
                                                                                batch_size=32,
                                                                                num_workers=4)                                                                                                                 
    """

    # use ImageFolder to create datasets 
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)

    test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # get class names 
    class_names = train_data.classes 

    # turn images into dataloaders 
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names