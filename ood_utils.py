"""
Helper functions or OOD experiments.
"""
import torch
from torchvision.datasets import CelebA, CIFAR10, SVHN, DTD, CIFAR100
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from mpi4py import MPI

import argparse
import os
import subprocess
from pathlib import Path
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO

def download_celeba_alternative(data_dir):
    """Attempts to download CelebA from alternative sources if it's not found."""
    celeba_path = Path(data_dir) / "celeba"
    img_path = celeba_path / "img_align_celeba"

    if img_path.exists():
        print("CelebA dataset found locally. Skipping download.")
        return

    print("Dataset not found. Attempting alternative sources...")

    try:
        print("Trying to download and extract ...")
        url = "https://cseweb.ucsd.edu/~weijian/static/datasets/celeba/img_align_celeba.zip"
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure the download succeeded

        with ZipFile(BytesIO(response.content)) as zfile:
            zfile.extractall(celeba_path)

        print("Dataset downloaded and extracted successfully.")

        # Move files from nested folder to desired folder structure
        nested_dir = celeba_path / "img_align_celeba" / "img_align_celeba"

        if nested_dir.exists():
            for item in nested_dir.iterdir():
                shutil.move(str(item), celeba_path / "img_align_celeba")
            nested_dir.rmdir()  # Remove empty nested directory

        print("CelebA downloaded successfully.")
        return
    except Exception as e:
        print("CelebA download failed:", str(e))


def get_interpolation_mode(mode):
    if mode == 'bilinear':
        return transforms.InterpolationMode.BILINEAR
    elif mode =='nearest':
        return transforms.InterpolationMode.NEAREST
    elif mode =='nearest_exact':
        return transforms.InterpolationMode.NEAREST_EXACT
    elif mode =='bicubic':
        return transforms.InterpolationMode.BICUBIC
    elif mode =='box':
        return transforms.InterpolationMode.BOX
    elif mode =='hamming':
        return transforms.InterpolationMode.HAMMING
    elif mode =='lanczos':
        return transforms.InterpolationMode.LANCZOS
    else:
        print('not a valid interpolation mode')
        exit()

def build_subset_per_process(dataset):
    """
    Partitions dataset so each process (GPU) trains on a unique subset.
    """
    n_processes  = MPI.COMM_WORLD.Get_size()
    n_current_rank = MPI.COMM_WORLD.Get_rank()
    n_indices = torch.arange(0, len(dataset), dtype=int)

    indices_chunks = torch.chunk(n_indices, chunks=n_processes)
    indices_for_current_rank = indices_chunks[n_current_rank]
    subset = Subset(dataset, indices_for_current_rank)
    return subset


def yield_(loader):
    while True:
        yield from loader


def load_celeba(data_dir, batch_size, image_size, train=False, interpolation_mode='bilinear', shuffle=True):
    # Ensure dataset is available
    download_celeba_alternative(data_dir)

    transform = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize((image_size, image_size), interpolation=get_interpolation_mode(interpolation_mode)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    dataset = CelebA(data_dir, download=False, transform=transform, split='train' if train else 'test')
    subset = build_subset_per_process(dataset)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=False)
    return loader

def load_celeba_resized(data_dir, batch_size, image_size, train=False, interpolation_mode='bilinear', shuffle=True):
    transform = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize((32, 32), interpolation=get_interpolation_mode(interpolation_mode)),
        transforms.Resize((image_size, image_size), interpolation=get_interpolation_mode(interpolation_mode)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    dataset = CelebA(data_dir, download=True, transform=transform, split='train' if train else 'test')
    subset = build_subset_per_process(dataset)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=False)
    return loader


def load_cifar10(data_dir, batch_size, image_size, train=False, interpolation_mode='bilinear', shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=get_interpolation_mode(interpolation_mode)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    dataset = CIFAR10(data_dir, download=True, transform=transform, train=train)
    subset = build_subset_per_process(dataset)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=False)
    return loader


def load_svhn(data_dir, batch_size, image_size, train=False, interpolation_mode='bilinear', shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=get_interpolation_mode(interpolation_mode)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    dataset = SVHN(data_dir, download=True, transform=transform, split='train' if train else 'test')
    subset = build_subset_per_process(dataset)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=False)
    return loader

def load_textures(data_dir, batch_size, image_size, train=False, interpolation_mode='bilinear', shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=get_interpolation_mode(interpolation_mode)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    dataset = DTD(data_dir, download=True, transform=transform, split='train' if train else 'test')
    subset = build_subset_per_process(dataset)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=False)
    return loader

def load_textures_resized(data_dir, batch_size, image_size, train=False, interpolation_mode='bilinear', shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((32, 32), interpolation=get_interpolation_mode(interpolation_mode)),
        transforms.Resize((image_size, image_size), interpolation=get_interpolation_mode(interpolation_mode)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    dataset = DTD(data_dir, download=True, transform=transform, split='train' if train else 'test')
    subset = build_subset_per_process(dataset)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=False)
    return loader

def load_cifar100(data_dir, batch_size, image_size, train=False, interpolation_mode='bilinear', shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=get_interpolation_mode(interpolation_mode)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    dataset = CIFAR100(data_dir, download=True, transform=transform, train=train)
    subset = build_subset_per_process(dataset)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=False)
    return loader


def load_data(dataset, data_dir, batch_size, image_size, train, interpolation_mode='bilinear', shuffle=True):

    if dataset == "cifar10":
        dataloader = load_cifar10(data_dir, batch_size, image_size, train, interpolation_mode, shuffle)
    elif dataset == "celeba":
        dataloader = load_celeba(data_dir, batch_size, image_size, train, interpolation_mode, shuffle)
    elif dataset == "celeba_resized":
        dataloader = load_celeba_resized(data_dir, batch_size, image_size, train, interpolation_mode, shuffle)
    elif dataset == "svhn":
        dataloader = load_svhn(data_dir, batch_size, image_size, train, interpolation_mode, shuffle)
    elif dataset == "textures":
        dataloader = load_textures(data_dir, batch_size, image_size, train, interpolation_mode, shuffle)
    elif dataset == "textures_resized":
        dataloader = load_textures_resized(data_dir, batch_size, image_size, train, interpolation_mode, shuffle)
    elif dataset == "cifar100":
        dataloader = load_cifar100(data_dir, batch_size, image_size, train, interpolation_mode, shuffle)
    else:
        print("Wrong ID dataset!")
        exit()
    return dataloader


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
