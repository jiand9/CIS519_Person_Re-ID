import numpy as np
from torchvision import datasets, transforms
import torch
import os


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def get_dataloader(batch_size):
    transform_list_train = [
        transforms.Resize(size=(384, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_list_test = [
        transforms.Resize(size=(384, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    train = datasets.ImageFolder('./dataset/train', transforms.Compose(transform_list_train))
    query = datasets.ImageFolder('./dataset/query', transforms.Compose(transform_list_test))
    gallery = datasets.ImageFolder('./dataset/gallery', transforms.Compose(transform_list_test))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
    query_loader = torch.utils.data.DataLoader(query, batch_size=batch_size * 16, shuffle=False, num_workers=4)
    gallery_loader = torch.utils.data.DataLoader(gallery, batch_size=batch_size * 16, shuffle=False, num_workers=4)

    return train_loader, query_loader, gallery_loader


def get_camera_label(images):
    cameras = []
    labels = []
    for path, _ in images:
        label = int(path.split('_')[0][-4:])
        camera = int(path.split('_')[1][1])

        labels.append(label)
        cameras.append(camera)

    return np.array(cameras), np.array(labels)
