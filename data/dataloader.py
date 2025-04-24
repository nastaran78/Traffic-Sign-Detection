import os
from typing import List, Tuple, Dict
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class FasterRCNNDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 416):
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_paths = []
        self.label_paths = []

        images_dir = os.path.join(self.root_dir, 'images')
        for dirpath, _, files in os.walk(images_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.image_paths.append(os.path.join(dirpath, file))
                    self.label_paths.append(
                        os.path.join(dirpath.replace('/images', '/labels'), file.replace('.jpg', '.txt'))
                    )
            break

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load annotations
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.read().strip().split('\n')
                for line in lines:
                    if line:
                        cls, x_c, y_c, w, h = map(float, line.split(' '))
                        cls = int(cls)
                        x1 = (x_c - w / 2) * self.img_size
                        y1 = (y_c - h / 2) * self.img_size
                        x2 = (x_c + w / 2) * self.img_size
                        y2 = (y_c + h / 2) * self.img_size
                        annotations.append([cls, x1, y1, x2, y2])

        annotations = torch.tensor(annotations, dtype=torch.float32)
        if annotations.shape[0] == 0:
            return F.to_tensor(image), {"labels": torch.empty((0,), dtype=torch.int64), "boxes": torch.empty((0, 4))}

        return F.to_tensor(image), {"labels": annotations[:, 0].long(), "boxes": annotations[:, 1:]}


def collate_fn(batch: List[Tuple[Tensor, Dict[str, Tensor]]]) -> Tuple[Tuple[Tensor], Tuple[Dict[str, Tensor]]]:
    return tuple(zip(*batch))
