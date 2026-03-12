import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


class MedicalDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transform=None):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    # -----------------------------
    # Load NIfTI volume
    # -----------------------------
    def load_volume(self, path):

        volume = nib.load(path)
        volume = volume.get_fdata()

        return volume

    # -----------------------------
    # Compute Signed Distance Function
    # -----------------------------
    def compute_sdf(self, mask):

        mask = mask.astype(np.bool_)

        posmask = mask
        negmask = ~mask

        dist_out = distance_transform_edt(negmask)
        dist_in = distance_transform_edt(posmask)

        sdf = dist_out - dist_in

        return sdf

    # -----------------------------
    # Generate normalized coordinates
    # -----------------------------
    def generate_coords(self, num_tokens=2048):

        coords = torch.rand(num_tokens, 3)

        return coords

    # -----------------------------
    # Get item
    # -----------------------------
    def __getitem__(self, idx):

        # Load image and mask
        image = self.load_volume(self.image_paths[idx])
        mask = self.load_volume(self.mask_paths[idx])

        # Compute SDF
        sdf = self.compute_sdf(mask)

        # Convert to tensors
        image = torch.tensor(image).float()
        mask = torch.tensor(mask).float()
        sdf = torch.tensor(sdf).float()

        # Add channel dimension
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        sdf = sdf.unsqueeze(0)

        # Apply transforms if any
        if self.transform:
            image, mask = self.transform(image, mask)

        # Generate coordinates for graph clustering
        coords = self.generate_coords()

        # Return batch dictionary
        return {
            "image": image,
            "mask": mask,
            "sdf": sdf,
            "coords": coords
        }