import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse

from domainbed import augmentation_algorithms
from domainbed import augmentation_hparams_registry


class AugmentationVisualizer:
    def __init__(self, dataset_name: str, save_dir: str):
        """
        Visualizer class for data augmentations
        Args:
            dataset_name: Name of the dataset for hyperparameter selection
            save_dir: Directory to save visualization results
        """
        self.dataset_name = dataset_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.denorm = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def visualize_augmentation(self, image_dir: str, batch_size: int = 12):
        """
        Run all augmentations and save results:
        1. Dataset-level augmentation: Applied when creating dataset
        2. Batch-level augmentation: Applied to minibatches during processing
        
        Args:
            image_dir: Directory containing input images
            batch_size: Batch size for processing
        """
        # Prepare raw dataset without any transformations
        base_dataset = datasets.ImageFolder(
            root=image_dir,
            transform=None  # No transformation at this stage
        )
        
        # Get all available augmentation classes
        aug_names = augmentation_algorithms.ALGORITHMS
        
        # Process each augmentation
        for aug_name in aug_names:
            print(f"Processing {aug_name}...")
            
            # Get hyperparameters for this specific augmentation
            hparams = augmentation_hparams_registry.default_hparams(
                aug_name,  # Pass single augmentation as list
                self.dataset_name
            )
            
            # 1. Dataset-level augmentation
            aug_class = getattr(augmentation_algorithms, aug_name)
            augmented_dataset = aug_class(base_dataset, hparams)
            
            # Create dataloader for augmented dataset
            dataloader = torch.utils.data.DataLoader(
                augmented_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # Get a batch
            x, y = next(iter(dataloader))
            x, y = x.to(self.device), y.to(self.device)
            
            # 2. Batch-level augmentation
            x, y = augmented_dataset.augment_batch(x, y)
            
            # Denormalize and clip to [0,1] range
            images = torch.stack([self.denorm(img) for img in x])
            images = torch.clamp(images, 0, 1)
            
            # Save individual images from the batch
            self.save_dir.mkdir(exist_ok=True)
            
            for j in range(batch_size):
                vutils.save_image(
                    images[j],
                    self.save_dir / f'{aug_name}_{j}.jpg'
                )
            
            # Save grid visualization
            vutils.save_image(
                images,
                self.save_dir / f'{aug_name}_grid.jpg',
                nrow=4
            )


def main():
    parser = argparse.ArgumentParser(description='Augmentation Visualization')
    parser.add_argument('--image_dir', type=str, default="./PACS_images",
                      help='Directory containing test images')
    parser.add_argument('--dataset', type=str, default="PACS",
                      help='Dataset name for hyperparameter selection')
    parser.add_argument('--output_dir', type=str, default="./visualize_augs",
                      help='Directory to save visualization results')
    parser.add_argument('--batch_size', type=int, default=12,
                      help='Batch size for processing')
    args = parser.parse_args()

    # Run visualization
    visualizer = AugmentationVisualizer(
        dataset_name=args.dataset,
        save_dir=args.output_dir
    )
    visualizer.visualize_augmentation(args.image_dir, args.batch_size)


if __name__ == "__main__":
    main()