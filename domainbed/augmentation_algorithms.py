from domainbed.lib_augmentations.ipmix import ipmix
import torch
import torch.nn as nn
import random
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Tuple, Any, List
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder

from domainbed.lib_augmentations import augmix, strategy_augment, randconv
from domainbed.lib_augmentations.ipmix import ipmix

from domainbed.lib_augmentations.stylized.function import adaptive_instance_normalization
import domainbed.lib_augmentations.stylized.net as net
from pathlib import Path

import kornia
from PIL import ImageFile
from domainbed.lib_augmentations.cartoongan.Transformer import Transformer
ImageFile.LOAD_TRUNCATED_IMAGES = True

from domainbed.lib_augmentations.edgedetection.edge_detection import EdgeDetector

ALGORITHMS = [
	'Original', 
    'Test',
	'AugMix', 
	'MixUp',  
	'CutMix',  
	'IPMix',  
	'TrivialAugment',
	'RandAugment',
	'RandConv',
	'Stylized',
	'EdgeDetection',
	'CartoonGAN',
]

class Original(Dataset):
	"""Base class for all augmentation techniques"""
	def __init__(self, dataset: Dataset, hparams: dict):
		self.dataset = dataset
		self.hparams = hparams
		self.transform = transforms.Compose([
			transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
			transforms.RandomGrayscale(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], 
							   std=[0.229, 0.224, 0.225])
		])

	def __getitem__(self, i: int) -> Tuple[torch.Tensor, Any]:
		x, y = self.dataset[i]
		return self.transform(x), y

	def __len__(self) -> int:
		return len(self.dataset)

	def augment_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""No batch augmentation for Original class"""
		return x, y


class Test(Original):
	"""Test-time preprocessing without augmentations"""
	def __init__(self, dataset: Dataset, hparams: dict):
		super().__init__(dataset, hparams)
		self.transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], 
							   std=[0.229, 0.224, 0.225])
		])


class MixUp(Original):
	"""MixUp augmentation"""
	def augment_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Apply MixUp augmentation to the batch"""
		if not self.hparams.get('mixup_alpha'):
			return x, y

		batch_size = x.size(0)
		device = x.device
		alpha = self.hparams['mixup_alpha']

		# Generate mixing weights from beta distribution
		lam = np.random.beta(alpha, alpha)

		# Create shuffled indices
		index = torch.randperm(batch_size, device=device)

		# Perform mixup
		mixed_x = lam * x + (1 - lam) * x[index]

		return mixed_x, y


class CutMix(Original):
	"""CutMix augmentation"""
	def _rand_bbox(self, size: Tuple[int, int, int, int], lam: float) -> Tuple[int, int, int, int]:
		"""Generate random bounding box"""
		W = size[2]
		H = size[3]
		cut_rat = np.sqrt(1.0 - lam)
		cut_w = int(W * cut_rat)
		cut_h = int(H * cut_rat)

		# uniform
		cx = np.random.randint(W)
		cy = np.random.randint(H)

		x1 = np.clip(cx - cut_w // 2, 0, W)
		y1 = np.clip(cy - cut_h // 2, 0, H)
		x2 = np.clip(cx + cut_w // 2, 0, W)
		y2 = np.clip(cy + cut_h // 2, 0, H)

		return x1, y1, x2, y2

	def augment_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Apply CutMix augmentation to the batch"""
		if not self.hparams.get('cutmix_alpha'):
			return x, y

		batch_size = x.size(0)
		device = x.device
		alpha = self.hparams['cutmix_alpha']

		# Generate random mixing ratio
		lam = np.random.beta(alpha, alpha)

		# Create shuffled indices
		index = torch.randperm(batch_size, device=device)

		# Generate random bounding box
		x1, y1, x2, y2 = self._rand_bbox(x.size(), lam)

		# Perform cutmix
		mixed_x = x.clone()
		mixed_x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]

		return mixed_x, y


class AugMix(Original):
	"""AugMix augmentation"""
	def __init__(self, dataset: Dataset, hparams: dict):
		super().__init__(dataset, hparams)
		
		# Initial transform (without ToTensor and Normalize)
		self.transform = transforms.Compose([
			transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
			transforms.RandomGrayscale()
		])

		# Post transform
		self.post_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], 
							   std=[0.229, 0.224, 0.225])
		])

		# Augmentation operations
		if self.hparams["augmix_all_ops"]:
			self.aug_list = augmix.augmentations_all
		else:
			self.aug_list = augmix.augmentations

	def apply_augmix(self, image: Image.Image) -> torch.Tensor:
		"""Apply AugMix to a single image"""
		# Initial transform
		image = self.transform(image)
		
		# Sample mix weights and ratio
		ws = np.float32(np.random.dirichlet([1] * self.hparams["augmix_mixture_width"]))
		m = np.float32(np.random.beta(1, 1))

		# Base transform
		mixed = self.post_transform(image)
		aug_mixed = torch.zeros_like(mixed)

		# Apply chains of augmentations
		for i in range(self.hparams["augmix_mixture_width"]):
			chain_depth = self.hparams["augmix_chain_depth"] if self.hparams["augmix_chain_depth"] > 0 \
				else np.random.randint(1, 4)
			
			img_aug = image.copy()
			for _ in range(chain_depth):
				op = np.random.choice(self.aug_list)
				img_aug = op(img_aug, self.hparams["augmix_severity"])

			# Add to mixture
			aug_mixed += ws[i] * self.post_transform(img_aug)

		# Final mixing
		mixed = (1 - m) * self.post_transform(image) + m * aug_mixed
		return mixed

	def __getitem__(self, i: int) -> Tuple[torch.Tensor, Any]:
		x, y = self.dataset[i]
		x = self.apply_augmix(x)
		return x, y


class IPMix(Original):
    """IPMix augmentation"""
    def __init__(self, dataset: Dataset, hparams: dict):
        super().__init__(dataset, hparams)
        
        # Initial transform (without ToTensor and Normalize)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale()
        ])

        # Post transform
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        # Load mixing set
        self.mixing_set = ImageFolder(
            self.hparams['ipmix_mixing_set_path'],
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224)
            ])
        )

    def augment_input(self, image: Image.Image) -> Image.Image:
        """Apply random augmentation to input image"""
        aug_list = ipmix.augmentations_all if self.hparams["ipmix_all_ops"] else ipmix.augmentations
        op = np.random.choice(aug_list)
        return op(image.copy(), self.hparams["ipmix_severity"])

    def apply_ipmix(self, image: Image.Image) -> torch.Tensor:
        """Apply IPMix to a single image"""
        # Initial transform
        image = self.transform(image)

        # Get random mixing image
        mixing_idx = np.random.choice(len(self.mixing_set))
        mixing_pic, _ = self.mixing_set[mixing_idx]

        # Mixing parameters
        patch_sizes = [4, 8, 16, 32, 64, 224]
        mixing_op = ['Img', 'P']
        ws = np.float32(np.random.dirichlet([1] * self.hparams["ipmix_k"]))
        m = np.float32(np.random.beta(1, 1))

        # Initialize mix
        mix = torch.zeros_like(self.post_transform(image))

        # Apply mixing k times
        for i in range(self.hparams["ipmix_k"]):
            mixed = image.copy()
            mixing_ways = random.choice(mixing_op)

            if mixing_ways == 'P':
                # Patch-based mixing
                for _ in range(np.random.randint(self.hparams["ipmix_t"] + 1)):
                    patch_size = random.choice(patch_sizes)
                    mix_op = random.choice(ipmix.mixings)
                    if random.random() > 0.5:
                        mixed = ipmix.patch_mixing(mixed, mixing_pic, patch_size, 
                                                     mix_op, self.hparams["ipmix_beta"])
                    else:
                        mixed_copy = self.augment_input(image)
                        mixed = ipmix.patch_mixing(mixed, mixed_copy, patch_size, 
                                                     mix_op, self.hparams["ipmix_beta"])
            else:
                # Image-level mixing
                for _ in range(np.random.randint(self.hparams["ipmix_t"] + 1)):
                    mixed = self.augment_input(mixed)

            mix += ws[i] * self.post_transform(mixed)

        # Final mixing
        mixed = (1 - m) * self.post_transform(image) + m * mix
        return mixed

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, Any]:
        x, y = self.dataset[i]
        x = self.apply_ipmix(x)
        return x, y
	

class TrivialAugment(Original):
    """TrivialAugment augmentation"""
    def __init__(self, dataset: Dataset, hparams: dict):
        super().__init__(dataset, hparams)
        
        # Initial transform (without ToTensor and Normalize)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale()
        ])

        # Post transform
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        # Initialize TrivialAugment
        self.augmenter = strategy_augment.TrivialAugment()

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, Any]:
        x, y = self.dataset[i]
        x = self.transform(x)          # Initial transform
        x = self.augmenter(x)          # TrivialAugment
        x = self.post_transform(x)     # Post transform
        return x, y
	

class RandAugment(TrivialAugment):
    """RandAugment augmentation"""
    def __init__(self, dataset: Dataset, hparams: dict):
        super().__init__(dataset, hparams)
        # Override augmenter with RandAugment
        self.augmenter = strategy_augment.RandAugment()


class RandConv(Original):
    """RandConv augmentation"""
    def __init__(self, dataset: Dataset, hparams: dict):
        super().__init__(dataset, hparams)
        
        # Initialize RandConv augmenter for batch-level augmentation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.augmenter = randconv.RandConvModule(
            net=None,
            in_channels=3,
            out_channels=3,
            kernel_size=self.hparams.get('randconv_kernel_size', [1,3,5,7]),
            mixing=self.hparams.get('randconv_mixing', True),
            identity_prob=self.hparams.get('randconv_identity_prob', 0.0),
            rand_bias=self.hparams.get('randconv_rand_bias', False),
            distribution=self.hparams.get('randconv_distribution', 'kaiming_normal'),
            data_mean=[0.485, 0.456, 0.406],
            data_std=[0.229, 0.224, 0.225],
            clamp_output=self.hparams.get('randconv_clamp_output', False)
        ).to(self.device)

    def augment_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RandConv augmentation to the batch
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            y: Labels tensor
            
        Returns:
            Tuple of (augmented_x, y)
        """
        self.augmenter.randomize()
        x = x.to(self.device)
        x = self.augmenter(x)
        return x, y
	

class Stylized(Original):
    """Style Transfer data augmentation"""
    def __init__(self, dataset: Dataset, hparams: dict):
        super().__init__(dataset, hparams)
        
        # Set style image directory path
        self.style_dir = Path(hparams["style_dir"])
        assert self.style_dir.is_dir(), 'Style directory not found'
        
        # Get list of style images
        self.styles = []
        extensions = ['png', 'jpeg', 'jpg', 'JPEG']
        for ext in extensions:
            self.styles += list(self.style_dir.rglob('*.' + ext))
        assert len(self.styles) > 0, 'No style images found'
        
        # Setup models
        self.decoder = net.decoder
        self.vgg = net.vgg
        
        self.decoder.eval()
        self.vgg.eval()
        
        # Load model weights
        self.decoder.load_state_dict(torch.load(hparams['decoder_pth']))
        self.vgg.load_state_dict(torch.load(hparams['vgg_normalised_pth']))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
        
        # Move models to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg.to(self.device)
        self.decoder.to(self.device)
        
        # Setup style transform
        style_size = hparams.get("style_size", 512)
        self.style_transform = transforms.Compose([
            transforms.Resize((style_size, style_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.alpha = hparams["alpha"]

    def _style_transfer(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Apply style transfer to content image using given style image
        
        Args:
            content: Content image tensor
            style: Style image tensor
            
        Returns:
            Stylized content image tensor
        """
        with torch.no_grad():
            content_f = self.vgg(content)
            style_f = self.vgg(style)
            feat = adaptive_instance_normalization(content_f, style_f)
            feat = feat * self.alpha + content_f * (1 - self.alpha)
            return self.decoder(feat)

    def augment_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply style transfer augmentation to the batch
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            y: Labels tensor
            
        Returns:
            Tuple of (stylized_x, y)
        """
        batch_size = x.size(0)
        
        # Randomly select and preprocess style images
        style_paths = random.choices(self.styles, k=batch_size)
        style_images = [Image.open(style_path).convert('RGB') for style_path in style_paths]
        style_batch = torch.stack([self.style_transform(img) for img in style_images])
        
        # Apply style transfer
        x = x.to(self.device)
        style_batch = style_batch.to(self.device)
        x = self._style_transfer(x, style_batch)
        
        # Clean up style images
        for style_img in style_images:
            style_img.close()
            
        return x, y
    

class CartoonGAN(Original):
    """CartoonGAN Augmentation"""
    def __init__(self, dataset: Dataset, hparams: dict):
        super().__init__(dataset, hparams)

        # Override transform to exclude normalization
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor()
        ])

        # Define post transform (normalization only)
        self.post_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Setup CartoonGAN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cartoongan = Transformer()
        self.cartoongan.load_state_dict(torch.load(hparams["cartoongan_model_path"]))
        self.cartoongan.to(self.device)
        self.cartoongan.eval()

    def augment_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CartoonGAN transformation to the batch
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width), range [0, 1]
            y: Labels tensor
            
        Returns:
            Tuple of (cartoon_x, y) where cartoon_x is normalized
        """
        # Move to device
        x = x.to(self.device)

        # Preprocess: scale to [-1, 1] and convert RGB to BGR
        x = -1 + 2 * x
        x = kornia.color.rgb_to_bgr(x.float())

        # Apply CartoonGAN
        with torch.no_grad():
            cartoon_x = self.cartoongan(x)

        # Convert BGR to RGB
        cartoon_x = kornia.color.bgr_to_rgb(cartoon_x.float())

        # Scale back to [0, 1]
        cartoon_x = cartoon_x.data.float() * 0.5 + 0.5

        # Apply normalization
        cartoon_x = self.post_transform(cartoon_x)

        return cartoon_x, y
    

class EdgeDetection(Original):
    """Edge Detection Augmentation"""
    def __init__(self, dataset: Dataset, hparams: dict):
        super().__init__(dataset, hparams)

        # Override transform to exclude normalization
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor()
        ])

        # Define post transform (normalization only)
        self.post_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Setup edge detector
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_detector = EdgeDetector().to(self.device)
        self.edge_detector.eval()

    def augment_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply edge detection to the batch
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width), range [0, 1]
            y: Labels tensor
            
        Returns:
            Tuple of (edges_x, y) where edges_x is normalized
        """
        # Move to device
        x = x.to(self.device)

        # Scale to [0, 255] and convert RGB to BGR
        x = x * 255.0
        x = kornia.color.bgr_to_rgb(x.float())

        # Apply edge detection
        with torch.no_grad():
            edges = self.edge_detector(x)

        # Convert to 3-channel if necessary
        if edges.shape[1] == 1:
            edges = edges.repeat(1, 3, 1, 1)

        # Convert to float [0, 1]
        edges = edges.byte() / 255.0

        # Apply normalization
        edges = self.post_transform(edges)

        return edges, y