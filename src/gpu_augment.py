from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.transforms import functional as TF

from .data import IMAGENET_MEAN, IMAGENET_STD, ImageData


# ITU-R 601 luma, the same weights torchvision uses for grayscale conversion.
_LUMA = (0.299, 0.587, 0.114)
_RGB_TO_YIQ = torch.tensor(
    [[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]]
)


def square_image_tensor(data: ImageData, indices: list[int], size: int = 256) -> torch.Tensor:
    """Center-crop the cached images to size x size and stack them as uint8 (N, 3, size, size).

    The cache already resized the shorter side to `size`, so this equals the
    Resize(256) + CenterCrop(256) region that evaluation also looks at.
    """
    images = []
    for index in indices:
        tensor = torch.from_numpy(data.images[index]).permute(2, 0, 1)
        images.append(TF.center_crop(tensor, [size, size]))
    return torch.stack(images)


def _luma(images: torch.Tensor) -> torch.Tensor:
    r, g, b = images.unbind(dim=1)
    return (_LUMA[0] * r + _LUMA[1] * g + _LUMA[2] * b).unsqueeze(1)


class GPUViewAugmenter:
    """SimCLR view generation for a whole batch on the GPU.

    Same components and defaults as data.build_ssl_view_transform (random
    resized crop, horizontal flip, color jitter, random grayscale, Gaussian
    blur, optional solarization), but sampled per image and applied with batched
    tensor ops instead of per-image PIL calls, which cost ~50 ms per view on
    the CPU. Differences from torchvision: the four jitter operations are
    applied in a fixed order, and hue is rotated in YIQ space.
    """

    def __init__(self, image_size: int, augmentation: dict, device: torch.device, seed: int) -> None:
        self.image_size = image_size
        self.crop_min = float(augmentation.get("crop_min", 0.2))
        self.jitter = float(augmentation.get("color_jitter", 0.4))
        self.jitter_p = float(augmentation.get("jitter_p", 0.8))
        self.hue = min(0.1, self.jitter / 4)
        self.grayscale_p = float(augmentation.get("grayscale_p", 0.2))
        self.blur_p = float(augmentation.get("blur_p", 0.5))
        self.solarize_p = float(augmentation.get("solarize_p", 0.0))
        self.kernel_size = 2 * int(image_size * 0.05) + 1
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.rgb_to_yiq = _RGB_TO_YIQ.to(device)
        self.yiq_to_rgb = torch.linalg.inv(_RGB_TO_YIQ).to(device)

    def _uniform(self, count: int, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(count, device=self.device, generator=self.generator)

    def _bernoulli(self, count: int, p: float) -> torch.Tensor:
        return torch.rand(count, device=self.device, generator=self.generator) < p

    def _crop_boxes(self, count: int, height: int, width: int) -> torch.Tensor:
        """RandomResizedCrop box sampling (scale (crop_min, 1), ratio (3/4, 4/3)), 10 tries then full image."""
        attempts = 10
        area = height * width * self._uniform(count * attempts, self.crop_min, 1.0)
        log_ratio = self._uniform(count * attempts, math.log(3 / 4), math.log(4 / 3))
        ratio = log_ratio.exp()
        box_w = (area * ratio).sqrt().view(attempts, count)
        box_h = (area / ratio).sqrt().view(attempts, count)
        valid = (box_w <= width) & (box_h <= height)
        first = valid.float().argmax(dim=0)
        any_valid = valid.any(dim=0)
        columns = torch.arange(count, device=self.device)
        box_w = torch.where(any_valid, box_w[first, columns], torch.full_like(box_w[0], width))
        box_h = torch.where(any_valid, box_h[first, columns], torch.full_like(box_h[0], height))
        x0 = (width - box_w) * torch.rand(count, device=self.device, generator=self.generator)
        y0 = (height - box_h) * torch.rand(count, device=self.device, generator=self.generator)
        return torch.stack([columns.float(), x0, y0, x0 + box_w, y0 + box_h], dim=1)

    def _color_jitter(self, images: torch.Tensor) -> torch.Tensor:
        count = images.size(0)
        apply = self._bernoulli(count, self.jitter_p).view(-1, 1, 1, 1)

        def factor(low: float, high: float, neutral: float) -> torch.Tensor:
            values = self._uniform(count, low, high).view(-1, 1, 1, 1)
            return torch.where(apply, values, torch.full_like(values, neutral))

        strength = self.jitter
        images = (images * factor(1 - strength, 1 + strength, 1.0)).clamp(0, 1)
        mean = _luma(images).mean(dim=(1, 2, 3), keepdim=True)
        images = ((images - mean) * factor(1 - strength, 1 + strength, 1.0) + mean).clamp(0, 1)
        gray = _luma(images)
        images = ((images - gray) * factor(1 - strength, 1 + strength, 1.0) + gray).clamp(0, 1)
        if self.hue > 0:
            angle = factor(-self.hue, self.hue, 0.0).view(-1) * 2 * math.pi
            cos, sin = angle.cos(), angle.sin()
            rotation = torch.zeros(count, 3, 3, device=self.device)
            rotation[:, 0, 0] = 1
            rotation[:, 1, 1], rotation[:, 1, 2] = cos, -sin
            rotation[:, 2, 1], rotation[:, 2, 2] = sin, cos
            transform = self.yiq_to_rgb @ rotation @ self.rgb_to_yiq
            images = torch.einsum("bij,bjhw->bihw", transform, images).clamp(0, 1)
        return images

    def _blur(self, images: torch.Tensor) -> torch.Tensor:
        count, channels, height, width = images.shape
        sigma = self._uniform(count, 0.1, 2.0)
        apply = self._bernoulli(count, self.blur_p)
        radius = self.kernel_size // 2
        offsets = torch.arange(-radius, radius + 1, device=self.device, dtype=images.dtype)
        kernel = torch.exp(-(offsets.view(1, -1) ** 2) / (2 * sigma.view(-1, 1) ** 2))
        kernel = kernel / kernel.sum(dim=1, keepdim=True)
        identity = (offsets == 0).to(images.dtype).expand_as(kernel)
        kernel = torch.where(apply.view(-1, 1), kernel, identity)
        kernel = kernel.repeat_interleave(channels, dim=0)
        stacked = images.reshape(1, count * channels, height, width)
        stacked = F.pad(stacked, (radius, radius, radius, radius), mode="reflect")
        stacked = F.conv2d(stacked, kernel.view(-1, 1, 1, self.kernel_size), groups=count * channels)
        stacked = F.conv2d(stacked, kernel.view(-1, 1, self.kernel_size, 1), groups=count * channels)
        return stacked.view(count, channels, height, width)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 (B, 3, H, W) on the device -> normalized float views (B, 3, image_size, image_size)."""
        count, _, height, width = images.shape
        images = images.float() / 255
        boxes = self._crop_boxes(count, height, width)
        views = roi_align(
            images, boxes, output_size=(self.image_size, self.image_size),
            spatial_scale=1.0, sampling_ratio=2, aligned=True,
        )
        flip = self._bernoulli(count, 0.5).view(-1, 1, 1, 1)
        views = torch.where(flip, views.flip(dims=(3,)), views)
        if self.jitter > 0 and self.jitter_p > 0:
            views = self._color_jitter(views)
        if self.grayscale_p > 0:
            gray = self._bernoulli(count, self.grayscale_p).view(-1, 1, 1, 1)
            views = torch.where(gray, _luma(views).expand_as(views), views)
        if self.blur_p > 0:
            views = self._blur(views)
        if self.solarize_p > 0:
            solarize = self._bernoulli(count, self.solarize_p).view(-1, 1, 1, 1)
            views = torch.where(solarize & (views >= 0.5), 1 - views, views)
        return ((views - self.mean) / self.std).contiguous(memory_format=torch.channels_last)


class GPUMultiViewLoader:
    """Iterates over shuffled batches of GPU-resident images, yielding augmented views and labels.

    Drops the last incomplete batch (contrastive losses want full batches).
    With num_views == 1 it yields a tensor, otherwise a list of tensors, the
    same structure a DataLoader over data.MultiView would produce.
    """

    def __init__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        augmenter: GPUViewAugmenter,
        batch_size: int,
        num_views: int,
        seed: int,
    ) -> None:
        self.images = images
        self.targets = targets
        self.augmenter = augmenter
        self.batch_size = batch_size
        self.num_views = num_views
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.images.size(0) // self.batch_size

    def __iter__(self):
        order = torch.randperm(self.images.size(0), generator=self.generator).to(self.images.device)
        for step in range(len(self)):
            batch = order[step * self.batch_size : (step + 1) * self.batch_size]
            images = self.images[batch]
            views = [self.augmenter(images) for _ in range(self.num_views)]
            yield (views[0] if self.num_views == 1 else views), self.targets[batch]
