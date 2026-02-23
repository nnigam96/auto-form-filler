"""
Robust Image Preprocessing for Passport OCR.
Strategies:
1. None/Grayscale Only (Let OCR handle thresholding) -> Best for high-quality scans
2. Otsu Thresholding -> Best for high contrast images
3. Adaptive Thresholding -> Best for uneven lighting/shadows
4. Morphology -> Noise reduction with morphological operations
5. Gaussian Blur -> Blur + Otsu for noisy images
6. Upscale -> 2x upscaling for low-res images
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union


def preprocess_none(image_path: Union[str, Path]) -> Image.Image:
    """No preprocessing - return original image as grayscale."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(gray)


def preprocess_grayscale(image_path: Union[str, Path]) -> Image.Image:
    """Simple Grayscale (alias for none, least destructive)."""
    return preprocess_none(image_path)


def preprocess_otsu(image_path: Union[str, Path]) -> Image.Image:
    """Otsu's thresholding - good for high contrast images.
    
    Automatically determines optimal threshold value.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


def preprocess_adaptive(image_path: Union[str, Path]) -> Image.Image:
    """Adaptive thresholding - good for varying lighting conditions.
    
    Uses local thresholding that adapts to different regions.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh)


def preprocess_morphology(image_path: Union[str, Path]) -> Image.Image:
    """Morphological operations for noise reduction.
    
    Applies Otsu + morphological closing to remove small noise.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological closing to remove noise
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return Image.fromarray(thresh)


def preprocess_gaussian_blur(image_path: Union[str, Path]) -> Image.Image:
    """Gaussian blur + Otsu - can help with noisy images."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


def preprocess_upscale(image_path: Union[str, Path]) -> Image.Image:
    """2x Upscaling + Grayscale (For low-res images)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    # Upscale 2x using Cubic interpolation
    height, width = img.shape[:2]
    img = cv2.resize(img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(gray)


# Map of preprocessing methods
PREPROCESSING_METHODS = {
    "none": preprocess_none,
    "grayscale": preprocess_grayscale,  # Alias for none
    "otsu": preprocess_otsu,
    "adaptive": preprocess_adaptive,
    "morphology": preprocess_morphology,
    "gaussian_blur": preprocess_gaussian_blur,
    "upscale": preprocess_upscale,
}