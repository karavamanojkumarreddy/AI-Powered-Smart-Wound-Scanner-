"""Wound image preprocessing with OpenCV.

Steps:
1. Load an RGB image.
2. Resize to 224x224.
3. Normalize to [0, 1].
4. Apply Gaussian blur for noise reduction.
"""

import cv2
import numpy as np


def preprocess_wound_image(image_path: str) -> dict[str, np.ndarray]:
    """Preprocess a wound image and return each intermediate step.

    Args:
        image_path: Path to the input wound image.

    Returns:
        Dictionary containing each preprocessing step with clear variable names.
    """
    # 1) Load image with OpenCV (BGR), then convert to RGB.
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise FileNotFoundError(f"Unable to load image at path: {image_path}")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # 2) Resize RGB image to 224x224.
    resized_rgb_image = cv2.resize(rgb_image, (224, 224), interpolation=cv2.INTER_AREA)

    # 3) Normalize pixel values to range [0, 1].
    normalized_rgb_image = resized_rgb_image.astype(np.float32) / 255.0

    # 4) Apply Gaussian blur for noise reduction.
    gaussian_blurred_image = cv2.GaussianBlur(normalized_rgb_image, (5, 5), sigmaX=0)

    return {
        "rgb_image": rgb_image,
        "resized_rgb_image": resized_rgb_image,
        "normalized_rgb_image": normalized_rgb_image,
        "gaussian_blurred_image": gaussian_blurred_image,
    }


if __name__ == "__main__":
    # Example usage
    input_image_path = "sample_wound.jpg"  # Replace with your image path
    preprocessing_outputs = preprocess_wound_image(input_image_path)

    print("Preprocessing completed. Output shapes and dtypes:")
    print(f"rgb_image: {preprocessing_outputs['rgb_image'].shape}, {preprocessing_outputs['rgb_image'].dtype}")
    print(
        f"resized_rgb_image: {preprocessing_outputs['resized_rgb_image'].shape}, "
        f"{preprocessing_outputs['resized_rgb_image'].dtype}"
    )
    print(
        f"normalized_rgb_image: {preprocessing_outputs['normalized_rgb_image'].shape}, "
        f"{preprocessing_outputs['normalized_rgb_image'].dtype}, "
        f"min={preprocessing_outputs['normalized_rgb_image'].min():.3f}, "
        f"max={preprocessing_outputs['normalized_rgb_image'].max():.3f}"
    )
    print(
        f"gaussian_blurred_image: {preprocessing_outputs['gaussian_blurred_image'].shape}, "
        f"{preprocessing_outputs['gaussian_blurred_image'].dtype}, "
        f"min={preprocessing_outputs['gaussian_blurred_image'].min():.3f}, "
        f"max={preprocessing_outputs['gaussian_blurred_image'].max():.3f}"
    )
