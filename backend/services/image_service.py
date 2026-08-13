"""Image validation, metadata extraction, and reusable preprocessing helpers."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from backend.utils.validators import allowed_file


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a valid supported image."""


def inspect_image(file_storage):
    """Validate image bytes and return safe metadata."""
    if not file_storage or not file_storage.filename:
        raise ImageValidationError("Please select an image first.")
    if not allowed_file(file_storage.filename):
        raise ImageValidationError("Unsupported format. Use JPG, JPEG, PNG, or WEBP.")
    try:
        file_storage.stream.seek(0)
        with Image.open(file_storage.stream) as image:
            image.verify()
        file_storage.stream.seek(0)
        with Image.open(file_storage.stream) as image:
            width, height = image.size
            image_format = image.format or "Unknown"
    except (UnidentifiedImageError, OSError, ValueError):
        raise ImageValidationError("This file is not a valid image.")
    finally:
        file_storage.stream.seek(0)
    return {"file_type": image_format.upper(), "width": width, "height": height, "resolution": f"{width} × {height}"}


def preprocess_image(image_path, target_size=(224, 224)):
    """Return normalized RGB pixels for image-processing helpers."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError("Unable to read the saved image.")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def image_statistics(image_path):
    image = preprocess_image(Path(image_path))
    return float(image.mean()), float(image.std())
