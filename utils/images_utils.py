# for image processing.
import os
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import io


def load_image(path):
    """Load an image from path and return PIL Image object"""
    try:
        if os.path.exists(path):
            return Image.open(path).convert('RGB')
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def resize_image(image, width=None, height=None, keep_aspect_ratio=True):
    """Resize an image to the specified dimensions"""
    if image is None:
        return None

    if width is None and height is None:
        return image

    if keep_aspect_ratio:
        if width is None:
            # Calculate width based on height while maintaining aspect ratio
            aspect_ratio = image.width / image.height
            width = int(height * aspect_ratio)
        elif height is None:
            # Calculate height based on width while maintaining aspect ratio
            aspect_ratio = image.width / image.height
            height = int(width / aspect_ratio)

        # Resize the image
        return image.resize((width, height), Image.LANCZOS)
    else:
        # Resize without maintaining aspect ratio
        return image.resize((width or image.width, height or image.height), Image.LANCZOS)


def apply_zoom(image, zoom_factor):
    """Apply zoom to an image"""
    if image is None or zoom_factor == 1.0:
        return image

    new_width = int(image.width * zoom_factor)
    new_height = int(image.height * zoom_factor)

    return image.resize((new_width, new_height), Image.LANCZOS)


def crop_image(image, x, y, width, height):
    """Crop a region from an image"""
    if image is None:
        return None

    # Ensure coordinates are within image bounds
    x = max(0, min(x, image.width))
    y = max(0, min(y, image.height))
    width = max(1, min(width, image.width - x))
    height = max(1, min(height, image.height - y))

    return image.crop((x, y, x + width, y + height))


def image_to_bytes(image, format='PNG'):
    """Convert PIL Image to bytes"""
    if image is None:
        return None

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()


def bytes_to_image(image_bytes):
    """Convert bytes to PIL Image"""
    if image_bytes is None:
        return None

    return Image.open(io.BytesIO(image_bytes))


def enhance_image(image, brightness=1.0, contrast=1.0, color=1.0, sharpness=1.0):
    """Apply enhancements to an image"""
    if image is None:
        return None

    # Apply brightness adjustment
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)

    # Apply contrast adjustment
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)

    # Apply color adjustment
    if color != 1.0:
        image = ImageEnhance.Color(image).enhance(color)

    # Apply sharpness adjustment
    if sharpness != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)

    return image


def draw_bounding_boxes(image, annotations, color_map=None):
    """Draw bounding boxes on an image based on annotations"""
    if image is None or not annotations:
        return image

    # Create a copy of the image
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Default color map if none provided
    if color_map is None:
        color_map = {
            'person': (255, 0, 0),  # Red
            'face': (255, 165, 0),  # Orange
            'animal': (0, 255, 0),  # Green
            'vehicle': (0, 0, 255),  # Blue
            'text': (255, 255, 0),  # Yellow
            'object': (255, 0, 255)  # Magenta
        }

    # Draw each annotation
    for annotation in annotations:
        label = annotation['label']
        bbox = annotation['bbox']

        # Get color for this label
        color = color_map.get(label, (255, 255, 255))  # Default to white

        # Draw rectangle
        draw.rectangle(
            [
                bbox['x'],
                bbox['y'],
                bbox['x'] + bbox['width'],
                bbox['y'] + bbox['height']
            ],
            outline=color,
            width=2
        )

        # Draw label text
        draw.text((bbox['x'], bbox['y'] - 15), label, fill=color)

    return img_draw