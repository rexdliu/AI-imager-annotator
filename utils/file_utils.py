"""File system utilities for dataset management, file operations, and annotation handling."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError

from utils.schema_utils import VLMSchema, BBox
from utils.logger import get_logger

# Get logger for this module
logger = get_logger("file_utils")

# Define root directories
DATASET_ROOT = Path("dataset").resolve()
ANNOT_ROOT = Path("annotated_dataset").resolve()


def list_images() -> List[str]:
    """Return all supported image file paths under dataset/ (relative str to CWD)."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".heic", ".heif"}
    imgs: List[str] = []

    if not DATASET_ROOT.exists():
        st.warning(f"Dataset directory '{DATASET_ROOT}' not found!")
        return imgs

    if not any(DATASET_ROOT.iterdir()):
        st.warning(f"Dataset directory '{DATASET_ROOT}' is empty.")
        return imgs

    logger.debug(f"Searching for images in: {DATASET_ROOT}")
    for p in DATASET_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            try:
                # Store path relative to CWD
                rel_path_str = str(p.relative_to(Path.cwd()))
                imgs.append(rel_path_str)
            except ValueError:
                try:
                    # Fallback relative to DATASET_ROOT
                    rel_path_str = str(p.relative_to(DATASET_ROOT))
                    full_rel_path = str(Path("dataset") / rel_path_str)
                    imgs.append(full_rel_path)
                except ValueError:
                    # Absolute path as last resort
                    abs_path_str = str(p.resolve())
                    imgs.append(abs_path_str)

    logger.debug(f"Found {len(imgs)} images")
    if not imgs and any(DATASET_ROOT.iterdir()):
        st.warning(f"No image files found with extensions: {extensions}. Check file types.")

    return sorted(imgs)


def derive_full_relative_path(img_path: Path | str) -> str:
    """
    Derive the relative path structure within DATASET_ROOT.
    Example: 'Food/Chinese', 'Transportation/Local', or '' if in root.
    """
    try:
        p = Path(img_path).resolve()
        rel_path = p.relative_to(DATASET_ROOT)
        parent_path = rel_path.parent.as_posix()
        result = parent_path if parent_path != "." else ""
        return result
    except ValueError:
        logger.warning(f"Path {img_path} is not inside {DATASET_ROOT}")
        return "(external)"
    except Exception as e:
        logger.error(f"Error deriving relative path for {img_path}: {e}")
        return "(error_deriving_path)"


def _get_output_subdir(base_prefix: str, relative_structure: str) -> Path:
    """Helper to construct the nested output subdirectory path."""
    if relative_structure and relative_structure not in ["(external)", "(error_deriving_path)"]:
        # Split the relative path (e.g., "Food/Chinese") into parts
        parts = Path(relative_structure).parts
        # Create the base output dir (e.g., "annotated_Food" or "schema_Food")
        output_base = ANNOT_ROOT / f"{base_prefix}_{parts[0]}"
        # Join the remaining parts (e.g., "Chinese")
        output_dir = output_base.joinpath(*parts[1:])
    elif relative_structure == "":  # Image was in dataset root
        output_dir = ANNOT_ROOT / f"{base_prefix}_(root)"
    else:  # Handle external or error cases
        output_dir = ANNOT_ROOT / f"{base_prefix}_{relative_structure}"

    return output_dir


def load_and_convert_image(image_path: str | Path) -> Optional[Image.Image]:
    """Load an image from path, convert to RGB, and return PIL Image object."""
    logger.debug(f"Loading image: {image_path}")
    try:
        resolved_path = Path(image_path).resolve()
        if not resolved_path.exists():
            st.error(f"Error: Image file not found at {resolved_path}")
            return None
        img = Image.open(resolved_path).convert("RGB")
        return img
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None


def save_annotated_image(
        original_path_str: str,
        image_id: str,
        rects: List[BBox],
        rect_colors: List[str] = None,
        rotated_image=None,
        rotation_angle: int = 0
) -> Path:
    """
    Load original image, convert to JPG, draw rectangles, and save to annotated_<category>/.../<image_id>.jpg

    Args:
        original_path_str: Relative path string to the original image
        image_id: The ID (stem or UUID) used for the output filename
        rects: List of bounding boxes (scaled to original dimensions)
        rect_colors: List of colors for each rectangle (hex strings)
        rotated_image: Optional pre-rotated PIL image to use
        rotation_angle: Rotation angle to apply if rotated_image not provided

    Returns:
        Path to the saved annotated JPG image
    """
    original_path = Path(original_path_str)
    if not original_path.exists() and rotated_image is None:
        raise FileNotFoundError(f"Original image not found: {original_path_str}")

    relative_structure = derive_full_relative_path(original_path)
    out_dir = _get_output_subdir("annotated", relative_structure)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_id}.jpg"  # Use image_id for filename

    logger.debug(f"Saving annotated image to: {out_path}")

    try:
        # Use provided rotated image if available, otherwise load and rotate
        if rotated_image is not None:
            img = rotated_image
            logger.debug(f"Using provided rotated image: {img.size}")
        else:
            img = Image.open(original_path).convert("RGB")
            if rotation_angle != 0:
                img = img.rotate(-rotation_angle, expand=True, resample=Image.Resampling.BILINEAR)
                logger.debug(f"Applied rotation of {rotation_angle}° to image: {img.size}")
            else:
                logger.debug(f"Loaded original image without rotation: {img.size}")

        if rects:
            draw = ImageDraw.Draw(img)
            # Default colors if none provided
            default_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]

            for i, bbox in enumerate(rects):
                if len(bbox) == 4:
                    # Use provided color if available, otherwise use default
                    if rect_colors and i < len(rect_colors):
                        color = rect_colors[i]
                    else:
                        color = default_colors[i % len(default_colors)]

                    draw.polygon(bbox, outline=color, width=3)
                else:
                    logger.warning(f"Skipping invalid bbox for drawing: {bbox}")

        # Save as JPEG
        img.save(out_path, "JPEG", quality=95)
        return out_path
    except UnidentifiedImageError:
        st.error(f"Could not identify image format: {original_path_str}")
        raise
    except Exception as e:
        st.error(f"Error processing/saving annotated image {image_id}.jpg: {str(e)}")
        raise


def save_schema(schema: VLMSchema) -> Path:
    """
    Save schema to JSON file named <image_id>.json in schema_<category>/.../
    using nested structure derived from the original image path.

    Args:
        schema: The schema object to save

    Returns:
        Path to the saved schema file
    """
    relative_structure = derive_full_relative_path(schema.image_path)
    out_dir = _get_output_subdir("schema", relative_structure)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{schema.image_id}.json"

    logger.debug(f"Saving schema to: {out_path}")

    if out_path.exists():
        logger.info(f"Updating existing schema: {out_path}")

    try:
        schema.to_json(out_path)
        return out_path
    except Exception as e:
        st.error(f"Error saving schema {schema.image_id}.json: {str(e)}")
        raise


def check_existing_annotation(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Check if an annotation JSON file exists for this image using its filename stem.
    Looks in the corresponding schema_<category>/.../ directory.

    Args:
        image_path: Path to the original image

    Returns:
        Existing schema as dict if found, None otherwise
    """
    img_path_obj = Path(image_path)
    image_stem = img_path_obj.stem  # Get ID from filename stem
    relative_structure = derive_full_relative_path(img_path_obj)
    schema_dir = _get_output_subdir("schema", relative_structure)

    if not schema_dir.exists():
        return None

    json_path = schema_dir / f"{image_stem}.json"  # Look for <stem>.json
    if not json_path.exists():
        return None

    try:
        return json.loads(json_path.read_text("utf-8"))
    except Exception as e:
        st.error(f"Error reading existing schema {json_path.name}: {e}")
        return None


def get_annotated_image_stems() -> Set[str]:
    """Return the set of image stems (filenames without ext) that have schema files."""
    annotated_stems = set()
    if not ANNOT_ROOT.exists():
        return annotated_stems

    # Check every potential schema file recursively
    for json_file in ANNOT_ROOT.rglob("schema_*/**/*.json"):
        if json_file.is_file():
            annotated_stems.add(json_file.stem)  # Store the stem

    return annotated_stems


def get_annotated_image_path(original_path: str, image_id: str) -> Optional[Path]:
    """Get the path to an annotated image if it exists.

    Args:
        original_path: Path to the original image
        image_id: Image ID (stem)

    Returns:
        Path to annotated image if exists, None otherwise
    """
    try:
        relative_structure = derive_full_relative_path(original_path)
        out_dir = _get_output_subdir("annotated", relative_structure)
        out_path = out_dir / f"{image_id}.jpg"

        if out_path.exists():
            return out_path
        return None
    except Exception as e:
        logger.error(f"Error finding annotated image: {e}")
        return None


def rename_dataset_files_to_uuid(progress_bar=None) -> Tuple[int, int, int]:
    """
    Renames image files in the 'dataset' folder to <uuid><ext>.
    Attempts to find corresponding schema files and rename them + update internal paths.

    !! WARNING: This modifies your original dataset and annotations. BACK UP FIRST. !!

    Args:
        progress_bar: Streamlit progress bar object (optional)

    Returns:
        Tuple: (success_count, annotation_updated_count, error_count)
    """
    logger.info("Starting dataset file renaming to UUIDs")
    st.warning(
        "**WARNING:** This operation will rename files in your original `dataset` folder and "
        "attempt to update corresponding annotations in `annotated_dataset`. "
        "**BACK UP BOTH FOLDERS BEFORE PROCEEDING.** This cannot be easily undone."
    )

    all_images = list_images()
    total_files = len(all_images)
    success_count = 0
    annotation_updated_count = 0
    error_count = 0
    processed_stems = set()  # To avoid processing duplicates

    status_placeholder = st.empty()

    for i, img_path_str in enumerate(all_images):
        status_placeholder.text(f"Processing file {i + 1}/{total_files}...")
        if progress_bar:
            progress_bar.progress((i + 1) / total_files)

        try:
            original_path = Path(img_path_str).resolve()  # Use absolute path for safety
            original_stem = original_path.stem
            original_suffix = original_path.suffix.lower()  # Keep original extension

            # Skip if already processed
            if str(original_path) in processed_stems:
                logger.debug(f"Skipping already processed path: {original_path}")
                continue
            processed_stems.add(str(original_path))

            # Skip if filename already looks like a UUID
            try:
                uuid.UUID(original_stem)
                logger.debug(f"Skipping potential UUID filename: {original_path.name}")
                continue  # Assume already renamed
            except ValueError:
                pass  # Not a UUID, proceed

            logger.debug(f"Processing: {original_path}")

            # Generate New UUID Name
            new_uuid = str(uuid.uuid4())
            new_filename = f"{new_uuid}{original_suffix}"
            new_path = original_path.with_name(new_filename)

            # Get new relative path for updating schema
            try:
                new_relative_path_str = str(new_path.relative_to(Path.cwd()))
            except ValueError:
                new_relative_path_str = str(Path("dataset") / new_path.relative_to(DATASET_ROOT))

            # Check for and Update Annotation
            annotation_updated = False
            relative_structure = derive_full_relative_path(original_path)
            schema_dir = _get_output_subdir("schema", relative_structure)
            old_schema_path = schema_dir / f"{original_stem}.json"
            new_schema_path = schema_dir / f"{new_uuid}.json"

            if old_schema_path.exists():
                logger.info(f"Found existing schema: {old_schema_path}")
                try:
                    # Load schema data
                    schema_data = json.loads(old_schema_path.read_text("utf-8"))

                    # Update fields
                    schema_data["image_id"] = new_uuid
                    schema_data["image_path"] = new_relative_path_str

                    # Rename schema file then update content
                    os.rename(old_schema_path, new_schema_path)
                    new_schema_path.write_text(json.dumps(schema_data, indent=2), encoding="utf-8")

                    annotation_updated = True
                    annotation_updated_count += 1
                except Exception as e_schema:
                    logger.error(f"Error updating schema for {original_stem}: {e_schema}")
                    st.error(f"Failed to update schema for {original_path.name}: {e_schema}. Image was NOT renamed.")

                    # Attempt to rename schema back if rename succeeded but content update failed
                    if new_schema_path.exists() and not old_schema_path.exists():
                        try:
                            os.rename(new_schema_path, old_schema_path)
                            logger.debug(f"Rolled back schema rename for {new_schema_path.name}")
                        except Exception as e_rollback:
                            logger.error(f"Error rolling back schema rename: {e_rollback}")
                    error_count += 1
                    continue  # Skip image rename if schema update failed

            # Rename Original Image File
            os.rename(original_path, new_path)
            logger.info(f"Renamed image: {original_path.name} -> {new_path.name}")
            success_count += 1

        except Exception as e:
            logger.error(f"Error processing file {img_path_str}: {e}")
            st.error(f"Failed to process {Path(img_path_str).name}: {e}")
            error_count += 1
            continue  # Move to next file

    status_placeholder.text(
        f"Renaming finished: {success_count} succeeded, {annotation_updated_count} annotations updated, {error_count} errors."
    )
    logger.info(
        f"Renaming completed. Success: {success_count}, Annotations: {annotation_updated_count}, Errors: {error_count}")
    return success_count, annotation_updated_count, error_count


def get_schema_stats() -> Dict[str, Any]:
    """Get statistics about annotated schemas by scanning schema_* dirs recursively."""
    stats = {
        "total": 0,
        "captioning": 0,
        "vqa": 0,
        "instruction": 0,
        "with_boxes": 0,
        "categories": set()  # Store full category paths like 'Food/Chinese'
    }

    if not ANNOT_ROOT.exists():
        logger.debug("Annotation root directory not found.")
        return stats

    schema_files_found = list(ANNOT_ROOT.rglob("schema_*/**/*.json"))
    logger.debug(f"Found {len(schema_files_found)} potential schema JSON files.")

    for json_path in schema_files_found:
        if json_path.is_file():
            try:
                # Derive category structure from the json path relative to ANNOT_ROOT
                rel_path = json_path.relative_to(ANNOT_ROOT)
                # Remove the 'schema_' prefix from the first part and the filename stem
                category_parts = list(rel_path.parent.parts)
                if category_parts:
                    # Ensure the first part starts with schema_ before replacing
                    if category_parts[0].startswith("schema_"):
                        category_parts[0] = category_parts[0].replace("schema_", "", 1)
                        category_str = "/".join(category_parts)
                        stats["categories"].add(category_str)

                # Load data and update stats
                data = json.loads(json_path.read_text("utf-8"))
                stats["total"] += 1
                task_type = data.get("task_type", "unknown")
                if task_type in stats:
                    stats[task_type] += 1

                # Count schemas with non-empty bounding boxes
                if data.get("bounding_box") and isinstance(data["bounding_box"], list) and len(
                        data["bounding_box"]) > 0:
                    stats["with_boxes"] += 1
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON file: {json_path}")
                continue
            except Exception as e:
                logger.warning(f"Error processing schema file {json_path}: {e}")
                continue

    stats["category_count"] = len(stats["categories"])
    stats["categories"] = sorted(list(stats["categories"]))
    return stats