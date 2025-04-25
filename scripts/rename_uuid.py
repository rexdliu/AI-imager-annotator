"""CLI: Rename all files in the dataset directory to UUIDs and update annotations."""

import os
import sys
import uuid
import json
from pathlib import Path

# Adjust path if scripts are run from root or scripts/
try:
    from utils.file_utils import list_images, derive_full_relative_path, _get_output_subdir
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))  # Add parent to path
    from utils.file_utils import list_images, derive_full_relative_path, _get_output_subdir

DATASET_ROOT = Path("dataset").resolve()
ANNOT_ROOT = Path("annotated_dataset").resolve()

def rename_files_to_uuid():
    """
    Rename all files in the dataset directory to UUIDs and update annotations.

    This script:
    1. Finds all image files in the dataset directory
    2. Generates a UUID for each file
    3. Renames the file to <uuid>.<extension>
    4. Updates any annotation files that reference the original filename

    Returns:
        tuple: (success_count, annotation_updated_count, error_count)
    """
    print("\n=== BATCH RENAME TO UUIDs ===")
    print("WARNING: This will rename all files in your dataset and update annotations.")
    print("Make sure you have backed up your dataset before continuing.")

    # Confirm with user
    confirm = input("\nDo you want to continue? (y/n): ")
    if confirm.lower() != "y":
        print("Operation cancelled.")
        return 0, 0, 0

    all_images = list_images()
    total_files = len(all_images)
    success_count = 0
    annotation_updated_count = 0
    error_count = 0
    processed_stems = set()  # To avoid processing duplicates

    print(f"\nProcessing {total_files} files...")

    for i, img_path_str in enumerate(all_images):
        print(f"Processing {i+1}/{total_files}: {img_path_str}")

        try:
            original_path = Path(img_path_str).resolve()  # Use absolute path
            original_stem = original_path.stem
            original_suffix = original_path.suffix.lower()

            # Skip if already processed
            if str(original_path) in processed_stems:
                print(f"  Skipping already processed path: {original_path}")
                continue
            processed_stems.add(str(original_path))

            # Skip if filename already looks like a UUID
            try:
                uuid.UUID(original_stem)
                print(f"  Skipping potential UUID filename: {original_path.name}")
                continue  # Assume already renamed
            except ValueError:
                pass  # Not a UUID, proceed

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
                print(f"  Found existing schema: {old_schema_path}")
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
                    print(f"  Updated schema: {old_schema_path.name} -> {new_schema_path.name}")
                except Exception as e:
                    print(f"  Error updating schema: {e}")

                    # Attempt to rename schema back if rename succeeded but update failed
                    if new_schema_path.exists() and not old_schema_path.exists():
                        try:
                            os.rename(new_schema_path, old_schema_path)
                            print(f"  Rolled back schema rename")
                        except Exception as e_rollback:
                            print(f"  Error rolling back schema rename: {e_rollback}")
                    error_count += 1
                    continue  # Skip image rename if schema update failed

            # Rename Original Image File
            os.rename(original_path, new_path)
            print(f"  Renamed image: {original_path.name} -> {new_path.name}")
            success_count += 1

        except Exception as e:
            print(f"  Error processing file {img_path_str}: {e}")
            error_count += 1
            continue  # Move to next file

    print("\n=== SUMMARY ===")
    print(f"Renamed files: {success_count}")
    print(f"Updated annotations: {annotation_updated_count}")
    print(f"Errors: {error_count}")

    return success_count, annotation_updated_count, error_count

if __name__ == "__main__":
    rename_files_to_uuid()