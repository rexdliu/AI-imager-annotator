"""CLI: Validate every JSON under annotated_dataset/schema_* using Pydantic."""

from pathlib import Path
import sys
import json

# Adjust path if scripts are run from root or src/scripts
try:
    from utils.schema_utils import VLMSchema
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))  # Add parent to path
    from utils.schema_utils import VLMSchema

ANNOT_ROOT = Path("annotated_dataset")

def validate_schemas():
    """Validate all schema files in the annotated_dataset directory."""
    print("Starting schema validation...")

    errors = 0
    count = 0

    # Find all JSON files in schema_* directories
    for p in ANNOT_ROOT.rglob("schema_*/**/*.json"):
        if p.is_file():
            count += 1
            try:
                # Use model_validate_json in Pydantic V2
                VLMSchema.model_validate_json(p.read_text("utf-8"))
                print(f"[OK]    {p.relative_to(ANNOT_ROOT)}")
            except Exception as e:  # Catch Pydantic's ValidationError and others
                print(f"[ERROR] {p.relative_to(ANNOT_ROOT)}: {e}")
                errors += 1

    print("-" * 50)
    if errors:
        print(f"Finished validating {count} files with {errors} invalid schema file(s).")
        sys.exit(1)
    else:
        print(f"All {count} schema files validated successfully.")
        sys.exit(0)

if __name__ == "__main__":
    validate_schemas()