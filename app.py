#newest version
"""Vision Schema Generator - Main App Entry Point

A Streamlit application for annotating images with bounding boxes and generating
bilingual question-answer pairs using Gemini AI for vision model training.
"""

import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import streamlit as st
from PIL import Image

# Import components
from components.canvas_box import draw_canvas
from components.json_viewer import show_json, interactive_json_editor, qa_card_selector
from components.sidebar import image_selector

# Import utils
from utils.gemini_api import generate_qa, GeminiQA
from utils.file_utils import (
    save_annotated_image,
    save_schema,
    check_existing_annotation,
    get_schema_stats,
    get_annotated_image_path,
    ANNOT_ROOT
)
from utils.logger import get_app_logger
from utils.schema_utils import VLMSchema, BBox, Metadata
from utils.env_utils import get_env_var

# Get logger for this module
logger = get_app_logger()

# --- Page Config ---
st.set_page_config(
    page_title="Vision Schema Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
default_keys = {
    "rects": [],  # Drawn boxes (display coords)
    "rect_colors": [],  # Colors for drawn boxes
    "schema": None,  # Current schema object
    "selected_image_path": None,  # Path selected via sidebar
    "current_image_path": None,  # Path being actively displayed
    "rotation_angle": 0,  # Canvas rotation
    "image_scale_factor": 1.0,  # Canvas display scale
    "last_action_time": time.time(),
    "processing_qa": False,  # Flag to disable buttons during QA generation
    "processing_confirm": False,  # Flag to disable buttons during confirmation
    "displayed_image": None,  # Displayed (possibly rotated) image
    "qa_pairs": None,  # Store multiple QA pairs from Gemini
    "use_annotated_image": False,  # Flag to use annotated image for Gemini
    "error_messages": [],  # Error messages that persist across reruns
    "box_color": "#FF0000",  # Default color for bounding boxes
    "zoom_level": 100,  # Default zoom level
}

for key, default_value in default_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# --- Helper Functions ---

def render_header():
    """Render the app header with stats."""
    st.title("📑 Vision Schema Generator")

    # Check if API key is configured
    api_key = get_env_var("GEMINI_API_KEY")
    if not api_key:
        st.warning("⚠️ Gemini API key not found in environment variables. Set the GEMINI_API_KEY in your .env file.")
    else:
        model = get_env_var("GEMINI_MODEL", "gemini-pro-vision")
        st.success(f"✅ Gemini API configured - Using model: {model}")

    try:
        stats = get_schema_stats()
        if stats["total"] > 0:
            category_list = ", ".join(stats.get('categories', []))
            if len(category_list) > 100:
                category_list = category_list[:100] + "..."
            st.caption(
                f"📊 Stats: {stats['total']} annotations ({stats['with_boxes']} with boxes) "
                f"across {stats.get('category_count', 0)} categories."
            )
        else:
            st.caption("📊 Stats: No annotations found yet.")
    except Exception as e:
        logger.error(f"Could not retrieve annotation stats: {e}", exc_info=True)
        st.warning(f"Could not retrieve annotation stats: {e}")
    st.markdown("---")


def display_persisted_errors():
    """Display any errors that have been persisted in the session state."""
    if st.session_state.error_messages:
        for error in st.session_state.error_messages:
            st.error(error)
        # Clear errors after displaying them once
        st.session_state.error_messages = []


def add_error(message: str):
    """Add an error message that will persist across reruns."""
    logger.error(message)
    st.session_state.error_messages.append(message)


def check_and_load_annotation(img_path: str) -> bool:
    """Check if annotation exists, load if checkbox checked. Return True if loaded."""
    logger.debug(f"Checking for existing annotation: {img_path}")
    existing_dict = check_existing_annotation(img_path)
    loaded = False
    if existing_dict:
        # Use image path in the key for uniqueness
        checkbox_key = f"load_existing_{img_path}"
        # Default to True only if schema is None or for a different path
        load_checked = st.session_state.schema is None or st.session_state.schema.image_path != img_path

        if st.sidebar.checkbox("Load existing annotation?", value=load_checked, key=checkbox_key):
            if st.session_state.schema is None or st.session_state.schema.image_path != img_path:
                logger.info(f"Loading existing annotation for: {img_path}")
                try:
                    schema = VLMSchema.model_validate(existing_dict)
                    st.session_state.schema = schema
                    st.session_state.rects = []  # Keep canvas empty initially when loading schema
                    st.sidebar.success(f"Loaded existing annotation for {Path(img_path).name}")
                    loaded = True
                except Exception as e:
                    logger.error(f"Error loading annotation for {img_path}: {e}", exc_info=True)
                    add_error(f"Error loading annotation: {str(e)}")
                    st.session_state.schema = None  # Reset on error
        elif st.session_state.schema is None:
            logger.debug(f"Existing annotation found for {img_path}, but checkbox unchecked.")
            st.sidebar.info(f"Existing annotation found but not loaded.")
    else:
        logger.debug(f"No existing annotation found for {img_path}.")
    return loaded


def handle_confirm_annotation(img_path: str, scaled_back_boxes: List[BBox], box_colors: List[str], rotated_img=None) -> Optional[VLMSchema]:
    """Handle Confirm: create/update schema, save schema & image."""
    st.session_state.last_action_time = time.time()
    saved_schema_obj = None
    logger.info(f"Handling confirm annotation for: {img_path}")

    try:
        img_path_str = str(Path(img_path))
        image_stem = Path(img_path_str).stem

        # Core data for schema creation/update
        core_data = {
            "image_path": img_path_str,
            "bounding_box": scaled_back_boxes,
            "task_type": "vqa" if scaled_back_boxes else "captioning",
        }

        # If schema is already loaded for this path, merge relevant fields
        existing_schema = st.session_state.schema
        if existing_schema and existing_schema.image_path == img_path_str:
            logger.debug("Merging with existing loaded schema...")
            # Preserve fields not directly set by drawing boxes
            core_data["difficulty"] = existing_schema.difficulty
            core_data["tags"] = existing_schema.tags
            core_data["text_en"] = existing_schema.text_en
            core_data["answer_en"] = existing_schema.answer_en
            core_data["text_ms"] = existing_schema.text_ms  # Updated to text_ms
            core_data["answer_ms"] = existing_schema.answer_ms  # Updated to answer_ms
            core_data["task_type"] = existing_schema.task_type
            core_data["language"] = existing_schema.language
            core_data["split"] = existing_schema.split
            core_data["source"] = existing_schema.source

            # Update metadata timestamp
            if existing_schema.metadata:
                existing_schema.metadata.timestamp = datetime.now()
                core_data["metadata"] = existing_schema.metadata
            else:
                core_data["metadata"] = Metadata()
        else:
            logger.debug("Creating new schema from scratch or overwriting different image's schema.")
            core_data["metadata"] = Metadata()

        # Create/Validate the Schema Object
        schema_obj = VLMSchema.model_validate(core_data)
        logger.debug(f"Validated Schema. Image ID set to: {schema_obj.image_id}")

        # Save the Schema JSON file
        schema_file_path = save_schema(schema_obj)
        logger.info(f"Schema saved/updated: {schema_file_path}")
        saved_schema_obj = schema_obj

        # Save Annotated Image with boxes
        try:
            rot_angle = st.session_state.rotation_angle if rotated_img else 0
            saved_img_path = save_annotated_image(
                img_path_str,
                schema_obj.image_id,
                scaled_back_boxes,
                box_colors,
                rotated_img,
                rot_angle
            )
            logger.info(f"Annotated image saved: {saved_img_path}")
        except Exception as img_e:
            logger.error(f"Failed to save annotated image copy: {img_e}", exc_info=True)
            add_error(f"Failed to save annotated image copy: {img_e}")

    except Exception as e:
        logger.error(f"Error processing or saving annotation: {e}", exc_info=True)
        add_error(f"Error processing or saving annotation: {str(e)}")
        return None

    return saved_schema_obj


def handle_qa_selection(qa: GeminiQA, schema: VLMSchema) -> Optional[VLMSchema]:
    """Update schema with the selected QA pair and save it."""
    try:
        # Make a deep copy of the schema
        updated_schema = schema.model_copy(deep=True)

        # Update schema fields with the selected QA pair
        updated_schema.task_type = qa.task_type
        updated_schema.text_en = qa.text_en
        updated_schema.text_ms = qa.text_ms  # Updated to text_ms
        updated_schema.answer_en = qa.answer_en
        updated_schema.answer_ms = qa.answer_ms  # Updated to answer_ms
        updated_schema.difficulty = qa.difficulty
        updated_schema.tags = qa.tags or []

        # Update metadata
        if updated_schema.metadata is None:
            updated_schema.metadata = Metadata()
        updated_schema.metadata.language_quality_score = qa.language_quality_score
        updated_schema.metadata.timestamp = datetime.now()

        # Save schema
        save_schema(updated_schema)
        logger.info("Schema updated with selected QA pair")

        # Clear QA pairs from session state
        st.session_state.qa_pairs = None

        return updated_schema
    except Exception as e:
        logger.error(f"Error updating schema with QA pair: {e}", exc_info=True)
        add_error(f"Error updating schema with QA pair: {e}")
        return None


def handle_gemini_qa(img_path: str, schema: VLMSchema, use_annotated_image: bool) -> None:
    """Call Gemini, get multiple QA pairs, and store them in session state for selection."""
    # Check if API key is set
    api_key = get_env_var("GEMINI_API_KEY")
    if not api_key:
        st.error("Gemini API key not found. Please set it in your .env file.")
        return

    st.session_state.last_action_time = time.time()
    st.session_state.processing_qa = True  # Set processing flag to disable buttons

    # Save current displayed image before processing
    current_displayed_image = st.session_state.displayed_image

    logger.info(f"Generating QA for image: {img_path}")
    logger.info(f"Using annotated image: {use_annotated_image}")

    try:
        # Get the existing schema as dict for context
        schema_dict = schema.model_dump()

        # Generate multiple QA pairs
        qa_pairs = generate_qa(
            img_path,
            existing_schema=schema_dict,
            use_annotated_image=use_annotated_image
        )

        # Store QA pairs in session state for selection
        st.session_state.qa_pairs = qa_pairs
        logger.info(f"Received {len(qa_pairs)} QA pairs from Gemini")

        # Restore the displayed image
        st.session_state.displayed_image = current_displayed_image

    except Exception as e:
        logger.error(f"Error generating Q/A: {e}", exc_info=True)
        add_error(f"Error generating Q/A: {str(e)}")
        st.session_state.qa_pairs = None

        # Still restore the displayed image on error
        st.session_state.displayed_image = current_displayed_image
    finally:
        st.session_state.processing_qa = False  # Reset processing flag


def get_scaled_boxes(boxes: List[BBox], scale_factor: float) -> List[BBox]:
    """Scale the bounding boxes by the given factor."""
    if abs(scale_factor - 1.0) < 1e-6:
        return boxes

    scaled_boxes = []
    for box in boxes:
        if isinstance(box, list) and len(box) == 4:
            scaled_box = [
                (int(round(x * scale_factor)), int(round(y * scale_factor)))
                for x, y in box
            ]
            scaled_boxes.append(scaled_box)
        else:
            logger.warning(f"Skipping invalid box during scaling: {box}")
    return scaled_boxes


def load_annotated_image(img_path: str, img_stem: str) -> Optional[Image.Image]:
    """Load the annotated image from disk if it exists."""
    try:
        # Try to get the annotated image path
        annotated_path = get_annotated_image_path(img_path, img_stem)

        if annotated_path and annotated_path.exists():
            logger.info(f"Loading annotated image from: {annotated_path}")
            return Image.open(annotated_path).convert("RGB")
        else:
            logger.debug("No annotated image found")
            return None
    except Exception as e:
        logger.error(f"Error loading annotated image: {e}", exc_info=True)
        return None


# --- Main App ---
def main():
    # --- Initial setup & Static Sidebar Elements ---
    logger.debug("Starting main application flow")

    render_header()
    display_persisted_errors()

    # Define persistent sidebar widgets
    st.sidebar.header("Dataset Images")
    search_term = st.sidebar.text_input("🔍 Search images", key="search_images")
    filter_options = ["All", "Annotated", "Not Annotated"]
    selected_filter = st.sidebar.radio(
        "Filter by status:", filter_options, horizontal=True, key="filter_status"
    )

    # Display sidebar list (and rename button)
    image_selector(search_term, selected_filter)

    # --- Get state potentially updated by sidebar ---
    img_path_selected = st.session_state.selected_image_path

    # --- State transition: Image Change Detection ---
    logger.debug("Checking for image change")
    rerun_needed = False
    if img_path_selected != st.session_state.current_image_path:
        logger.info(f"Image changed: '{st.session_state.current_image_path}' -> '{img_path_selected}'")
        st.session_state.current_image_path = img_path_selected
        # Reset state for the new image
        st.session_state.schema = None
        st.session_state.rects = []
        st.session_state.rect_colors = []
        st.session_state.rotation_angle = 0
        st.session_state.image_scale_factor = 1.0
        st.session_state.displayed_image = None
        st.session_state.qa_pairs = None
        st.session_state.use_annotated_image = False
        st.session_state.processing_qa = False
        st.session_state.processing_confirm = False
        rerun_needed = True

    # --- Load annotation IF image selected AND schema not loaded ---
    current_img_path = st.session_state.current_image_path
    annotation_loaded = False
    if current_img_path and st.session_state.schema is None:
        annotation_loaded = check_and_load_annotation(current_img_path)
        if annotation_loaded:
            logger.debug("Annotation was loaded, signaling rerun needed")
            rerun_needed = True

    # --- Handle QA Pair Selection (if pairs exist in session state) ---
    if st.session_state.qa_pairs and st.session_state.schema:
        logger.debug("Displaying QA pair selector")

        # Keep track of the current displayed image and current path
        current_displayed_image = st.session_state.displayed_image
        current_img_path = st.session_state.current_image_path
        current_img_stem = Path(current_img_path).stem if current_img_path else None

        # Function to handle QA selection while preserving the displayed image
        def handle_qa_selection_with_image_preserved(qa):
            # Update the schema
            updated_schema = handle_qa_selection(qa, st.session_state.schema)
            if updated_schema:
                st.session_state.schema = updated_schema

                # First try to preserve the current displayed image
                if current_displayed_image:
                    logger.debug("Preserving current displayed image")
                    st.session_state.displayed_image = current_displayed_image
                # If no current image or it was lost, try to load the saved annotated image
                elif current_img_path and current_img_stem:
                    logger.debug("Loading annotated image from disk")
                    annotated_img = load_annotated_image(current_img_path, current_img_stem)
                    if annotated_img:
                        logger.info("Successfully loaded annotated image")
                        st.session_state.displayed_image = annotated_img
                    else:
                        logger.warning("Could not load annotated image")

        # Display the card-based QA selector
        qa_card_selector(
            st.session_state.qa_pairs,
            handle_qa_selection_with_image_preserved
        )

        # If user has selected a QA pair, schema will have been updated and we need to rerun
        if st.session_state.qa_pairs is None:
            logger.debug("QA pair selection complete, rerunning")
            rerun_needed = True

    # --- Trigger Rerun if needed AFTER state updates ---
    if rerun_needed:
        logger.debug("Triggering rerun due to state transition")
        st.rerun()

    # --- Main Content Area ---
    if current_img_path:
        logger.debug(f"Rendering main content for: {current_img_path}")
        current_img_stem = Path(current_img_path).stem

        # --- Layout Containers ---
        st.header("📜 Schema & Actions")
        schema_placeholder = st.container()
        st.markdown("---")
        st.header(f"🖼️ Canvas: {Path(current_img_path).name}")
        rotation_controls_placeholder = st.container()
        canvas_placeholder = st.container()

        # --- Rotation ---
        with rotation_controls_placeholder:
            col_rot_1, col_rot_2 = st.columns([1, 4])
            with col_rot_1:
                rotate_button_disabled = st.session_state.processing_qa or st.session_state.processing_confirm
                if st.button("🔄 Rotate 90° CW", key=f"rotate_{current_img_path}",
                             disabled=rotate_button_disabled):
                    logger.info(
                        f"Rotating image 90° clockwise, angle: {st.session_state.rotation_angle} -> {(st.session_state.rotation_angle + 90) % 360}")
                    st.session_state.rotation_angle = (st.session_state.rotation_angle + 90) % 360
                    st.session_state.rects = []  # Clear boxes on rotation
                    st.session_state.rect_colors = []  # Clear colors on rotation
                    st.session_state.displayed_image = None  # Reset displayed image on rotation
                    st.warning("Bounding boxes cleared due to rotation.")
                    st.rerun()  # Rerun needed to redraw canvas rotated
            with col_rot_2:
                st.caption(f"Current display rotation: {st.session_state.rotation_angle}° clockwise")

        # --- Canvas ---
        with canvas_placeholder:
            try:
                # Check if we already have a displayed image from a previous operation
                if st.session_state.displayed_image is not None and not st.session_state.qa_pairs:
                    # Use the existing image but still allow drawing on it
                    boxes_display, scale_factor, displayed_image, box_colors = draw_canvas(
                        current_img_path,
                        st.session_state.rotation_angle
                    )
                    st.session_state.image_scale_factor = scale_factor
                    st.session_state.rects = boxes_display
                    st.session_state.rect_colors = box_colors
                    st.session_state.displayed_image = displayed_image

                    # Only update the displayed_image if we got a new one
                    if displayed_image is not None:
                        st.session_state.displayed_image = displayed_image
                else:
                    # Normal flow - draw canvas fresh
                    boxes_display, scale_factor, displayed_image, box_colors = draw_canvas(
                        current_img_path,
                        st.session_state.rotation_angle
                    )
                    st.session_state.image_scale_factor = scale_factor
                    st.session_state.rects = boxes_display
                    st.session_state.rect_colors = box_colors
                    st.session_state.displayed_image = displayed_image
            except Exception as e:
                logger.error(f"Failed to render canvas: {e}", exc_info=True)
                add_error(f"Failed to render canvas: {e}")
                st.session_state.rects = []
                st.session_state.image_scale_factor = 1.0
                st.session_state.displayed_image = None

        # --- Schema and Actions ---
        with schema_placeholder:
            schema_changed_in_section = False
            current_schema = st.session_state.schema

            if current_schema:
                # Check consistency: Schema ID should match current image stem
                if current_schema.image_id != current_img_stem:
                    logger.error(
                        f"State inconsistency: Schema ID {current_schema.image_id} ≠ Image stem {current_img_stem}")
                    add_error(
                        f"State inconsistency: Loaded schema ID '{current_schema.image_id}' does not match current "
                        f"image stem '{current_img_stem}'. Please re-select image.")
                    st.stop()

                # --- Schema Editor ---
                logger.debug("Rendering schema editor")
                updated_schema_obj = interactive_json_editor(
                    current_schema, key=f"editor_{current_img_path}"
                )
                if updated_schema_obj:
                    logger.info("Schema modified in editor")
                    st.session_state.schema = updated_schema_obj
                    try:
                        save_schema(updated_schema_obj)
                        st.success("Schema updated via editor and saved.")
                        schema_changed_in_section = True
                        current_schema = updated_schema_obj
                    except Exception as e:
                        logger.error(f"Error saving schema after edit: {e}", exc_info=True)
                        add_error(f"Error saving schema after edit: {e}")

                # --- Schema Display ---
                logger.debug(f"Displaying schema preview for: {current_schema.image_id}")
                show_json(current_schema, label=f"Schema Preview ({current_schema.image_id})")

            else:
                logger.debug("No schema loaded, showing info message")
                st.info("👆 Draw boxes (optional) and click Confirm to create the first schema.")

            # --- Action Buttons ---
            col1, col2 = st.columns(2)
            with col1:
                # Confirm button
                confirm_key = f"confirm_{current_img_path}"
                confirm_button_disabled = st.session_state.processing_qa or st.session_state.processing_confirm

                if st.button("✅ Confirm", use_container_width=True, type="primary",
                             key=confirm_key, disabled=confirm_button_disabled):
                    logger.info(f"Confirm button clicked for: {current_img_path}")
                    st.session_state.processing_confirm = True

                    try:
                        current_drawn_boxes = st.session_state.rects
                        current_box_colors = st.session_state.rect_colors
                        current_scale = st.session_state.image_scale_factor
                        scaled_back_boxes = get_scaled_boxes(current_drawn_boxes, current_scale)

                        # Show pending indicator
                        with st.spinner("Processing annotation..."):
                            # Pass the displayed image and colors
                            new_or_updated_schema = handle_confirm_annotation(
                                current_img_path,
                                scaled_back_boxes,
                                current_box_colors,
                                st.session_state.displayed_image
                            )

                            if new_or_updated_schema:
                                logger.info("Confirm annotation successful")
                                st.session_state.schema = new_or_updated_schema
                                st.success("✅ Annotation confirmed and saved successfully!")
                                schema_changed_in_section = True
                            else:
                                logger.warning("Confirm annotation failed or returned None")
                    finally:
                        # Re-enable buttons after processing completes
                        st.session_state.processing_confirm = False

            with col2:
                # Generate Q/A button
                qa_key = f"qa_btn_{current_img_path}"

                # Check if API key is available
                api_key = get_env_var("GEMINI_API_KEY")

                # Disable during any processing or if API key not provided
                qa_button_disabled = (current_schema is None or
                                     st.session_state.processing_qa or
                                     st.session_state.processing_confirm or
                                     not api_key)

                # If we have a schema, add image selection option for QA generation
                if current_schema:
                    # Check if annotated image exists
                    annotated_img_path = get_annotated_image_path(current_img_path, current_img_stem)
                    has_annotated_image = annotated_img_path is not None

                    # Only show the option if an annotated image exists
                    if has_annotated_image:
                        st.radio(
                            "Image to send to AI:",
                            ["Original Image", "Annotated Image (with boxes)"],
                            key="image_choice_radio",
                            index=1 if st.session_state.use_annotated_image else 0,
                            horizontal=True,
                            on_change=lambda: setattr(st.session_state, "use_annotated_image",
                                                     st.session_state.image_choice_radio == "Annotated Image (with boxes)")
                        )
                    else:
                        # If no annotated image yet, force original and show info
                        st.info("Save annotation first to use annotated image with AI.")
                        st.session_state.use_annotated_image = False

                if qa_button_disabled:
                    if not api_key:
                        st.caption("Gemini API key not found. Check your .env file.")
                    else:
                        st.caption("Confirm annotation first.")

                if st.button("🤖 Generate Q/A", type="secondary", use_container_width=True,
                             disabled=qa_button_disabled, key=qa_key):
                    if not qa_button_disabled:
                        logger.info(f"Generate Q/A button clicked for: {current_img_path}")

                        # Keep a reference to the display state
                        current_display_image = st.session_state.displayed_image

                        with st.spinner("Generating Q/A pairs with Gemini..."):
                            # Call Gemini to get multiple QA pairs
                            handle_gemini_qa(
                                current_img_path,
                                current_schema,
                                st.session_state.use_annotated_image
                            )

                        # Preserve the displayed image state
                        st.session_state.displayed_image = current_display_image

                        # Force rerun to show QA pair selection UI
                        st.rerun()

            # --- Rerun if schema changed by Confirm/QA/Edit in this section ---
            if schema_changed_in_section:
                logger.debug("Schema changed from actions, triggering rerun")
                st.rerun()

    else:
        # No image selected
        logger.debug("No image selected, showing initial info message")
        st.info("👈 Select an image from the sidebar to start annotating.")

        with st.expander("📘 How to use this app", expanded=True):
            st.markdown("""
             ### Quick Guide

             1. **Set up your environment**: Make sure your Gemini API key is in the `.env` file
             2. **Select an image** from the sidebar. Use search/filters. Annotation status (`✅`/`⚪`) is shown.
             3. **Choose a color** for your bounding boxes from the dropdown or color picker.
             4. (Optional) **Rotate** the image using `🔄 Rotate`.
             5. **Draw bounding boxes** (optional) on the canvas.
             6. **Confirm** (`✅ Confirm`) to save boxes and create/update the schema JSON and annotated image copy.
             7. **Generate Q/A** (`🤖 Generate Q/A`) via Gemini API (requires confirmed schema).
                - You can choose to use the original image or the annotated image with boxes.
                - Select from multiple AI-generated QA pairs presented as cards.
             8. (Optional) **Edit schema fields** (`✏️ Edit Schema Values`) for more options including language settings, metadata, and tags.
             9. (Optional) **Rename** original dataset files to UUIDs using the sidebar button (⚠️ **BACKUP FIRST**).
             """)


# --- Entry Point ---
if __name__ == "__main__":
    logger.info("***** Vision Schema Generator Starting *****")
    ANNOT_ROOT.mkdir(parents=True, exist_ok=True)  # Ensure output root exists
    main()