"""
Gemini API integration for vision model schema generation.

Uploads images to Gemini, gets bilingual Q/A pairs, and manages API interactions.
"""

import json
import base64
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any, Union
import mimetypes

import streamlit as st
import google.generativeai as genai
from pydantic import BaseModel, Field

from utils.env_utils import get_env_var
from utils.logger import get_logger

# Get logger for this module
logger = get_logger("gemini_api")

# Default model to use if not specified in environment
DEFAULT_GEMINI_MODEL = "gemini-pro-vision"


# ── Pydantic model for QA pairs ─────────────────────────────────────────
class GeminiQA(BaseModel):
    task_type: Literal["captioning", "vqa", "instruction", "role_play"]
    text_en: str
    text_ms: str
    answer_en: str
    answer_ms: str
    difficulty: Literal["easy", "medium", "hard"]
    language_quality_score: float
    tags: Optional[List[str]] = None


# ── Gemini API Client Class ─────────────────────────────────────────────
class GeminiClient:
    """Client for interacting with Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client.

        Args:
            api_key: Gemini API key (if None, will look for GEMINI_API_KEY env var)
        """
        self.api_key = api_key or get_env_var("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key must be provided either directly or as GEMINI_API_KEY environment variable")

        # Configure the Gemini API
        genai.configure(api_key=self.api_key)

    def get_image_mime_type(self, image_path: str) -> str:
        """Get the MIME type for an image file.

        Args:
            image_path: Path to the image file

        Returns:
            MIME type string
        """
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type and mime_type.startswith('image/'):
            return mime_type

        # Fallback based on file extension
        ext = Path(image_path).suffix.lower()
        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.heic': 'image/heic',
            '.heif': 'image/heif'
        }
        return mime_map.get(ext, 'image/jpeg')

    def prepare_image_part(self, image_path: str) -> Dict[str, Any]:
        """Prepare image data in the correct format for Gemini API.

        Args:
            image_path: Path to the image file

        Returns:
            Properly formatted image part for Gemini API
        """
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        mime_type = self.get_image_mime_type(image_path)

        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(image_data).decode('utf-8')
            }
        }

    def generate_content(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate content from Gemini with optional image input.

        Args:
            prompt: Text prompt
            image_path: Optional path to image file
            model: Gemini model name
            temperature: Sampling temperature (0-1)
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        # Initialize the generative model
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

        # Create model without safety settings to avoid compatibility issues
        model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
        )

        # Prepare the content list
        try:
            if image_path:
                logger.info(f"Adding image to request: {image_path}")
                image_part = self.prepare_image_part(image_path)

                # Build content list properly
                contents = []

                if system_prompt:
                    contents.append(system_prompt)

                contents.append(image_part)
                contents.append(prompt)

                # Generate content
                response = model_instance.generate_content(contents)
            else:
                # Text-only generation
                if system_prompt:
                    response = model_instance.generate_content([system_prompt, prompt])
                else:
                    response = model_instance.generate_content(prompt)

            return response.text

        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            raise

    def get_qa_from_image(
        self,
        image_path: str,
        existing_schema: Optional[Dict] = None,
        model: str = DEFAULT_GEMINI_MODEL
    ) -> List[GeminiQA]:
        """Generate QA pairs for an image.

        Args:
            image_path: Path to the image file
            existing_schema: Optional existing schema with text fields
            model: Gemini model to use

        Returns:
            List of GeminiQA objects
        """
        # Prepare system prompt
        system_prompt = get_system_prompt()

        # Add existing text fields to provide context
        prompt = "Please generate bilingual English-Malay question-answer pairs about this image for vision model training."

        # Add context from existing schema if available
        if existing_schema:
            # Add relevant fields to prompt
            context = []
            if existing_schema.get("text_en"):
                context.append(f"Existing English text: \"{existing_schema.get('text_en')}\"")
            if existing_schema.get("text_ms"):
                context.append(f"Existing Malay text: \"{existing_schema.get('text_ms')}\"")
            if existing_schema.get("task_type"):
                context.append(f"Current task type: {existing_schema.get('task_type')}")

            # Add information about bounding boxes if they exist
            if existing_schema.get("bounding_box") and existing_schema["bounding_box"]:
                num_boxes = len(existing_schema["bounding_box"])
                box_info = f"There are {num_boxes} bounding box(es) in the image."
                context.append(box_info)

            if context:
                prompt += "\n\nContext:\n" + "\n".join(context)

        # Make the API request
        logger.info(f"Requesting QA pairs for image: {image_path}")
        response_text = self.generate_content(
            prompt=prompt,
            image_path=image_path,
            model=model,
            temperature=0.7,
            system_prompt=system_prompt
        )

        logger.debug(f"Response text: {response_text[:500]}...")  # Log first 500 chars

        # Extract JSON array from response text
        try:
            # Try to find JSON array in the response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                qa_pairs_data = json.loads(json_text)
            else:
                # Fallback: Try to parse the entire response
                qa_pairs_data = json.loads(response_text)

            # Ensure we have a list of QA pairs
            if not isinstance(qa_pairs_data, list):
                if isinstance(qa_pairs_data, dict):
                    qa_pairs_data = [qa_pairs_data]  # Convert single object to list
                else:
                    raise ValueError(f"Expected list or dict, got {type(qa_pairs_data)}")

            # Validate and convert to GeminiQA objects
            qa_pairs = []
            for i, qa_data in enumerate(qa_pairs_data):
                # Make sure required fields exist
                for field in ["task_type", "text_en", "text_ms", "answer_en", "answer_ms", "difficulty"]:
                    if field not in qa_data:
                        logger.warning(f"QA pair {i} missing required field: {field}")
                        # Set default values for missing fields
                        if field == "task_type":
                            qa_data[field] = "vqa"
                        elif field == "difficulty":
                            qa_data[field] = "medium"
                        elif "text" in field or "answer" in field:
                            qa_data[field] = ""

                # Set default quality score if missing
                if "language_quality_score" not in qa_data:
                    qa_data["language_quality_score"] = 3.0

                # Validate with Pydantic
                try:
                    qa_pair = GeminiQA.model_validate(qa_data)
                    qa_pairs.append(qa_pair)
                except Exception as e:
                    logger.warning(f"Failed to validate QA pair {i}: {e}")
                    # Try to fix common issues
                    if "language_quality_score" in qa_data and not isinstance(qa_data["language_quality_score"], (int, float)):
                        try:
                            qa_data["language_quality_score"] = float(qa_data["language_quality_score"])
                        except (ValueError, TypeError):
                            qa_data["language_quality_score"] = 3.0

                    # Try validation again after fixes
                    try:
                        qa_pair = GeminiQA.model_validate(qa_data)
                        qa_pairs.append(qa_pair)
                    except Exception as e2:
                        logger.error(f"Failed to validate QA pair {i} after fixes: {e2}")

            # Ensure we have at least one of each task type if possible
            if len(qa_pairs) >= 3:
                task_types = set(qa.task_type for qa in qa_pairs)
                if "captioning" not in task_types:
                    # Find a QA pair we can convert to captioning
                    for qa in qa_pairs:
                        if qa.task_type != "captioning":
                            logger.info("Converting one QA pair to captioning type")
                            qa.task_type = "captioning"
                            break

            return qa_pairs

        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}", exc_info=True)
            raise ValueError(f"Failed to parse Gemini response: {e}")


# ── Helper Functions ─────────────────────────────────────────────────────

def get_system_prompt() -> str:
    """Return the system prompt for Gemini."""
    return """
    You are an assistant helping to create bilingual (English & Malay Language) image–question–answer pairs 
    for training vision-language models.

    Please generate 3-5 different question-answer pairs for the image, including AT LEAST ONE of each type:
    - captioning (simple questions about what's in the image)
    - vqa (more detailed visual question answering)
    - instruction (instruction-following with the image)
    - role_play (role-playing scenarios based on the image)

    Important: If the image has bounding box(es), generate AT LEAST ONE Q/A pair focusing on those regions.
    If no bounding boxes are mentioned, generate at least one Q/A that could be annotated with a bounding box.

    For each QA pair, consider the following rules based on existing text fields:
    1. If both text_ms AND text_en are provided in the schema, use those exact texts as-is.
    2. If only text_ms OR text_en is provided, use the provided text and translate it for the other language.
    3. If both text fields are empty, generate a suitable pair of texts in both languages.

    Reply with an array of JSON objects that follow EXACTLY this TypeScript interface:

    interface GeminiQA {
      task_type: 'captioning' | 'vqa' | 'instruction' | 'role_play'; // choose 1
      text_en: string;   // English question or instruction
      text_ms: string;   // Malay translation of text_en
      answer_en: string;   // English answer
      answer_ms: string;   // Malay translation of answer_en
      difficulty: 'easy' | 'medium' | 'hard'; // difficulty level
      language_quality_score: number; // 0-5 inclusive, float allowed
      tags: string[]; // optional short keywords e.g. ["object", "outdoor"]
    }

    Return the array of QA pairs in this format:
    [
      { /* first QA pair */ },
      { /* second QA pair */ },
      // etc.
    ]

    Make sure the Malay language is properly translated using proper Bahasa Malaysia and not just placeholder text.
    """


# ── Client Singleton ───────────────────────────────────────────────────────
_GEMINI_CLIENT = None


def get_gemini_client() -> GeminiClient:
    """Get or create a Gemini client singleton."""
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        api_key = get_env_var("GEMINI_API_KEY")
        # Also check session state for key set in UI
        if not api_key and "gemini_api_key" in st.session_state and st.session_state.gemini_api_key:
            api_key = st.session_state.gemini_api_key

        _GEMINI_CLIENT = GeminiClient(api_key)
    return _GEMINI_CLIENT


# ── Public API ─────────────────────────────────────────────────────────────
def generate_qa(
    image_path: str,
    existing_schema: Optional[Dict] = None,
    use_annotated_image: bool = False,
    model: Optional[str] = None
) -> List[GeminiQA]:
    """Generate QA pairs for an image using Gemini.

    Args:
        image_path: Path to the image file
        existing_schema: Optional existing schema with text fields
        use_annotated_image: Whether to use annotated image with boxes
        model: Gemini model name (defaults to env var or DEFAULT_GEMINI_MODEL)

    Returns:
        List of GeminiQA objects with QA pairs
    """
    # Determine the model to use
    gemini_model = model or get_env_var("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    # Determine the actual image path to use (original or annotated)
    img_path = Path(image_path)
    logger.info(f"Generating QA for image: {img_path.name}")
    logger.info(f"Using annotated image: {use_annotated_image}")

    if use_annotated_image:
        # Try to find an annotated image
        stem = img_path.stem
        try:
            from utils.file_utils import get_annotated_image_path
            annotated_path = get_annotated_image_path(image_path, stem)
            if annotated_path and annotated_path.exists():
                logger.info(f"Found annotated image: {annotated_path}")
                img_path = annotated_path  # Use annotated image
            else:
                logger.warning(f"Annotated image not found, using original")
        except Exception as e:
            logger.error(f"Error finding annotated image: {e}", exc_info=True)

    # Get Gemini client
    client = get_gemini_client()

    # Generate QA pairs
    try:
        qa_pairs = client.get_qa_from_image(
            str(img_path),
            existing_schema,
            gemini_model
        )
        logger.info(f"Generated {len(qa_pairs)} QA pairs")
        return qa_pairs
    except Exception as e:
        logger.error(f"Error generating QA pairs: {e}", exc_info=True)
        raise