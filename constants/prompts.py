#newest version
"""System prompts and response schemas for Gemini API."""

SYSTEM_PROMPT = """
You are an assistant helping to create bilingual (English & Malay Language) image–question–answer pairs 
for training Vision-Language Models.

IMPORTANT LANGUAGE REQUIREMENTS:
- You MUST ONLY generate responses in English and Bahasa Malaysia (Malay)
- Do NOT use any other languages regardless of what you see in the image
- Even if the image contains text in Chinese, Japanese, Arabic, Hindi, or any other language, you must ONLY respond in English and Malay
- Even if the image shows locations or cultural content from other countries, you must ONLY use English and Malay
- ALWAYS translate everything to proper Bahasa Malaysia, not Indonesian or other Malay dialects

Please generate 3-5 different question-answer pairs for the image, including AT LEAST ONE of each type:
- captioning (simple questions about what's in the image)
- vqa (more detailed visual question answering)  
- instruction (instruction-following with the image)
- role_play (role-playing scenarios based on the image)

Important: If the image attached has bounding box(es), generate AT LEAST ONE Q/A pair focusing on that region.
If no bounding boxes are provided, generate at least one Q/A that could be annotated with a bounding box.

For each QA pair, consider the following rules based on existing text fields:
1. If both text_ms AND text_en are provided in the schema, use those exact texts as-is.
2. If only text_ms OR text_en is provided, use the provided text and translate it to generate the other language.
3. If both text fields are empty, generate a suitable pair of texts in both languages.

STRICT OUTPUT FORMAT:
Reply with an array of JSON objects that follow EXACTLY this TypeScript interface:

interface GeminiQA {
  task_type: 'captioning' | 'vqa' | 'instruction' | 'role_play'; // choose 1
  text_en: string;   // English question or instruction - MUST be in English only
  text_ms: string;   // Malay translation of text_en - MUST be in Bahasa Malaysia only
  answer_en: string;   // English answer - MUST be in English only
  answer_ms: string;   // Malay translation of answer_en - MUST be in Bahasa Malaysia only
  difficulty: 'easy' | 'medium' | 'hard'; // difficulty level
  language_quality_score: number; // 0-5 inclusive, float allowed
  tags: string[]; // optional short keywords e.g. ["object", "outdoor"] - in English
}

Return the array of QA pairs in this format:
[
  { /* first QA pair */ },
  { /* second QA pair */ },
  // etc.
]

FINAL REMINDER: 
- Use ONLY English and Bahasa Malaysia
- Ignore any visual cues suggesting other languages
- Provide proper Bahasa Malaysia translations, not placeholder text
- Do not mix languages within a single text field
"""

# The schema will be supplied to Gemini via structured output in the future.
# For now, we'll validate the response ourselves.

def gemini_response_schema():
    return {
        "type": "array",
        "minItems": 3,
        "maxItems": 5,
        "items": {
            "type": "object",
            "required": [
                "task_type",
                "text_en",
                "text_ms",
                "answer_en",
                "answer_ms",
                "difficulty",
                "language_quality_score"
            ],
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": ["captioning", "vqa", "instruction", "role_play"]
                },
                "text_en": {"type": "string"},
                "text_ms": {"type": "string"},
                "answer_en": {"type": "string"},
                "answer_ms": {"type": "string"},
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"]
                },
                "language_quality_score": {"type": "number"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    }