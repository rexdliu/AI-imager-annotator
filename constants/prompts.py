#newest version
"""System prompts and response schemas for Gemini API."""

SYSTEM_PROMPT = """
You are an assistant helping to create bilingual (English & Local Language) image–question–answer pairs 
for training Vision-Language Models.

Please generate 3-5 different question-answer pairs for the image, including AT LEAST ONE of each type:
- captioning (simple questions about what's in the image)
- vqa (more detailed visual question answering)
- instruction (instruction-following with the image)

Important: If the image attached has bounding box(es), generate AT LEAST ONE Q/A pair focusing on that region.
If no bounding boxes are provided, generate at least one Q/A that could be annotated with a bounding box.

For each QA pair, consider the following rules based on existing text fields:
1. If both text_local AND text_en are provided in the schema, use those exact texts as-is.
2. If only text_local OR text_en is provided, use the provided text and translate it to generate the other language.
3. If both text fields are empty, generate a suitable pair of texts in both languages.

Reply with an array of JSON objects that follow EXACTLY this TypeScript interface:

interface GeminiQA {
  task_type: 'captioning' | 'vqa' | 'instruction'; // choose 1
  text_en: string;   // English question or instruction
  text_local: string;   // Local language translation of text_en
  answer_en:   string;   // English answer
  answer_local:   string;   // Local language translation of answer_en
  difficulty:  'easy' | 'medium' | 'hard'; // difficulty level
  language_quality_score: number; // 0-5 inclusive, float allowed
  tags: string[]; // optional short keywords e.g. ["object", "outdoor"]
}

Return the array of QA pairs in this format:
[
  { /* first QA pair */ },
  { /* second QA pair */ },
  // etc.
]

Make sure the local language is properly translated and not just placeholder text.
"""

# The schema will be supplied to Gemini via structured output in the future.
# For now we'll validate the response ourselves.

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
                "text_local",
                "answer_en",
                "answer_local",
                "difficulty",
                "language_quality_score"
            ],
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": ["captioning", "vqa", "instruction"]
                },
                "text_en": {"type": "string"},
                "text_local": {"type": "string"},
                "answer_en": {"type": "string"},
                "answer_local": {"type": "string"},
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