import re
from typing import Any, Callable, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

@register_validator(name="guardrails/persona_consistency", data_type="string")
class PersonaConsistency(Validator):
    """Validates that the output maintains a consistent persona.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `guardrails/persona_consistency`  |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        persona_description (string): A description of the expected persona.
        similarity_threshold (float): The minimum similarity score required to pass. Defaults to 0.7.
    """

    def __init__(
        self,
        persona_description: str,
        similarity_threshold: float = 0.7,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail, persona_description=persona_description, similarity_threshold=similarity_threshold)
        self._persona_description = persona_description
        self._similarity_threshold = similarity_threshold
        self._model = SentenceTransformer('all-MiniLM-L6-v2')
        self._persona_embedding = self._model.encode([self._persona_description])

    def validate(self, value: Any, metadata: Dict = {}) -> ValidationResult:
        """Validates that the output maintains a consistent persona."""
        if not isinstance(value, str):
            return FailResult(
                error_message="Input must be a string.",
            )

        # Encode the persona description and the output
        output_embedding = self._model.encode([value])

        # Calculate cosine similarity
        similarity = cosine_similarity(persona_embedding, output_embedding)[0][0]

        if similarity >= self._similarity_threshold:
            return PassResult()
        else:
            return FailResult(
                error_message=f"Output does not maintain the expected persona. Similarity score: {similarity:.2f}",
            )
