"""
Local Small Language Model agent for semantic extraction.

Provides a mock implementation that simulates the intent classification
and structured JSON generation that a local SLM (e.g., Phi-3-Mini via
llama-cpp-python) would perform.

In production, this would use GBNF grammar-constrained generation to
guarantee valid JSON output matching the expected schema.
"""

import json
import requests
from typing import Literal


class LLMAgent:
    """
    Mock local LLM agent for intent extraction.

    Simulates the behavior of a quantized Phi-3-Mini model running locally,
    classifying text into intent, urgency, and sentiment categories.
    """

    # Keyword heuristics simulating LLM classification
    _INTENT_KEYWORDS = {
        "focus": ["focus", "concentrate", "deep work", "coding", "writing", "design"],
        "learning": ["learn", "study", "course", "lecture", "training", "workshop"],
        "alertness": ["urgent", "alert", "breaking", "important", "deadline", "asap"],
        "relaxation": ["relax", "break", "lunch", "evening", "meditation", "yoga"],
    }

    _URGENCY_KEYWORDS = {
        "critical": ["critical", "emergency", "asap", "immediately"],
        "high": ["urgent", "important", "deadline", "priority"],
        "medium": ["scheduled", "planned", "upcoming"],
        "low": ["optional", "casual", "whenever", "low priority"],
    }

    _SENTIMENT_KEYWORDS = {
        "positive": ["breakthrough", "success", "great", "exciting", "achievement", "improved"],
        "negative": ["problem", "issue", "failure", "crisis", "decline", "concern"],
    }

    def extract_metadata(self, text: str, data_source: str = "unknown") -> str:
        """
        Extract structured semantic metadata from raw text.

        In production, this would prompt a local SLM with GBNF grammar
        constraints. For now, uses keyword matching as a mock.

        Args:
            text: Raw text from a calendar event, RSS feed item, etc.
            data_source: Source identifier (e.g., "calendar", "rss_feed").

        Returns:
            JSON string matching the schema expected by SemanticMapper:
            {"intent": str, "urgency": str, "sentiment": str,
             "data_source": str, "content_summary": str}
        """
        text_lower = text.lower()

        intent = self._classify(text_lower, self._INTENT_KEYWORDS, default="relaxation")
        urgency = self._classify(text_lower, self._URGENCY_KEYWORDS, default="medium")
        sentiment = self._classify(text_lower, self._SENTIMENT_KEYWORDS, default="neutral")

        metadata = {
            "intent": intent,
            "urgency": urgency,
            "sentiment": sentiment,
            "data_source": data_source,
            "content_summary": text[:200],
        }

        return json.dumps(metadata)

    @staticmethod
    def _classify(text: str, keyword_map: dict[str, list[str]], default: str) -> str:
        """Match text against keyword lists, returning the first matching category."""
        for category, keywords in keyword_map.items():
            if any(kw in text for kw in keywords):
                return category
        return default


class OllamaLLMAgent:
    """
    Ollama-backed LLM agent for semantic extraction.

    Uses a local Ollama instance with the qwen3.5-4b model to perform
    real semantic understanding and classification of text into structured
    metadata.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3.5-9b", timeout: int = 30):
        """
        Initialize the Ollama LLM agent.

        Args:
            base_url: Base URL for the Ollama API.
            model: Name of the Ollama model to use.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.api_url = f"{base_url}/api/generate"

    def extract_metadata(self, text: str, data_source: str = "unknown") -> str:
        """
        Extract structured semantic metadata from raw text using Ollama.

        Args:
            text: Raw text from a calendar event, RSS feed item, etc.
            data_source: Source identifier (e.g., "calendar", "rss_feed").

        Returns:
            JSON string matching the schema expected by SemanticMapper:
            {"intent": str, "urgency": str, "sentiment": str,
             "data_source": str, "content_summary": str}
        """
        prompt = self._build_prompt(text, data_source)

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 150},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            llm_output = result.get("response", "")

            # Parse and validate the JSON response
            metadata = self._parse_llm_output(llm_output, text, data_source)
            return json.dumps(metadata)

        except (requests.RequestException, json.JSONDecodeError, KeyError, Exception) as e:
            # If anything fails, return a default structured response
            return self._fallback_response(text, data_source)

    def _build_prompt(self, text: str, data_source: str) -> str:
        """Build the prompt for the LLM to extract structured metadata."""
        return f"""/no_think
Classify this text into a JSON object. Output ONLY valid JSON.

Text: "{text}"

Output exactly this JSON structure:
{{"intent": "<INTENT>", "urgency": "<URGENCY>", "sentiment": "<SENTIMENT>", "data_source": "{data_source}", "content_summary": "<SUMMARY>"}}

INTENT must be one of:
- "focus" = deep work, coding, writing, designing, problem-solving, concentration tasks
- "learning" = studying, courses, training, reading educational material
- "alertness" = urgent matters, breaking news, emergencies, deadlines
- "relaxation" = breaks, rest, yoga, meditation, leisure, winding down

URGENCY must be one of: "low", "medium", "high", "critical"
SENTIMENT must be one of: "positive", "neutral", "negative"

JSON:"""

    def _parse_llm_output(self, llm_output: str, text: str, data_source: str) -> dict:
        """Parse and validate LLM output, ensuring it matches the expected schema."""
        # Strip thinking blocks (qwen3.5 outputs <think>...</think> before JSON)
        llm_output = llm_output.strip()
        if "</think>" in llm_output:
            llm_output = llm_output.split("</think>", 1)[-1].strip()

        # Find the first complete JSON object by matching balanced braces
        start_idx = llm_output.find("{")
        if start_idx != -1:
            depth = 0
            for i in range(start_idx, len(llm_output)):
                if llm_output[i] == "{":
                    depth += 1
                elif llm_output[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = llm_output[start_idx : i + 1]
                        parsed = json.loads(json_str)
                        break
            else:
                parsed = json.loads(llm_output)
        else:
            parsed = json.loads(llm_output)

        # Validate and normalize the response
        metadata = {
            "intent": self._validate_field(
                parsed.get("intent"),
                ["focus", "relaxation", "alertness", "learning"],
                "relaxation"
            ),
            "urgency": self._validate_field(
                parsed.get("urgency"),
                ["low", "medium", "high", "critical"],
                "medium"
            ),
            "sentiment": self._validate_field(
                parsed.get("sentiment"),
                ["positive", "neutral", "negative"],
                "neutral"
            ),
            "data_source": data_source,
            "content_summary": parsed.get("content_summary", text[:200]),
        }

        return metadata

    @staticmethod
    def _validate_field(value: str | None, valid_options: list[str], default: str) -> str:
        """Validate that a field value is in the set of valid options."""
        if value and value.lower() in valid_options:
            return value.lower()
        return default

    def _fallback_response(self, text: str, data_source: str) -> str:
        """Generate a fallback response when Ollama is unavailable."""
        metadata = {
            "intent": "relaxation",
            "urgency": "medium",
            "sentiment": "neutral",
            "data_source": data_source,
            "content_summary": text[:200],
        }
        return json.dumps(metadata)


def create_agent(backend: Literal["ollama", "mock"] = "ollama") -> LLMAgent | OllamaLLMAgent:
    """
    Factory function to create an LLM agent.

    Args:
        backend: The backend to use ("ollama" for real LLM, "mock" for keyword matching).

    Returns:
        An instance of LLMAgent (mock) or OllamaLLMAgent.
    """
    if backend == "ollama":
        return OllamaLLMAgent()
    elif backend == "mock":
        return LLMAgent()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'mock'.")
