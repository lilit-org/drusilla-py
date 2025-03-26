from dataclasses import dataclass


@dataclass
class Usage:
    """Tracks token usage and request counts for LLM API calls."""

    requests: int = 0
    """Number of API requests made."""

    input_tokens: int = 0
    """Total tokens sent to the API."""

    output_tokens: int = 0
    """Total tokens received from the API."""

    total_tokens: int = 0
    """Combined input and output tokens."""

    def add(self, other: "Usage") -> None:
        """Combines usage statistics from another Usage instance.

        Args:
            other: Usage instance to combine with
        """
        self.requests += other.requests or 0
        self.input_tokens += other.input_tokens or 0
        self.output_tokens += other.output_tokens or 0
        self.total_tokens += other.total_tokens or 0
