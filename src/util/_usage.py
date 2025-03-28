from dataclasses import dataclass

########################################################
#               Usage Tracking
########################################################


@dataclass
class Usage:
    """Track LLM API token usage and requests."""

    requests: int = 0
    """API request count."""

    input_tokens: int = 0
    """Tokens sent to API."""

    output_tokens: int = 0
    """Tokens received from API."""

    total_tokens: int = 0
    """Total tokens used."""

    def add(self, other: "Usage") -> None:
        """Merge usage stats from another instance."""
        self.requests += other.requests or 0
        self.input_tokens += other.input_tokens or 0
        self.output_tokens += other.output_tokens or 0
        self.total_tokens += other.total_tokens or 0
