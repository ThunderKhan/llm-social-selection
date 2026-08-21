from dataclasses import dataclass


def _require_non_empty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    prompt: str
    expected_answer: str | None
    scorer_version: str

    def __post_init__(self) -> None:
        _require_non_empty(self.task_id, "task_id")
        _require_non_empty(self.family, "family")
        _require_non_empty(self.prompt, "prompt")
        _require_non_empty(self.scorer_version, "scorer_version")
        if self.expected_answer is not None and not isinstance(
            self.expected_answer, str
        ):
            raise ValueError("expected_answer must be a string or None")
