import json

from .models import PromptProfile
from ..tasks import Task


def render_prompt(profile: PromptProfile, task: Task) -> str:
    """Render the minimal versioned prompt used by provider requests."""
    parameters = json.dumps(
        dict(profile.parameters),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (
        "SYSTEM ROLE:\n"
        f"Profile: {profile.profile_id}\n"
        f"Template version: {profile.template_version}\n"
        f"Parameters: {parameters}\n\n"
        "TASK:\n"
        f"{task.prompt}"
    )
