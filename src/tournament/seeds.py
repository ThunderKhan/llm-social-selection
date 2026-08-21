import hashlib
import json


def derive_seed(
    trial_seed: int,
    round_index: int,
    namespace: str,
    *components: str,
) -> int:
    """Derive a stable 64-bit child seed for one isolated operation."""
    if not isinstance(trial_seed, int) or isinstance(trial_seed, bool):
        raise ValueError("trial_seed must be an integer")
    if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 0:
        raise ValueError("round_index must be a non-negative integer")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")
    if any(not isinstance(component, str) for component in components):
        raise ValueError("seed components must be strings")

    payload = json.dumps(
        {
            "components": components,
            "namespace": namespace,
            "round_index": round_index,
            "trial_seed": trial_seed,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
