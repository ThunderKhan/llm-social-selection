import argparse
import sys

from .config import ConfigError, hash_config, load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="llm-social-selection",
        description="Reproducible LLM social-selection experiment framework.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an experiment configuration.",
    )

    validate_parser.add_argument(
        "config",
        help="Path to the YAML experiment configuration.",
    )

    args = parser.parse_args()

    if args.command == "validate":
        try:
            config = load_config(args.config)
        except ConfigError as error:
            print(f"Configuration invalid:\n{error}", file=sys.stderr)
            return 1

        print("Configuration valid")
        print(f"Schema version: {config.experiment.schema_version}")
        print(f"Selection mechanism: {config.selection.mechanism}")
        print(f"Population size: {config.population.size}")
        print(f"Config hash: {hash_config(config)[:12]}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
