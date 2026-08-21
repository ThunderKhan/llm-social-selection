import argparse


def main() -> None:
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
        print(f"Validating configuration: {args.config}")


if __name__ == "__main__":
    main()