from __future__ import annotations

import argparse

from model.stable_audio_integration import write_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = write_dataset_config(output_path=args.output, audio_dir=args.audio_dir)
    print(path)


if __name__ == "__main__":
    main()
