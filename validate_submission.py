"""Validate the frame,label submission contract using only the Python standard library."""

import argparse
import csv
import os
import sys

N_FRAMES = 199319
VALID_LABELS = {"H", "A", "D", "X"}


def validate(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [column.strip().lower() for column in next(reader)]
        except StopIteration:
            return "file is empty"
        if header != ["frame", "label"]:
            return "header must be exactly frame,label"

        rows = 0
        counts = {label: 0 for label in sorted(VALID_LABELS)}
        for rows, row in enumerate(reader, 1):
            if len(row) != 2:
                return "row {} must have exactly two cells".format(rows)
            frame_text, label_text = (cell.strip() for cell in row)
            if not frame_text or not frame_text.isascii() or not frame_text.isdecimal():
                return "frame at data row {} is not an integer".format(rows)
            frame = int(frame_text)
            if frame != rows - 1:
                return "expected frame {}, got {} at data row {}".format(
                    rows - 1, frame, rows
                )
            if not label_text:
                return "blank label at frame {}".format(frame)
            label = label_text.upper()
            if label not in VALID_LABELS:
                return "invalid label {!r} at frame {}".format(label_text, frame)
            counts[label] += 1

    if rows != N_FRAMES:
        return "expected {} data rows, found {}".format(N_FRAMES, rows)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", help="CSV to validate")
    args = parser.parse_args()
    if not os.path.isfile(args.submission):
        print("FAIL  no such file: {}".format(args.submission))
        return 1

    result = validate(args.submission)
    if isinstance(result, str):
        print("FAIL  {}".format(result))
        return 1

    print("OK    {} rows; label counts: {}".format(
        N_FRAMES, ", ".join("{} {}".format(k, v) for k, v in result.items())
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
