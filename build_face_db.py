#!/usr/bin/env python3
"""
build_face_db.py — SENTINEL face database builder

Usage:
    python build_face_db.py --input ~/known_faces --output ~/face_db.pkl

    # or with defaults:
    python build_face_db.py

Folder structure expected under --input:
    known_faces/
        Hemkesh/
            photo1.jpg
            photo2.jpg
        Yogesh/
            photo1.jpg

Each sub-folder name becomes the identity label.
The database stores one mean-normalised Facenet512 embedding per person,
which is exactly the format surveillance3_10.py expects.

Requirements: deepface, opencv-python (or JetPack OpenCV), numpy
"""

import argparse
import os
import pickle
import sys
import numpy as np

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")

DEFAULT_INPUT  = "~/known_faces"
DEFAULT_OUTPUT = "~/face_db.pkl"
MODEL_NAME     = "Facenet512"
DETECTOR       = "retinaface"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SENTINEL face database from a folder of reference photos."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Root folder with one sub-folder per person (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output path for the pickle database (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--detector",
        default=DETECTOR,
        choices=["retinaface", "yunet", "opencv", "ssd", "mtcnn"],
        help=f"Face detector backend (default: {DETECTOR}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir   = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)

    if not os.path.isdir(input_dir):
        sys.exit(f"❌  Input folder not found: {input_dir}")

    try:
        from deepface import DeepFace
    except ImportError:
        sys.exit("❌  DeepFace not installed. Run: pip install deepface")

    print("=" * 52)
    print("  SENTINEL — face database builder")
    print("=" * 52)
    print(f"  Input    : {input_dir}")
    print(f"  Output   : {output_path}")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Detector : {args.detector}")
    print()
    print("Building face database...")

    known_faces = {}

    for person_name in sorted(os.listdir(input_dir)):
        person_dir = os.path.join(input_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []

        for img_file in sorted(os.listdir(person_dir)):
            if not img_file.lower().endswith(SUPPORTED_EXTS):
                continue

            img_path = os.path.join(person_dir, img_file)
            try:
                result = DeepFace.represent(
                    img_path          = img_path,
                    model_name        = MODEL_NAME,
                    detector_backend  = args.detector,
                    enforce_detection = True,
                )
                emb = np.array(result[0]["embedding"])
                emb = emb / np.linalg.norm(emb)   # L2 normalise
                embeddings.append(emb)
                print(f"  ✅ {person_name}: {img_file}")

            except Exception as e:
                print(f"  ❌ {person_name}/{img_file}: {e}")

        if embeddings:
            # Store a single mean embedding — same format surveillance3_10.py loads
            known_faces[person_name] = np.mean(embeddings, axis=0)
            print(f"✅ {person_name} enrolled with {len(embeddings)} photo(s)")
        else:
            print(f"❌ {person_name}: no valid faces found")

    if not known_faces:
        sys.exit(
            "\n❌  No embeddings generated. Check that:\n"
            "  1. Sub-folders exist under the input directory\n"
            "  2. Photos contain a clearly visible face\n"
            "  3. retinaface model can download (needs internet on first run)"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(known_faces, f)

    print(f"\n✅ Database saved! {len(known_faces)} persons: {list(known_faces.keys())}")
    print(f"\n  Set in your .env:")
    print(f"    FACE_DB_PATH={output_path}")


if __name__ == "__main__":
    main()
