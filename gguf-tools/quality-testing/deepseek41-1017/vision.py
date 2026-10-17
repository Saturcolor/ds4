"""Probe the compiled image preprocessing and freeze actual vision input identities."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(records, images, binary, out):
    if out.exists() or not os.access(out.parent, os.W_OK):
        raise ValueError("fresh writable task output required")
    out.mkdir()
    rows = []
    for case in records:
        for identity in case["image_identity"]:
            image = (images / identity["file"]).resolve()
            if not image.is_relative_to(images.resolve()):
                raise ValueError("image path escapes the provided asset root")
            if digest(image) != identity["raw_sha256"]:
                raise ValueError("frozen raw image differs")
            stem = identity["raw_sha256"]
            prefix = out / stem
            metadata_path = prefix.with_suffix(".json")
            if not metadata_path.exists():
                metadata = json.loads(subprocess.check_output([str(binary), str(image), str(prefix)], text=True))
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            metadata = json.loads(metadata_path.read_text())
            types = prefix.with_suffix(".types").read_bytes()
            mapping = {0: 0, 2: 1, 3: 2, 4: 3}
            semantic = [mapping[value] for value in types]
            start = identity["start"]
            expected = case["token_types"][start:start+len(types)]
            if semantic != expected:
                raise ValueError("semantic image token layout differs")
            if case["input_ids"][start:start+len(types)] != [129264]*len(types):
                raise ValueError("frozen image placeholder span differs")
            if [metadata["patch_count"], 3, 14, 14] != identity["patch_shape"]:
                raise ValueError("image patch geometry differs")
            if [metadata["grid_height"], metadata["grid_width"]] != [identity["n_vit_h"], identity["n_vit_w"]]:
                raise ValueError("native vision grid differs")
            rows.append({"case_id": case["case_id"], "image": identity["file"],
                         "raw_sha256": stem, "start": start,
                         "native_patches_sha256": identity["patches_sha256"],
                         "patch_values": metadata["patch_values"],
                         "semantic_token_layout_matches": True,
                         "image_token_count": len(types)})
            print(f'\rChecking image layout: {case["case_id"]}                    ', end="", flush=True)
    (out / "capture.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")


def summarize(out):
    import numpy as np
    import ml_dtypes
    rows = json.loads((out / "capture.json").read_text())["rows"]
    for row in rows:
        patches = np.fromfile(out / (row["raw_sha256"] + ".f32"), dtype="<f4")
        if patches.size != row["patch_values"] or not np.isfinite(patches).all():
            raise ValueError("incomplete or nonfinite candidate patches")
        sha = hashlib.sha256(patches.astype(ml_dtypes.bfloat16).tobytes()).hexdigest()
        row.update(candidate_patches_sha256=sha,
                   candidate_fp32_patches_sha256=digest(out / (row['raw_sha256'] + '.f32')),
                   bf16_patch_bytes_match=sha == row["native_patches_sha256"])
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "probe_build_source": json.loads((out / (rows[0]['raw_sha256'] + '.json')).read_text())["build_source"],
              "cases": len(set(row["case_id"] for row in rows)),
              "image_occurrences": len(rows),
              "unique_images": len(set(row["raw_sha256"] for row in rows)),
              "raw_images_geometry_and_semantic_layout_match": True,
              "bf16_patch_byte_matches": sum(row["bf16_patch_bytes_match"] for row in rows),
              "actual_model_inference_performed": False,
              "interpretation": "Patch differences describe existing preprocessing implementations; not evidence of a quantization defect. Preserve them separately in whole-pipeline vision comparisons.",
              "rows": rows}
    (out / "VISION-PREPROCESSING-ASSESSMENT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(flush=True)
    return out / "VISION-PREPROCESSING-ASSESSMENT.json"
