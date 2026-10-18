# Captures from another engine

Use the native panel's case directories and frozen `input_ids`, `positions` and `target_ids`. For each scored position `p`, feed `input_ids[:p+1]` and save the resulting next-token logits as an FP32 row with all 129,280 vocabulary entries. Preserve the tokenizer and checkpoint identities. A generated continuation with a different history is not comparable to these references.

Write one directory per case, with a copy of the reference `case.json` and the following candidate fields:

```python
metadata = dict(reference_case)
metadata.update(
    native_outputs_collected=False,
    outputs_collected=True,
    logit_dtype="float32",
    logit_chunks=[{
        "file": "logits.npy",
        "start_row": 0,
        "end_row": len(reference_case["positions"]),
        "sha256": sha256_of_saved_numpy_file,
    }],
)
```

Save the rows as a two-dimensional NumPy array with dtype `<f4`, using `np.save(..., allow_pickle=False)`. Chunk files must cover every scored row exactly once, in order. Do not change `input_ids_sha256`, `target_ids_sha256`, `positions`, `vocab`, `thinking_mode` or `reasoning_effort`. The candidate still identifies the original 1017 source checkpoint; add its engine, model/quantization and runtime details under a separate provenance field.

Run the complete text panel with:

```sh
python3 gguf-tools/quality-testing/deepseek41-1017/score_collections.py \
  --reference references-1017/short100 \
  --candidate candidate/short100 --out comparison
```

This route compares captures from any backend without using the ROCm collector. It does not provide a new Metal or CUDA capture implementation. Vision additionally requires matching raw images, geometry, token spans and explicit image-pipeline provenance; use the included ROCm runner for the tested vision path.
