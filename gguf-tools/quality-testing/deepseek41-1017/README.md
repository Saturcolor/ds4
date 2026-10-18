# DeepSeek V4.1 Flash1017 deep scorer

Compare a 1017 GGUF with saved full-vocabulary reference logits from the original checkpoint, using identical frozen token histories. This is separate from `score_official`: it adds expanded text, vision, long continuations and 4K–128K context panels. It measures quantization and runtime divergence, not task accuracy.

Exact checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`. Older V4.1 and V4 Flash 0731 artifacts are rejected. The native reference uses the original mixed FP8/FP4 weights; it is not a BF16 replacement or hosted top-N approximation.

## Build once

Use Python 3.11+ and the ROCm development environment for this fork:

```sh
python3 -m pip install -r gguf-tools/quality-testing/deepseek41-1017/requirements.txt
make -j4 deep-score-rocm ROCM_ARCH=gfx1151
```

Use the same source overlays and runtime as your inference build. The included 1017 vision checkpoint change only validates the matching encoder revision; it changes no preprocessing or inference equations. The build target is ROCm 10.0/gfx1151; use the matching 1017 BF16 vision encoder with the Q2 language model.

## Run

With the reference bundle and model already on disk:

```sh
python3 gguf-tools/quality-testing/deepseek41-1017/score.py \
  --model /path/DeepSeek-V4.1-Flash-1017-Q2.gguf \
  --vision /path/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf \
  --references /path/1017-references --out /path/results
```

The default is all nine panels, sequentially. Run with the GPU available for this model. The reference runtime and original native model weights are not needed; no API key, hosted inference or paid server is used. GGUF checks read metadata and file stats, without an independent full-model checksum pass.

| Suite | Coverage | Cases | Scored positions |
| --- | --- | ---: | ---: |
| `short` | Routine 100-case comparison | 100 | 6,159 |
| `text` | Short plus expanded text | 1,000 | 42,856 |
| `vision` | 32 image cases,40 image occurrences | 32 | 252 |
| `long` | Context compatibility, four native generation histories, fixed long continuations | 24 | 66,299 |
| `deep` | Six task types across 4K, 8K, 16K, 32K, 64K, 128K context buckets | 36 | 13,824 |
| `controls` | Native generation pilot and prefill control on the same 128-token history | 2 | 256 |
| `all` | All nine panels | 1,094 | 123,487 |

For a shorter run, add `--suite short`. Select individual panels with repeated `--panel`, for example `--panel compatibility-long12`. For an explicit smoke subset, add `--limit 1`; the report marks it as partial coverage. Full long/deep runs can take hours. Candidate raw logits use about 64 GB for all panels and remain available for later analysis.

Resume with the same command plus `--resume`. Completed panels are reused. A successful native collection can be compared again without rerunning inference; interrupted collections are retained in separate attempts. There is no inference timeout. Ctrl-C preserves evidence and stops the owned native process.

## Results

Open `report.md` for panel NLL, ΔNLL, forward KL and strict top-token agreement, with task-group, context-bucket, scored-position-type and image-aligner summaries. Rows weight scored positions; groups can have very different token counts, which are shown explicitly. `summary.json` and each panel's `SCORE.json` retain Jensen–Shannon divergence, total variation, entropy, probability regret, centered-logit RMSE, case-bootstrap confidence intervals and worst cases. Per-position arrays and raw full-vocabulary logits are retained.

Missing cases, wrong token IDs/positions, nonfinite vectors, changed files and mismatched checkpoint identities are errors. There is no combined quality percentage or universal pass threshold. Results include both quantization and runtime arithmetic. Image results include the complete preprocessing/encoder pipeline; aligner cosine/RMSE compare natural features before prompt-sentinel insertion.

The generation-history panels teacher-force saved native continuations. They do not measure independent candidate task success; all four long native traces reached their 4,096-token caps. Pilot/control histories overlap, so case counts are occurrences, not independent questions. Native numerical controls are shown separately and are never subtracted as an error floor. Hidden states, router traces and the original head remain available in the reference bundle as reference-only diagnostics; no uncollected candidate probes are claimed.

The deep-context panel scores 384 selected positions per case: prompt/boundary probes and tokens of a known oracle answer. Context buckets label the case's total input length; the scored-position-type rows separate answer-token metrics from prompt probes. Oracle answers are teacher-forced, so these measurements do not establish independently generated answer accuracy.

## Reuse existing captures

Compare saved `PANEL/pack` outputs or this runner's completed native captures entirely on CPU:

```sh
python3 gguf-tools/quality-testing/deepseek41-1017/score.py \
  --references /path/1017-references --existing-runs /path/candidate-runs \
  --suite short --out /path/recomputed-results
```

This validates every selected full vector and retains any original exit records. It does not infer a successful original collector exit from the presence of raw files.

Point `--existing-runs` at the original result directory to recompute metrics with a changed analysis script, without loading the model or repeating GPU inference. New native captures retain relative payload locations so the result directory can be moved together.

## Reference bundle

Large data stays outside Git. [references.json](references.json) pins all nine collection identities, case counts and token denominators. Build a portable directory from the saved native captures, evaluated images and numerical-control archive:

```sh
python3 gguf-tools/quality-testing/deepseek41-1017/bundle.py \
  --source /path/native-references --images /path/vision-qa-images \
  --controls /path/native-controls-v1.tar.gz --out /path/1017-references
```

The bundler preserves all reference payloads, including hidden/router data and vision features. It hardlinks on the same filesystem and copies across filesystems. It never changes source permissions or deletes source data. Logit payload integrity is verified when scoring.

## Checks

After building the helper, run the protocol/coverage tests without loading a model:

```sh
python3 -m unittest discover -s gguf-tools/quality-testing/deepseek41-1017/tests -v
```
