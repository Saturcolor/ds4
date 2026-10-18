# DeepSeek V4.1 Flash 1017 scorer

Compare a 1017 GGUF with saved full-vocabulary logits from the original checkpoint. The default is the 100-case panel; vision, longer continuations and 4K–128K context checks are optional. The [reference dataset](https://huggingface.co/datasets/kyuz0/deepseek-v4.1-flash-1017-qa) contains the exact prompts, token histories and native outputs, so running the original model again is unnecessary.

## Run the 100-case panel

Use a ROCm development container with enough memory for the GGUF. The collector is tested on ROCm 10.0/gfx1151; it has not been qualified on Metal. Run these commands from the repository root:

```sh
python3 -m pip install -r gguf-tools/quality-testing/deepseek41-1017/requirements.txt
make -j4 deep-score-rocm ROCM_ARCH=gfx1151

python3 gguf-tools/quality-testing/deepseek41-1017/download.py --out references-1017

python3 gguf-tools/quality-testing/deepseek41-1017/score.py \
  --model /models/DeepSeek-V4.1-Flash-1017-Q2.gguf \
  --references references-1017 --out results-1017
```

The downloader fetches only the selected panel, pins the dataset revision and resumes interrupted transfers through the Hub client. It checks the extracted collection identity and case coverage. Completed installations are reused; existing unrelated directories are not overwritten. Allow about 7.5 GB of free disk for the default download and extraction, plus space for the candidate captures.

Open `results-1017/report.md`. It shows native and candidate NLL, their difference, forward KL and top-token agreement. Lower KL means closer agreement with the native distribution. ΔNLL is candidate NLL minus native NLL; a positive value means less probability assigned to the frozen targets on average. Top-token agreement is not task accuracy. The reference uses the original mixed FP8/FP4 release, not a BF16 replacement or hosted top-k log probabilities.

The published Q2 completed all 100 cases and 6,159 scored positions:

| Native NLL | Q2 NLL | ΔNLL | Forward KL | Top-token agreement |
|---:|---:|---:|---:|---:|
| 0.363260 | 0.456892 | +0.093632 | 0.129874 | 90.03% |

The NLL definition matches DwarfStar's `avg_nll`. These prompts and continuation targets come from the September 19 V4.1 Flash panel, re-tokenized and evaluated on the 1017 checkpoint. Historical scores from a different checkpoint are not interchangeable.

## Choose another panel

Pass the same selection to both `download.py` and `score.py`:

| Selection | Cases | Scored positions |
|---|---:|---:|
| `--suite short` (default) | 100 | 6,159 |
| `--suite text` | 1,000 | 42,856 |
| `--suite vision` | 32 | 252 |
| `--suite long` | 24 | 66,299 |
| `--suite deep` | 36 | 13,824 |
| `--suite controls` | 2 | 256 |
| `--suite all` | 1,094 | 123,487 |

Use repeated `--panel NAME` arguments to select individual panels. Add `--vision /models/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf` when scoring vision; its image assets are downloaded automatically. Add `--limit N` to the scorer for an explicitly partial run. Long and deep panels repeat long prefills and take substantially longer than the default.

Use a new output directory for each candidate, or add `--resume` to continue an interrupted run. Completed captures are retained, including their raw FP32 vectors and per-position metrics. Candidate logits need about 3.2 GB for the default panel or 64 GB for all panels.

All comparisons use identical frozen histories and temperature-1 distributions. The long-generation panels score saved native continuations rather than independent candidate solutions; all four native long traces reached their 4,096-token cap. Deep-context oracle answers are teacher-forced. The pilot and prefill control overlap, so suite totals count case occurrences rather than independent questions.

## Analyse saved captures on CPU

No model load, API key or GPU is needed to recompute an existing capture:

```sh
python3 gguf-tools/quality-testing/deepseek41-1017/score.py \
  --references references-1017 --existing-runs candidate-runs \
  --out recomputed-results
```

For captures exported by another engine as per-case `case.json` and FP32 logit chunks, use `score_collections.py --reference references-1017/short100 --candidate candidate/short100 --out comparison`. The candidate must preserve the frozen token IDs, targets, scored positions and tokenizer identity; see [the capture format](CAPTURE-FORMAT.md).

The panel archives also preserve hidden states, expert routing and vision features for further analysis. `download.py --diagnostics` adds the shared original output head and raw numerical controls; ordinary scoring does not use it. Native numerical controls are reported separately and are not subtracted from divergence. Missing cases, wrong token histories, nonfinite vectors and changed payloads are rejected.
