"""Identity and distribution metrics for DeepSeek V4.1 Flash1017 only."""
import hashlib
import numpy as np

MODEL_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
MODEL_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
FIXTURE_CHECKPOINT = "DeepSeek-V4.1-Flash"


def require_native_identity(meta):
    if meta.get("model_repo") != MODEL_REPO or meta.get("model_revision") != MODEL_REVISION:
        raise ValueError("reference must identify the exact DeepSeek V4.1 Flash1017 checkpoint")


def token_digest(ids):
    values = np.asarray(ids)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= 129280):
        raise ValueError("invalid V4.1 token IDs")
    return hashlib.sha256(values.astype("<u4").tobytes()).hexdigest()


def require_matching_case(reference, candidate, *, vision_whole_pipeline=False):
    require_native_identity(reference)
    require_native_identity(candidate)  # candidate records its original source checkpoint
    for name in ("case_id", "input_ids_sha256", "target_ids_sha256", "positions", "vocab", "tokenizer_sha256"):
        if name not in reference or reference[name] != candidate.get(name):
            raise ValueError(f"reference/candidate contract mismatch: {name}")
    for name in ('thinking_mode','reasoning_effort','token_types_sha256'):
        if reference.get(name) != candidate.get(name):
            raise ValueError(f"reference/candidate contract mismatch: {name}")


    if not vision_whole_pipeline:
        if reference.get('image_identity') != candidate.get('image_identity'):
            raise ValueError('reference/candidate contract mismatch: image_identity')
    else:
        native,actual=reference.get('image_identity'),candidate.get('image_identity')
        if not native or not actual or len(native)!=len(actual) or not candidate.get('vision_comparison'):
            raise ValueError('explicit whole-pipeline vision provenance required')
        if candidate.get('reference_image_identity')!=native:
            raise ValueError('whole-pipeline reference image identity differs')
        for p,q in zip(native,actual):
            for key in ('file','raw_sha256','start','n_vit_h','n_vit_w','patch_shape','types_sha256'):
                if p[key]!=q.get(key):raise ValueError(f'whole-pipeline image contract mismatch: {key}')
            if not q.get('patches_sha256') or q.get('patch_dtype')!='float32' or not q.get('patches_diagnostic_bf16_sha256'):
                raise ValueError('actual FP32 and diagnostic BF16 candidate patch identities required')


def require_token_contract(meta):
    require_native_identity(meta)
    for field in ('input_ids','target_ids'):
        if meta.get(field+'_sha256') != token_digest(meta[field]):
            raise ValueError(f'frozen {field} identity mismatch')
    positions = meta['positions']
    if not positions or positions != sorted(set(positions)) or not all(isinstance(i,int) for i in positions):
        raise ValueError('invalid prediction positions')
    if positions[0] < 0 or positions[-1] >= len(meta['input_ids']) or len(positions) != len(meta['target_ids']):
        raise ValueError('prediction position/target coverage mismatch')
    if not isinstance(meta['vocab'],int) or meta['vocab'] < 2 or max(meta['target_ids']) >= meta['vocab'] or max(meta['input_ids']) >= meta['vocab']:
        raise ValueError('invalid reference vocabulary')


def distribution_metrics(reference, candidate, target_ids):
    """Per-position metrics, with float64 normalization of saved raw FP32 rows."""
    p_logits, q_logits = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target_ids)
    if p_logits.ndim != 2 or q_logits.shape != p_logits.shape or target.shape != p_logits.shape[:1]:
        raise ValueError("logit/target dimensions differ")
    if target.dtype.kind not in "iu" or np.any(target < 0) or np.any(target >= p_logits.shape[1]):
        raise ValueError("target ID out of range")
    if not np.all(np.isfinite(p_logits)) or not np.all(np.isfinite(q_logits)):
        raise ValueError("nonfinite logits")
    p_log = p_logits - np.logaddexp.reduce(p_logits, axis=1, keepdims=True)
    q_log = q_logits - np.logaddexp.reduce(q_logits, axis=1, keepdims=True)
    p, q = np.exp(p_log), np.exp(q_log)
    mid_log = np.logaddexp(p_log, q_log) - np.log(2.)
    rows = np.arange(target.size)
    p_top, q_top = p_logits.argmax(1), q_logits.argmax(1)
    centered_delta = ((q_logits - q_logits.mean(1, keepdims=True))
                      - (p_logits - p_logits.mean(1, keepdims=True)))
    top2 = np.partition(p_logits, -2, axis=1)[:, -2:]
    return {
        "native_nll": -p_log[rows, target],
        "candidate_nll": -q_log[rows, target],
        "nll_delta": p_log[rows, target] - q_log[rows, target],
        "forward_kl": np.sum(p * (p_log - q_log), axis=1),
        "jensen_shannon": .5 * (np.sum(p * (p_log - mid_log), axis=1) + np.sum(q * (q_log - mid_log), axis=1)),
        "total_variation": .5 * np.sum(np.abs(p-q), axis=1),
        "native_entropy": -np.sum(p * p_log, axis=1),
        "candidate_entropy": -np.sum(q * q_log, axis=1),
        "top1_same": p_top == q_top,
        "native_top1_margin": top2.max(1) - top2.min(1),
        "native_probability_regret": p[rows, p_top] - p[rows, q_top],
        "centered_logit_rmse": np.sqrt(np.mean(centered_delta**2, axis=1)),
    }
