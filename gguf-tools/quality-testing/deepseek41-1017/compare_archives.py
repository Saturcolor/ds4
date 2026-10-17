"""Compare exact-token native and candidate reference-format cases on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from reference_contract import require_matching_case, require_token_contract, distribution_metrics


class Case:
    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root/"case.json").read_text())
        require_token_contract(self.meta)
        if self.meta.get('outputs_collected',self.meta.get("native_outputs_collected")) is not True or self.meta.get("logit_dtype") != "float32":
            raise ValueError("incomplete or unsupported raw-logit case")
        self.chunks = []
        expected = 0
        for row in self.meta["logit_chunks"]:
            if row["start_row"] != expected or row["end_row"] <= expected:
                raise ValueError("missing or duplicate logit rows")
            path = self.root/row["file"]
            if Path(row['file']).name != row['file']:
                raise ValueError('invalid logit chunk path')
            with path.open("rb") as f:
                if hashlib.file_digest(f,"sha256").hexdigest() != row["sha256"]:
                    raise ValueError("raw-logit checksum mismatch")
            data = np.load(path,mmap_mode="r",allow_pickle=False)
            if data.dtype != np.dtype("<f4") or data.shape != (row["end_row"]-expected,self.meta["vocab"]):
                raise ValueError("wrong raw-logit shape/dtype")
            self.chunks.append((expected,row["end_row"],data))
            expected = row["end_row"]
        if expected != len(self.meta["positions"]) or expected != len(self.meta["target_ids"]):
            raise ValueError("incomplete target coverage")

    def rows(self,start,end):
        parts=[data[max(start-left,0):min(end-left,right-left)] for left,right,data in self.chunks if left<end and right>start]
        if sum(len(part) for part in parts) != end-start:
            raise ValueError("row coverage mismatch")
        return parts[0] if len(parts)==1 else np.concatenate(parts)


def compare(reference, candidate, out, *, vision_whole_pipeline=False):
    p,q=Case(reference),Case(candidate)
    require_matching_case(p.meta,q.meta,vision_whole_pipeline=vision_whole_pipeline)
    arrays={}
    for start in range(0,len(p.meta["positions"]),32):
        end=min(start+32,len(p.meta["positions"]))
        metrics=distribution_metrics(p.rows(start,end),q.rows(start,end),np.asarray(p.meta["target_ids"][start:end]))
        for name,values in metrics.items():
            arrays.setdefault(name,[]).append(values)
    arrays={name:np.concatenate(parts).astype(np.float64) for name,parts in arrays.items()}
    result={"case_id":p.meta["case_id"],"model_repo":p.meta["model_repo"],"model_revision":p.meta["model_revision"],
            "positions":len(p.meta["positions"]),"status":"complete exact-token pair; differences measured, no acceptance threshold implied",
            "metrics":{name:{"mean":float(values.mean()),"p50":float(np.quantile(values,.5)),
                              "p95":float(np.quantile(values,.95)),"p99":float(np.quantile(values,.99)),
                              "maximum":float(values.max())} for name,values in arrays.items()}}
    if vision_whole_pipeline:
        result['vision_comparison']=q.meta['vision_comparison']
        result['interpretation']='Whole image pipeline includes preprocessing, encoder, quantization and engine arithmetic; not isolated quantization error.'
    out=Path(out)
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+"\n")
    np.savez(out.with_suffix(".per-position.npz"),**arrays)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("reference",type=Path)
    parser.add_argument("candidate",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument('--vision-whole-pipeline',action='store_true')
    args=parser.parse_args()
    compare(args.reference,args.candidate,args.out,vision_whole_pipeline=args.vision_whole_pipeline)
