"""E09 experiment 3, step 1 (laptop): build intervention vectors and configs from the experiment 1 probes.

Everything is chosen on train / val records only:
    layer      the resid index with the highest VAL AUROC of the model's susceptibility probe (obey_base; for the base
               model this is its own obedience), excluding index 0; the hook goes on decoder layer (index - 1), whose
               output is that residual
    direction  unit raw-space probe direction (e09_probes probe_directions.npz)
    unit       sigma = std over the model's TRAIN injected rows of the projection of the final-token residual onto the
               obedience direction at that layer; strengths are multiples of sigma, and every control vector at the
               same strength has the same L2 norm (matched magnitude)
    controls   two random unit directions (seeded Gaussian) and the model's detection-probe direction at that layer
    prompts    the VAL records (all 7 conditions each); test records are reserved for a final confirmation run

Writes OUT/vectors.npz, OUT/records_val.json, OUT/records_test.json, OUT/configs_<model>.json, OUT/plan.json.
Usage: python e09_make_interventions.py --probes PROBES_DIR --acts ACTS_DIR --out OUT [--models base,routed]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e09_probes as P  # noqa: E402

# signed multiples of sigma; negative = against the "will obey" direction
STRENGTHS = {"base": {"obey": [-8, -4, -2, 4], "detect": [-8, -4, 4], "rand1": [8, 4], "rand2": [8]},
             "routed": {"obey": [4, 8], "detect": [8], "rand1": [8]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="base,routed")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = json.loads((Path(args.probes) / "results.json").read_text())
    dirs = np.load(Path(args.probes) / "probe_directions.npz")
    split = json.loads(P.SPLIT.read_text())["assign"]
    for part in ("val", "test"):
        (out / f"records_{part}.json").write_text(json.dumps({"ids": sorted(r for r, p in split.items() if p == part)}) + "\n")

    vectors, plan = {}, {}
    keys = None
    for m in args.models.split(","):
        keys, X, _ = P.load_model(m, Path(args.acts), keys)
        rows = [r for r in R["probes"][m]["obey_base"]["layers"] if r and r["layer"] > 0]
        best = max(rows, key=lambda r: r["val_auroc"])
        l = best["layer"]
        obey, det = dirs[f"{m}__obey_base__L{l}"], dirs[f"{m}__detect__L{l}"]
        rng = np.random.default_rng(1000 + l)
        r1, r2 = (v / np.linalg.norm(v) for v in rng.standard_normal((2, obey.shape[0])))
        tr = np.array([split[k[0]] == "train" and k[1] != "clean" for k in keys])
        proj = X[tr, l] @ obey
        sigma = float(proj.std())
        resid_norm = float(np.linalg.norm(X[tr, l], axis=1).mean())
        names = {"obey": f"{m}__obey_base__L{l}", "detect": f"{m}__detect__L{l}", "rand1": f"rand1__L{l}",
                 "rand2": f"rand2__L{l}"}
        vectors |= {names["obey"]: obey, names["detect"]: det, names["rand1"]: r1.astype(np.float32),
                    names["rand2"]: r2.astype(np.float32)}
        cfgs = [{"name": "control", "layer": l - 1, "vector": None, "norm": 0.0}]
        for kind, ks in STRENGTHS[m].items():
            for k in ks:
                # signed norm: delta = norm * v, so a negative norm subtracts the direction
                cfgs.append({"name": f"{kind}_L{l}_{'+' if k > 0 else '-'}{abs(k)}s", "layer": l - 1,
                             "vector": names[kind], "norm": float(k * sigma)})
        (out / f"configs_{m}.json").write_text(json.dumps(cfgs, indent=1) + "\n")
        plan[m] = {"resid_index": l, "hook_decoder_layer": l - 1, "val_auroc": best["val_auroc"], "test_auroc": best["auroc"],
                   "sigma": round(sigma, 4), "mean_resid_norm_last_token": round(resid_norm, 2),
                   "cos_obey_detect": round(float(obey @ det), 4), "cos_obey_rand1": round(float(obey @ r1), 4),
                   "configs": [c["name"] for c in cfgs]}
    np.savez(out / "vectors.npz", **vectors)
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
