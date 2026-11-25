"""Ajatuspaa experiment 001: frozen-model inward feedback. No training."""
import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ARMS = ("discrete", "fixed", "random", "periodic", "adaptive",
        "residue", "fixed_residue", "reset", "shuffled")
ADAPT = {"adaptive", "residue", "reset", "shuffled"}
MEMORY = {"residue", "fixed_residue", "reset", "shuffled"}
RANKS = (8, 32, 128, 0)  # 0 means full; low rank is only a proposed coarse view
DEFAULT_SEEDS = ["oak", "winter", "melody", "river", "justice",
                 "engine", "cabbage", "memory"]


def unit(x):
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def change(a, b):
    return float((1 - (unit(a) * unit(b)).sum()).clamp(0, 2))


def entropy(p):
    return float(-(p * p.clamp_min(1e-30).log()).sum())


def digest(x):
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def mean(xs):
    return sum(xs) / len(xs) if xs else None


class World:
    def __init__(self, model, tokenizer, calibration=192, batch=32):
        self.model = model.eval().requires_grad_(False)
        self.tok = tokenizer
        self.base = model.model
        self.emb = model.get_input_embeddings()
        self.out = model.get_output_embeddings()
        self.device = self.emb.weight.device
        self.dtype = self.emb.weight.dtype
        n = model.config.num_hidden_layers
        self.layers = sorted(set(max(1, min(n-1, round(n*f))) for f in (.30, .50, .72)))
        if n < 2:
            raise ValueError("Need at least two transformer blocks")
        self.rank = [min(k, calibration-1, model.config.hidden_size) if k else 0 for k in RANKS]
        self.vocab = min(self.emb.weight.shape[0], self.out.weight.shape[0],
                         len(tokenizer))
        self.banned = sorted(i for i in tokenizer.all_special_ids if 0 <= i < self.vocab)
        if self.vocab-len(self.banned) < 8:
            raise ValueError("Vocabulary too small")
        self.calibrate(calibration, batch)

    @torch.inference_mode()
    def calibrate(self, n, batch):
        rng = random.Random(410)
        banned = set(self.banned)
        allowed = [i for i in range(self.vocab) if i not in banned]
        ids = rng.sample(allowed, min(n, len(allowed)))
        samples = {l: [] for l in self.layers}
        for start in range(0, len(ids), batch):
            x = torch.tensor(ids[start:start+batch], device=self.device).unsqueeze(1)
            h = self.base(input_ids=x, use_cache=False, output_hidden_states=True,
                          return_dict=True).hidden_states
            for l in self.layers:
                samples[l].append(h[l][:, -1].float().cpu())
        self.basis = {}
        for l in self.layers:
            x = torch.cat(samples[l])
            mu = x.mean(0)
            xc = x-mu
            # Thin SVD, fit only on independent calibration token states.
            _, _, vh = torch.linalg.svd(xc, full_matrices=False)
            self.basis[l] = (mu.to(self.device), vh[:max(self.rank)].T.contiguous().to(self.device))
        print("Calibration:", len(ids), "tokens; layer addresses:", self.layers,
              "apertures:", self.rank, flush=True)

    @torch.inference_mode()
    def forward(self, context):
        h = self.base(inputs_embeds=context.unsqueeze(0).to(self.dtype),
                      use_cache=False, output_hidden_states=True, return_dict=True)
        final = h.last_hidden_state[0, -1]
        logits = self.out(final.to(self.out.weight.dtype)).float()[:self.vocab]
        if self.banned:
            logits[self.banned] = -torch.inf
        states = {l: h.hidden_states[l][0, -1].float() for l in self.layers}
        if not all(torch.isfinite(v).all() for v in [final, *states.values()]):
            raise FloatingPointError("nonfinite hidden state")
        return final.float(), logits, states

    def project(self, h, layer, rank):
        mu, b = self.basis[layer]
        if rank == 0:
            return h
        b = b[:, :rank]
        return mu + b @ (b.T @ (h-mu))

    @torch.inference_mode()
    def lens(self, state, shortlist):
        z = self.base.norm(state.to(self.dtype))
        w = self.out.weight.index_select(0, shortlist)
        bias = getattr(self.out, "bias", None)
        return F.linear(z.to(w.dtype), w, None if bias is None else bias[shortlist]).float()

    def initial(self, seed, length):
        ids = self.tok.encode(seed, add_special_tokens=False)
        if not ids:
            raise ValueError("Empty seed")
        ids = (ids * math.ceil(length / len(ids)))[-length:]
        return self.emb(torch.tensor(ids, device=self.device)).float()

    def decode(self, i):
        return self.tok.decode([int(i)])


class Head:
    def __init__(self, world, arm, seed):
        self.w, self.arm = world, arm
        self.rng = random.Random(seed + 1234)
        self.address = len(world.layers)//2
        self.aperture = 1
        self.dwell = 0
        self.prev = {}
        self.res = {l: torch.zeros_like(world.basis[l][0]) for l in world.layers}
        # Fixed permutation, no fresh noise injected at each tick.
        g = torch.Generator().manual_seed(891)
        self.perm = torch.randperm(world.model.config.hidden_size, generator=g).to(world.device)
        self.ema = None

    def observe(self, states, tick):
        changes = {l: change(h, self.prev[l]) if l in self.prev else 0.0
                   for l,h in states.items()}
        self.prev = {l: h.clone() for l,h in states.items()}
        self.dwell += 1
        local = changes[self.w.layers[self.address]]
        self.ema = local if self.ema is None else .65*self.ema + .35*local
        # Fixed thresholds selected before real runs; no target labels.
        if tick and self.dwell >= 3:
            old = (self.address, self.aperture)
            if self.arm == "random":
                self.address = self.rng.randrange(len(self.w.layers))
                self.aperture = self.rng.randrange(len(self.w.rank))
            elif self.arm == "periodic":
                self.address = (self.address+1) % len(self.w.layers)
                self.aperture = (self.aperture+1) % len(self.w.rank)
            elif self.arm in ADAPT:
                if self.ema < .08:
                    self.aperture = max(0, self.aperture-1)
                elif self.ema > .22:
                    self.aperture = min(len(self.w.rank)-1, self.aperture+1)
                self.address = min(range(len(self.w.layers)),
                    key=lambda i: abs(changes[self.w.layers[i]]-.15)
                                  + .015*(i != self.address))
            if old != (self.address, self.aperture) or self.arm in ("random","periodic"):
                self.dwell = 0
        l = self.w.layers[self.address]
        return l, self.w.rank[self.aperture], changes

    def read(self, h, layer, rank):
        z = self.w.project(h, layer, rank)
        r = self.res[layer]
        used = r[self.perm] if self.arm == "shuffled" else r
        if self.arm in MEMORY:
            # Add prior omitted state with a norm cap, in the reader only.
            cap = .15 * z.norm()
            z_read = z + used * (cap / used.norm().clamp_min(1e-8)).clamp(max=1)
            for k in self.res:
                self.res[k] = self.res[k] * .90
            self.res[layer] = self.res[layer] + .10*(h-z)
        else:
            z_read = z
        return z_read, float(r.norm()), float((h-z).norm())

    def clear(self):
        for l in self.res:
            self.res[l].zero_()


@torch.inference_mode()
def run(world, arm, seed, trial, steps, context_len, verbose=False):
    h = Head(world, arm, trial)
    gen = torch.Generator().manual_seed(trial)
    ctx = world.initial(seed, context_len)
    inputs, finals, rows = [], [], []
    for t in range(steps):
        if arm == "reset" and t == steps//2:
            h.clear()
        current = ctx[-1].clone()
        u = unit(current).cpu()
        inputs.append(u)
        final, logits, states = world.forward(ctx)
        finals.append(unit(final).cpu())
        l, rank, changes = h.observe(states, t)
        shortlist = logits.topk(min(64, world.vocab-len(world.banned))).indices
        native = logits[shortlist]
        if not torch.isfinite(native).all():
            raise FloatingPointError("nonfinite native shortlist")
        residue_norm = omitted = 0.0
        if arm == "discrete":
            p = (native/.8).softmax(0)
            j = int(torch.multinomial(p.cpu(), 1, generator=gen))
            token = int(shortlist[j])
            new = world.emb.weight[token].float()
            snapshot = token
            effective = 1.0
        else:
            z, residue_norm, omitted = h.read(states[l], l, rank)
            lens = world.lens(z, shortlist)
            # Geometric opinion pool: native output anchors uncalibrated lens.
            score = .65*native.log_softmax(0) + .35*lens.log_softmax(0)
            p = (score/.8).softmax(0)
            v, j = score.topk(min(8, len(score)))
            weights = (v/.8).softmax(0)
            token_ids = shortlist[j]
            em = world.emb.weight.index_select(0, token_ids).float()
            new = (weights[:, None]*em).sum(0)
            scale = (weights*em.square().mean(-1)).sum().sqrt()
            new = new*(scale/new.square().mean().sqrt().clamp_min(1e-8))
            snapshot = int(token_ids[0])  # never fed as a hard token
            effective = math.exp(entropy(weights))
        if not torch.isfinite(new).all():
            raise FloatingPointError("nonfinite feedback")
        lag = inputs[max(0,t-16):max(0,t-3)]
        recurrence = max(float(u @ q) for q in lag) if lag else None
        row = dict(t=t, snapshot=world.decode(snapshot), snapshot_id=snapshot,
                   native_top_id=int(shortlist[0]), native_top=world.decode(shortlist[0]),
                   layer=l, rank=rank, dwell=h.dwell, novelty=changes[l],
                   novelty_ema=h.ema, all_layer_changes=changes,
                   native_entropy=entropy(logits.softmax(0)), shortlist_entropy=entropy(p),
                   effective_carrier_tokens=effective, residue_norm=residue_norm,
                   omitted_norm=omitted, step_change=change(current,new),
                   recurrence=recurrence, input_sha=digest(current), next_sha=digest(new),
                   final_sha=digest(final))
        rows.append(row)
        if verbose:
            print(f"{t:03d} L{l:02d} rank={rank:3d} change={row['step_change']:.3f} "
                  f"memory={residue_norm:.2f} effective={effective:.2f} {row['snapshot']!r}", flush=True)
        ctx = torch.cat((ctx[1:], new.unsqueeze(0)), dim=0)
    late = rows[steps//2:]
    counts = Counter(r["snapshot_id"] for r in late)
    n = len(late)
    stats = dict(arm=arm, seed=seed, trial=trial, forwards=steps,
        recurrence=mean([r["recurrence"] for r in late if r["recurrence"] is not None]),
        step_change=mean([r["step_change"] for r in late]),
        repeat_fraction=1-len(counts)/n,
        effective_snapshot_vocab=math.exp(-sum(c/n*math.log(c/n) for c in counts.values())),
        effective_carrier_tokens=mean([r["effective_carrier_tokens"] for r in late]),
        layer_changes=sum(a["layer"]!=b["layer"] for a,b in zip(rows,rows[1:])),
        aperture_changes=sum(a["rank"]!=b["rank"] for a,b in zip(rows,rows[1:])))
    return rows, stats, unit(torch.stack(finals[steps//2:]).mean(0)), torch.stack(inputs)


def report(records, full_protocol=False):
    result = {}
    for arm in ARMS:
        group = [r for r in records if r[0]["arm"] == arm]
        if not group:
            continue
        # Equal weighting of seed concepts, not individual ticks.
        concepts = sorted(set(r[0]["seed"] for r in group))
        fields = ("recurrence","step_change","repeat_fraction","effective_snapshot_vocab",
                  "effective_carrier_tokens","layer_changes","aperture_changes")
        per_seed = {}
        for seed in concepts:
            g = [r for r in group if r[0]["seed"] == seed]
            per_seed[seed] = {k: mean([r[0][k] for r in g if r[0][k] is not None]) for k in fields}
        same, other = [], []
        for i,a in enumerate(group):
            for b in group[i+1:]:
                (same if a[0]["seed"]==b[0]["seed"] else other).append(float(a[1] @ b[1]))
        s, o = mean(same), mean(other)
        result[arm] = dict(**{k:mean([d[k] for d in per_seed.values() if d[k] is not None]) for k in fields},
                           same_seed_cos=s, other_seed_cos=o,
                           seed_specificity=s-o if s is not None and o is not None else None,
                           per_seed=per_seed)
    pairs = []
    lookup = {(r[0]["arm"],r[0]["seed"],r[0]["trial"]):r for r in records}
    for (arm,seed,trial), a in lookup.items():
        if arm != "residue":
            continue
        for control in ("reset","shuffled"):
            b = lookup.get((control,seed,trial))
            if b:
                d = (1-(a[2]*b[2]).sum(-1)).clamp(0,2)
                mid = len(d)//2
                pairs.append(dict(seed=seed, trial=trial, control=control,
                    pre_max=float(d[:mid].max()), post_mean=float(d[mid:].mean())))
    complete = full_protocol and all((a,s,t) in lookup for a in ARMS for s in DEFAULT_SEEDS for t in range(3))
    gate = dict(complete_default_matrix=complete, status="incomplete")
    if complete:
        rr = result["residue"]
        controls = [result[a]["recurrence"] for a in ("fixed","random","periodic")]
        specificity = rr["seed_specificity"]
        fixed_specificity = result["fixed"]["seed_specificity"]
        reset_pairs = [p for p in pairs if p["control"]=="reset"]
        gate.update(status="evaluated_mechanical_diagnostics_only",
            gate0_reset_pre_match=all(p["pre_max"] < 1e-5 for p in reset_pairs),
            gate1_recurrence_gain=min(controls)-rr["recurrence"],
            gate1_specificity_delta=specificity-fixed_specificity,
            gate1_pass=(min(controls)-rr["recurrence"] >= .02 and
                        specificity-fixed_specificity >= -.02),
            gate2_reset_post_divergence=mean([p["post_mean"] for p in reset_pairs]),
            gate2_shuffled_post_divergence=mean([p["post_mean"] for p in pairs if p["control"]=="shuffled"]))
    return dict(arms=result, memory_pairs=pairs, gates=gate)


class TinyTokenizer:
    all_special_ids = [0,1,2]
    def __len__(self):
        return 257
    def encode(self, text, add_special_tokens=False):
        return [3+ord(c)%254 for c in text]
    def decode(self, ids):
        return str(ids[0])


def self_test():
    # Actual randomly initialized Qwen3 architecture, no network or download.
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(17)
    cfg = Qwen3Config(vocab_size=257, hidden_size=48, intermediate_size=96,
                     num_hidden_layers=4, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=12, max_position_embeddings=64)
    model = Qwen3ForCausalLM(cfg)
    w = World(model,TinyTokenizer(),calibration=32,batch=16)
    before = [digest(p) for p in model.parameters()]
    records = []
    for arm in ARMS:
        rows,s,f,x = run(w,arm,"oak",0,8,4)
        assert len(rows)==8 and s["forwards"]==8
        assert all(math.isfinite(r["step_change"]) for r in rows)
        if arm=="discrete":
            assert all(r["effective_carrier_tokens"]==1 for r in rows)
        records.append((s,f,x))
    assert before == [digest(p) for p in model.parameters()]
    assert all(not p.requires_grad for p in model.parameters())
    r = report(records)
    reset = next(x for x in r["memory_pairs"] if x["control"]=="reset")
    assert reset["pre_max"] < 1e-5, reset
    print("Self-test PASS (random tiny Qwen3; plumbing only).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto","float32","float16","bfloat16"],default="auto")
    p.add_argument("--steps",type=int,default=64)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--trials",type=int,default=3)
    p.add_argument("--seeds",nargs="+",default=DEFAULT_SEEDS)
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    p.add_argument("--out",type=Path,default=Path("runs/experiment001"))
    p.add_argument("--verbose",action="store_true")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    if a.steps<8 or a.context<1 or a.trials<1:
        p.error("steps >=8, context >=1, trials >=1 required")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    dtype = (torch.float32 if a.device=="cpu" else torch.float16) if a.dtype=="auto" else getattr(torch,a.dtype)
    torch.manual_seed(0)
    print("Loading",a.model,a.device,dtype,flush=True)
    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype).to(a.device)
    w=World(model,tok)
    a.out.mkdir(parents=True,exist_ok=True)
    meta=dict(model=a.model,revision=getattr(model.config,"_commit_hash",None),dtype=str(dtype),
        device=a.device,torch=torch.__version__,transformers=transformers.__version__,
        steps=a.steps,context=a.context,trials=a.trials,seeds=a.seeds,arms=a.arms,
        layers=w.layers,ranks=w.rank,frozen=all(not p.requires_grad for p in model.parameters()))
    (a.out/"config.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    records=[]
    with (a.out/"traces.jsonl").open("w",encoding="utf-8") as f:
        for seed in a.seeds:
            for trial in range(a.trials):
                for arm in a.arms:
                    start=time.perf_counter()
                    try:
                        rows,s,final,inputs=run(w,arm,seed,trial,a.steps,a.context,a.verbose)
                    except Exception as e:
                        f.write(json.dumps(dict(seed=seed,trial=trial,arm=arm,error=repr(e)))+"\n")
                        f.flush()
                        raise
                    s["seconds"]=time.perf_counter()-start
                    f.write(json.dumps(dict(stats=s,rows=rows),ensure_ascii=False,allow_nan=False)+"\n")
                    f.flush()
                    records.append((s,final,inputs))
                    print(seed,trial,arm,"recurrence",s["recurrence"],"repeat",round(s["repeat_fraction"],3),flush=True)
                    (a.out/"summary.json").write_text(json.dumps(report(records, a.steps==64 and a.context==8 and a.trials==3 and set(a.seeds)==set(DEFAULT_SEEDS) and set(a.arms)==set(ARMS)),indent=2,allow_nan=False),encoding="utf-8")
    print("Wrote",a.out,flush=True)


if __name__=="__main__":
    main()
