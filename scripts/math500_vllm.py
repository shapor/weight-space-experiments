"""MATH-500 with the post's think35 protocol, plus two cheap baselines the post didn't run.

usage: math500_vllm.py <model> <out_dir> <mode>
  think   thinking on, T0.6/top_p .95/top_k 20/presence 1.5, 32,768 new tokens (the post's protocol, 1 sample)
  nothink thinking off, the model card's non-thinking setting, 8,192 new tokens
  force   reads <out_dir>/think.jsonl; every answer cut at the limit gets Qwen's own "answer now" cue appended
          and up to 1,024 more tokens (budget forcing); uncut answers keep their grade
Graded like the post: last \\boxed{} answer, checked with math-verify.
Problems run in chunks of 100 appended to <out_dir>/<mode>.jsonl, so a restart resumes where it stopped.
"""
import json, os, sys
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
FORCE = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
THINK = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, seed=0)
NOTHINK = dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, seed=0)
CHUNK = 100


def boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + 7:j - 1] if depth == 0 else None


def correct(pred, gold):
    return pred is not None and bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))


def run_chunk(llm, mode, rows, prompts, prev):
    if mode == "force":
        recs = [dict(p) for p in prev]
        cut = [i for i, p in enumerate(prev) if p["cut"]]
        outs = llm.generate([prompts[i] + prev[i]["text"] + FORCE for i in cut], SamplingParams(max_tokens=1024, **THINK))
        for i, o in zip(cut, outs):
            recs[i].update(text=prev[i]["text"] + FORCE + o.outputs[0].text, pred=boxed(o.outputs[0].text))
    else:
        sp = SamplingParams(max_tokens=32768 if mode == "think" else 8192, **(THINK if mode == "think" else NOTHINK))
        recs = [dict(id=r["unique_id"], cut=o.outputs[0].finish_reason == "length", tokens=len(o.outputs[0].token_ids),
                     text=o.outputs[0].text) for r, o in zip(rows, llm.generate(prompts, sp))]
        for rec in recs:
            rec["pred"] = boxed(rec["text"])
    for r, rec in zip(rows, recs):
        rec["correct"] = correct(rec["pred"], r["answer"])
    return recs


def main():
    model, out, mode = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{mode}.jsonl")
    done = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    rows = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    prev = [json.loads(l) for l in open(os.path.join(out, "think.jsonl"))] if mode == "force" else None
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype="bfloat16", max_model_len=32768 + 2048 + 1024, gpu_memory_utilization=0.9, seed=0)
    for s in range(len(done), len(rows), CHUNK):
        part = rows[s:s + CHUNK]
        prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + "\n" + INSTR}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=(mode != "nothink")) for r in part]
        recs = run_chunk(llm, mode, part, prompts, prev[s:s + CHUNK] if prev else None)
        with open(path, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
        done += recs
        print(f"{mode} {len(done)}/{len(rows)}", flush=True)
    n = len(done)
    summary = dict(mode=mode, model=model, n=n, acc=sum(r["correct"] for r in done) / n,
                   cut_share=sum(r["cut"] for r in done) / n, mean_tokens=sum(r["tokens"] for r in done) / n)
    json.dump(summary, open(os.path.join(out, f"{mode}.summary.json"), "w"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":  # vLLM spawns workers, which re-import this file
    main()
