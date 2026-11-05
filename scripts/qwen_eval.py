# Scaled-down CPU replica of the post's exp8 protocol: thinking on, T=0.6 top_p=0.95 top_k=20, presence penalty 1.5,
# GSM8K test, but max_new_tokens capped (default 4096 vs the post's 32,768).
import sys, re, json, torch, time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
model_dir, out, n, cap, bs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
torch.manual_seed(0); torch.set_num_threads(int(sys.argv[6]) if len(sys.argv) > 6 else 12)
class Presence(LogitsProcessor):
    def __init__(s, plen, p): s.plen, s.p = plen, p
    def __call__(s, ids, scores):
        gen = ids[:, s.plen:]
        if gen.shape[1]: scores.scatter_add_(1, gen, torch.full(gen.shape, -s.p, dtype=scores.dtype))  # once per seen token
        return scores
tok = AutoTokenizer.from_pretrained(model_dir); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
def gold(a): return a.split("####")[-1].strip().replace(",", "")
def pred(t):
    b = re.findall(r"\\boxed\{([^{}]*)\}", t.split("</think>")[-1]) or re.findall(r"\\boxed\{([^{}]*)\}", t)
    return re.sub(r"[^\d.\-]", "", b[-1]) if b else None
res = []
for i in range(0, n, bs):
    rows = ds.select(range(i, min(i + bs, n)))
    prompts = [tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
               tokenize=False, add_generation_prompt=True, enable_thinking=True) for q in rows["question"]]
    enc = tok(prompts, return_tensors="pt", padding=True); t0 = time.time()
    # presence penalty must count each distinct token once: dedupe via a set-style mask
    class P(LogitsProcessor):
        def __call__(s, ids, scores):
            g = ids[:, enc.input_ids.shape[1]:]
            if g.shape[1]:
                m = torch.zeros_like(scores, dtype=torch.bool); m.scatter_(1, g, True); scores = scores - 1.5 * m
            return scores
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=cap, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                           logits_processor=LogitsProcessorList([P()]), pad_token_id=tok.pad_token_id)
    for j, r in enumerate(rows):
        g = o[j, enc.input_ids.shape[1]:]; L = int((g != tok.pad_token_id).sum()); text = tok.decode(g, skip_special_tokens=True)
        p = pred(text); ok = p is not None and gold(r["answer"]) == p.rstrip(".").removesuffix(".0").removesuffix(".00")
        res.append({"i": i + j, "len": L, "hit_cap": L >= cap, "pred": p, "gold": gold(r["answer"]), "correct": ok, "text": text})
    print(f"batch {i}: {time.time()-t0:.0f}s acc so far {sum(r['correct'] for r in res)}/{len(res)} cap-hits {sum(r['hit_cap'] for r in res)}", flush=True)
    json.dump(res, open(out, "w"))
print(json.dumps({"model": model_dir.split('/')[-3], "n": len(res), "acc": sum(r["correct"] for r in res) / len(res),
                  "hit_cap": sum(r["hit_cap"] for r in res) / len(res), "mean_len": sum(r["len"] for r in res) / len(res)}))
