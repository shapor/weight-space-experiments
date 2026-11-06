# Re-running "Sparks of Weight-Space Programming"

An independent replication of [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang & Khattab, MIT, November 2026), built on the authors' public logs
([RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)).

By [shapor](https://shapor.com). The analysis was run with Claude (Anthropic) as the agent; GPT-6 Astra was run through the
Codex CLI. The write-up with charts is [`site/index.html`](site/index.html): one self-contained file, so open it in
a browser.

## What we found

- **The Qwen MATH-500 jump is mostly about finishing answers.** The post reports 7.9% → 48.1% after Astra's weight
  edit. In the post's setup, the unedited model runs out of tokens on 93% of answers. With thinking turned off (the
  model card's other recommended setting), we measured the unedited model at 45.2% and the edited model at 44.2%, on
  all 500 problems with the post's grading. Across the post's own released runs and ours, accuracy tracks the share of
  answers that finish.
- **The other released results reproduce.** Astra's hand-written MinAtar networks (Breakout 500.5 vs the post's 517.4),
  the Go networks against the released PPO opponents, and the car fix all land within a few percent of the post.
- **The car result is one run.** The level and physics are deterministic, so "1,024 of 1,024 episodes" is the same
  73-step episode 1,024 times.
- **Reading small networks isn't unique to Astra.** Given the same prompts blind, Claude Opus and Claude Fable named
  the games and diffusion shapes as well as Astra did. Astra stood clearly ahead on finding copying heads in a
  transformer.

## Follow-up: a programmed galaxy classifier, with the control it was missing

[weight-space-galaxy10](https://github.com/DaybreakWannaCode/weight-space-galaxy10) (by
[@hanshananyue](https://x.com/hanshananyue)) has Claude write all 475k weights of a CNN for Galaxy10 DECaLS: 151
hand-designed morphology measurements feeding a closed-form QDA classifier, 73.7% test accuracy with no gradient step.
Its report notes that no control had been run, so it was unclear how much came from the programmed measurements and
how much from the closed-form head, which will score something on almost any features.

We kept the head exactly as the author built it (hyperparameters frozen, fit on train only) and swapped only the 151
measurements. Our numpy version of the head reproduces the author's score on their measurements, so the comparison is
like for like.

| Measurements fed to the same head | Test accuracy | Test cross-entropy |
|---|---|---|
| The author's programmed measurements | 73.5% | 0.806 |
| Author's network, random first-layer filters (3 seeds) | 53–56% | 1.29–1.36 |
| Same architecture, all random weights (3 seeds) | 43–46% | 1.52–1.57 |
| PCA of 32×32 pixels, 151 components | 45.5% | 1.546 |
| Class prior / trained CNN (author's baselines) | 14.9% / 82.4% | 2.220 / 0.489 |

Here the control confirms the result. On generic features the head reaches about 45%, and the programmed measurements
add about 28 points on top. The baselines are generic rather than tuned, and the measurements were chosen with 23
validation passes, so the true margin is probably somewhat smaller, but it is clearly real.

## How the experiment is designed, and why

**Blind prompts.** Each model sees only the original prompt and weights, under a random six-character ID, with no
access to the post or to an answer key. Without that, a model that has read the post could recall the answers. The
key that maps IDs to truth is in `keys/` and was never in any model's working directory.

**Controls that keep the weights but destroy the structure.** For the game task, the Breakout network's board cells
are shuffled consistently across channels; for the diffusion task, pixels are shuffled through the first and last
layers so the network computes the same function on a scrambled image. A model that truly reads the weights should
notice. The pixel control was rejected by every model. The cell control wasn't: the game can be named without the
board layout (only left and right matter in the output layer, and exactly 8 input cells carry the paddle), so that
control turned out to be weaker than intended. We report it anyway.

**Same tools as the post.** For the game and shape tasks, Astra ran with every Codex tool disabled, as in the post.
The transformer task allows code in the original, so all three models could run Python on the weights there. Claude
models ran as fresh agents with the same prompt and nothing else in context.

**Our own scorers, not the authors'.** MinAtar scoring, the 9x9 Go engine and the Kinetix rollouts are written from
the rules stated in the prompts, so a bug in the authors' harness wouldn't silently reproduce. The Go engine follows
the exp6 prompt: simple ko is illegal, whole-board repetition loses for the mover, two passes or 162 moves end the
game, area scoring, komi 7.5.

**Qwen: the post's protocol plus two alternatives it didn't run.** We use the post's thinking-on sampling and
32,768-token cap, graded on the last `\boxed{}` answer with math-verify. We add thinking off (the model card's
non-thinking setting) and a forced answer at the limit (Qwen's own "answer now" cue appended to any cut-off answer).
Both ask the same question: does the edit add math ability, or does it mainly make the model stop? We use one sample
per problem rather than the post's four, to keep the GPU bill near $2. That adds about ±2 points of noise, far
smaller than the gaps being tested.

## What's here

| Path | What it holds |
|---|---|
| `site/index.html` | The write-up with charts |
| `blind/` | The blind prompts for experiments 1 and 2, including the two scrambled controls; `blind/exp3/task.txt` is the transformer task |
| `keys/` | Which blind ID is which network |
| `prompts/` | The Codex model-instructions file used for Astra |
| `results/astra`, `opus`, `fable` | Each model's answer per blind ID for experiments 1 and 2 (Astra also has its raw Codex event stream) |
| `results/astra3`, `opus3`, `fable3` | Answers for the transformer task |
| `results/numbers/` | Raw output of the MinAtar, Go and Kinetix runs |
| `results/qwen_cpu/` | An early 32-problem GSM8K check on CPU (directional only) |
| `results/qwen_gpu/` | MATH-500 runs on an NVIDIA L4 with vLLM |
| `results/galaxy/` | The Galaxy10 control: test and val scores per feature set |
| `scripts/` | Everything we ran, one file per job; each has its usage at the top |

## Reproducing

1. Clone the authors' logs: `git clone https://github.com/RikiyaT/lm_understand-logs logs`.
2. Rebuild the blind set with `scripts/mkblind.py logs blind`; it writes the key to `blind_key.json` beside the folder. The seeds are fixed, so you get the same IDs.
   For the transformer task, copy `logs/exp3_language/networks/<corpus>/blocks.npz` into a folder per ID as
   `weights.npz` (the mapping is in `keys/blind3_key.json`).
3. Run a model on a blind prompt: `scripts/run_astra.sh blind/<id>.txt results/astra` for Astra. For Claude, give a
   fresh agent the prompt file and ask it to answer in the format the prompt specifies.
4. MinAtar: `scripts/minatar_eval.py <game> <hand.npz|random|follow_ball> 1000` (MinAtar 1.0.15, sticky actions 0.1,
   10,000-step cap).
5. Go: `scripts/go_h2h.py <network-dir-A> <network-dir-B|random> 400`.
6. Car: `scripts/kx_det.py` checks determinism; `scripts/kx_policy.py <actor.npz>...` runs 1,024 episodes per actor
   (needs Kinetix).
7. Qwen weight diff: `scripts/qwen_diff.py <base-dir> <edited-dir>` shows the rank-1 edit to 64 columns of
   `mlp.down_proj` in layers 15 and 19.
8. Qwen MATH-500 on a GPU: `scripts/vm_run.sh <base|edit> <gs://bucket/prefix>` on a fresh Ubuntu GPU machine
   installs vLLM and runs `scripts/math500_vllm.py` in each mode, resuming after an interruption.

9. Galaxy10 control: download `Galaxy10_DECals.h5` (link in the galaxy repo's README), clone that repo, then
   `scripts/galaxy_control.py <h5> <clone>` (needs torch and h5py; no GPU).

## Not covered

- The sycophancy experiment (exp7): its checkpoints weren't released.
- The post's headline Go opponents: not released, so we compare only against the released same-size PPO networks.
- One sample per blind task per model. Repeating each a few times would show how reliable each model's reading is.

## License and credit

Our code is MIT-licensed (`LICENSE`). The blind prompts and keys are derived from the authors' logs, which are
CC BY 4.0: credit to Takehi, Teoh, Zhang and Khattab. We changed the prompts only by
renaming them, by scrambling the two controls, and in the transformer task by replacing the paragraph on how to
compute.
