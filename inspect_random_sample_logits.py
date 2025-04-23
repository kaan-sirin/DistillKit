###############################################################################
#  Usage:  python inspect_random_sample_logits.py  sparse_logits.pt   --n 3
###############################################################################
import argparse, torch, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer
# --------------------------------------------------------------------------- #
def inspect(path: Path, n_examples: int = 3):
    data = torch.load(path, map_location="cpu")   # list[ sequences ]

    n_seq           = len(data)
    steps_per_seq   = [len(seq)               for seq in data]
    tokens_per_step = [len(step)
                       for seq in data
                       for step in seq]

    print(f"\nFile   : {path}")
    print(f"Sequences             : {n_seq}")
    print(f"Steps / sequence      : avg {st.mean(steps_per_seq):.1f} | "
          f"min {min(steps_per_seq)}  max {max(steps_per_seq)}")
    print(f"Unique tokens / step  : avg {st.mean(tokens_per_step):.2f} | "
          f"min {min(tokens_per_step)}  max {max(tokens_per_step)}")
    print(f"Histogram (token‑count → #steps) :")
    hist = Counter(tokens_per_step)
    for k in sorted(hist):
        print(f"  {k:2d} → {hist[k]}")

    print("\nExamples:")
    for i in range(min(n_examples, n_seq)):
        print(f"\nSequence {i}")
        for t, step in enumerate(data[i][:3]):        # first 3 steps
            pairs = ', '.join(f"({tid},{prob:.3f})" for tid, prob in step[:6])
            print(f"  step {t:2} | {len(step)} ids | {pairs} ...")

# --------------------------------------------------------------------------- #

def print_first_n_sequences(path: Path, n: int = 3):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    data = torch.load(path, map_location="cpu")   # list[ sequences ]
    print(f"Printing first {n} sequences out of {len(data)}")
    for i in range(min(n, len(data))):
        print(f"\nSequence {i}")
        for t, step in enumerate(data[i]):
            step.sort(key=lambda x: x[1], reverse=True)
            print(tokenizer.decode(step[0][0], skip_special_tokens=False), end="")
        print()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=str, help="*.pt file produced by sampling script")
    ap.add_argument("--n", type=int, default=3, help="# sequences to print")
    args = ap.parse_args()

    inspect(Path(args.file), n_examples=args.n)
    # print_first_n_sequences(Path("generated_tokens/sassy/teacher_random_logits_320_R300_tau1.0.pt"), n=3)