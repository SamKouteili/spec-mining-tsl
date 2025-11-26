import os
import json

DATA_DIR = "/Users/will/github/spec-mining-tsl/SygusFunctionSolver/examples/temp_data_1"
OUTPUT_DIR = "output_pairs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# clean function
def clean_line(raw):
    return (
        raw.replace("\ufeff", "")  # BOM
           .replace("\xa0", " ")   # NBSP
           .strip()
    )

def classify_and_store(prev_obj, next_obj, source, pairs):
    X1, Y1 = prev_obj["X"], prev_obj["Y"]
    X2, Y2 = next_obj["X"], next_obj["Y"]

    mappings = {
        "X1_to_X2":  (X1, X2),
        "X1_to_Y2":  (X1, Y2),
        "Y1_to_X2":  (Y1, X2),
        "Y1_to_Y2":  (Y1, Y2),
        "XY1_to_X2": ((X1, Y1), X2),
        "XY1_to_Y2": ((X1, Y1), Y2),
    }

    for cls, val in mappings.items():
        pairs[cls].append({
            "source": source,
            "input": val[0],
            "output": val[1]
        })

if __name__ == "__main__":

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".jsonl"):
            continue

        file_path = os.path.join(DATA_DIR, fname)

        # ---- create per-trace directory ----
        trace_dir = os.path.join(OUTPUT_DIR, fname.replace(".jsonl", ""))
        os.makedirs(trace_dir, exist_ok=True)

        # initialize fresh containers for THIS TRACE ONLY
        pairs = {
            "X1_to_X2": [],
            "X1_to_Y2": [],
            "Y1_to_X2": [],
            "Y1_to_Y2": [],
            "XY1_to_X2": [],
            "XY1_to_Y2": [],
        }

        # ---- load lines ----
        lines = []
        with open(file_path, "r") as f:
            for raw_line in f:
                clean = clean_line(raw_line)
                if not clean:
                    continue
                try:
                    obj = json.loads(clean)
                    lines.append(obj)
                except:
                    print("Bad JSON in", fname, "line:", repr(raw_line))

        # ---- process transitions ----
        for i in range(len(lines)-1):
            classify_and_store(
                lines[i], lines[i+1],
                f"{fname}:line_{i}→{i+1}",
                pairs
            )

        # ---- save per-trace output ----
        for cls, items in pairs.items():
            out_path = os.path.join(trace_dir, f"{cls}.jsonl")
            with open(out_path, "w") as f:
                for obj in items:
                    f.write(json.dumps(obj) + "\n")

        print(f"Processed trace: {fname}")

    print("Done!")
