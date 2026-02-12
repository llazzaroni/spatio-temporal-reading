from pathlib import Path
import json

def select_best_model(root):
    best = None
    best_loss = float("inf")
    for run in root.iterdir():
        metrics = run / "metrics.json"
        weights = run / "best_model_baseline.pt"
        if metrics.exists() and weights.exists():
            with metrics.open() as f:
                val_loss = json.load(f).get("val_loss")
            if val_loss is not None and val_loss < best_loss:
                best_loss = val_loss
                best = run
    return best


root = Path("/Users/lorenzolazzaroni/Documents/Programming/Python/Research in DS/spatio-temporal-reading-proj/data")
models = ["BASE_LF_FILTERED", "BASE_LF_RAW"]
#models = ["RME_CSS_WS_FILTERED", "RME_CSS_WS_RAW", "BASE_LF_RAW", "BASE_SHP_RAW", "CSS_RAW", "poisson_raw_baseline", "RME_CSS_CHAR_LEN_FREQ_RAW", "RME_CSS_CHAR_WORD_LEN_FREQ_RAW", "RME_CSS_CS_RAW", "RME_CSS_DUR_RAW", "RME_CSS_FREQ_RAW", "RME_CSS_LEN_FREQ_RAW", "RME_CSS_LEN_RAW", "RME_CSS_RAW", "RME_CSS_WORD_LEN_FREQ_RAW", "RME_CSS_WS_RAW", "BASE_LF_FILTERED", "BASE_SHP_FILTERED", "CSS_FILTERED", "poisson_filtered_baseline", "RME_CSS_CHAR_LEN_FREQ_FILTERED", "RME_CSS_CHAR_WORD_LEN_FREQ_FILTERED", "RME_CSS_CS_FILTERED", "RME_CSS_DUR_FILTERED", "RME_CSS_FREQ_FILTERED", "RME_CSS_LEN_FREQ_FILTERED", "RME_CSS_LEN_FILTERED", "RME_CSS_FILTERED", "RME_CSS_WORD_LEN_FREQ_FILTERED", "RME_CSS_WS_FILTERED"]
for model in models:
    root_model = root / model
    print(select_best_model(root_model))