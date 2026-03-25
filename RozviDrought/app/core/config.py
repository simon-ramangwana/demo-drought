from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

MASTER_INPUTS_PATH = (
    Path(r"C:\Projects\Infer RozviDrought\data\master_inputs\master_inputs_long_198001_205012.parquet")
)

DEFAULT_GRID_PIXELS = 26775
DEFAULT_MODEL = "hybrid" # Options : "logit", "hybrid"