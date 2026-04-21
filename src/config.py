SEED = 314

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / 'results'
PLOTS_PATH = RESULTS_PATH / 'plots'

LOG_PATH = RESULTS_PATH / 'experiment_log.csv'


BEESWARM_PATH = PLOTS_PATH / 'beeswarm.png'
WATERFALL_PATH = PLOTS_PATH / 'waterfall.png'
CONFUSION_MATRIX_PATH = PLOTS_PATH / 'confusion_matrix.png'
CALIBRATION_CURVE_PATH = PLOTS_PATH / 'calibration_curve.png'

MODEL_PATH = RESULTS_PATH / 'model.pkl'
