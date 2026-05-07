# ETTh1 LSTM Forecasting

Multi-step time series forecasting on the **ETTh1** (Electricity Transformer Temperature) dataset using a from-scratch **LSTM** in PyTorch.

**Task.** Predict the Oil Temperature (`OT`) over the next 96 hours (24 hours ahead at hourly resolution) from a 192-hour lookback window.

**Status.** 🚧 Work in progress.

## Project structure

configs/    # YAML configurations
src/        # core library (data, models, training, evaluation, utils)
scripts/    # entry points: train.py, evaluate.py, predict.py
notebooks/  # exploratory analysis

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**For GPU (CUDA) support**, install the matching PyTorch build before the rest of the dependencies:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) to match your CUDA version.

## Dataset

ETTh1 is publicly available at [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset). Place `ETTh1.csv` under `data/raw/`.

## License

MIT