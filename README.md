# F1 Lap Time Predictor (MSML 612 Final Project)

A decoder only transformer that predicts Formula 1 lap times using real race telemetry from the 2024-2025 seasons. Given the last 15 laps of a driver's race, the model forecasts the next 5 laps, informed by tyre compound, weather and circuit context.

## Architecture

```
Input (15 laps × 14 features)
  ├── 12 numerical features → linear projection → d_model=64
  ├── Positional embeddings (learned, max_len=20)
  ├── Driver embedding (27 drivers)
  └── Track embedding (24 circuits)
          ↓
  2× DecoderBlock
    └── Causal multi-head attention (4 heads)
        + Layer norm + Feedforward (d_ff=256)
          ↓
  Output head → 5 residual predictions
```

| Hyperparameter | Value |
|----------------|-------|
| Input features | 12 numerical + 2 IDs |
| Model dim (`d_model`) | 64 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feedforward dim | 256 |
| Dropout | 0.1 |
| Context window | 15 laps |
| Prediction horizon | 5 laps |


## Data

**Source**: FastF1 API — 48 races from the 2024 and 2025 Formula 1 seasons

**Features per lap**:
| Feature | Description |
|---------|-------------|
| `LapTime` | Total lap duration (seconds) |
| `Sector1Time`, `Sector2Time`, `Sector3Time` | Sector splits |
| `SpeedST` | Speed trap reading |
| `TyreLife` | Laps completed on current tyre set |
| `CompoundEncoded` | Tyre type (SOFT=0, MEDIUM=1, HARD=2, WET=3, INTER=4) |
| `AirTemp`, `TrackTemp` | Environmental conditions |
| `Rainfall` | Precipitation level |
| `StintNumber` | Current stint index |
| `PrevCompound` | Compound used in previous stint |
| `DriverID` | Integer ID per driver (27 total) |
| `TrackID` | Integer ID per circuit (24 total) |

**Preprocessing**:
- Lap times filtered to 60–200 seconds; lap 1 and pit stop laps removed
- Groups split 70/15/15 (train/val/test) at the driver-race level — no leakage across splits
- Sliding windows of 15 context laps → 5 prediction laps
- Numerical features standardized with a `StandardScaler` fit on training data only

| Split | Windows |
|-------|---------|
| Train | 20,068 |
| Validation | 4,449 |
| Test | 4,217 |

27 drivers and 24 tracks in total

## Training


## Repository Structure

```
MSML612Project/
├── app.py                    # Streamlit web app
├── data_main.py              # Run full data pipeline end-to-end
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── fetch.py          # Pull race data via FastF1
│   │   ├── preprocess.py     # Cleaning, encoding, windowing, normalization
│   │   ├── dataset_file.py   # PyTorch Dataset / DataLoader
│   │   └── inference.py      # Online inference pipeline for the app
│   └── model/
│       ├── transformer.py    # LapTimeTransformer architecture
│       ├── training.py       # Training loop
│       └── evaluate.py       # Batch evaluation on test split
├── data/
│   ├── raw/                  # all_laps.csv from FastF1
│   └── splits/               # Preprocessed .npy arrays, scaler, metadata
├── models/
│   └── best_model.pth        # Saved checkpoint
└── results/
    └── best_model_history.csv
```


## Setup

Run the following commands in order, the virtual environment command will differ based on operating system
```bash
Windows : python -m venv .venv
MacOS/Linux : python3 -m venv .venv
```

Activate the virtual environment 
```bash
Windows : .venv\Scripts\activate
Mac : source .venv/bin/activate
```

Finally, run the following command to install all packages.
```bash
pip install -r requirements.txt
```

## Usage

### 1. Build the dataset

Fetches all 2024–2025 race sessions, cleans and windows the lap data, and writes train/val/test splits to `data/splits/`:

```bash
python data_main.py
```

FastF1 caches session data locally after the first download, so subsequent runs are much faster.

### 2. Train the model

```bash
python src/model/training.py
```

Saves the best checkpoint to `models/best_model.pth` and logs per-epoch metrics to `results/best_model_history.csv`.

### 3. Evaluate on the test set

```bash
python -m src.model.evaluate
```

Prints overall MAE/RMSE and breaks down error by prediction step (+1 through +5 laps), driver, and circuit.

### 4. Run the web app

```bash
streamlit run app.py
```

## Web App

The Streamlit interface has two tabs:

**Live Prediction**
- Select a season, circuit, and driver
- Use the lap slider to choose a starting point in the race
- The model predicts the next 5 laps and plots context (grey), actuals (blue), and predictions (red dashed)
- Per-step metrics shown for laps +1, +3, and +5
- Lap-by-lap forecast table with error column when actuals are available

**Model Performance**
- Overall MAE, RMSE, and test window count
- Per-driver accuracy leaderboard, filterable by season and race
- Per-circuit error bar chart with color-coded thresholds (green < 0.4s, yellow < 0.7s, red ≥ 0.7s)


## Results

| Metric | Value |
|--------|-------|
| Average Test MAE | ~0.6 seconds |
| Prediction horizon | 5 laps |
| Training seasons | 2024, 2025 |
| Drivers covered | 27 |
| Circuits covered | 24 |

Prediction error increases with step distance — lap +1 is most accurate, lap +5 accumulates the most uncertainty.
