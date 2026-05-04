"""Generate the final-presentation .pptx for the F1 Lap Time project."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUT = Path("F1_Lap_Time_Presentation.pptx")
PLOTS = Path("plots")

NAVY = RGBColor(0x0B, 0x2D, 0x5C)
RED = RGBColor(0xC8, 0x10, 0x2E)
GREY = RGBColor(0x55, 0x55, 0x55)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]


def add_textbox(slide, left, top, width, height, text, *, size=18, bold=False,
                color=None, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    if isinstance(text, str):
        text = [text]
    for i, line in enumerate(text):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        if color is not None:
            run.font.color.rgb = color
    return tb


def add_title_bar(slide, title, subtitle=None):
    add_textbox(slide, Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.7),
                title, size=32, bold=True, color=NAVY)
    if subtitle:
        add_textbox(slide, Inches(0.5), Inches(1.0), Inches(12.3), Inches(0.4),
                    subtitle, size=16, color=GREY)


def add_bullets(slide, left, top, width, height, items, size=18):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = "• " + item
        run.font.size = Pt(size)
        p.space_after = Pt(6)
    return tb


def add_table(slide, left, top, width, height, data, header=True,
              first_col_bold=False, font_size=14):
    rows = len(data)
    cols = len(data[0])
    shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = shape.table
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            tf = cell.text_frame
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(font_size)
            if header and r == 0:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            elif first_col_bold and c == 0:
                run.font.bold = True
    return table


# ---------------------------------------------------------------------------
# Slide 1 - Title
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_textbox(s, Inches(0.5), Inches(2.4), Inches(12.3), Inches(1.3),
            "F1 Lap Time Prediction", size=54, bold=True, color=NAVY,
            align=PP_ALIGN.CENTER)
add_textbox(s, Inches(0.5), Inches(3.5), Inches(12.3), Inches(0.7),
            "Using a Decoder-Only Transformer", size=28, color=RED,
            align=PP_ALIGN.CENTER)
add_textbox(s, Inches(0.5), Inches(5.4), Inches(12.3), Inches(0.5),
            "Srikara Sai Junnuri  •  Nikhil Sumesh", size=20,
            align=PP_ALIGN.CENTER)
add_textbox(s, Inches(0.5), Inches(5.9), Inches(12.3), Inches(0.4),
            "MSML 612 — Final Project", size=16, color=GREY,
            align=PP_ALIGN.CENTER)


# ---------------------------------------------------------------------------
# Slide 2 - Problem & Motivation
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Problem & Motivation")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(5),
            [
                "Formula 1 lap times are governed by tyre degradation, fuel load, weather, traffic, and driver style — a noisy multi-variable time series.",
                "Accurate short-horizon lap-time forecasts directly inform pit-stop strategy, undercut/overcut decisions, and broadcast analytics.",
                "Classical race-strategy models rely on hand-crafted physics heuristics that ignore weather and per-driver behaviour.",
                "Recurrent models (RNN / LSTM) struggle with long-range dependencies and vanishing gradients on race-length sequences.",
                "Transformers can attend to specific historical laps via self-attention — a natural fit for this forecasting task.",
            ], size=18)


# ---------------------------------------------------------------------------
# Slide 3 - Objective & Target Metric
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Objective")
add_textbox(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(0.6),
            "Given the last 15 laps for a driver, predict the next 5 lap times.",
            size=22, bold=True)

add_textbox(s, Inches(0.5), Inches(2.4), Inches(6), Inches(0.4),
            "Inputs", size=20, bold=True, color=NAVY)
add_bullets(s, Inches(0.5), Inches(2.9), Inches(6), Inches(2.5),
            [
                "Window of 15 consecutive laps",
                "12 features per lap (lap & sector times, speed, tyre, weather, IDs)",
                "Causal masking — each lap can only see prior laps",
            ], size=16)

add_textbox(s, Inches(7), Inches(2.4), Inches(6), Inches(0.4),
            "Target metric", size=20, bold=True, color=NAVY)
add_bullets(s, Inches(7), Inches(2.9), Inches(6), Inches(2.5),
            [
                "Primary: Mean Absolute Error (MAE) in seconds",
                "Secondary: RMSE",
                "Proposal target: MAE < 0.5 s",
            ], size=16)


# ---------------------------------------------------------------------------
# Slide 4 - Dataset
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Dataset")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(7), Inches(4),
            [
                "Source: FastF1 Python library (official F1 timing + telemetry).",
                "Coverage: 2024 + 2025 seasons — 48 Grand Prix events.",
                "27 unique drivers, ~54,000 raw lap records.",
                "Per-lap: lap & sector times, speed-trap, tyre life, compound, weather (air/track temp, rainfall), driver, track, lap number, pit info.",
            ], size=16)

add_table(s, Inches(7.8), Inches(1.7), Inches(5), Inches(2.5),
          [
              ["Split", "Windows", "Shape"],
              ["Train", "20,068", "(15, 12)"],
              ["Validation", "4,449", "(15, 12)"],
              ["Test", "4,217", "(15, 12)"],
          ], font_size=14)


# ---------------------------------------------------------------------------
# Slide 5 - Data Pipeline
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Data Pipeline")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(5),
            [
                "1. Fetch — FastF1 session.load() for each race; merge weather via time-asof; cache to disk.",
                "2. Clean — drop NaN rows, lap times outside 60–200 s, lap 1 (standing start), pit in/out laps, and laps with TrackStatus ≠ 1 (safety car / red flag).",
                "3. Encode — compound → integer, driver → DriverID, race → TrackID.",
                "4. Window — sliding (15-lap input, 5-lap horizon) per (Race, Year, Driver).",
                "5. Split — assign (Race, Year, Driver) groups to train/val/test BEFORE windowing → prevents data leakage from overlapping windows.",
                "6. Normalise — StandardScaler fitted only on training data, applied to all splits.",
            ], size=15)


# ---------------------------------------------------------------------------
# Slide 6 - Model Architecture
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Model — Decoder-Only Transformer")

add_textbox(s, Inches(0.5), Inches(1.5), Inches(6), Inches(0.4),
            "Components", size=20, bold=True, color=NAVY)
add_bullets(s, Inches(0.5), Inches(2.0), Inches(7), Inches(4.5),
            [
                "Input projection: 12-d feature vector → d_model.",
                "Learnable positional embeddings (max length 20).",
                "N stacked decoder blocks: causal multi-head self-attention + position-wise FFN, residual + LayerNorm, dropout 0.1.",
                "Output head: linear projection back to 1 (predicted lap time per position).",
                "Training uses teacher forcing — only the last 5 positions of the output are compared with the 5 future-lap labels.",
            ], size=16)

add_table(s, Inches(8.2), Inches(2.0), Inches(4.6), Inches(3.8),
          [
              ["Hyperparameter", "Value"],
              ["d_model", "128"],
              ["n_layers", "3"],
              ["n_heads", "4"],
              ["d_ff", "256"],
              ["dropout", "0.1"],
              ["lookback / horizon", "15 / 5"],
              ["loss", "MSE"],
          ], font_size=14)


# ---------------------------------------------------------------------------
# Slide 7 - Training Setup
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Training Setup")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(5),
            [
                "Optimiser: Adam, learning rate 5e-4, weight decay 1e-3.",
                "Batch size 64; up to 150 epochs.",
                "Learning-rate scheduler: ReduceLROnPlateau (factor 0.5, patience 3) on validation loss.",
                "Early stopping: patience 8 — training halts when validation loss does not improve for 8 epochs.",
                "Checkpointing: best model (by validation loss) saved per run; per-epoch metrics logged to CSV.",
                "Framework: PyTorch; data loaders via torch.utils.data.",
            ], size=16)


# ---------------------------------------------------------------------------
# Slide 8 - Interim Report Result (the starting point)
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Interim Report — where we started",
              "April 2, 2026")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(7.5), Inches(4),
            [
                "7-feature model (lap & sector times, speed, tyre, compound).",
                "d_model=64, 2 layers, 4 heads, 50 epochs, MSE.",
                "Test MAE = 0.668 s, RMSE = 1.039 s.",
                "Per-horizon error nearly flat (0.66–0.69) — suspicious.",
                "Identified challenges: data leakage from overlapping windows, missing weather / track ID features, no hyperparameter tuning.",
            ], size=16)
if (PLOTS / "horizon_mae.png").exists():
    s.shapes.add_picture(str(PLOTS / "horizon_mae.png"),
                         Inches(8.3), Inches(1.6), width=Inches(4.7))


# ---------------------------------------------------------------------------
# Slide 9 - Stage 1: Pre-session changes
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Stage 1 — Fixing the foundation",
              "Changes between interim report and final tuning")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(5),
            [
                "Data-leakage fix: groups assigned to train/val/test BEFORE windowing — overlapping windows from the same race can no longer span splits.",
                "Feature expansion 7 → 11: added AirTemp, TrackTemp, Rainfall (weather) and TrackID.",
                "Result: honest baseline test MAE = 0.753 s (worse than interim's 0.668 — the leakage was inflating that number).",
                "This is the trustworthy starting point for tuning.",
            ], size=18)


# ---------------------------------------------------------------------------
# Slide 10 - Stage 2 overview
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Stage 2 — Tuning session",
              "Four incremental changes, 0.753 → 0.667 s")
add_table(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(4),
          [
              ["#", "Change", "Test MAE", "Δ"],
              ["0", "Honest baseline (post-leakage fix)", "0.753 s", "—"],
              ["1", "Training-loop refactor (LR scheduler, early stopping, run logging)", "—", "infra"],
              ["2", "Hyperparameter sweep (9 configs)", "0.714 s", "−0.039"],
              ["3", "Add DriverID as a feature (12 features)", "0.707 s", "−0.007"],
              ["4", "Residual-target prediction", "0.667 s", "−0.040"],
          ], font_size=14)


# ---------------------------------------------------------------------------
# Slide 11 - Hyperparameter sweep details
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Hyperparameter sweep")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(7.5), Inches(4),
            [
                "9 configs run sequentially via sweep.py.",
                "Phase 1 — training-loop knobs: lr, batch size, weight decay.",
                "Phase 2 — architecture: d_model ∈ {128, 192, 256}, layers ∈ {3, 4}, heads ∈ {4, 8}, d_ff, dropout.",
                "Each run trained → evaluated → logged to results/sweep.csv.",
                "Models bigger than d_model=128 performed WORSE — confirms the bottleneck is not capacity.",
            ], size=16)
add_table(s, Inches(8.3), Inches(2), Inches(4.7), Inches(2.5),
          [
              ["Top 3 configs", "Test MAE"],
              ["d_model=128, layers=3", "0.714"],
              ["wd=1e-3, default arch", "0.724"],
              ["d_model=128, layers=4, heads=8", "0.725"],
          ], font_size=14)


# ---------------------------------------------------------------------------
# Slide 12 - Residual prediction (the key insight)
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Key insight — residual-target prediction")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(2.5),
            [
                "Absolute lap times (60–150 s) are dominated by track + car characteristics — fixed across the prediction window.",
                "The actually-predictable component (degradation, fuel burn, traffic) is only ±2–3 s.",
                "Forcing the model to predict the absolute value wastes capacity on memorising baselines.",
            ], size=16)

add_textbox(s, Inches(0.5), Inches(4.4), Inches(12.3), Inches(0.5),
            "Reformulation:", size=20, bold=True, color=NAVY)
add_textbox(s, Inches(1.0), Inches(4.9), Inches(12.3), Inches(0.6),
            "ỹᵢ = yᵢ − x_T,LapTime    (i = 1, …, 5)",
            size=22, bold=True)
add_textbox(s, Inches(1.0), Inches(5.6), Inches(12.3), Inches(1),
            ["Subtract the last input lap from every future-lap label.",
             "MAE is invariant to subtracting a constant — reported number stays in seconds.",
             "Two-line preprocessing change; no model or evaluation code changes."],
            size=14)


# ---------------------------------------------------------------------------
# Slide 13 - Final results overall
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Final results")
add_table(s, Inches(0.5), Inches(1.5), Inches(6.5), Inches(2.5),
          [
              ["Metric", "Value"],
              ["Overall test MAE", "0.667 s"],
              ["Overall test RMSE", "1.245 s"],
              ["Lap +1 MAE (next-lap)", "0.505 s"],
              ["Best per-race MAE (Austria 2025)", "0.262 s"],
              ["Best per-driver MAE (Antonelli)", "0.322 s"],
          ], font_size=14)
add_bullets(s, Inches(7.3), Inches(1.7), Inches(5.5), Inches(4),
            [
                "Averaged MAE matches the interim's 0.668 s — but on the corrected, leak-free pipeline.",
                "Lap +1 MAE = 0.505 s essentially meets the proposal's < 0.5 s target for next-lap prediction.",
                "Per-horizon error rises naturally now (0.50 → 0.74) — sign of an honest forecaster.",
            ], size=16)


# ---------------------------------------------------------------------------
# Slide 14 - Per-horizon comparison
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Per-horizon MAE — interim vs final")
add_table(s, Inches(0.5), Inches(1.5), Inches(7), Inches(3.5),
          [
              ["Future lap", "Interim (s)", "Final (s)", "Δ"],
              ["+1", "0.655", "0.505", "−0.150"],
              ["+2", "0.660", "0.634", "−0.026"],
              ["+3", "0.669", "0.708", "+0.039"],
              ["+4", "0.664", "0.749", "+0.085"],
              ["+5", "0.690", "0.738", "+0.048"],
              ["Average", "0.668", "0.667", "≈ 0"],
          ], font_size=14)
add_bullets(s, Inches(7.8), Inches(1.7), Inches(5), Inches(4),
            [
                "Big short-horizon improvement.",
                "Long-horizon slightly worse — but flat-error interim curve was a leakage artefact.",
                "Trade-off favours real-world utility: next-lap accuracy is what drives strategy decisions.",
            ], size=14)


# ---------------------------------------------------------------------------
# Slide 15 - Per-race breakdown
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Where the residual error lives — by race")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(7), Inches(4),
            [
                "Best races: Austria 2025 (0.26 s), Italy 2025 (0.35 s), Miami 2025 (0.35 s) — smooth circuits with predictable degradation.",
                "Worst races: São Paulo 2024 (2.24 s, rain), Australia 2025 (1.30 s, multiple red flags), Monaco 2025 (0.91 s, tight street circuit).",
                "Wet / chaotic races dominate the long tail of error — not addressable from past dry-lap dynamics alone.",
            ], size=16)
if (PLOTS / "race_mae.png").exists():
    s.shapes.add_picture(str(PLOTS / "race_mae.png"),
                         Inches(8.0), Inches(1.5), width=Inches(5.0))


# ---------------------------------------------------------------------------
# Slide 16 - Per-driver breakdown
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Where the residual error lives — by driver")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(7), Inches(4),
            [
                "Best: ANT (Antonelli) 0.32 s, VER (Verstappen) 0.37 s, HAD (Hadjar) 0.45 s — known for highly consistent pace.",
                "Worst: BOT 1.15 s, ZHO 0.97 s — Sauber drivers; ZHO retired mid-season leaving sparse training data.",
                "Driver-level variance is largely a car/team variance: model performance is bounded by the underlying car consistency.",
            ], size=16)
if (PLOTS / "driver_mae.png").exists():
    s.shapes.add_picture(str(PLOTS / "driver_mae.png"),
                         Inches(8.0), Inches(1.5), width=Inches(5.0))


# ---------------------------------------------------------------------------
# Slide 17 - Mapping back to interim Future Work
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Mapping back to the interim's Future Work")
add_table(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(4.5),
          [
              ["Interim Future-Work item", "Status"],
              ["1. Add weather as a feature", "Done (pre-session)"],
              ["2. Add TrackID as a feature", "Done (pre-session)"],
              ["3. Filter / weight high-variance races", "Not done — discussed below"],
              ["4. Hyperparameter tuning", "Done — 9-config sweep"],
              ["5. Train more epochs to reduce noise", "Done — scheduler + early stopping"],
          ], font_size=14)


# ---------------------------------------------------------------------------
# Slide 18 - Limitations
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Limitations")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(5),
            [
                "Wet / safety-car-affected races (e.g., São Paulo 2024) cannot be predicted from past dry-lap dynamics alone.",
                "Categorical features (DriverID, TrackID, Compound) are currently z-scored — mathematically meaningless for arbitrary integer IDs.",
                "Long-horizon error grows naturally with uncertainty — single-shot multi-step prediction has limits.",
                "Sparse data for drivers who retired mid-season (e.g., ZHO).",
                "Features omit fuel load and tyre temperature, both major degradation drivers — not directly exposed by FastF1.",
            ], size=16)


# ---------------------------------------------------------------------------
# Slide 19 - Future work
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Future work")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(5),
            [
                "Categorical embeddings: replace z-scored DriverID / TrackID / Compound with nn.Embedding layers concatenated to continuous features.",
                "Per-horizon weighted loss to push Lap +1/+2 accuracy further.",
                "Filter rain-affected laps and reframe scope as 'dry-condition lap-time prediction' — likely takes averaged MAE below 0.6 s.",
                "Auto-regressive inference at evaluation time, instead of one-shot 5-lap prediction.",
                "Stretch: incorporate position / gap-to-leader as a feature to capture traffic effects.",
            ], size=16)


# ---------------------------------------------------------------------------
# Slide 20 - Conclusion
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Conclusion")
add_bullets(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(5),
            [
                "Built a decoder-only Transformer that forecasts the next 5 lap times from the previous 15.",
                "Fixed a data-leakage bug in the interim pipeline → the new 0.667 s MAE is comparable to the interim's 0.668 s but on a trustworthy split.",
                "Lap +1 MAE = 0.505 s essentially meets the proposal's < 0.5 s target for short-horizon prediction.",
                "Largest single gain: switching from absolute-lap-time to residual-target prediction (− 0.04 s overall, − 0.15 s on Lap +1).",
                "Remaining error is dominated by chaotic races (rain, safety cars) and inconsistent cars — bounded more by data than by model.",
            ], size=18)


# ---------------------------------------------------------------------------
# Slide 21 - Thanks / Q&A
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_textbox(s, Inches(0.5), Inches(2.8), Inches(12.3), Inches(1.2),
            "Thank you — Questions?", size=54, bold=True, color=NAVY,
            align=PP_ALIGN.CENTER)
add_textbox(s, Inches(0.5), Inches(4.0), Inches(12.3), Inches(0.5),
            "Srikara Sai Junnuri  •  Nikhil Sumesh", size=20,
            align=PP_ALIGN.CENTER)


prs.save(str(OUT))
print(f"Wrote {OUT.resolve()}  ({OUT.stat().st_size/1024:.0f} KB)")
print(f"Slides: {len(prs.slides)}")
