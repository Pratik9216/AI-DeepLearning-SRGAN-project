# SRGAN for Super-Resolution + Downstream Classification (Dogs vs Cats)

**Course:** AI / Deep Learning — Midterm Exam

This project builds a **Super-Resolution GAN (SRGAN)** that upscales **32×32** low-resolution
images to **128×128** high-resolution images (×4 super-resolution), and then studies whether
those GAN-generated images can replace real high-resolution images when training a **binary
classifier** for the *dogs-vs-cats* problem from Assignment 1.

Everything is implemented in one reproducible notebook:
[`SRGAN_Midterm.ipynb`](SRGAN_Midterm.ipynb).

**Pipeline:** Model **A** (classifier on real 128×128) → **SRGAN** (32→128, ≥150 epochs) →
Model **B** (classifier on SRGAN-generated 128×128) → **compare A vs B** (Accuracy, F1, AUC).

---

## 1. Requirements mapping (each assignment step → where it happens)

| # | Requirement | Where in the notebook |
|---|-------------|-----------------------|
| 1 | Binary classifier **A** via transfer learning, images at **128×128** | §4 — `train_classifier("A_real", …)`, MobileNetV2 head + fine-tune |
| 2 | Train **SRGAN** to generate **128×128** from **32×32** | §5 — generator/discriminator + custom training loop |
| 3 | Show examples of the **scaled images** | §5 — "Show some examples of the scaled images" (LR → bicubic → HR) |
| 4 | Train a new model **B** on **SRGAN-generated** images | §6 — `train_classifier("B_srgan", …)` |
| 5 | Train the SRGAN for **≥150 epochs** | §1 — `SRGAN_EPOCHS = 150` when `QUICK_RUN = False` |
| 6 | Split dataset **70% train / 30% test** | §2 — stratified `train_test_split(test_size=0.30)` |
| 7 | **Normalization** + image transformation, show samples | §3 — `tf.data` pipeline + augmentation preview grid |
| 8 | Compare A vs B with **F1, Accuracy, AUC** | §7 — metrics table, bar chart, ROC curves, confusion matrices |
| 9 | **Save models every n epochs** (Colab 12-hour limit) | §5 — `tf.train.CheckpointManager`, `SAVE_EVERY`, `RESUME` |

---

## 2. Dataset

Kaggle **dogs-vs-cats-classification** (~25k images: `cats` and `dogs`), the same dataset as
Assignment 1. Expected layout (any nesting is auto-detected):

```
dogs-vs-cats-classification/
  train/        cats/*.jpg   dogs/*.jpg
  validation/   cats/*.jpg   dogs/*.jpg
  test/         cats/*.jpg   dogs/*.jpg
```

The notebook **pools all splits and re-splits 70/30 stratified by class** (fixed seed) to
satisfy the split requirement, then carves a small **validation** set out of the 70% train
portion so the classifiers can monitor/checkpoint **without ever touching the held-out 30%
test set**. Corrupt/truncated JPEGs (a few exist in this dataset) are recovered
(`decode_jpeg(try_recover_truncated=True)`) or skipped (`ignore_errors`), so training never
crashes on a bad file.

> Image files are **not** committed to git (see `.gitignore`). Provide the dataset yourself
> (Kaggle download, or on Google Drive — see below).

---

## 3. How to run

### Option A — Google Colab (recommended: free T4 GPU + Drive persistence)

1. Upload `SRGAN_Midterm.ipynb` to Colab; `Runtime → Change runtime type → T4 GPU`.
2. Put the dataset on your Google Drive (a **`.zip` is preferred** — it reads/unzips far
   faster than thousands of loose files). Note its folder.
3. In the **Config cell (§1)** set the Drive paths to match your account:
   ```python
   USE_DRIVE      = True                         # mount Drive so training survives disconnects
   DRIVE_ROOT     = "/content/drive/MyDrive/Colab Notebooks/DSBA AI and Deep Learning"
   DRIVE_DATA_DIR = DRIVE_ROOT + "/dogs-vs-cats-classification/dogs-vs-cats-classification"
   QUICK_RUN      = False                        # full 150-epoch run
   BATCH_SIZE     = 32                            # good default for a T4 (16 → 32 is faster)
   ```
4. `Runtime → Run all`. The notebook mounts Drive, imports the dataset (unzips a Drive/`/content`
   zip, or copies a Drive folder, to fast local disk), trains, and saves checkpoints/models
   **into `DRIVE_ROOT`** so they persist across sessions.

> **Why Drive matters:** Colab wipes `/content` when a session ends. Checkpoints on local disk
> are lost and training restarts at epoch 1. With `USE_DRIVE = True` they land on Drive, so
> re-running the training cell in a new session prints `Resumed from checkpoint at epoch N`
> and continues.

### Option B — Local machine

```bash
pip install -r requirements.txt
jupyter notebook SRGAN_Midterm.ipynb
```
Set `USE_DRIVE = False` (artifacts go to local `models/`, `checkpoints/`, …) and, without a
GPU, keep `QUICK_RUN = True` for a fast end-to-end smoke test on a small subset.

---

## 4. Configuration knobs (all in the Config cell, §1)

| Variable | Full-run value | Purpose |
|----------|----------------|---------|
| `QUICK_RUN` | `False` | `True` = tiny smoke test (few epochs, capped images); `False` = full run |
| `HR_SIZE` / `LR_SIZE` | `128` / `32` | high-res / low-res sides → ×4 super-resolution |
| `BATCH_SIZE` | `16` (use **`32`** on a T4) | larger batch → faster on GPU (watch for OOM); no help on CPU |
| `TEST_FRAC` | `0.30` | 70/30 train/test split |
| `SRGAN_EPOCHS` | `150` | SRGAN training length (≥150 required) |
| `SAVE_EVERY` | `5` (set **`2`** to checkpoint more often) | checkpoint cadence, in epochs |
| `RESUME` | `True` | restore the latest checkpoint and continue instead of restarting |
| `MAX_IMAGES_PER_CLASS` | `None` | cap images per class (`None` = use all) |
| `USE_DRIVE` | `True` | mount Drive; point `MODELS_DIR`/`CKPT_DIR`/`LOG_DIR` into `DRIVE_ROOT` |
| `DRIVE_ROOT` | your Drive folder | where checkpoints/models/logs are saved (persists) |
| `DRIVE_DATA_DIR` | your dataset folder | searched first for the dataset (zip or folder) |
| `DATA_DIR` | `None` | set explicitly to a folder holding `cats/` & `dogs/` to skip auto-detection |

---

## 5. Surviving Colab's 12-hour limit (checkpoint & resume)

The SRGAN loop checkpoints every `SAVE_EVERY` epochs via `tf.train.CheckpointManager`
(generator + discriminator + optimizer state) and records the last completed epoch in
`<CKPT_DIR>/epoch.json`. On the next run, with `RESUME = True`, it restores the latest
checkpoint and continues; the best generator is also exported to `<MODELS_DIR>/generator.keras`.

**Multi-session workflow:** run → checkpoints to Drive every `SAVE_EVERY` epochs → session
ends → next session `Run all` → prints `Resumed from checkpoint at epoch N` → repeat until 150.

**Continuing on a different Google account (Drive sharing blocked):** checkpoints are small,
so carry them manually. On account A: `shutil.make_archive('/content/ckpt','zip',CKPT_DIR)`
then download it. On account B: upload and `extractall` into that account's `CKPT_DIR`, keep
`RESUME = True`, and run — it resumes from the saved epoch. (Prefer simply waiting for the
GPU quota to reset, which avoids account juggling.)

---

## 6. Architecture summary

**Model A / Model B (identical architecture):** MobileNetV2 (ImageNet) backbone, frozen for
head training, then the top 20 layers unfrozen and fine-tuned at a 10× smaller LR;
`GlobalAveragePooling2D → Dropout(0.3) → Dense(1, sigmoid)`. MobileNetV2's `[-1,1]`
normalization is a `Rescaling(2.0, -1.0)` first layer (serialization-safe under Keras 3), so
the same `[0,1]` input pipeline feeds both real and generated images. **A** trains on real
128×128; **B** trains on the SRGAN's reconstructions of the same images.

**SRGAN generator (SRResNet):** 9×9 conv + PReLU → 16 residual blocks (conv-BN-PReLU-conv-BN
+ skip) → global skip → two ×2 sub-pixel (`depth_to_space`) upsampling blocks → 9×9 conv with
`tanh`. Input LR `[0,1]` → output HR `[-1,1]`.

**SRGAN discriminator:** strided conv-BN-LeakyReLU blocks (64→512) → Dense(1024) → sigmoid.

**Losses:** generator = **VGG19 content loss** (`block5_conv4` features) + `1e-3 ·` adversarial
(BCE) + pixel MSE; discriminator = BCE with one-sided label smoothing.

---

## 7. Outputs & artifacts

With `USE_DRIVE = True` these live under `DRIVE_ROOT`; locally they are top-level folders.

| Path | Contents |
|------|----------|
| `models/` | `A_real_*.keras`, `B_srgan_*.keras`, `generator.keras` |
| `checkpoints/` | resumable SRGAN checkpoints + `epoch.json` |
| `logs/` | TensorBoard logs (`%tensorboard --logdir <LOG_DIR>`) |
| `assets/` | all figures + `comparison_metrics.csv` |

Figures produced: `transforms_preview.png`, `scaled_examples.png`, `srgan_results.png`,
`model_B_inputs.png`, `comparison_bar.png`, `roc_confusion.png`.

---

## 8. Reproducibility notes

* All randomness is seeded (`SEED = 42`: Python, NumPy, TensorFlow, and the split).
* Image geometry is a single source of truth in the Config cell (`HR_SIZE=128`, `LR_SIZE=32`).
* The A-vs-B comparison evaluates **both** models on the **same real 128×128 test set**, so any
  performance gap is attributable to the training data (real vs SRGAN), not a different test set.
* The notebook can be regenerated deterministically from source:
  `python scripts/build_notebook.py`.

---

## 9. Repository layout

```
SRGAN_Midterm.ipynb        # the full pipeline (run this)
README.md                  # this documentation
requirements.txt           # Python dependencies
scripts/build_notebook.py  # generator script that produces the notebook
models/  checkpoints/  logs/  assets/   # created/populated at runtime (git-ignored)
```

---

## 10. How to reproduce, end to end

1. Clone the repo and open `SRGAN_Midterm.ipynb` in Colab (T4 GPU).
2. Put the dogs-vs-cats dataset (zip) on your Drive; set `DRIVE_ROOT` / `DRIVE_DATA_DIR`.
3. Set `QUICK_RUN = False`, `USE_DRIVE = True`, `BATCH_SIZE = 32`.
4. `Run all`. Across sessions, re-run the training cell to resume until the SRGAN reaches 150
   epochs.
5. Run the remaining cells to train Model B and produce the A-vs-B metrics
   (`assets/comparison_metrics.csv`) and figures.
