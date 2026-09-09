# EEG Emotion Recognition with a Bipartite Graph-Transformer Network

Subject-independent emotion recognition from EEG signals, using a Domain-Adversarial Neural Network (DANN) enhanced with **bipartite (BP) graphs** at both the spatial and temporal level. This repository implements the architecture described in our paper published in *IEEE Journal of Biomedical and Health Informatics* (2025), using the SEED dataset as the base case.

> **Paper:** Niaki, M., Dharia, S. Y., Chen, Y., & Valderrama, C. E. (2025). *Bipartite Graph Adversarial Network for Subject-Independent Emotion Recognition.* IEEE JBHI, 29(10), 7234–7247. [DOI: 10.1109/JBHI.2025.3570187](https://doi.org/10.1109/JBHI.2025.3570187)

---

## Motivation

EEG signals are less susceptible to conscious manipulation than facial expressions or speech, which motivates their use for emotion recognition in applications such as mental health monitoring and human-computer interaction.

A known obstacle to deploying EEG-based emotion recognition is **domain shift**: inter-subject variability in EEG signals means a model trained on one group of subjects does not necessarily generalize to a new subject. This project evaluates the proposed method using **Leave-One-Subject-Out Cross-Validation (LOSOCV)**, in which the test subject's data is entirely excluded from training. This is the standard protocol for supporting a subject-independence claim, since it is the setting in which the model cannot rely on any information from the individual being evaluated.

## Method

Prior work has used Domain-Adversarial Neural Networks (DANN) to align feature distributions between training subjects (source domain) and new subjects (target domain) at the level of the overall feature distribution. This project introduces **bipartite (BP) graphs**, adapted from prior work in video domain adaptation [Luo et al., 2020], to align source and target samples at the level of individual sample pairs. BP graphs are applied at two points in the network: after the spatial transformer (over EEG channels) and after the temporal transformer (over time windows).

---

## Datasets

The paper evaluates the architecture on five SEED-family datasets, differing mainly in the number of subjects and emotion classes:

| Dataset | Subjects | Emotions | Reported accuracy (paper) |
|---|---|---|---|
| SEED | 15 | negative / neutral / positive | 85.4% |
| SEED-IV | 15 | happy / sad / fear / neutral | 77.3% |
| SEED-V | 16 | happy / sad / fear / disgust / neutral | 82.1% |
| SEED-FRA | 8 | negative / neutral / positive | 90.7% |
| SEED-GER | 8 | negative / neutral / positive | 87.6% |

This repository implements the base case, **SEED** (3-class: negative/neutral/positive, 15 subjects), and is structured so that adapting it to the other four datasets requires changing the classifier output size (`nn.Linear(310, num_classes)`) and the subject/video counts in `load_data()`, without changing the model architecture itself.

*SEED-family datasets are third-party and not redistributed in this repository — see the [SEED dataset page](https://bcmi.sjtu.edu.cn/home/seed/) for access.*

---

## Data

### Provenance

The 62-channel, 1000 Hz EEG signals were preprocessed and reduced to differential entropy (DE) features upstream of this repository — this repository does not perform EEG preprocessing, windowing, or DE extraction. Each 4-second window is represented as 310 features (62 channels × 5 frequency bands: delta, theta, alpha, beta, gamma).

### CSV structure (`data_basicSEED.csv`)

Each row is one 4-second window from one subject watching one video clip:

| Column(s) | Meaning |
|---|---|
| `user` | Subject ID |
| `session` | Recording session |
| `trial` | Which of the 15 video clips was shown |
| `window` | Sequential position of this 4-second segment within the clip (0, 1, 2, …) |
| `label` | Emotion for the clip: `-1` = negative, `0` = neutral, `1` = positive |
| `{channel}_{band}` | 310 columns, e.g. `0_delta`, `0_theta`, …, `61_gamma` — one DE value per channel × frequency band |

**Grouping logic:** the triple `(user, session, trial)` identifies one video-watching trial. All rows sharing that triple, sorted by `window`, reconstruct the ordered sequence of 4-second segments for that clip. `label` is constant within a group, since one video elicits one target emotion. Clip length varies (video durations differ), so the number of rows per `(user, session, trial)` group varies accordingly.



---

## Repository structure

```
.
├── data_basicSEED.csv        # DE-feature table (sample; full file needed for training)
├── dann.py                   # model architecture: transformers, BP graph modules, DANN wrapper
├── utils_dann.py             # data loading, loss functions, train/evaluate loops
├── training_dann.py          # entry point: LOSOCV orchestration loop
├── all_participants_results.txt   # per-subject accuracy logs from prior runs
├── model_features/           # saved per-subject feature embeddings (.npz), created at runtime
├── models_per_subject/       # saved per-subject model checkpoints (.pth), created at runtime
└── README.md
```

### How the three code files relate

```
training_dann.py
   │
   ├── imports load_data(), train(), evaluate()   from utils_dann.py
   │        └── load_data() reads data_basicSEED.csv, builds source/target DataLoaders
   │
   ├── imports DomainAdaptationModel(_spatial/_temporal)   from dann.py
   │        └── each wraps a feature extractor (model_0 / model_0_spatial / model_temporal)
   │            built from shared building blocks: TransformerEncoderStack,
   │            EdgeUpdateNetwork, NodeUpdateNetwork, GradientReversalFunction
   │
   └── runs the LOSOCV loop: for each of the 15 subjects, builds a fresh model,
       trains it with utils_dann.train(), evaluates with utils_dann.evaluate(),
       and logs/saves the results
```

`dann.py` defines *what the model is*. `utils_dann.py` defines *how data flows through it and how it's trained/scored*. `training_dann.py` defines *the experiment*: which subjects to hold out, which model variant to run, and for how long.

---

## Model architecture [(`dann.py`)](https://github.com/Marzieniaki/Subject-independent-EEG-emotion-recognition-using-a-bipartite-graph-transformer-code/blob/main/dann.py)

### Input

Each training step receives a batch of `B` video clips per side (source and target), each represented as a fixed-length sequence of `W` windows (`W = 66` for SEED, the maximum clip length; shorter clips are zero-padded) of 310 features each:

```
source, target: (B, W, 310)
```

### Spatial module

**Reshape.** The video and window dimensions are merged, then the 310 features are split back into 62 channels × 5 bands:
```
(B, W, 310) → (B·W, 310) → (B·W, 62, 5)
```
Every window, from every clip in the batch, becomes its own sample: a sequence of 62 channel-tokens, each a 5-dim (delta/theta/alpha/beta/gamma) vector. Zero-padded windows are filtered out before proceeding.

**Spatial transformer** (`TransformerEncoderStack`, `temporal=False`). Standard multi-head self-attention (`embed_dim=5`, `num_heads=5`) runs across the 62 channel-tokens *within each window independently* — modeling which channels co-activate together in that window. Output shape unchanged.

**Spatial BP graph** (`EdgeUpdateNetwork` → `NodeUpdateNetwork`). This module treats each of the 62 channels as an independent problem: for channel `c`, it builds one bipartite graph whose nodes are *windows* (all windows from all clips in the batch, source side vs. target side), fully connecting every source window to every target window within that channel. Edge weights are computed from the pairwise absolute difference between each pair's 5-dim feature vector, passed through a small CNN and normalized (sigmoid, then L1-normalized across rows and columns):
```
x_ij = |x_i - x_j|                      # (62, N_s, N_t, 5)
edge_weights = normalize(sigmoid(CNN(x_ij)))   # (62, N_s, N_t)
```
`NodeUpdateNetwork` then lets each node absorb a similarity-weighted combination of the opposite domain's nodes (`bmm(edge_weights, opposite_nodes)`), concatenates it with the node's original features, and passes the result through another CNN to produce the updated node representation. Output shape is unchanged (`(N_valid, 62, 5)` per side); values are updated.

The result is reshaped back into `(B, W, 310)` to close out the spatial module.

### Temporal module

**Temporal transformer** (`temporal=True`). A learnable positional embedding (`self.position_embedding`) is added first, since self-attention has no inherent notion of window order. Multi-head attention then runs across the `W` window-tokens *within each clip independently* (`embed_dim=310`), modeling how a clip's emotional signal evolves over its duration. A `key_padding_mask` ensures padded window positions are excluded from attention and zeroed out after each sub-layer.

**Temporal BP graph.** Structurally the mirror image of the spatial BP graph: here each *window position* is treated as an independent problem, and the nodes are *whole video clips* (source side vs. target side). Since clips vary in length, the graph is computed only up to `maxWindows` (the shortest number of jointly-valid windows across all clip-pairs in the batch); window positions beyond that are filled in with the batch-averaged edge matrix rather than a directly computed one, as a practical accommodation for variable-length clips.
```
edge_weights: (maxWindows, B, B)
```
The same `EdgeUpdateNetwork` / `NodeUpdateNetwork` pattern (edge computation, then similarity-weighted aggregation and CNN update) is applied, producing updated per-window-position clip representations.

**Masked average pooling.** The temporal module ends by averaging each clip's representation across its valid (non-padded) windows, collapsing `(B, W, 310) → (B, 310)` — one feature vector per clip.

### DANN wrapper and classifier heads

`DomainAdaptationModel` (and its `_spatial`/`_temporal` variants) wrap the feature extractor above with two small classifier heads:

- **Class classifier**: `Dropout(0.3) → Linear(310, 3)` — predicts the emotion (3 classes for SEED)
- **Domain classifier**: `Dropout(0.3) → Linear(310, 2)` — predicts source (0) vs. target (1)

Before reaching the domain classifier, features pass through `GradientReversalFunction`, a custom autograd function that acts as identity on the forward pass but negates and scales (`alpha`) the gradient on the backward pass — the mechanism that forces the feature extractor to produce features the domain classifier cannot use to tell source from target.

### Architecture variants (ablation)

Three feature-extractor classes share the same building blocks:

| Class | Spatial module | Temporal module | Corresponds to |
|---|---|---|---|
| `model_0` | ✓ | ✓ | full model |
| `model_0_spatial` | ✓ | ✗ (removed) | temporal-ablation experiment |
| `model_temporal` | ✗ (removed) | ✓ | spatial-ablation experiment |

`training_dann.py` selects between these via the `modelToRun` variable.

---

## Data loading and training loop [(`utils_dann.py`)](https://github.com/Marzieniaki/Subject-independent-EEG-emotion-recognition-using-a-bipartite-graph-transformer-code/blob/main/utils_dann.py)

### `load_data(file_path, subject, batch_size)`

1. Reads the CSV and determines `max_windows` (66 for SEED).
2. For each `(user, session, trial)` group, extracts its DE-feature sequence, z-score normalizes it (`StandardScaler`, fit per video clip), and zero-pads it to `max_windows`.
3. Stacks all clips into an array of shape `(num_subjects × videos_per_subject, max_windows, 310)`.
4. Splits by `subject`: the **target set** is every clip belonging to the held-out subject; the **source set** is every clip from all other subjects.
5. Wraps both in `DataLoader`s (`shuffle=True`) and returns them.

Labels are remapped from `{-1, 0, 1}` to `{2, 0, 1}` so they start at 0, as required by `nn.CrossEntropyLoss`.

### Loss functions

- **Task loss** — categorical cross-entropy on the source batch's emotion predictions, with per-batch inverse-class-frequency weighting to counteract class imbalance.
- **Domain loss** — binary cross-entropy for the domain classifier, computed separately for source and target and summed.
- **Entropy loss** (`entropy_minimization_loss`) — Shannon entropy of the target batch's emotion predictions, discouraging ambiguous (high-entropy) predictions on the unlabeled target domain.
- **MMD loss** (`compute_mmd_loss`) — Maximum Mean Discrepancy between source and target feature embeddings, computed with multi-scale RBF kernels (`σ ∈ {1, 2, 4, 8, 16}`).

```python
total_loss = task_loss + (domain_loss_s + domain_loss_t) + 0.1 * entropy_loss + 0.1 * mmd_loss
```

### `train(model, source_loader, target_loader, ...)`

One epoch iterates once over `source_loader` (`steps_per_epoch = len(source_loader)`). Since the source pool (14 subjects × 45 clips ≈ 630) is much larger than the target pool (1 held-out subject × 45 clips), the target iterator is reshuffled and restarted whenever it runs out mid-epoch — the target subject's clips are seen multiple times per epoch, the source clips once. Each step, the target batch is trimmed to match the source batch size, so both sides feed the model an equal number of samples per step. `alpha` (the gradient-reversal strength) is scheduled upward over training via `adjust_alpha`, following standard DANN practice; the learning rate is scheduled downward via `optimizer_scheduler`.

### `evaluate(model, source_loader, target_loader, ...)`

Runs the held-out subject's data through the model with `alpha=0` (no domain-adversarial signal at evaluation), and reports loss, overall accuracy, per-class accuracy, a confusion matrix, and the raw target-domain feature embeddings (saved for later analysis, e.g. the paper's t-SNE and channel-importance work).

---

## Experiment orchestration [(`training_dann.py`)](https://github.com/Marzieniaki/Subject-independent-EEG-emotion-recognition-using-a-bipartite-graph-transformer-code/blob/main/training_dann.py)

Runs a full LOSOCV loop: for each of the 15 SEED subjects, a fresh model is initialized (variant selected via `modelToRun`), trained for `num_epochs`, and evaluated with that subject held out as target. Per-subject results are appended to `all_participants_results.txt`; per-subject model weights and target-domain feature embeddings are saved to `models_per_subject/` and `model_features/` respectively.

**Current hyperparameters in this script** (edit `training_dann.py` to change):

| Hyperparameter | Value |
|---|---|
| `batch_size` | 32 |
| `num_epochs` | 60 |
| `num_layers` | 1 | 
| `num_heads` (spatial) | 5 | grid-searched over {1, 5} |
| `learning_rate` | 5e-4 (scaled by batch size) | grid-searched over {1e-4, 5e-4, 1e-3} |
| optimizer | AdamW, weight_decay=1e-4 | AdamW |

The current values are convenient for quick iteration/debugging; reproducing the paper's reported accuracies requires setting `num_epochs=60` (and optionally revisiting the other grid-searched hyperparameters per Section IV-B of the paper).

---

## Installation

```bash
pip install torch einops pandas numpy scikit-learn torchmetrics matplotlib
```

## Usage

```bash
python training_dann.py
```

By default this runs LOSOCV across all 15 SEED subjects using the model variant set by `modelToRun` in `training_dann.py` (`1` = full model, `2` = spatial-only, `3` = temporal-only), appending results to `all_participants_results.txt` and saving per-subject checkpoints and features to `models_per_subject/` and `model_features/`.

---

## Results

Results reported in the paper (5 datasets, LOSOCV, model as described in Section III-F):

- Accuracy matched or exceeded the reported state-of-the-art on 4 of the 5 datasets (comparable, within ~2%, on the fifth: SEED), with lower standard deviation across subjects than the comparison models on all five.
- An ablation study isolated the contribution of each architectural component: removing the temporal module reduced accuracy by an average of 33.6% across the five datasets; removing both BP graphs reduced it by 24.5%; removing the adversarial DANN component alone reduced it by 1.2%.
- Statistical testing (Friedman test, p < 0.05) identified EEG channels in the frontal, temporal, and parietal regions as showing significant feature differences across emotion classes in at least three of the five datasets.



## Citation

If you use this code, please cite:

```bibtex
@article{niaki2025bipartite,
  title={Bipartite Graph Adversarial Network for Subject-Independent Emotion Recognition},
  author={Niaki, Marzieh and Dharia, Shyamal Y. and Chen, Yangjun and Valderrama, Camilo E.},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={29},
  number={10},
  pages={7234--7247},
  year={2025},
  publisher={IEEE},
  doi={10.1109/JBHI.2025.3570187}
}
```


## Author
Copyright (c) 2026 Marzieh Niaki, Shyamal Y. Dharia, Yangjun Chen, and Camilo E. Valderrama

**Marzieh Niaki** — [LinkedIn](https://www.linkedin.com/in/marzie-pouresmaeil-niaki-ba73561bb/)
