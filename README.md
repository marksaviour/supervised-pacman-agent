# Supervised Pacman Agent — Neural Network Move Classifier

## About

A supervised-learning agent that controls Pac-Man by **predicting moves from a
neural network it trains from scratch**. The classifier is a **multi-layer
perceptron (MLP)** implemented end-to-end in NumPy — no scikit-learn, no deep
learning framework — that learns to map the binary feature vector describing
Pac-Man's surroundings (walls, food, ghosts) to one of four movement actions
(North, East, South, West).

It learns by **imitation**: the training data (`good-moves.txt`) is a log of
feature-vector → move pairs recorded from gameplay, and the network learns to
reproduce those decisions, a form of behavioural cloning. The brief only
required a simple classifier (even 1-NN would satisfy it); this implementation
goes well beyond that with a fully hand-written MLP, Adam optimisation,
backpropagation, and early stopping.

This was built as a coursework project for the **Machine Learning** module at
**King's College London**.

---

## Attribution

The Pac-Man framework this agent plugs into was originally developed at **UC
Berkeley** for their CS188 Intro to AI course (`pacman.py`, `game.py`, `api.py`,
`classifierAgents.py`, the layouts, etc.). The Berkeley homepage for the projects
is http://ai.berkeley.edu/.

> **Only `classifier.py` (the `Classifier` class) is my own work.** Every other
> file in the Pac-Man distribution is unmodified Berkeley-provided code and is not
> included here — the coursework explicitly allows modifying only this one file.
> To run the agent, drop `classifier.py` into a clean copy of the Berkeley
> Pac-Man distribution.

---

## What it does

The agent decides Pac-Man's move at every step of the game by:

1. Receiving a **binary feature vector** from the game's `api.getFeatureVector()`
   (a list of 1s and 0s encoding walls, food, and ghost positions in the cells
   around Pac-Man).
2. Feeding that vector through a trained neural network.
3. Returning the predicted direction — with a guarantee that the returned move is
   always **legal**, so the game never crashes.

The network is trained once at game start-up on the labelled `good-moves.txt`
data, then queried for a prediction on every subsequent turn.

---

## How it works

### Architecture
A feed-forward MLP:

```
input (feature vector)  ->  Dense(64) + ReLU  ->  Dense(32) + ReLU  ->  Dense(4) + softmax
```

The input width is **inferred from the data at training time** rather than
hardcoded, so the same code works for any feature-vector length (different Pac-Man
layouts produce different-length vectors). The four output neurons correspond to
North / East / South / West.

### Training (`fit`)
- **Loss:** categorical cross-entropy with **L2 regularisation** to discourage
  overfitting on the small (~127-example) training set.
- **Optimiser:** **Adam** (adaptive per-parameter learning rates) for fast, stable
  convergence with little tuning, implemented manually with bias-corrected first
  and second moment estimates (`_adam_update`, `_init_adam`).
- **Weight initialisation:** **He initialisation** (`std = sqrt(2 / n_in)`), chosen
  because ReLU zeroes roughly half its inputs, so this keeps gradient magnitudes
  stable across layers (`_init_weights`).
- **Optimisation loop:** **mini-batch stochastic gradient descent** — data is
  shuffled each epoch and processed in batches, with a full forward pass
  (`_forward`) and a hand-derived **backpropagation** pass (`_backward`) computing
  the gradients for every layer.
- **Early stopping:** a held-out validation split is scored each epoch; the
  best-validation weights are cached and restored, and training halts if the
  validation loss fails to improve for `patience` epochs — preventing overfitting.

### Inference (`predict`)
A forward pass produces a softmax probability over the four actions. The agent
takes the highest-probability move, but includes a **legality safety net**: if the
top prediction isn't currently legal, it falls back to the highest-probability
move that *is* legal (`_legal_to_ints`, `_safe_random_legal`). This ensures the
controller never sends an illegal action to the game engine.

### Supporting pieces
The class implements every numerical component by hand, including:
`_relu` / `_relu_derivative`, `_softmax`, `_cross_entropy_loss`, `_one_hot`
encoding, and `_train_val_split`. A `reset()` method clears learned state so the
agent can be re-trained cleanly across consecutive games.

---

## Key hyperparameters

Set in `__init__` and tuned empirically for the small dataset:

| Parameter | Value | Role |
|-----------|-------|------|
| `hidden_sizes` | `[64, 32]` | Two hidden layers — enough capacity for non-linear boundaries without overfitting |
| `n_classes` | `4` | Output actions (N/E/S/W) |
| `learning_rate` | `0.001` | Initial Adam learning rate |
| `epochs` | `500` | Maximum training epochs (early stopping usually triggers first) |
| `batch_size` | `32` | Mini-batch size |
| `l2_lambda` | `1e-4` | L2 regularisation strength |

---

## Running

Place `classifier.py` in a clean Berkeley Pac-Man distribution (the coursework
`pacman-base.zip` folder), then:

```bash
python3 pacman.py -p ClassifierAgent
```

The agent requires only **NumPy** (the one external dependency used here); the
network itself is implemented from scratch.

---

## Notes

This was written as an academic exercise. The point of the task was to build *and
use* a classifier to drive Pac-Man — game-winning performance is explicitly not
the objective, only that a genuine classifier makes the decisions. It is shared as
a portfolio piece demonstrating a neural network, Adam, backpropagation, and
regularisation implemented from first principles in NumPy.