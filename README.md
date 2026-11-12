# Flappy Bird Reinforcement Learning

A tabular Q-learning agent for Flappy Bird, built on [FlapPyBird](https://github.com/sourabhv/FlapPyBird). The state tracks distance to the next pipe, vertical offset from its gap, and the bird's vertical velocity. Training uses epsilon-greedy exploration and updates from completed games.

## Run

Run from the repository root:

```bash
python -m pip install pygame numpy dill
python flappy_rl_test.py
```

The viewer loads the included `model_params/40000.params` checkpoint. To train from scratch:

```bash
python flappy_rl_train.py
```

Learning settings are in `q_learning.py`; the training duration is set by `TRAIN_ITER` in `flappy_rl_train.py`. Checkpoints are saved every 10,000 games. Run `python flappy.py` to play manually.
