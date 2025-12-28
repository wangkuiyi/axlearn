---
marp: true
class: invert
---

# An Introduction to AXLearn

Yi Wang
<wyi@apple.com>

---

## What Is AXLearn

- A trainer, like TorchTitan or Megatron-LM.

- More than a trianer -- covers experiment management.

---

## The Trainer

The trainer `axlearn.common.trainer.SpmdTrainer` is configurable.

```python
import axlearn.common as ax

class SpmdTrainer(ax.Module):
    class Config(ax.Module.Config):
        model: ax.config.Required[ax.BaseModel.Config] # model's config
        input: ax.config.Required[ax.Input.Config]     # data's config
        learner: ax.config.Required[ax.Learner.Config] # optimizer's config
        ...

    def __init__(self, cfg: Config):
        ...

    def run(self, prng_key):
        ...
```

---

## Configure, Instantiate and Run

Configure `SpmdTrainer` to train an LLM `MyLLM` using a JSON dataset.

```python
if __name__ == "__main__":
    cfg = SpmdTrainer.default_config().set(
        model = MyLLM.default_config().set(layers=56, hidden_dim=2048, ...),
        input = MyJSONDataLoader.default_config().set(fn="/tmp/my.json"),
        learner = SomeCoolOptimizer.default_config().set(lr=1e-9),
    )
    # The following line recursively calls the instantiate method of
    # model, input, and learner.
    tnr: SpmdTrainer = cfg.instantiate(parent=None)
    tnr.run(jax.random.PRNGKey(42))
```

---

## Everything is Configurable

```python

class MyLLM(ax.BaseModel):
    class Config(BaseModel.Config):
        embedding_layer: ax.config.Required[EmbeddingLayer.Config]
        ...
    def __init__(self, cfg: Config):
        ...
    def forward(self, batch):
        ...

class EmbeddingLayer(BaseLayer):
    class Config(BaseLayer.Config):
        vocab_size: ax.config.Required[int]
        dim: ax.config.Required[int]
    def __init__(self, cfg: Config):
        ...
    def forward(self, batch):
        ...
```
