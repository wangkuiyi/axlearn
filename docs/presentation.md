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
- More than a trainer -- covers experiment management.

---

## A Trainer

---

## `SpmdTrainer`

The trainer `axlearn.common.trainer.SpmdTrainer` can be configured to train an arbitrary model, using arbitrary dataset and optimizer.

```python
import axlearn.common as ax

class SpmdTrainer(ax.Module):
    class Config(ax.Module.Config):
        model: ax.config.Required[ax.BaseModel.Config] # model's config
        input: ax.config.Required[ax.Input.Config]     # data's config
        learner: ax.config.Required[ax.Learner.Config] # optimizer's config
        ...

    def __init__(self, cfg: Config): ...
    def run(self, prng_key): ...
```

---

## Configure, Instantiate and Run the Trainer

Configure `SpmdTrainer` to train an LLM `MyLLM` using a JSON dataset.

```python
cfg: SpmdTrainer.Config = SpmdTrainer.default_config().set(
    model = MyLLM.default_config().set(layers=56, hidden_dim=2048, ...),
    input = MyJSONDataLoader.default_config().set(fn="/tmp/my.json"),
    learner = SomeCoolOptimizer.default_config().set(lr=1e-9),
)
```

We instantiate a trainer given its config by calling the `instantiate` method, which recursively instantiates `model`, `input`, and `learner`.

```python
if __name__ == "__main__":
    tnr: SpmdTrainer = cfg.instantiate(parent=None)
    tnr.run(jax.random.PRNGKey(42))
```

---

## Everything is Configurable and Instantiable

```python

class MyLLM(ax.BaseModel):
    class Config(BaseModel.Config):
        embedding_layer: ax.config.Required[EmbeddingLayer.Config]
        ...
    def __init__(self, cfg: Config): ...
    def forward(self, batch):
        self.embedding_layers.forward(batch)    ------\
        ...                                           |
                                                      |
class EmbeddingLayer(BaseLayer):                      |
    class Config(BaseLayer.Config):                   | calls
        vocab_size: ax.config.Required[int]           |
        dim: ax.config.Required[int]                  |
    def __init__(self, cfg: Config): ...              |
    def forward(self, batch): ...               <-----/
```

---

## The Class Hierarchy

AXLearn provides a wide range of configurable and instantiable components.

```plaintext
Configurable                           # Root class (config.py)
├── Module                             # Hierarchical composition (module.py)
│   ├── BaseLayer                      # Neural network layers (base_layer.py)
│   │   ├── MultiheadAttention         # Attention primitive
│   │   ├── TransformerAttentionLayer  # Attention + norm + residual
│   │   ├── TransformerLayer           # Self-attn + FFN block
│   │   ├── Decoder                    # Transformer stack + embeddings
│   │   └── BaseModel                  # Trainable model with loss
│   │       └── causal_lm.Model        # Autoregressive LLM
│   │
│   ├── Input                          # Data input (input_base.py)
│   │   ├── input_tf_data.Input        # TensorFlow tf.data
│   │   └── input_grain.Input          # PyGrain (preferred)
│   │
│   └── LearnerModule                  # Optimizer state (learner_base.py)
│       └── BaseLearner
│           └── Learner                # Wraps optimizer + EMA
```

---

## The Root of Class Hierarchy

All classes in AXLearn are derived from `ax.Configurable`. Their config classes are derived from `ax.Configurable.Config`.

```python
class Configurable
    class Config(InstantiableConfig[C]):
        def instantiate(self, **kwargs) -> C: ...

    def default_config(cls: type[C]) -> Config[C]:
        return cls.Config(klass=cls)

    def __init__(self, cfg:Config):
        self._config = copy.deepcopy(cfg)

class InstantiableConfig(Generic[T], ConfigBase):
    def instantiate(self, **kwargs) -> T:
        raise NotImplementedError(type(self))
```

---

## Integrating Third-Party Classes

Any Python class can be made configurable and instantable via `config_for_class`:

```python
from axlearn.common.config import config_for_class
import torch.nn as nn

cfg = config_for_class(nn.Linear).set(in_features=512, out_features=256)
layer: nn.Linear = cfg.instantiate()  # nn.Linear(512, 256)
```

`config_for_class` defines a config class including parameters of `nn.Linear.__init__`.

We could integrate TorchTitan trainer, Ray data loader, etc. to support PyTorch/GPU. But does this make a point?

---

## More Than a Trainer
