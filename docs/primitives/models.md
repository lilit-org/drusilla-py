# models

<br>

## tl; dr

<br>

* an overview on how drusilla handles LLM models

<br>

---

### contents

<br>

- [model settings](#model-settingss)

<br>

---

## model settings

<br>

the agent's underlying model is set by `ModelSettings` (inside [`models/settings.py`](../../src/models/settings.py)), with the following parameters:

<br>

| attribute          | type      | description                          |
|--------------------|-----------|--------------------------------------|
| `temperature`      | `float`   | the temperature to use (0.0 to 2.0)  |
| `top_p`            | `float`   | the top_p to use (0.0 to 1.0)       |
| `max_tokens`       | `int`     | output to generate (must be >= 1)    |
| `sword_choice`     | `literal` | the sword selection to use ("auto", "required", or "none") |
| `parallel_sword_calls`| `bool`    | whether to run sword calls in parallel |


<br>

all parameters are optional and can be set to `None`. validation is performed when settings are used. 