# orbs

<br>

## tl; dr

<br>

orbs serve as intelligent intermediaries that:

- facilitate dynamic task transfer between agents
- maintain context and state during delegation
- enable flexible agent-to-agent communication
- support complex multi-agent workflows

<br>

---

## contents

<br>

- [overview of the shield module](#overview-of-the-shield-module)
    - [the base class for `Orbs`](#the-base-class-for-orbs)
- [tips and best practices](#tips-and-best-practices)
  - [customizing error messages](#customizing-error-messages)
  - [running tests](#running-tests)
- [available examples](#available-examples)

<br>

---

## overview of the orbs module

<br>

### the base class for `Orbs`

<br>


<br>



`@function_sword` is implemented through a decorator factory utilizing the `@overload` pattern to provide better type hint and documentation for different ways the function can be called.



----

## tips and best practices

<br>

### customizing error messages

<br>

in the code above, error handlers (and their messages) are stored inside `ORBS_ERROR_HANDLER`, which is defined in the top of the file with:

```python
ORBS_ERROR_HANDLER = create_error_handler(ERROR_MESSAGES.ORBS_ERROR.message)
```

<br>

`create_error_handler()` is a method defined in [util/_exceptions.py](../../src/util/_exceptions.py) and is not intended to be modified. however, the string `ERROR_MESSAGES.ORBS_ERROR.message` (which is imported from [util/_constants.py](../../src/util/_constants.py)) can be directly customized inside your [`.env`](../../.env.example).

<br>

---

### running tests

<br>

unit tests for the `Shield` module can be run with:

<br>

```shell
poetry run pytest tests/gear/test_shield.py -v
```

<br>

---

## available examples

<br>

* [agent world traveler](../../examples/agents/world_traveler.py)
* [agent friend with benefit](../../examples/agents/friend_with_benefits.py)
