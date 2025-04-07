# shields

<br>

## tl; dr

<br>

shields are used to validate the agent's inputs and outputs, running in parallel and ensuring data integrity and safety throughout the agent execution pipeline.

there are two types of shields:
- `InputShield`: validates and sanitizes agent input before execution
- `OutputShield`: validates and formats agent output after execution

<br>

the sword primitive is defined in the [src/gear/sword.py](../../src/gear/swords.py) module. swords can be defined by:

- functions and decorators (function swords), or
- an agent (agent swords)

<br>

---

## contents


<br>

---

## overview of the shield module

<br>

### the main class `Shield`

<br>
