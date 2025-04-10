# context

<br>

## tl; dr

<br>

context is a core concept in drusilla that enables passing dependencies and data to custom code. it's not passed to the llm.

<br>

----

## table of contents

<br>

- [`RunContextWrapper`](#runcontextwrapper)
- [context switch](#context-switch)
- [T typevar](#t-typevar)
- [llm context](#llm-context)
- [example](#example)

<br>

___

### `RunContextWrapper`

<br>

`RunContextWrapper` is a generic wrapper class that encapsulates context objects passed to `runner.run()`. it's used to maintain type safety and provide a consistent interface for accessing context data.

<br>

```python
@dataclass
class RunContextWrapper(Generic[TContext]):
    context: TContext
    usage: Usage = field(default_factory=Usage)
```

<br>

___

### context switch

<br>

context switch refers to the process of transferring context between different agents during execution. this happens when:

1. an agent delegates a task to another agent using orbs
2. the context is wrapped in a `RunContextWrapper` and passed through the agent lifecycle
3. charms and shields can access and modify the context during execution

<br>

___

### T typevar

<br>

`T` is a type variable used for generic type annotations in drusilla. it's commonly used with `RunContextWrapper` to specify the type of context data:

<br>

```python
TContext = TypeVar("TContext")
```

<br>

this enables type-safe context handling throughout the framework.

<br>

___

### llm context

<br>

when an llm is called, it can only see data from the conversation history. to make new data available to the llm, you can:

1. add it to the agent instructions (system prompt)
2. add it to the input when calling `Runner.run`
3. expose it via function tools
4. use retrieval or web search tools

<br>

___

### example

<br>

```python
from dataclasses import dataclass
from drusilla import Agent, RunContextWrapper, Runner, function_tool

@dataclass
class UserInfo:
    name: str
    uid: int

@function_tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo]) -> str:
    return f"User {wrapper.context.name} is 47 years old"

async def main():
    user_info = UserInfo(name="John", uid=123)
    agent = Agent[UserInfo](
        name="Assistant",
        tools=[fetch_user_age],
    )
    
    result = await Runner.run(
        starting_agent=agent,
        input="What is the age of the user?",
        context=user_info,
    )
    
    print(result.final_output)
    # The user John is 47 years old.

if __name__ == "__main__":
    asyncio.run(main()) 