# customizing error messages

<br>

## tl; dr

<br>

the application uses a separate `.error_messages` file to manage all error messages. this allows for easy customization of error messages without modifying the application code.

<br>

---

### file location

<br>

error messages are stored in `.error_messages` in the root directory of the project. this file follows the same format as `.env` files.

<br>

----

### available error messages

<br>


the following error messages can be customized:

- `SWORD_ERROR`: error messages related to the sword component
- `SHIELD_ERROR`: error messages related to the shield component
- `RUNCONTEXT_ERROR`: error messages related to the runcontextwrapper
- `RUNNER_ERROR`: error messages related to the runner component
- `ORBS_ERROR`: error messages related to the orbs component
- `AGENT_EXEC_ERROR`: error messages related to agent execution
- `MODEL_ERROR`: error messages related to the model
- `TYPES_ERROR`: error messages related to type checking
- `OBJECT_ADDITIONAL_PROPERTIES_ERROR`: error messages related to object property validation

<br>

---

### customization


<br>

to customize an error message:

1. open the `.error_messages` file
2. find the error message you want to customize
3. modify the message while keeping the `{error}` placeholder
4. save the file

example:
```env
SWORD_ERROR="custom sword error: {error}"
```

<br>

---

###### placeholders


<br>

all error messages must include the `{error}` placeholder, which will be replaced with the actual error message at runtime.


<br>

---

### default messages


<br>

if the `.error_messages` file is not found, the application will use default error messages defined in the code. a warning will be logged to indicate that the custom error messages file is missing.


<br>

