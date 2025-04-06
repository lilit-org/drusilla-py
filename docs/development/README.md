# local development

<br>

## running noctira

<br>

set up your python environment:

```shell
python3 -m venv venv
source venv/bin/activate
```

<br>

install dependencies:

```shell
make install
```

<br>

create a `.env` file in your project root with your deepseek api endpoint and any customization (or you can leave the default values):

```shell
BASE_URL = "http://localhost:11434"
MODEL = "deepseek-r1"
MAX_TURNS = 10
MAX_SHIELD_QUEUE_SIZE = 1000
LRU_CACHE_SIZE = 128
LOG_LEVEL = "DEBUG"
HTTP_TIMEOUT_TOTAL = 120.0
HTTP_TIMEOUT_CONNECT = 30.0
HTTP_TIMEOUT_READ = 90.0
HTTP_MAX_KEEPALIVE_CONNECTIONS = 5
HTTP_MAX_CONNECTIONS = 10

```

<br>

start [ollama](https://ollama.com/) in another terminal window (after downloading [deepseek-r1](https://ollama.com/library/deepseek-r1)):

```shell
ollama serve
```
