# installation and development

<br>

## running noctira

<br>

pre-requisites
  - python 3.9+
  - [poetry](https://python-poetry.org/) for dependency management

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

<br>

---

## local development

<br>

### testing github actions locally

<br>

pre-requisites:
- [docker](https://www.docker.com/) for running github actions locally
- [act](https://github.com/nektos/act) for testing github actions locally

<br>

you can test github actions workflows locally using act. this is useful for debugging ci issues before pushing changes.

1. make sure you have act installed (see prerequisites above)
2. run the tests:
   ```bash
   make test-actions  # or make test-actions-mac for apple silicon
   ```

this will run both the ci and test workflows locally, allowing you to catch any issues before pushing your changes.


<br>

#### installing docker

- **macos**:
  ```bash
  brew install --cask docker
  ```
  after installation, open docker desktop and wait for it to start

- **linux**:
  ```bash
  # ubuntu/debian
  sudo apt-get update
  sudo apt-get install docker.io
  sudo systemctl start docker
  sudo systemctl enable docker
  ```

- **windows**:
  download and install [docker desktop](https://www.docker.com/products/docker-desktop)

<br>

#### installing act

- **macos (intel)**:
  ```bash
  brew install act
  ```

- **macos (apple silicon/m1/m2)**:
  ```bash
  brew install act
  ```
  note: when running act on apple silicon, use `make test-actions-mac` instead of `make test-actions`

- **linux**:
  ```bash
  curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
  ```

- **windows**:
  ```bash
  choco install act-cli
  ```

<br>

#### troubleshooting docker issues

if you encounter docker connectivity issues when running act:

1. verify docker is running:
   ```bash
   docker info
   ```

2. check docker network connectivity:
   ```bash
   docker run hello-world
   ```

3. if you're behind a corporate proxy, configure docker to use it:
   ```bash
   # create or edit ~/.docker/config.json
   {
     "proxies": {
       "default": {
         "httpProxy": "http://proxy.example.com:8080",
         "httpsProxy": "http://proxy.example.com:8080",
         "noProxy": "localhost,127.0.0.1"
       }
     }
   }
   ```

4. if you're still having issues, try pulling the image manually first:
   ```bash
   docker pull nektos/act-environments-ubuntu:18.04
   ```
