# spacy-phrases

This repository contains spacy-based **`noun-phrases extraction`** script. 

## Usage

To use or contribute to this repository, first checkout the code. 
Then create a new virtual environment:

```console
$ git clone https://github.com/hcss-utils/spacy-phrases.git
$ cd spacy-phrases
$ python3 -m venv env
$ . env/bin/activate
$ pip install -r requirements.txt
```

To run the app, use `python main.py`: 

```console
Usage: main.py [OPTIONS] INPUT_TABLE OUTPUT_JSONL

Arguments:
  INPUT_TABLE   [required]
  OUTPUT_JSONL  [required]

Options:
  --model TEXT                    [default: en-core-web-sm]
  --text-field TEXT               [default: fulltext]
  --uuid-field TEXT               [default: uuid]
  --pattern TEXT                  [default: influenc]
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.
```