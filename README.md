# spacy-phrases

This repository contains spaCy's **`rule-based matching`** & **`noun-chunk extraction`** scripts.

## Google colabs

| Colab | Description |
| --- | --- |
| [`sentencizer`](https://colab.research.google.com/drive/11dzY5m3swlIDfw7VTKz4StpvjTbi0MAn?usp=sharing) | Transform document-based dataset into a sentence-based one |
| [`noun_chunks`](https://colab.research.google.com/drive/1yPwCk-ptJ9QQlzqUiNHeB3NR9Jnc0EIw?usp=sharing) | Extract noun phrases using spaCy's noun_chunks attribute |
| [`dep_matcher`](https://colab.research.google.com/drive/17CDLmxSD0usg4dcJl1XcuTUejOLHyknj?usp=sharing) | Match documents/sentences on dependecy tree |


## Installation (locally)

To use or contribute to this repository, first checkout the code. 
Then create a new virtual environment:

<details>
<summary>Windows</summary>
<p>

```console
$ git clone https://github.com/hcss-utils/spacy-phrases.git
$ cd spacy-phrases
$ python -m venv env 
$ . env/Scripts/activate
$ pip install -r requirements.txt
```
</p>
</details>

<details>
<summary>MacOS / Linux</summary>
<p>

```console
$ git clone https://github.com/hcss-utils/spacy-phrases.git
$ cd spacy-phrases
$ python3 -m venv env 
$ . env/bin/activate
$ pip install -r requirements.txt
```
</p>
</details>
  
## Usage
### Data transformation

As in some cases we want to have a couple of 'versions' (document-, paragraph-, and sentence-based) of our corpora, 
there's a [scripts/make_dataset.py](scripts/make_dataset.py) that transforms document-based datasets into a sentence-based ones.

<details>
<summary>Data prep</summary>
<p>

To prepare dataset, run `python scripts/make_dataset.py`: 

```console
Usage: make_dataset.py [OPTIONS] INPUT_TABLE OUTPUT_TABLE

  Typer app that processes datasets.

Arguments:
  INPUT_TABLE   [required]
  OUTPUT_TABLE  [required]

Options:
  --lang [en|ru]                  sentecizer's base model  [default:
                                  Languages.EN]
  --docs-max-length INTEGER       Doc's max length.  [default: 2000000]
  --paragraph / --sentence        [default: sentence]
  --text TEXT                     [default: fulltext]
  --uuid TEXT                     [default: uuid]
  --lemmatize / --no-lemmatize    [default: no-lemmatize]
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.
```
</p>
</details>

### Matching phrases

We've developed two different approaches to extracting noun phrases:
- our first guess was to use `Doc`'s `noun_chunks` attribute (we iterate 
over noun_chunks and keep those that fit out criteria). 
But this approach isn't perfect and doesn't for work ru models.
- we then moved to `Rule-based matching` which is more flexible as long as you write accurate patterns 
(and works for both en and ru models).

<details>
<summary>Noun_chunks</summary>
<p>

To extract phrases using noun_chunks approach, run `python scripts/noun_chunks.py`: 

```console
Usage: noun_chunks.py [OPTIONS] INPUT_TABLE OUTPUT_JSONL

  Extract noun phrases using spaCy.

Arguments:
  INPUT_TABLE   [required]
  OUTPUT_JSONL  [required]

Options:
  --model TEXT                    [default: en_core_web_sm]
  --docs-max-length INTEGER       [default: 2000000]
  --batch-size INTEGER            [default: 50]
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

</p>
</details>

<details>
<summary>Dependency Matcher</summary>
<p>

To extract phrases using Dependency Matcher approach, run `python scripts/dep_matcher.py`: 

```console
Usage: dep_matcher.py [OPTIONS] INPUT_TABLE PATTERNS OUTPUT_JSONL

  Match dependencies using spaCy's dependency matcher.

Arguments:
  INPUT_TABLE   Input table containing text & metadata  [required]
  PATTERNS      Directory or a single pattern file with rules  [required]
  OUTPUT_JSONL  Output JSONLines file where matches will be stored  [required]

Options:
  --model TEXT                    SpaCy model's name  [default:
                                  en_core_web_sm]
  --docs-max-length INTEGER       Doc's max length.  [default: 2000000]
  --text-field TEXT               [default: fulltext]
  --uuid-field TEXT               [default: uuid]
  --batch-size INTEGER            [default: 50]
  --context-depth INTEGER
  --merge-entities / --no-merge-entities
                                  [default: no-merge-entities]
  --merge-noun-chunks / --no-merge-noun-chunks
                                  [default: no-merge-noun-chunks]
  --keep-sentence / --no-keep-sentence
                                  [default: no-keep-sentence]
  --keep-fulltext / --no-keep-fulltext
                                  [default: no-keep-fulltext]
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.
```
</p>
</details>

### Counting

Once phrases/matches extracted, you could transform them into a usable format, or/and 
count their frequencies:
- to extract phrases from matches (process rule-based matching output), 
see [notebooks/count-matcher-phrases.ipynb](notebooks/count-matcher-phrases.ipynb)
- to count extacted phrases, 
see [notebooks/count-noun-chunk-phrases.ipynb](notebooks/count-noun-chunk-phrases.ipynb)
