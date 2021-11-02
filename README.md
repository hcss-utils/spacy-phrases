# spacy-phrases

This repository contains spacy-based **`noun-phrases extraction`** scripts. 

## Installation

To use or contribute to this repository, first checkout the code. 
Then create a new virtual environment:

```console
$ git clone https://github.com/hcss-utils/spacy-phrases.git
$ cd spacy-phrases
$ python3 -m venv env
$ . env/bin/activate
$ pip install -r requirements.txt
```

## Usage

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
  --models-max-length INTEGER     [default: 2000000]
  --table-chunksize INTEGER       [default: 10]
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

  Extract noun phrases using spaCy's dependency matcher.

Arguments:
  INPUT_TABLE   [required]
  PATTERNS      [required]
  OUTPUT_JSONL  [required]

Options:
  --model TEXT                    [default: en_core_web_sm]
  --models-max-length INTEGER     [default: 2000000]
  --table-chunksize INTEGER       [default: 10]
  --text-field TEXT               [default: fulltext]
  --uuid-field TEXT               [default: uuid]
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.
```
</p>
</details>

---

Once phrases/matches extracted, you could transform them into a usable format, or/and 
count their frequencies:
- to extract phrases from matches (process rule-based matching output), 
see [notebooks/count-matcher-phrases.ipynb](notebooks/count-matcher-phrases.ipynb)
- to count extacted phrases, 
see [notebooks/count-noun-chunk-phrases.ipynb](notebooks/count-noun-chunk-phrases.ipynb)
