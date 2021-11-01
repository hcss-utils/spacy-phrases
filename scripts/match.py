# -*- coding: utf-8 -*-
import time
import json
from pathlib import Path
from typing import Dict, Iterator, List, Union

import pandas as pd  # type: ignore
import spacy
from spacy.matcher import DependencyMatcher
import typer

Phrases = Dict[str, List[str]]
Patterns = List[Dict[str, Union[str, Dict[str, str]]]]


def read_pattern(path: Path) -> Patterns:
    with path.open("r", encoding="utf-8") as file_content:
        pattern = json.load(file_content)["pattern"]
    return pattern


def update_jsonl(path: Path, lines: Phrases) -> None:
    """Update JSONLines file with content."""
    with path.open("a", encoding="utf-8") as output:
        json.dump(lines, output, ensure_ascii=False)
        output.write("\n")


def extract_chunks(
    nlp: spacy.language.Language,
    matcher: DependencyMatcher,
    csv_reader: pd.io.parsers.TextFileReader,
    text: str = "fulltext",
    uuid: str = "uuid",
) -> Iterator[Phrases]:
    """Extract noun-chunks from streamed 'csv_reader'.

    Parameters
    ----------
    nlp: spacy.language.Language
        language model that identifies phrases and takes care of lemmatization
    matcher: DependencyMatcher
        spaCy's rule-based matcher
    csv_reader: pd.io.parsers.TextFileReader
        pandas' file reader that processes .csv file in chunks
    text: str
        text column that we extract phrases from (stored as dict values)
    uuid: str
        id column that we use to identify phrases (stored as dict keys)
    """
    for idx, df in enumerate(csv_reader):
        if df.empty:
            continue
        data_tuples = ((df.loc[idx, text], df.loc[idx, uuid]) for idx in df.index)
        for doc, _id in nlp.pipe(data_tuples, as_tuples=True):
            phrases = {}
            for match_id, (start, end) in matcher(doc):
                label = nlp.vocab[match_id].text
                if _id not in phrases:
                    phrases[_id] = {}
                if label not in phrases[_id]:
                    phrases[_id][label] = []
                phrases[_id][label].append(f"{doc[end]} {doc[start]}")
            if phrases:
                yield phrases
        if idx % 10 == 0:
            typer.echo(idx * 10)
            


def main(
    input_table: Path = typer.Argument(..., exists=True, dir_okay=False),
    pattern_json1: Path = typer.Argument(..., exists=True, dir_okay=False),
    pattern_json2: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_jsonl: Path = typer.Argument(..., dir_okay=False),
    model: str = "en_core_web_sm",
    models_max_length: int = 2_000_000,
    table_chunksize: int = 10,
    text_field: str = "fulltext",
    uuid_field: str = "uuid"
) -> None:
    """Extract noun phrases using spaCy."""
    nlp = spacy.load(model, disable=["ner"])
    nlp.max_length = models_max_length

    pattern_vli = read_pattern(pattern_json1)
    pattern_voz = read_pattern(pattern_json2)
    matcher = DependencyMatcher(nlp.vocab)
    matcher.add("Влияние", [pattern_vli])
    matcher.add("Воздействие", [pattern_voz])

    csv_reader = pd.read_csv(input_table, chunksize=table_chunksize, encoding="utf-8")
    for document in extract_chunks(
        nlp=nlp,
        matcher=matcher,
        csv_reader=csv_reader,
        text=text_field,
        uuid=uuid_field,
    ):
        update_jsonl(output_jsonl, document)


if __name__ == "__main__":
    typer.run(main)
