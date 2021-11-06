# -*- coding: utf-8 -*-
"""Typer app that extracts noun phrases using spaCy's noun_chunks."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import spacy
import typer

DataTuple = Tuple[str, str]
Phrases = Dict[str, List[str]]


def update_jsonl(path: Path, lines: Phrases) -> None:
    """Update JSONLines file with content."""
    with path.open("a", encoding="utf-8") as output:
        json.dump(lines, output)
        output.write("\n")


def build_tuples(path: Path, uuid: str, text: str) -> Iterator[DataTuple]:
    """Build data tuples (text, identifier) for spaCy's pipes."""
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        for row in csv_reader:
            yield row[text], row[uuid]


def extract_chunks(
    nlp: spacy.language.Language,
    data_tuples: Iterator[DataTuple],
    batch_size: int,
    pattern: str,
) -> Iterator[Phrases]:
    """Extract noun-chunks from streamed 'csv_reader'.

    Parameters
    ----------
    nlp: spacy.language.Language
        language model that identifies phrases and takes care of lemmatization
    data_tuples: DataTuple
        tuple of text and its identifier
    batch_size: int
        the number of texts to buffer
    pattern: str
        noun that identifies relevant noun chunks
    """
    for doc, _id in nlp.pipe(data_tuples, as_tuples=True, batch_size=batch_size):
        phrases = defaultdict(list)
        for noun_chunk in doc.noun_chunks:
            if (
                pattern not in noun_chunk.lemma_
                or any(t.is_stop or t.is_digit for t in noun_chunk)
                or len(noun_chunk) < 2
            ):
                continue
            phrases[_id].append(noun_chunk.lemma_.lower())
        if phrases:
            yield phrases


def main(
    input_table: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_jsonl: Path = typer.Argument(..., dir_okay=False),
    model: str = "en_core_web_sm",
    docs_max_length: int = 2_000_000,
    batch_size: int = 50,
    text_field: str = "fulltext",
    uuid_field: str = "uuid",
    pattern: str = "influenc",
) -> None:
    """Extract noun phrases using spaCy."""
    nlp = spacy.load(model, disable=["ner"])
    nlp.max_length = docs_max_length
    csv.field_size_limit(docs_max_length)
    data_tuples = build_tuples(input_table, uuid=uuid_field, text=text_field)
    for document in extract_chunks(
        nlp=nlp,
        data_tuples=data_tuples,
        batch_size=batch_size,
        pattern=pattern,
    ):
        update_jsonl(output_jsonl, document)


if __name__ == "__main__":
    typer.run(main)
