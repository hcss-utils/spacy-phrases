# -*- coding: utf-8 -*-
"""Typer app that extracts noun phrases using spaCy's noun_chunks."""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd  # type: ignore
import spacy
import typer

Phrases = Dict[str, List[str]]


def update_jsonl(path: Path, lines: Phrases) -> None:
    """Update JSONLines file with content."""
    with path.open("a", encoding="utf-8") as output:
        json.dump(lines, output)
        output.write("\n")


def extract_chunks(
    nlp: spacy.language.Language,
    csv_reader: pd.io.parsers.TextFileReader,
    text: str = "fulltext",
    uuid: str = "uuid",
    pattern: str = "influenc",
) -> Iterator[Phrases]:
    """Extract noun-chunks from streamed 'csv_reader'.

    Parameters
    ----------
    nlp: spacy.language.Language
        language model that identifies phrases and takes care of lemmatization
    csv_reader: pd.io.parsers.TextFileReader
        pandas' file reader that processes .csv file in chunks
    text: str
        text column that we extract phrases from (stored as dict values)
    uuid: str
        id column that we use to identify phrases (stored as dict keys)
    pattern: str
        regex pattern that identifies relevant texts
    """
    for chunk in csv_reader:
        mask = chunk[text].str.contains(pattern, na=False, case=False)
        df = chunk.loc[mask].reset_index(drop=True)
        if df.empty:
            continue
        data_tuples = ((df.loc[idx, text], df.loc[idx, uuid]) for idx in df.index)
        for doc, _id in nlp.pipe(data_tuples, as_tuples=True):
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
    models_max_length: int = 2_000_000,
    table_chunksize: int = 10,
    text_field: str = "fulltext",
    uuid_field: str = "uuid",
    pattern: str = "influenc",
) -> None:
    """Extract noun phrases using spaCy."""
    nlp = spacy.load(model, disable=["ner"])
    nlp.max_length = models_max_length
    csv_reader = pd.read_csv(input_table, chunksize=table_chunksize)
    for document in extract_chunks(
        nlp=nlp,
        csv_reader=csv_reader,
        text=text_field,
        uuid=uuid_field,
        pattern=pattern,
    ):
        update_jsonl(output_jsonl, document)


if __name__ == "__main__":
    typer.run(main)
