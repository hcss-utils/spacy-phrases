# -*- coding: utf-8 -*-
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Union

import typer
import spacy
import pandas as pd

JSONL = Dict[str, List[str]]
Phrases = Dict[str, List[str]]


def update_jsonl(p: Path, lines: JSONL) -> None:
    """Update JSONLines file with content."""
    with p.open("a", encoding="utf-8") as output:
        json.dump(lines, output)
        output.write("\n")


def extract_chunks(
    nlp: spacy.language.Language,
    csv_reader: pd.io.parsers.TextFileReader,
    text_column: str = "fulltext",
    uuid_column: str = "uuid",
    pattern: str = "influenc",
) -> Iterator[Phrases]:
    """Extract noun-chunks from streamed 'csv_reader'."""
    for csv_chunk in csv_reader:
        _mask = csv_chunk[text_column].str.contains(pattern, na=False, case=False)
        df = csv_chunk.loc[_mask].reset_index(drop=True)
        if df.empty:
            continue
        data_tuples = (
            (df.loc[idx, text_column], df.loc[idx, uuid_column]) for idx in df.index
        )
        for doc, uuid in nlp.pipe(data_tuples, as_tuples=True):
            dd = defaultdict(list)
            for chunk in doc.noun_chunks:
                if (
                    pattern not in chunk.lemma_
                    or any(t.is_stop or t.is_digit for t in chunk)
                    or len(chunk) < 2
                ):
                    continue
                dd[uuid].append(chunk.lemma_)
            if dd:
                yield dd


def main(
    input_table: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_jsonl: Path = typer.Argument(..., dir_okay=False),
    model: str = "en_core_web_sm",
    text_field: str = "fulltext",
    uuid_field: str = "uuid",
    pattern: str = "influenc",
) -> None:
    nlp = spacy.load(model, disable=["ner"])
    nlp.max_length = 2_000_000  # based on previous cases
    csv_reader = pd.read_csv(input_table, chunksize=10)
    for document in extract_chunks(
        nlp=nlp,
        csv_reader=csv_reader,
        text_column=text_field,
        uuid_column=uuid_field,
        pattern=pattern,
    ):
        update_jsonl(output_jsonl, document)


if __name__ == "__main__":
    typer.run(main)
