# -*- coding: utf-8 -*-
"""Transform document-based dataset into a sentence-based one."""
import csv
from enum import Enum
from pathlib import Path
from typing import Iterator, Tuple, Dict, Union

import typer
import spacy

DocumentBased = Tuple[str, str]
SentenceBased = Dict[str, Union[str, int]]


class Language(str, Enum):
    """Languages that we pass into spaCy's blank model."""
    EN = "en"
    RU = "ru"


def check_extension(path: Path) -> Path:
    """Typer's callback that validates .csv extension."""
    if path.suffix != ".csv":
        typer.echo("You need to pass .csv file")
        raise typer.Exit(code=1)
    return path


def write_csv(path: Path, data: SentenceBased, write_header: bool = False) -> None:
    """Write processed data to .csv file, in chunks."""
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["document_id", "sentence_id", "sentence"]
        writer = csv.DictWriter(csv_file, delimiter=",", fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(data)


def build_tuples(path: Path, uuid: str, text: str) -> Iterator[DocumentBased]:
    """Builds data tuples (text, identifier) for spaCy's pipes."""
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        for row in csv_reader:
            if row[text] is not None and row[uuid] is not None:
                yield row[text], row[uuid]


def sentencize(
    nlp: spacy.language.Language, data: Iterator[DocumentBased]
) -> Iterator[SentenceBased]:
    """Transform document-based dataset into a sentence-based one."""
    for doc, document_id in nlp.pipe(data, as_tuples=True, batch_size=25):
        for sentence_id, sentence in enumerate(doc.sents, start=1):
            yield {
                "document_id": document_id,
                "sentence_id": sentence_id,
                "sentence": sentence.text,
            }


def main(
    input_table: Path = typer.Argument(
        ..., exists=True, dir_okay=False, callback=check_extension
    ),
    output_table: Path = typer.Argument(..., dir_okay=False, callback=check_extension),
    lang: Language = typer.Option(Language.EN, help="sentecizer's base model"),
    text: str = "fulltext",
    uuid: str = "uuid",
) -> None:
    """Typer app that processes datasets."""
    csv.field_size_limit(2_000_000)
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")

    data_tuples = build_tuples(input_table, uuid, text)
    sentencizer = sentencize(nlp, data_tuples)
    for idx, sentence in enumerate(sentencizer):
        write_csv(output_table, sentence, idx == 0)


if __name__ == "__main__":
    typer.run(main)
