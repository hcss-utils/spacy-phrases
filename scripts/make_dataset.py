# -*- coding: utf-8 -*-
"""Transform document-based dataset into a sentence-based one."""
from enum import Enum
from pathlib import Path
from typing import Iterator, Tuple, Dict, Union

import typer
import spacy
import pandas as pd  # type: ignore

DocumentBased = Tuple[str]
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


def build_tuples(data: pd.DataFrame, uuid: str, text: str) -> DocumentBased:
    """Builds data tuples (text, identifier) for spaCy's pipes."""
    return ((data.loc[idx, text], data.loc[idx, uuid]) for idx in data.index)


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
    output_table: Path = typer.Argument(
        ..., dir_okay=False, callback=check_extension
    ),
    lang: Language = typer.Option(Language.EN, help="sentecizer's base model"),
    text: str = "fulltext",
    uuid: str = "uuid",
) -> None:
    """Typer app that processes datasets."""
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")

    data = pd.read_csv(input_table)
    data_tuples = build_tuples(data, uuid, text)
    sentencizer = sentencize(nlp, data_tuples)
    pd.DataFrame(sentencizer).to_csv(output_table, index=False)


if __name__ == "__main__":
    typer.run(main)
