# -*- coding: utf-8 -*-
"""Transform document-based dataset into a paragraph/sentence-based one."""
import csv
from enum import Enum
from pathlib import Path
from typing import Iterator, Tuple, Dict, Union

import typer
import spacy
from spacy.tokens import Doc
from spacy.language import Language

DocumentBased = Tuple[str, str]
SentenceBased = Dict[str, Union[str, int]]


class Languages(str, Enum):
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
        fieldnames = ["document_id", "unit_id", "fulltext", "lemmas"]
        writer = csv.DictWriter(csv_file, delimiter=",", fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(data)


@Language.component("set_paragraph_boundaries")
def set_paragraph_boundaries(doc: Doc) -> Doc:
    """Split Doc on newline chars instead of on the default punct_chars."""
    for token in doc[:-1]:
        if token.text.startswith("\n"):
            doc[token.i + 1].is_sent_start = True
    return doc


def create_nlp(
    lang: Languages, on_paragraph: bool, max_length: int
) -> Language:
    """Customize spaCy's built-in loader."""
    nlp = spacy.blank(lang)
    custom_pipe = "set_paragraph_boundaries" if on_paragraph else "sentencizer"
    nlp.add_pipe(custom_pipe)
    nlp.max_length = max_length
    return nlp


def build_tuples(path: Path, uuid: str, text: str) -> Iterator[DocumentBased]:
    """Builds data tuples (text, identifier) for spaCy's pipes."""
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        for row in csv_reader:
            if row[text] is not None and row[uuid] is not None:
                yield row[text], row[uuid]


def transform(
    nlp: Language,
    data: Iterator[DocumentBased],
    add_lemmas: bool = False,
) -> Iterator[SentenceBased]:
    """Transform document-based dataset into a paragraph/sentence-based one."""
    for doc, document_id in nlp.pipe(data, as_tuples=True, batch_size=25):
        for sentence_id, sentence in enumerate(doc.sents, start=1):
            processed = {
                "document_id": document_id,
                "unit_id": sentence_id,
                "fulltext": sentence.text,
            }
            if add_lemmas:
                processed["lemmas"] = [
                    t.lemma_ for t in sentence if t.is_alpha and not t.is_stop
                ]
            else:
                processed["lemmas"] = []
            yield processed


def main(
    input_table: Path = typer.Argument(
        ..., exists=True, dir_okay=False, callback=check_extension
    ),
    output_table: Path = typer.Argument(..., dir_okay=False, callback=check_extension),
    lang: Languages = typer.Option(Languages.EN, help="sentecizer's base model"),
    docs_max_length: int = typer.Option(2_000_000, help="Doc's max length."),
    on_paragraph: bool = typer.Option(False, "--paragraph/--sentence"),
    text: str = "fulltext",
    uuid: str = "uuid",
    lemmatize: bool = False,
) -> None:
    """Typer app that processes datasets."""
    csv.field_size_limit(docs_max_length)
    nlp = create_nlp(lang, on_paragraph=on_paragraph, max_length=docs_max_length)
    data_tuples = build_tuples(input_table, uuid, text)
    transformer = transform(nlp, data_tuples, add_lemmas=lemmatize)
    for idx, sentence in enumerate(transformer):
        write_csv(output_table, sentence, idx == 0)


if __name__ == "__main__":
    typer.run(main)
