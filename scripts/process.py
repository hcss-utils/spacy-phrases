# -*- coding: utf-8 -*-
"""Text preprocessing."""
import csv
import json
from functools import partial
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple, Union

import spacy
import typer
from textacy import extract, preprocessing  # type: ignore

DataTuple = Tuple[str, str]
ProcessedDoc = Dict[str, Union[str, List[str], List[Dict[str, List[str]]]]]

preproc = preprocessing.make_pipeline(
    preprocessing.replace.emails,
    preprocessing.replace.hashtags,
    preprocessing.replace.emojis,
    preprocessing.replace.phone_numbers,
    preprocessing.replace.urls,
    preprocessing.remove.accents,
    preprocessing.remove.brackets,
    preprocessing.remove.html_tags,
    preprocessing.normalize.bullet_points,
    preprocessing.normalize.hyphenated_words,
    preprocessing.normalize.quotation_marks,
    preprocessing.normalize.unicode,
    preprocessing.normalize.whitespace,
)


def read_processed_data(path: Path) -> Set[str]:
    """Collect processed uuids."""
    seen: Set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        for row in csv_reader:
            seen.add(row["document_id"])
    return seen


def create_nlp(model: str, max_length: int) -> spacy.language.Language:
    """Customize spaCy's built-in loader."""
    nlp = spacy.load(model)
    nlp.max_length = max_length
    return nlp


def build_tuples(
    path: Path, uuid: str, text: str, processed_uuids: Set[str]
) -> Iterator[DataTuple]:
    """Builds data tuples (text, identifier) for spaCy's pipes."""
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        for row in csv_reader:
            if row[uuid] in processed_uuids:
                continue
            yield row[text], (row[uuid], row["unit_id"])


def write_csv(path: Path, data: ProcessedDoc, write_header: bool = False) -> None:
    """Write processed data to .csv file, in chunks."""
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "document_id",
            "unit_id",
            "fulltext",
            "lemmas",
            "terms",
            "subject_verb_object_triples",
        ]
        writer = csv.DictWriter(csv_file, delimiter=",", fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(data)


def process(
    nlp: spacy.language.Language,
    data: Iterator[DataTuple],
    batch_size: int,
    preprocess: bool,
    lemmas: bool,
    terms: bool,
    sovs: bool,
) -> Iterator[ProcessedDoc]:
    """Preprocessing text."""
    for doc, (document_id, unit_id) in nlp.pipe(data, as_tuples=True, batch_size=batch_size):
        record: ProcessedDoc = {"document_id": document_id, "unit_id": unit_id}
        if preprocess:
            record["fulltext"] = preproc(doc.text)
        if lemmas:
            record["lemmas"] = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
        if terms:
            record["terms"] = [
                span.text
                for span in extract.terms(
                    doc,
                    ents=partial(
                        extract.entities, include_types=["PERSON", "ORG", "GPE"]
                    ),
                    ngs=lambda doc: extract.ngrams(doc, n=(1, 3)),
                )
            ]
        if sovs:
            record["subject_verb_object_triples"] = [
                dict(
                    s=[s.text for s in svo.subject],
                    v=[v.text for v in svo.verb],
                    o=[o.text for o in svo.object],
                )
                for svo in extract.subject_verb_object_triples(doc)
            ]
        yield record


def main(
    input_table: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input table containing text & metadata"
    ),
    output_table: Path = typer.Argument(
        ..., dir_okay=False, help="Output csv file where matches will be stored"
    ),
    model: str = typer.Option("en_core_web_sm", help="SpaCy model's name"),
    docs_max_length: int = typer.Option(2_000_000, help="Doc's max length."),
    text_field: str = "fulltext",
    uuid_field: str = "uuid",
    batch_size: int = 50,
    preprocess: bool = True,
    lemmas: bool = False,
    terms: bool = False,
    sovs: bool = False,
) -> None:
    """Preprocess dataset."""
    if not any(pipe for pipe in [preprocess, lemmas, terms, sovs]):
        raise ValueError("You should call at least 1 pipeline.")
    csv.field_size_limit(docs_max_length)
    nlp = create_nlp(model, docs_max_length)
    processed_uuids = read_processed_data(output_table)
    data_tuples = build_tuples(input_table, uuid_field, text_field, processed_uuids)
    transformer = process(
        nlp=nlp,
        data=data_tuples,
        batch_size=batch_size,
        preprocess=preprocess,
        lemmas=lemmas,
        terms=terms,
        sovs=sovs,
    )
    for idx, doc in enumerate(transformer):
        write_header = bool(idx == 0 and not processed_uuids)
        write_csv(output_table, doc, write_header)


if __name__ == "__main__":
    typer.run(main)
