#
""" """
import csv
import json
from pathlib import Path
from functools import partial
from typing import Iterator, Dict, Union, Tuple, List

import typer
import spacy
from textacy import preprocessing, extract  # type: ignore

DataTuple = Tuple[str, str]
ProcessedDoc = Dict[str, Union[str, List[str], List[Dict[str, str]]]]

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


def create_nlp(model: str, max_length: int) -> spacy.language.Language:
    """Customize spaCy's built-in loader."""
    nlp = spacy.load(model)
    nlp.max_length = max_length
    return nlp


def update_jsonl(path: Path, lines: ProcessedDoc) -> None:
    """Update JSONLines file with content."""
    with path.open("a", encoding="utf-8") as output:
        json.dump(lines, output, ensure_ascii=False)
        output.write("\n")


def build_tuples(path: Path, uuid: str, text: str) -> Iterator[DataTuple]:
    """Builds data tuples (text, identifier) for spaCy's pipes."""
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        for row in csv_reader:
            yield row[text], row[uuid]


def process(
    nlp: spacy.language.Language, data: Iterator[DataTuple], batch_size: int
) -> Iterator[ProcessedDoc]:
    for idx, (doc, document_id) in enumerate(
        nlp.pipe(data, as_tuples=True, batch_size=batch_size), start=1
    ):
        yield {
            "document_id": document_id,
            "processed_text": preproc(doc.text),
            "lemmas": [t.lemma_ for t in doc if t.is_alpha and not t.is_stop],
            "terms": [
                span.text
                for span in extract.terms(
                    doc,
                    ents=partial(
                        extract.entities, include_types=["PERSON", "ORG", "GPE"]
                    ),
                    ncs=True,
                    ngs=lambda doc: extract.ngrams(doc, n=(2, 3)),
                )
            ],
            "subject_verb_object_triples": [
                dict(
                    s=[s.text for s in svo.subject],
                    v=[v.text for v in svo.verb],
                    o=[o.text for o in svo.object],
                )
                for svo in extract.subject_verb_object_triples(doc)
            ],
        }


def main(
    input_table: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input table containing text & metadata"
    ),
    output_jsonl: Path = typer.Argument(
        ..., dir_okay=False, help="Output JSONLines file where matches will be stored"
    ),
    model: str = typer.Option("en_core_web_sm", help="SpaCy model's name"),
    docs_max_length: int = typer.Option(2_000_000, help="Doc's max length."),
    text_field: str = "fulltext",
    uuid_field: str = "uuid",
    batch_size: int = 50,
):
    csv.field_size_limit(docs_max_length)
    nlp = create_nlp(model, docs_max_length)
    data_tuples = build_tuples(input_table, uuid_field, text_field)
    for doc in process(nlp, data_tuples, batch_size):
        update_jsonl(output_jsonl, doc)


if __name__ == "__main__":
    typer.run(main)
