# -*- coding: utf-8 -*-
"""spaCy's Dependency Matcher exposed via Typer app."""
import csv
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

import typer
import spacy
from spacy.matcher import DependencyMatcher

DataTuple = Tuple[str, str]
Phrases = Dict[str, Dict[str, List[Dict[str, str]]]]
Patterns = List[Dict[str, Union[str, Dict[str, str]]]]


def read_pattern(path: Path) -> Patterns:
    """Read patterns JSON file."""
    with path.open("r", encoding="utf-8") as file_content:
        pattern = json.load(file_content)["pattern"]
    return pattern


def update_jsonl(path: Path, lines: Phrases) -> None:
    """Update JSONLines file with content."""
    with path.open("a", encoding="utf-8") as output:
        json.dump(lines, output, ensure_ascii=False)
        output.write("\n")


def create_nlp(
    model: str, max_length: int, merge_entities: bool, merge_noun_chunks: bool
) -> spacy.language.Language:
    """Customize spaCy's built-in loader."""
    nlp = spacy.load(model)
    nlp.max_length = max_length
    if merge_entities:
        nlp.add_pipe("merge_entities")
    if merge_noun_chunks:
        nlp.add_pipe("merge_noun_chunks")
    return nlp


def build_tuples(path: Path, uuid: str, text: str) -> Iterator[DataTuple]:
    """Build data tuples (text, identifier) for spaCy's pipes."""
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        for row in csv_reader:
            yield row[text], row[uuid]


def build_matcher(nlp: spacy.language.Language, patterns: Path) -> DependencyMatcher:
    """Build Dependency Matcher."""
    matcher = DependencyMatcher(nlp.vocab)
    if patterns.is_dir():
        for patterns_file in patterns.rglob("*.json"):
            pattern = read_pattern(patterns_file)
            matcher.add(patterns_file.stem, [pattern])
    elif patterns.is_file():
        pattern = read_pattern(patterns)
        matcher.add(patterns.stem, [pattern])
    else:
        raise ValueError("patterns should either be a non-empty dir or a file.")
    return matcher


def match(
    nlp: spacy.language.Language,
    data: Iterator[DataTuple],
    matcher: DependencyMatcher,
    batch_size: int,
    keep_text: bool,
) -> Iterator[Phrases]:
    """Match documents/sentences on dependecy tree.

    Parameters
    ----------
    nlp: spacy.language.Language
        language model that identifies phrases and takes care of lemmatization
    data_tuples: DataTuple
        tuple of text and its identifier
    matcher: DependencyMatcher
        spaCy's rule-based matcher
    keep_text: bool
        whether to keep or discard original text
    """
    for doc, _id in nlp.pipe(data, as_tuples=True, batch_size=batch_size):
        phrases: Phrases = {}
        for match_id, token_ids in matcher(doc):
            label = nlp.vocab[match_id].text
            _, pattern = matcher.get(label)
            token_matches = {
                pattern[0][i].get("RIGHT_ID"): doc[token_ids[i]].text
                for i in range(len(token_ids))
            }
            if keep_text:
                token_matches = dict(**token_matches, text=doc.text)
            if _id not in phrases:
                phrases[_id] = {}
            if label not in phrases[_id]:
                phrases[_id][label] = []
            phrases[_id][label].append(token_matches)
        if phrases:
            yield phrases


def main(
    input_table: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input table containing text & metadata"
    ),
    patterns: Path = typer.Argument(
        ...,
        file_okay=True,
        dir_okay=True,
        help="Directory or a single pattern file with rules",
    ),
    output_jsonl: Path = typer.Argument(
        ..., dir_okay=False, help="Output JSONLines file where matches will be stored"
    ),
    model: str = typer.Option("en_core_web_sm", help="SpaCy model's name"),
    docs_max_length: int = typer.Option(2_000_000, help="Doc's max length."),
    merge_entities: bool = False,
    merge_noun_chunks: bool = False,
    text_field: str = "fulltext",
    uuid_field: str = "uuid",
    batch_size: int = 50,
    keep_text: bool = False,
) -> None:
    """Match dependencies using spaCy's dependency matcher."""
    nlp = create_nlp(model, docs_max_length, merge_entities, merge_noun_chunks)
    matcher = build_matcher(nlp, patterns)
    csv.field_size_limit(docs_max_length)
    data_tuples = build_tuples(input_table, uuid=uuid_field, text=text_field)
    for document in match(
        nlp=nlp,
        data=data_tuples,
        matcher=matcher,
        batch_size=batch_size,
        keep_text=keep_text,
    ):
        update_jsonl(output_jsonl, document)


if __name__ == "__main__":
    typer.run(main)
