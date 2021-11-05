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


def build_tuples(path: Path, uuid: str, text: str) -> Iterator[DataTuple]:
    """Builds data tuples (text, identifier) for spaCy's pipes."""
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
    nlp: spacy.language.Language, data: DataTuple, matcher: DependencyMatcher, keep_text: bool
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
    for doc, _id in nlp.pipe(data, as_tuples=True, batch_size=50):
        phrases: Phrases = {}
        for match_id, token_ids in matcher(doc):
            label = nlp.vocab[match_id].text
            _patterns = matcher._raw_patterns.get(match_id)[0]
            token_matches = {
                _patterns[i].get("RIGHT_ID"): doc[token_ids[i]].text
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
    input_table: Path = typer.Argument(..., exists=True, dir_okay=False),
    patterns: Path = typer.Argument(..., file_okay=True, dir_okay=True),
    output_jsonl: Path = typer.Argument(..., dir_okay=False),
    model: str = "en_core_web_sm",
    models_max_length: int = 2_000_000,
    text_field: str = "fulltext",
    uuid_field: str = "uuid",
    keep_text: bool = False,
) -> None:
    """Extract noun phrases using spaCy's dependency matcher."""
    nlp = spacy.load(model)
    nlp.max_length = models_max_length
    csv.field_size_limit(models_max_length)
    typer.echo(f"loaded {model} spaCy model...")

    matcher = build_matcher(nlp, patterns)
    typer.echo("built matcher...")

    data_tuples = build_tuples(input_table, uuid=uuid_field, text=text_field)
    typer.echo("built data_tuples...")

    for document in match(
        nlp=nlp, data=data_tuples, matcher=matcher, keep_text=keep_text
    ):
        update_jsonl(output_jsonl, document)


if __name__ == "__main__":
    typer.run(main)
