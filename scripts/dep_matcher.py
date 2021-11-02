# -*- coding: utf-8 -*-
"""spaCy's Dependency Matcher exposed via Typer app."""
import json
from pathlib import Path
from typing import Dict, Iterator, List, Union

import pandas as pd  # type: ignore
import typer
import spacy
from spacy.matcher import DependencyMatcher

Phrases = Dict[str, Dict[str, List[str]]]
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


def extract_chunks(
    nlp: spacy.language.Language,
    matcher: DependencyMatcher,
    csv_reader: pd.io.parsers.TextFileReader,
    text: str = "fulltext",
    uuid: str = "uuid",
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
    for idx, df in enumerate(csv_reader, start=1):
        if df.empty:
            continue
        data_tuples = ((df.loc[idx, text], df.loc[idx, uuid]) for idx in df.index)
        for doc, _id in nlp.pipe(data_tuples, as_tuples=True):
            phrases: Phrases = {}
            for match_id, token_ids in matcher(doc):
                label = nlp.vocab[match_id].text
                _patterns = matcher._raw_patterns.get(match_id)[0]
                token_matches = {
                    _patterns[i].get("RIGHT_ID"): doc[token_ids[i]].text
                    for i in range(len(token_ids))
                }
                if _id not in phrases:
                    phrases[_id] = {}
                if label not in phrases[_id]:
                    phrases[_id][label] = []
                phrases[_id][label].append(token_matches)
            if phrases:
                yield phrases
        typer.echo(f"processed {idx} table chunks..")


def main(
    input_table: Path = typer.Argument(..., exists=True, dir_okay=False),
    patterns: Path = typer.Argument(..., file_okay=True, dir_okay=True),
    output_jsonl: Path = typer.Argument(..., dir_okay=False),
    model: str = "en_core_web_sm",
    models_max_length: int = 2_000_000,
    table_chunksize: int = 10,
    text_field: str = "fulltext",
    uuid_field: str = "uuid",
) -> None:
    """Extract noun phrases using spaCy's dependency matcher."""
    nlp = spacy.load(model, disable=["ner"])
    nlp.max_length = models_max_length
    typer.echo(f"loaded {model} spaCy model...")
    matcher = build_matcher(nlp, patterns)
    typer.echo("built matcher...")
    csv_reader = pd.read_csv(input_table, chunksize=table_chunksize, encoding="utf-8")
    for document in extract_chunks(
        nlp=nlp,
        matcher=matcher,
        csv_reader=csv_reader,
        text=text_field,
        uuid=uuid_field,
    ):
        update_jsonl(output_jsonl, document)


if __name__ == "__main__":
    typer.run(main)
