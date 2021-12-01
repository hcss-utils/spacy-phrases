# -*- coding: utf-8 -*-
"""spaCy's Dependency Matcher exposed via Typer app."""
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import typer
import spacy
from spacy.tokens import Doc, Span
from spacy.matcher import DependencyMatcher

DataTuple = Tuple[str, str]
Phrases = Dict[str, Dict[str, List[Dict[str, str]]]]
Patterns = List[Dict[str, Union[str, Dict[str, str]]]]


def read_processed_data(path: Path) -> Set[str]:
    """Collect processed uuids."""
    seen: Set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as lines:
        for line in lines:
            data = json.loads(line)
            for uuid in data.keys():
                seen.add(uuid)
    return seen


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


def build_tuples(
    path: Path, uuid: str, text: str, processed_uuids: Set[str]
) -> Iterator[DataTuple]:
    """Build data tuples (text, identifier) for spaCy's pipes."""
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        for row in csv_reader:
            if row[uuid] in processed_uuids:
                continue
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


def get_previous_sentence(doc: Doc, sent: Optional[Span]) -> Optional[Span]:
    """Extract previous sentence before `sent`."""
    if sent is None:
        return None
    if sent.start - 1 < 0:
        return None
    return doc[sent.start - 1].sent


def get_next_sentence(doc: Doc, sent: Optional[Span]) -> Optional[Span]:
    """Extract next sentence after `sent`."""
    if sent is None:
        return None
    if sent.end + 1 >= len(doc):
        return None
    return doc[sent.end + 1].sent


def get_context(doc: Doc, sent: Span, depth: int = 1) -> Span:
    """Structurally, extract context (`depth` sents before and after `sent`)."""
    previous_sent = get_previous_sentence(doc, sent)
    next_sent = get_next_sentence(doc, sent)
    for _ in range(1, depth):
        previous_sent = get_previous_sentence(doc, previous_sent)
        next_sent = get_next_sentence(doc, next_sent)
    previous_sent_i = None if previous_sent is None else previous_sent[0].i
    next_sent_i = None if next_sent is None else next_sent[-1].i
    return doc[previous_sent_i:next_sent_i].text


def match(
    nlp: spacy.language.Language,
    data_tuples: Iterator[DataTuple],
    matcher: DependencyMatcher,
    batch_size: int,
    keep_sentence: bool,
    keep_fulltext: bool,
) -> Iterator[Phrases]:
    """Match documents/sentences on dependecy tree.

    Parameters
    ----------
    nlp: spacy.language.Language
        spaCy's language model
    data_tuples: Iterator[DataTuple]
        tuple of text and its identifier
    matcher: DependencyMatcher
        spaCy's rule-based matcher
    batch_size: int
        the number of texts to buffer
    keep_sentence: bool
        whether to keep or discard sentence within which matches occur
    keep_fulltext: bool
        whether to keep or discard original text
    """
    for doc, _id in nlp.pipe(data_tuples, as_tuples=True, batch_size=batch_size):
        phrases: Phrases = defaultdict(lambda: defaultdict(list))
        for match_id, token_ids in matcher(doc):
            label = nlp.vocab[match_id].text
            _, pattern = matcher.get(label)
            token_matches = {
                pattern[0][i].get("RIGHT_ID"): doc[token_ids[i]].lemma_.lower()
                for i in range(len(token_ids))
            }
            if keep_sentence:
                sent = doc[min(token_ids)].sent
                token_matches["sentence"] = doc[sent.start:sent.end].text
                token_matches["sent_context"] = get_context(doc, sent)
            if keep_fulltext:
                token_matches["fulltext"] = doc.text
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
    text_field: str = "fulltext",
    uuid_field: str = "uuid",
    batch_size: int = 50,
    merge_entities: bool = False,
    merge_noun_chunks: bool = False,
    keep_sentence: bool = False,
    keep_fulltext: bool = False,
) -> None:
    """Match dependencies using spaCy's dependency matcher."""
    nlp = create_nlp(model, docs_max_length, merge_entities, merge_noun_chunks)
    matcher = build_matcher(nlp, patterns)
    csv.field_size_limit(docs_max_length)
    processed_uuids = read_processed_data(output_jsonl)
    data_tuples = build_tuples(
        input_table, uuid=uuid_field, text=text_field, processed_uuids=processed_uuids
    )
    for document in match(
        nlp=nlp,
        data_tuples=data_tuples,
        matcher=matcher,
        batch_size=batch_size,
        keep_sentence=keep_sentence,
        keep_fulltext=keep_fulltext,
    ):
        update_jsonl(output_jsonl, document)


if __name__ == "__main__":
    typer.run(main)
