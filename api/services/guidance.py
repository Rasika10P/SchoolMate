"""SQL curriculum guidance and separately scoped framework-prose retrieval."""

from dataclasses import dataclass
from typing import Any, Literal

from psycopg import Connection

from api.models import AchievementLevel, ProgressionStep, Standard
from api.services import progression, standards


@dataclass(frozen=True)
class Guidance:
    framework_id: str
    grade: int
    standards: tuple[Standard, ...]
    achievement_levels: tuple[AchievementLevel, ...]
    progression: tuple[ProgressionStep, ...]


def get_guidance(
    conn: Connection[Any], framework_id: str, grade: int,
    domain: str | None = None,
    direction: Literal["forward", "backward"] | None = None,
) -> Guidance:
    """Use current standards as roots for all three curriculum goals."""
    current = standards.get_standards(conn, framework_id, grade, domain)
    levels: list[AchievementLevel] = []
    steps: list[ProgressionStep] = []
    if direction is None:
        levels = standards.get_achievement_levels(conn, framework_id, grade)
    else:
        for standard in current:
            steps.extend(progression.walk(
                conn, standard.framework_id, standard.code, direction, depth=1,
            ))
    return Guidance(framework_id, grade, tuple(current), tuple(levels), tuple(steps))


# Framework prose retrieval is separate from get_guidance's SQL standards path.
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import re

from api.frameworks import FRAMEWORK_BY_SUBJECT

CORPUS_DIR = Path(__file__).resolve().parents[2] / 'data' / 'frameworks'
EMBED_MODEL = 'multilingual-e5-large'  # One 1024-dimensional cosine index.
logger = logging.getLogger(__name__)


def _subject(framework_id: str) -> str:
    for subject, expected in FRAMEWORK_BY_SUBJECT.items():
        if framework_id == expected:
            return subject
    raise ValueError('framework_id must name a supported, non-empty framework')


def _pinecone():
    key = os.environ.get('PINECONE_API_KEY')
    if not key:
        raise RuntimeError('PINECONE_API_KEY is not configured')
    from pinecone import Pinecone  # Optional until prose ingestion/search is used.
    client = Pinecone(api_key=key)
    host = os.environ.get('PINECONE_INDEX_HOST')
    name = os.environ.get('PINECONE_INDEX')
    if not host and not name:
        raise RuntimeError('PINECONE_INDEX_HOST or PINECONE_INDEX is not configured')
    index = client.Index(host=host) if host else client.Index(name)
    return client, index


def _chunk_ranges(text: str, size: int = 800, overlap: int = 100) -> list[tuple[str, int, int]]:
    """Pack paragraphs while retaining source offsets, including overlap."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < size')
    units = []

    def add(start, end):
        raw = text[start:end]
        start += len(raw) - len(raw.lstrip())
        end -= len(raw) - len(raw.rstrip())
        if start < end:
            units.append((text[start:end], start, end))

    start = 0
    boundaries = [(m.start(), m.end()) for m in re.finditer(r'\n\s*\n', text)]
    for end, following in [*boundaries, (len(text), len(text))]:
        if end - start > size:
            part_start = start
            for separator in re.finditer(r'(?<=[.!?])\s+(?=[A-Z“])', text[start:end]):
                add(part_start, start + separator.start())
                part_start = start + separator.end()
            add(part_start, end)
        else:
            add(start, end)
        start = following

    def joined(parts):
        return '\n\n'.join(part[0] for part in parts)

    def record(parts):
        return joined(parts), parts[0][1], parts[-1][2]

    result, current = [], []
    for unit in units:
        if current and len(joined([*current, unit])) > size:
            result.append(record(current))
            tail = []
            for previous in reversed(current):
                if len(joined([previous, *tail])) > overlap:
                    break
                tail.insert(0, previous)
            current = tail if len(joined([*tail, unit])) <= size else []
        current.append(unit)
    if current:
        result.append(record(current))
    return result


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Prefer paragraph boundaries; split oversized paragraphs between sentences.

    Sizes and overlap are approximate to avoid cutting sentences. Whole trailing
    paragraphs (or sentences in oversized paragraphs) overlap when they fit.
    """
    return [chunk for chunk, _, _ in _chunk_ranges(text, size, overlap)]


def _read_chunks(path: str, framework_id: str) -> list[dict[str, Any]]:
    subject = _subject(framework_id)
    source = Path(path)
    if source.suffix != '.txt':
        raise ValueError('Framework prose must be a .txt file, never a standards CSV')
    text = source.read_text(encoding='utf-8-sig')
    if re.search(r'Standard Identifier:|framework_id,code,grade|^\s*(?:[K0-9]+\.[A-Z]+[.:]|(?:HSS|PE)-\d|ELD\.P)', text, re.M):
        raise ValueError('Standards belong in Postgres, not the framework prose corpus')
    defaults = dict(framework_id=framework_id, subject=subject, grade=None, page=None,
                    document_title=source.stem, source_url='')
    chunks, lines, pages = [], [], []
    metadata = defaults.copy()
    current_page = None
    has_page_markers = False

    def flush():
        clean_text = ''.join(lines)
        for chunk, start, end in _chunk_ranges(clean_text):
            if len(chunk.encode('utf-8')) > 30000:
                raise ValueError('A prose paragraph is too large; add paragraph boundaries before ingestion')
            chunk_metadata = {**metadata, 'page': pages[start]}
            # End at the last content character, not a trailing empty page marker.
            if pages[end - 1] != pages[start]:
                chunk_metadata['page_end'] = pages[end - 1]
            chunks.append({'text': chunk, 'metadata': chunk_metadata})
        lines.clear()
        pages.clear()

    for line in text.splitlines():
        marker = re.fullmatch(r'@page\s+(\d+)', line)
        if marker:
            current_page = int(marker.group(1))
            has_page_markers = True
            continue
        if line.startswith('@metadata '):
            flush()
            supplied = json.loads(line.removeprefix('@metadata '))
            if not isinstance(supplied, dict) or set(supplied) - set(defaults):
                raise ValueError('Invalid prose metadata fields')
            metadata = {**defaults, **supplied}
            if metadata['framework_id'] != framework_id or metadata['subject'] != subject:
                raise ValueError('Prose metadata must match the framework and subject')
            grade = metadata['grade']
            if grade is not None and (isinstance(grade, bool) or not isinstance(grade, int) or not -1 <= grade <= 12):
                raise ValueError('Prose grade must be null or an integer from -1 to 12')
            if not isinstance(metadata['document_title'], str) or not metadata['document_title'].strip():
                raise ValueError('document_title is required')
            # @page is the source of pagination; legacy metadata.page is ignored.
        else:
            content_line = line + '\n'
            lines.append(content_line)
            pages.extend([current_page] * len(content_line))
    flush()
    if not has_page_markers:
        logger.warning('%s: no @page markers found; chunk pages remain unknown', source)
    if not chunks:
        raise ValueError('No framework prose found; namespace left unchanged')
    return chunks


def _values(client, texts: list[str], input_type: str) -> list[list[float]]:
    response = client.inference.embed(model=EMBED_MODEL, inputs=texts,
                                      parameters={'input_type': input_type, 'truncate': 'END'})
    data = response['data'] if isinstance(response, dict) else response.data
    values = [item['values'] if isinstance(item, dict) else item.values for item in data]
    if len(values) != len(texts):
        raise RuntimeError('Embedding service returned the wrong number of vectors')
    return values


def ingest(path: str, framework_id: str) -> int:
    """Embed prose into the mandatory framework namespace; return chunk count.

    Re-ingestion replaces deterministic framework/index IDs. Stale trailing IDs
    from a shorter file are removed only after every new batch upserts. This
    operation owns only its deterministic ID prefix within the namespace.
    """
    return _upsert_chunks(_read_chunks(path, framework_id), framework_id)


def _upsert_chunks(chunks: list[dict[str, Any]], framework_id: str) -> int:
    client, index = _pinecone()
    prefix = 'prose-' + hashlib.sha256(framework_id.encode()).hexdigest()[:16] + '-'
    ids = [f'{prefix}{i:08d}' for i in range(len(chunks))]
    existing = {key for page in index.list(prefix=prefix, namespace=framework_id) for key in page}
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        vectors = _values(client, [c['text'] for c in batch], 'passage')
        records = []
        for offset, (chunk, values) in enumerate(zip(batch, vectors)):
            # Pinecone rejects nulls; search restores omitted nullable fields.
            metadata = {key: value for key, value in chunk['metadata'].items() if value is not None}
            metadata['text'] = chunk['text']
            records.append(dict(id=ids[start + offset], values=values, metadata=metadata))
        index.upsert(vectors=records, namespace=framework_id)
    stale = sorted(existing - set(ids))
    for start in range(0, len(stale), 1000):
        index.delete(ids=stale[start:start + 1000], namespace=framework_id)
    return len(chunks)


def search(query: str, framework_id: str, grade: int | None = None, k: int = 5) -> dict[str, Any]:
    """Retrieve prose from one framework, including general prose alongside the requested grade."""
    _subject(framework_id)  # Fail before any network call for missing/unknown IDs.
    if not query.strip():
        raise ValueError('query must not be empty')
    if grade is not None and (isinstance(grade, bool) or not isinstance(grade, int) or not -1 <= grade <= 12):
        raise ValueError('grade must be null or an integer from -1 to 12')
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
        raise ValueError('k must be an integer from 1 to 100')
    filters: dict[str, Any] = {'framework_id': {'$eq': framework_id}}
    if grade is not None:
        filters['$or'] = [{'grade': {'$eq': grade}}, {'grade': {'$exists': False}}]
    try:
        client, index = _pinecone()
        vector = _values(client, [query], 'query')[0]
        response = index.query(namespace=framework_id, vector=vector, top_k=k,
                               filter=filters, include_metadata=True)
        matches = response['matches'] if isinstance(response, dict) else response.matches
        chunks = []
        for match in matches:
            item = match if isinstance(match, dict) else match.to_dict()
            metadata = dict(item.get('metadata') or {})
            if metadata.get('framework_id') != framework_id:
                continue  # Defense in depth against mislabelled/foreign results.
            if grade is not None and metadata.get('grade') not in (None, grade):
                continue
            text = metadata.pop('text', '')
            if not text:
                continue
            metadata.setdefault('grade', None)
            metadata.setdefault('page', None)
            metadata.setdefault('page_end', None)
            chunks.append(dict(id=item['id'], text=text, score=item.get('score'), metadata=metadata))
        if not chunks:
            return {'state': 'unavailable', 'reason': 'No framework prose is available for this subject and grade selection.'}
        return {'state': 'available', 'framework_id': framework_id, 'chunks': chunks}
    except Exception:
        logger.warning('Framework prose search unavailable for %s', framework_id, exc_info=True)
        return {'state': 'unavailable', 'reason': 'Framework teaching guidance is temporarily unavailable or has not been configured.'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Ingest framework prose into Pinecone namespaces.')
    parser.add_argument('--ingest', action='store_true', required=True)
    parser.add_argument('--framework', choices=list(FRAMEWORK_BY_SUBJECT.values()))
    parser.add_argument('--path', type=Path, help='Override text file; requires --framework')
    args = parser.parse_args(argv)
    if args.path and not args.framework:
        parser.error('--path requires --framework')
    failed = False
    for framework in ([args.framework] if args.framework else FRAMEWORK_BY_SUBJECT.values()):
        path = args.path or CORPUS_DIR / f'{framework}.txt'
        try:
            chunks = _read_chunks(str(path), framework)
            count = _upsert_chunks(chunks, framework)
            paginated = sum(chunk['metadata']['page'] is not None for chunk in chunks)
            print(f'{framework}: {count} chunks upserted; '
                  f'{paginated} with page numbers, {count - paginated} without page numbers', flush=True)
        except Exception as exc:
            failed = True
            print(f'{framework}: ingestion failed ({type(exc).__name__}: {exc})', flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
