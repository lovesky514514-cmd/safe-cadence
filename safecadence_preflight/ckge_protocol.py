"""Load incremental FastKGE-style snapshots without future-data leakage."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


@dataclass
class Stage:
    """One snapshot: incoming facts plus training facts seen so far."""

    index: int
    incoming: np.ndarray
    cumulative: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    n_entities: int
    n_relations: int


def _mapping(path: Path) -> dict[str, int]:
    """Map names to contiguous IDs, rejecting duplicates and negative IDs."""
    mapping: dict[str, int] = {}
    used: set[int] = set()
    for lineno, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f'{path}:{lineno}: expected NAME ID')
        name, raw_id = fields
        try:
            identifier = int(raw_id)
        except ValueError:
            raise ValueError(f'{path}:{lineno}: invalid numeric ID') from None
        if name in mapping or identifier in used or identifier < 0:
            raise ValueError(f'{path}:{lineno}: duplicate or negative mapping')
        mapping[name] = identifier
        used.add(identifier)
    if used != set(range(len(used))):
        raise ValueError(f'{path}: entity/relation IDs must be contiguous')
    return mapping


def _rows(path: Path, entities: dict[str, int], relations: dict[str, int],
          *, numeric: bool) -> np.ndarray:
    """Parse unique triples in (head, relation, tail) order."""
    rows: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for lineno, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f'{path}:{lineno}: expected h r t')
        if numeric:
            try:
                head, relation, tail = (int(v) for v in fields)
            except ValueError:
                raise ValueError(f'{path}:{lineno}: noninteger triple') from None
        else:
            try:
                head = entities[fields[0]]
                relation = relations[fields[1]]
                tail = entities[fields[2]]
            except KeyError as exc:
                raise ValueError(f'{path}:{lineno}: unknown entity/relation {exc}') from None
        if min(head, relation, tail) < 0 or head >= len(entities) or tail >= len(entities) or relation >= len(relations):
            raise ValueError(f'{path}:{lineno}: entity/relation ID out of range')
        triple = (head, relation, tail)
        if triple in seen:
            raise ValueError(f'{path}:{lineno}: duplicate triple')
        seen.add(triple)
        rows.append(triple)
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def _as_set(rows: np.ndarray) -> set[tuple[int, int, int]]:
    return set(map(tuple, rows.tolist()))


def load_ckge(root: str | Path, *, snapshot_count: int = 5,
              strict: bool = True) -> list[Stage]:
    """Load incremental train triples; do not merge validation or test into train.

    Existing entity IDs cannot move; relation mapping cannot change.
    Strict mode rejects train overlap across snapshots and overlap between
    the current snapshot's train/valid/test splits.
    """
    if not isinstance(snapshot_count, int) or not 1 <= snapshot_count <= 5:
        raise ValueError('snapshot_count must be between 1 and 5')

    root = Path(root)
    stages: list[Stage] = []
    old_entities: dict[str, int] = {}
    old_relations: dict[str, int] | None = None
    history: set[tuple[int, int, int]] = set()
    for index in range(snapshot_count):
        folder = root / str(index)
        if not folder.is_dir():
            raise FileNotFoundError(f'missing snapshot folder: {folder}')
        entities = _mapping(folder / 'entity2id.txt')
        relations = _mapping(folder / 'relation2id.txt')
        if not entities or not relations:
            raise ValueError(f'snapshot {index}: empty entity/relation mapping')
        if any(entities.get(name) != identifier for name, identifier in old_entities.items()):
            raise ValueError(f'snapshot {index}: entity mapping is not append-only')
        if old_relations is not None and relations != old_relations:
            raise ValueError(f'snapshot {index}: relation mapping changed')
        old_entities, old_relations = entities, relations

        incoming = _rows(folder / 'train_id.txt', entities, relations, numeric=True)
        if not len(incoming):
            raise ValueError(f'snapshot {index}: empty incoming train set')
        # A textual train file, when provided, must match the numeric file.
        if (folder / 'train.txt').exists():
            named = _rows(folder / 'train.txt', entities, relations, numeric=False)
            if _as_set(named) != _as_set(incoming):
                raise ValueError(f'snapshot {index}: train.txt and train_id.txt disagree')
        valid = _rows(folder / 'valid.txt', entities, relations, numeric=False)
        test = _rows(folder / 'test.txt', entities, relations, numeric=False)
        new, validation, held_out = map(_as_set, (incoming, valid, test))
        if strict:
            if history & new:
                raise ValueError(f'snapshot {index}: new train overlaps historical train')
            if new & validation or new & held_out or validation & held_out:
                raise ValueError(f'snapshot {index}: overlapping splits')
        history.update(new)
        cumulative = np.asarray(sorted(history), dtype=np.int64).reshape(-1, 3)
        stages.append(Stage(index, incoming, cumulative, valid, test,
                            len(entities), len(relations)))
    return stages


def audit_ckge(root: str | Path, *, snapshot_count: int = 5) -> dict:
    """Return stage sizes and source hashes for an auditable dataset copy."""
    stages = load_ckge(root, snapshot_count=snapshot_count)
    stats = []
    old_total = 0
    for stage in stages:
        stats.append({
            'snapshot': stage.index,
            'incoming_train': len(stage.incoming),
            'cumulative_train': len(stage.cumulative),
            'valid': len(stage.valid), 'test': len(stage.test),
            'total_entities': stage.n_entities,
            'new_entities': stage.n_entities - old_total,
            'total_relations': stage.n_relations,
        })
        old_total = stage.n_entities
    root = Path(root)
    hashes = {}
    recognized = {'train_id.txt', 'train.txt', 'entity2id.txt',
                  'relation2id.txt', 'valid.txt', 'test.txt'}
    for index in range(snapshot_count):
        for path in sorted((root / str(index)).glob('*.txt')):
            if path.name not in recognized:
                continue
            digest = sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
    return {'producer': 'FastKGE-style adapter; source identity not independently verified',
            'root': str(root.resolve()), 'snapshot_semantics': 'incremental',
            'stage_statistics': stats, 'file_sha256': hashes}


def save_report(report: dict, path: str | Path) -> None:
    """Store audit JSON without depending on the terminal locale."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                      encoding='utf-8')
