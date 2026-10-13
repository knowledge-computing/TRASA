import hashlib
import json
from pathlib import Path


def test_pinned_upstream_snapshots_are_complete_and_unmodified():
    root = Path(__file__).resolve().parents[1]
    files = json.loads((root / 'docs/upstream-files.json').read_text())
    assert len(files) >= 200
    for record in files:
        assert hashlib.sha256((root / record['path']).read_bytes()).hexdigest() == record['sha256']
    for relative in ('third_party/ldtr/train.py', 'third_party/ldtr/losses_map.py',
                     'third_party/ldtr/models/ops/setup.py', 'third_party/sam_road/train.py',
                     'third_party/sam_road/dataset.py', 'third_party/sam_road/model.py',
                     'third_party/sam_road/sam/segment_anything/build_sam.py'):
        assert (root / relative).is_file()
