"""Validate each registered G1-W/G5 numeric occurrence with the shared marker checker."""
import json
from pathlib import Path
import re

from doc_claims import check_document


def check(root, family, catalog):
    root = Path(root)
    documents = json.loads((root / 'scripts' / family / 'claims.json').read_text())
    problems = []
    for name, required in documents.items():
        body = (root / name).read_text()
        body = re.sub(r'<!-- claim:(?!' + re.escape(family) + r'\.)[^>]+-->.*?<!-- /claim -->',
                      '', body, flags=re.DOTALL)
        language = 'zh' if name.endswith('zh-CN.md') or '/zh/' in name else 'en'
        problems.extend(check_document(body, catalog, language, required=required, name=name))
    return problems
