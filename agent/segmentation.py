import json
import re
import sys
from typing import Any, Dict, List


def segment_contract(contract_text: str) -> List[Dict[str, Any]]:
    text = (contract_text or '').strip()
    if not text:
        return []

    pattern = re.compile(
        r"(?:^|\n\s*)(?:SECTION|CLAUSE|CLÁUSULA|CLAUSULA|ARTICLE|\d+\.?\s+[A-ZÁÉÍÓÚ][A-ZÁÉÍÓÚ0-9\s&/\-]*)",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))

    if not matches:
        return [{
            'title': 'Contract Body',
            'text': text,
            'index': 0,
        }]

    clauses: List[Dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        raw = text[start:end].strip()
        if not raw:
            continue

        title = re.split(r'\n|\r', raw)[0].strip()
        if not title:
            title = f'Clause {idx + 1}'
        if len(title) > 160:
            title = title[:157].rstrip() + '...'

        clauses.append({
            'index': idx,
            'title': title,
            'text': raw,
        })

    if not clauses:
        return [{
            'title': 'Contract Body',
            'text': text,
            'index': 0,
        }]

    return clauses


if __name__ == '__main__':
    payload = json.loads(sys.stdin.read() or '{}')
    text_value = payload.get('contractText') or payload.get('text') or ''
    print(json.dumps(segment_contract(text_value), ensure_ascii=False))
