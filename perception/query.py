"""Turn a text query into {target, relation, reference}. Labels are normalised to the detector vocabulary."""

import re

SYNONYMS = {
    "tv": "tv_monitor",
    "television": "tv_monitor",
    "monitor": "tv_monitor",
    "tv monitor": "tv_monitor",
    "fridge": "refrigerator",
    "couch": "sofa",
}

_LABEL = r"[a-z_]+(?: [a-z_]+)?"  # one word, or two for 'tv monitor'; longer phrases are not labels
_QUERY = re.compile(
    rf"^\s*(?:the\s+)?(?P<target>{_LABEL}?)"
    rf"(?:\s+(?P<relation>nearest)\s+(?:the\s+)?(?P<reference>{_LABEL}?))?\s*$"
)


def normalise_label(text, vocabulary=None):
    label = SYNONYMS.get(text.strip().lower(), text.strip().lower()).replace(" ", "_")
    if vocabulary is not None and label not in vocabulary:
        raise ValueError(f"unknown label {text!r} (normalised {label!r}); not in detector vocabulary")
    return label


def parse_query(text, vocabulary=None):
    """'the chair nearest the stove' -> {'target': 'chair', 'relation': 'nearest', 'reference': 'stove'}.
    Unqualified queries give relation = reference = None. 'nearest' is the only relation in queries.json;
    anything else raises instead of being silently misread."""
    m = _QUERY.match(text.lower())
    if not m:
        raise ValueError(f"cannot parse query {text!r}: expected 'the X' or 'the X nearest the Y'")
    return {
        "target": normalise_label(m["target"], vocabulary),
        "relation": m["relation"],
        "reference": normalise_label(m["reference"], vocabulary) if m["reference"] else None,
    }
