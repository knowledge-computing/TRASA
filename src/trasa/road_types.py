"""The five road symbol types of the TRASA benchmark (paper Sec. 3).

Graph JSON files store road types by name in ``edge_types``; in-memory graphs
store integer codes in ``Graph.edge_classes`` (-1 = unknown / not annotated).
"""

ROAD_TYPES = ('highway', 'secondary_highway', 'light_duty_road', 'light_duty_road_paved', 'trail')

# Wording used in prompts and figures ("[road type] roads", paper Table 3).
DISPLAY_NAMES = {
    'highway': 'highway',
    'secondary_highway': 'secondary highway',
    'light_duty_road': 'light duty road',
    'light_duty_road_paved': 'paved light duty road',
    'trail': 'trail',
}

# Default label widths in pixels. The paper gives a 2-10 pixel range by symbol;
# the dataset spec (datasets/<name>/dataset.yaml) is the authoritative source.
DEFAULT_WIDTHS = {
    'highway': 10,
    'secondary_highway': 8,
    'light_duty_road': 5,
    'light_duty_road_paved': 6,
    'trail': 2,
}

UNKNOWN = -1


def code(name):
    if name is None:
        return UNKNOWN
    if name not in ROAD_TYPES:
        raise ValueError(f'Unknown road type {name!r}; expected one of {", ".join(ROAD_TYPES)}')
    return ROAD_TYPES.index(name)


def name(value):
    value = int(value)
    if value == UNKNOWN:
        return None
    if not 0 <= value < len(ROAD_TYPES):
        raise ValueError(f'Invalid road type code {value}')
    return ROAD_TYPES[value]


def check_widths(widths):
    """Validate a {road_type: positive integer width} table covering every type."""
    if set(widths) != set(ROAD_TYPES):
        missing = sorted(set(ROAD_TYPES) - set(widths))
        extra = sorted(set(widths) - set(ROAD_TYPES))
        raise ValueError(f'Road width table must list every road type (missing {missing}, unknown {extra})')
    for key, value in widths.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'Road width for {key} must be a positive integer')
    return dict(widths)
