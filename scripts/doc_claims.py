"""Check named numeric references at their document locations.

Markers surround only the displayed quantity; prose and document order are free.
Every location is required once, so deleting a marker cannot disable its check.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import statistics

from g2.evidence import sig3


@dataclass(frozen=True)
class Quantity:
    value: str
    unit_en: str
    unit_zh: str
    source: str

    def display(self, language):
        unit = self.unit_zh if language == 'zh' else self.unit_en
        return self.value + (' ' + unit if unit else '')


# Location names identify individual references, including repeated quantities.
README_LOCATIONS = {
    'speed-card': 'g2.ane-share',
    'calls-card': 'g2.native-calls',
    'memory-card': 'g2.native-shrink',
    'temperature-card': 'g2.temperature-gap',
    'arithmetic-card': 'arithmetic.q8-summary',
    'ane-rate': 'g2.ane-rate',
    'gpu-rate': 'g2.gpu-rate',
    'service-share': 'g2.service-share',
    'equal-rate': 'g2.equal-rate-max',
    'fan-idle': 'g2.fan-idle',
    'temperature-detail': 'g2.temperature-gap',
    'ane-saturated': 'g2.ane-saturated',
    'gpu-saturated': 'g2.gpu-saturated',
    'fan-saturated': 'g2.fan-saturated',
}
DOCUMENTS = {'README.md': 'en', 'README.zh-CN.md': 'zh'}


def quantities(root, data):
    """Resolve formatted quantities from verified G2 rows and arithmetic records."""
    point = next(row for row in data['medians'] if row['positions'] == 1024)
    native = [row for row in data['native_growth'] if row['engine'] == 'C']
    shrink = sorted(abs(row['footprint_change_bytes']) / 2**20 for row in native)
    sat = {engine: [slot for name, slot in data['slots'].items()
                    if name.startswith('SAT-') and slot['config']['engine'] == engine]
           for engine in 'CG'}
    rate = lambda slot: slot['service']['inference']['observation_throughput_per_second']
    equal_rates = [rate(slot) for name, slot in data['slots'].items() if name.startswith('P3-')]
    temperature = [row['gpu_mean_G_minus_C'] for row in data['expected']['P3_pairs']]
    fans = [slot['tail_medians']['fan0_rpm'] for slot in sat['G']]
    arithmetic = json.loads((Path(root) / 'results/historical/cross-model-validation.json').read_text())
    models = arithmetic['models'].values()
    q8 = f"{sum(row['q8_numeric_mismatches'] for row in models):,} / {sum(row['real_values'] for row in models):,}"
    bundle = 'results/historical/g2-w4a16-night/'
    return {
        'g2.ane-share': Quantity(f"{100 * point['C_over_G']:.1f}%", '', '',
                                bundle + 'p2.jsonl.gz: N1024 median C rate / median G rate'),
        'g2.native-calls': Quantity(f"{native[0]['stage_calls']:,}", 'calls', '次调用',
                                   bundle + 'p0.json: native host.stage_calls'),
        'g2.native-shrink': Quantity(f'{shrink[0]:.0f}–{shrink[-1]:.0f}', 'MiB', 'MiB',
                                    bundle + 'p0.json: native checkpoint footprint decrease'),
        'g2.temperature-gap': Quantity(f'{min(temperature):.1f}–{max(temperature):.1f}', '°C', '°C',
                                      bundle + 'expected-summary.json: P3_pairs.gpu_mean_G_minus_C'),
        'arithmetic.q8-summary': Quantity(q8, '', '',
                                        'results/historical/cross-model-validation.json: '
                                        'sum(models.*.q8_numeric_mismatches) / sum(models.*.real_values), Q8 outputs'),
        'g2.ane-rate': Quantity(sig3(point['C']['positions_per_second']), 'positions/s', '位置/s',
                               bundle + 'p2.jsonl.gz: N1024 median C positions_per_second'),
        'g2.gpu-rate': Quantity(sig3(point['G']['positions_per_second']), 'positions/s', '位置/s',
                               bundle + 'p2.jsonl.gz: N1024 median G positions_per_second'),
        'g2.service-share': Quantity(f"{100 * data['saturated_ratio']:.1f}%", '', '',
                                    bundle + 'slots.json: SAT mean C rate / mean G rate'),
        'g2.equal-rate-max': Quantity(f'{max(equal_rates):.1f}', 'requests/s', '请求/s',
                                     bundle + 'slots.json: maximum P3 observation throughput'),
        'g2.fan-idle': Quantity(f"{round(data['fan_floor']['fan0_rpm'], -1):,.0f}", 'RPM', 'RPM',
                               bundle + 'thermal-starts.json: median block-start medians.fan0_rpm'),
        'g2.ane-saturated': Quantity(f"{statistics.mean(map(rate, sat['C'])):.1f}", 'requests/s', '请求/s',
                                    bundle + 'slots.json: mean SAT C observation throughput'),
        'g2.gpu-saturated': Quantity(f"{statistics.mean(map(rate, sat['G'])):.1f}", 'requests/s', '请求/s',
                                    bundle + 'slots.json: mean SAT G observation throughput'),
        'g2.fan-saturated': Quantity(f'{round(min(fans), -2):,.0f}–{round(max(fans), -2):,.0f}', 'RPM', 'RPM',
                                    bundle + 'sensors.jsonl.gz: SAT G tail median fan0_rpm range'),
    }


TOKEN = re.compile(r'<!--\s*(?:claim:([a-z0-9.-]+)@([a-z0-9.-]+)|(/claim))\s*-->')


def check_document(body, catalog, language, required=README_LOCATIONS, name='document'):
    problems, seen = [], set()
    opening = None
    for token in TOKEN.finditer(body):
        if token.group(3):
            if opening is None:
                problems.append(f'{name}: orphan /claim marker')
                continue
            claim, location, start, line = opening
            opening = None
            identity = f'{claim}@{location}'
            if location in seen:
                problems.append(f'{name}:{line}: duplicate claim location {location}')
            seen.add(location)
            if required.get(location) != claim or claim not in catalog:
                problems.append(f'{name}:{line}: unknown or changed claim {identity}')
                continue
            expected = catalog[claim].display(language)
            actual = ' '.join(body[start:token.start()].split())
            if actual != expected:
                problems.append(f'{name}:{line}: {identity}: expected {expected!r}, got {actual!r}; '
                                f'source: {catalog[claim].source}')
        else:
            if opening is not None:
                problems.append(f'{name}: nested claim marker')
            opening = (token.group(1), token.group(2), token.end(), body.count('\n', 0, token.start()) + 1)
    if opening is not None:
        problems.append(f'{name}: unclosed claim {opening[0]}@{opening[1]}')
    if re.search(r'<!--\s*/?claim\b', TOKEN.sub('', body)):
        problems.append(f'{name}: malformed claim marker')
    for location in sorted(required.keys() - seen):
        problems.append(f'{name}: missing claim {required[location]}@{location}')
    return problems


def check_local_quotes(root, data):
    catalog = quantities(root, data)
    problems = []
    for name, language in DOCUMENTS.items():
        problems.extend(check_document((Path(root) / name).read_text(), catalog, language, name=name))
    return problems


def contains_number(body, value):
    """Require a complete numeric token; 13 must not match 113 or 13.50."""
    body = re.sub(r'<!--.*?-->', '', body, flags=re.DOTALL).replace('\u2212', '-')
    value = value.replace('\u2212', '-')
    return re.search(r'(?<![\dA-Za-z_.,])' + re.escape(value) +
                     r'(?![\dA-Za-z_%]|[.,]\d)', body) is not None
