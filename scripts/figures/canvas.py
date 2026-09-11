"""Shared SVG canvas for the documentation figures.

Standard library only: every figure is assembled as text, so generation is
deterministic, needs no plotting stack and can run in CI.  Colours are emitted
as CSS classes with a ``prefers-color-scheme`` override, so one asset stays
legible on light and dark document backgrounds.  Renderers that ignore the
media query fall back to the light palette.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

WIDTH = 900

FONT = '-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif'
MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'

# token -> (light, dark).  Series colours are chosen to stay distinguishable
# when printed in grey and to keep contrast against both grounds.
TOKENS = {
    'bg': ('#fbfcfd', '#0d1117'),
    'card': ('#ffffff', '#161b22'),
    'rule': ('#e3e8ef', '#2f3742'),
    'grid': ('#eef2f7', '#242c36'),
    'ink': ('#1f2937', '#e6edf3'),
    'muted': ('#65748b', '#9aa7b6'),
    'slate': ('#94a3b8', '#8492a3'),
    'blue': ('#3b7dd8', '#6ea8f0'),
    'teal': ('#0e8a7d', '#33b3a4'),
    'amber': ('#c8791d', '#e0a44e'),
    'red': ('#c0524a', '#e0776d'),
    'plum': ('#7c6bd6', '#a394ec'),
}
TINTS = {
    'blue': ('#eff5fd', '#16212f'),
    'teal': ('#ecf8f6', '#10241f'),
    'amber': ('#fdf5ea', '#241a10'),
    'red': ('#fdf0ef', '#241413'),
    'slate': ('#f5f7fa', '#1b212a'),
}

# Type scale: four sizes only, so every figure reads at the same density.
T_TITLE, T_SUB, T_BODY, T_SMALL = 17, 12, 11.5, 10.5


def _style() -> str:
    light, dark = [], []
    for name, (a, b) in TOKENS.items():
        light += [f'.t-{name}{{fill:{a}}}', f'.f-{name}{{fill:{a}}}', f'.s-{name}{{stroke:{a}}}']
        dark += [f'.t-{name}{{fill:{b}}}', f'.f-{name}{{fill:{b}}}', f'.s-{name}{{stroke:{b}}}']
    for name, (a, b) in TINTS.items():
        light.append(f'.tint-{name}{{fill:{a}}}')
        dark.append(f'.tint-{name}{{fill:{b}}}')
    common = ('text{dominant-baseline:auto}'
              '.num{font-variant-numeric:tabular-nums}'
              '.b{font-weight:700}.sb{font-weight:600}'
              f'.mono{{font-family:{MONO}}}')
    return ('<style>' + common + ''.join(light)
            + '@media (prefers-color-scheme:dark){' + ''.join(dark) + '}</style>')


class Canvas:
    """Append-only SVG builder.  Coordinates are plain user units."""

    def __init__(self, title: str, description: str, height: float):
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{num(height)}" '
            f'viewBox="0 0 {WIDTH} {num(height)}" role="img" '
            f'aria-labelledby="fig-title fig-desc" font-family="{FONT}">',
            f'<title id="fig-title">{escape(title)}</title>',
            f'<desc id="fig-desc">{escape(description)}</desc>',
            _style(),
            '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
            'orient="auto-start-reverse"><path d="M0 1 L8 4.5 L0 8" fill="none" '
            'class="s-muted" stroke-width="1.6" stroke-linecap="round" '
            'stroke-linejoin="round"/></marker></defs>',
            f'<rect width="{WIDTH}" height="{num(height)}" class="f-bg"/>',
        ]

    # -- primitives ------------------------------------------------------
    def text(self, x, y, value, size=T_BODY, color='ink', weight='', anchor='start', cls=''):
        classes = ' '.join(filter(None, [f't-{color}', weight, cls]))
        anchor = f' text-anchor="{anchor}"' if anchor != 'start' else ''
        self.parts.append(f'<text x="{num(x)}" y="{num(y)}" font-size="{num(size)}" '
                          f'class="{classes}"{anchor}>{escape(value)}</text>')

    def rect(self, x, y, w, h, fill='card', stroke='rule', radius=0, width=1):
        classes = ' '.join(filter(None, [f'f-{fill}' if fill else '', f's-{stroke}' if stroke else '']))
        self.parts.append(
            f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" '
            f'rx="{num(radius)}" class="{classes}"'
            + (f' stroke-width="{num(width)}"' if stroke else ' stroke="none"')
            + (' fill="none"' if not fill else '') + '/>')

    def tint(self, x, y, w, h, name, radius=8, stroke=True):
        classes = f'tint-{name}' + (f' s-{name}' if stroke else '')
        self.parts.append(
            f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" '
            f'rx="{num(radius)}" class="{classes}"'
            + (' stroke-width="1" stroke-opacity=".35"' if stroke else ' stroke="none"') + '/>')

    def line(self, x1, y1, x2, y2, color='rule', width=1, dash='', cap=''):
        extra = f' stroke-dasharray="{dash}"' if dash else ''
        extra += f' stroke-linecap="{cap}"' if cap else ''
        self.parts.append(f'<path d="M{num(x1)} {num(y1)} L{num(x2)} {num(y2)}" fill="none" '
                          f'class="s-{color}" stroke-width="{num(width)}"{extra}/>')

    def path(self, d, color='muted', width=1.6, arrow=False, dash=''):
        extra = ' marker-end="url(#arrow)"' if arrow else ''
        extra += f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<path d="{d}" fill="none" class="s-{color}" stroke-width="{num(width)}" '
                          f'stroke-linejoin="round"{extra}/>')

    def arrow(self, x1, y1, x2, y2):
        self.path(f'M{num(x1)} {num(y1)} L{num(x2)} {num(y2)}', arrow=True)

    def dot(self, x, y, r, color):
        self.parts.append(f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(r)}" class="f-{color}"/>')

    def square(self, x, y, s, color):
        self.parts.append(f'<rect x="{num(x - s / 2)}" y="{num(y - s / 2)}" width="{num(s)}" '
                          f'height="{num(s)}" rx="1.5" class="f-{color}"/>')

    def bar(self, x, y, w, h, color, radius=3):
        self.parts.append(f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" '
                          f'rx="{num(radius)}" class="f-{color}"/>')

    # -- composites ------------------------------------------------------
    def header(self, title, subtitle='', meta=''):
        """Title block shared by every figure."""
        self.text(40, 30, title, T_TITLE, 'ink', 'b')
        if subtitle:
            self.text(40, 50, subtitle, T_SUB, 'muted')
        if meta:
            self.text(40, 68, meta, T_SMALL, 'muted')

    def footnote(self, y, *lines):
        for i, line in enumerate(lines):
            self.text(40, y + i * 16, line, T_SMALL, 'muted')

    def card(self, x, y, w, h, title, lines, color='blue', title_size=None):
        """Bordered card with a coloured left edge and a bold heading."""
        self.rect(x, y, w, h, 'card', 'rule', radius=8)
        self.parts.append(f'<path d="M{num(x + 1)} {num(y + 8)} v{num(h - 16)}" class="s-{color}" '
                          f'stroke-width="3" stroke-linecap="round"/>')
        self.text(x + 16, y + 22, title, title_size or 12.5, color, 'b')
        for i, line in enumerate(lines):
            self.text(x + 16, y + 41 + i * 16, line, T_BODY, 'ink')

    def legend(self, x, y, items):
        """items: (marker, colour, label) with marker in {'dot', 'square', 'line'}."""
        for marker, color, label in items:
            if marker == 'dot':
                self.dot(x + 5, y - 4, 4.5, color)
            elif marker == 'square':
                self.square(x + 5, y - 4, 8, color)
            else:
                self.line(x, y - 4, x + 12, y - 4, color, 3, cap='round')
            self.text(x + 18, y, label, T_SMALL, 'muted')
            x += 18 + len(label) * 5.6 + 22

    def finish(self) -> str:
        return '\n'.join(self.parts + ['</svg>', ''])


def num(value) -> str:
    """Compact, locale-free, deterministic number formatting."""
    text = f'{float(value):.3f}'.rstrip('0').rstrip('.')
    return text if text not in ('', '-0') else '0'
