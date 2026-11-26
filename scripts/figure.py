#!/usr/bin/env python3
"""Write the picture of the README: the probe of Inkling-2-Mini against the full gate, and the gates of K 112.

The numbers are the results of the windows of 2026-11-25 (plans/PLAN_PRUNE_MOE_KIT.md §6). The script writes one SVG
file for the light theme and one for the dark theme into docs/img/. It needs only Python: `python3 scripts/figure.py`.
"""

from pathlib import Path

LIMIT = 3.0   # the limit of the gate `nll`, in percent
# (experts kept of 128, GiB freed, probe, low and high end of the 90% band, full gate, full gate passes)
CUTS = [(120, 0.91, 0.72, 0.24, 1.21, 0.69, True),
        (112, 1.82, 2.01, 1.14, 3.29, 1.97, True),
        (104, 2.74, 3.75, 2.42, 5.53, 4.24, False)]
TILES = [("Weights of the model", "−1.82 GiB", "of 18.78 GiB"),          # the gates window of K 112
         ("Tool-call tasks, of 85", "51", "full model: 49"),
         ("Blind judge, 30 pairs", "No loss", "13 better, 6 worse, 11 equal"),
         ("Decode speed", "−0.2%", "37.5 against 37.6 tokens per second")]
THEMES = {
    "light": dict(surface="#fcfcfb", border="#e1e0d9", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", probe="#2a78d6", wash=0.22, full="#eb6834", good="#0ca30c",
                  bad="#d03b3b"),
    "dark": dict(surface="#1a1a19", border="#2c2c2a", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", probe="#3987e5", wash=0.34, full="#d95926", good="#0ca30c",
                 bad="#d03b3b"),
}
W, H = 880, 510
X0, X1, Y0, Y1 = 84, 548, 395, 165   # the plot: 0% is at Y0, 6% is at Y1
XS = [170, 316, 462]


def y(value: float) -> float:
    return round(Y0 - value * (Y0 - Y1) / 6, 1)


def svg(c: dict) -> str:
    out = []
    add = out.append
    text = lambda x, yy, s, size=14, fill=c["ink2"], weight=400, anchor="start": add(
        f'<text x="{x}" y="{yy}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{s}</text>')
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        'font-family="system-ui, -apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">')
    add("<title>A 41-minute probe shows how deep a cut can go</title>")
    add(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="{c["surface"]}" '
        f'stroke="{c["border"]}"/>')
    text(32, 46, "A 41-minute probe shows how deep a cut can go", 23, c["ink"], 650)
    text(32, 72, "Inkling-2-Mini with coding sessions. The loss of three cuts against the full model: lower is better.")
    text(32, 92, "The full gate needs 103 minutes, and each of its results is inside the band of the probe.")
    # The legend: two series, so the legend is always there.
    add(f'<rect x="34" y="113" width="14" height="18" rx="4" fill="{c["probe"]}" fill-opacity="{c["wash"]}"/>')
    add(f'<line x1="30" y1="122" x2="52" y2="122" stroke="{c["probe"]}" stroke-width="2.5" '
        'stroke-linecap="round"/>')
    text(60, 127, "Probe: 20 turns, with the band of 90% of new samples", 13)
    add(f'<circle cx="41" cy="145" r="5.5" fill="{c["full"]}"/>')
    text(60, 150, "Full gate: 60 turns", 13)
    # The grid, the axis and the limit.
    for tick in (0, 2, 4, 6):
        line = c["axis"] if tick == 0 else c["grid"]
        add(f'<line x1="{X0}" y1="{y(tick)}" x2="{X1}" y2="{y(tick)}" stroke="{line}"/>')
        text(X0 - 10, y(tick) + 4.5, f"+{tick}%" if tick else "0%", 12.5, c["muted"], anchor="end")
    add(f'<line x1="{X0}" y1="{y(LIMIT)}" x2="{X1}" y2="{y(LIMIT)}" stroke="{c["ink2"]}" stroke-width="1.5"/>')
    text(X0 + 6, y(LIMIT) - 8, "Limit of the gate: +3% of loss (NLL)", 12.5, c["ink2"], 600)
    for x, (kept, gib, probe, low, high, full, ok) in zip(XS, CUTS):
        add(f'<rect x="{x - 9}" y="{y(high)}" width="18" height="{round(y(low) - y(high), 1)}" rx="4" '
            f'fill="{c["probe"]}" fill-opacity="{c["wash"]}"/>')
        add(f'<line x1="{x - 16}" y1="{y(probe)}" x2="{x + 16}" y2="{y(probe)}" stroke="{c["probe"]}" '
            'stroke-width="2.5" stroke-linecap="round"/>')
        add(f'<circle cx="{x}" cy="{y(full)}" r="5.5" fill="{c["full"]}" stroke="{c["surface"]}" stroke-width="2"/>')
        text(x + 24, y(full) + 5, f"+{full:.2f}%", 14, c["ink"], 650)
        text(x, 419, f"{kept} of 128", 14, c["ink"], 650, "middle")
        text(x, 437, f"frees {gib:.2f} GiB", 12.5, anchor="middle")
        # The result of the full gate: an icon and a word, never the color alone.
        add(f'<circle cx="{x - 22}" cy="452" r="6.5" fill="{c["good"] if ok else c["bad"]}"/>')
        mark = (f"M{x - 25} 452.2l2.2 2.2 3.8-4.4" if ok else f"M{x - 24.5} 449.5l5 5m0-5l-5 5")
        add(f'<path d="{mark}" fill="none" stroke="#ffffff" stroke-width="1.6" stroke-linecap="round" '
            'stroke-linejoin="round"/>')
        text(x - 11, 456.5, "passes" if ok else "fails", 12.5)
    text((X0 + X1) // 2, 481, "Experts that each layer keeps, and the result of the full gate", 12.5, c["muted"],
         anchor="middle")
    # The tiles: the other gates of the cut to 112 experts.
    add(f'<line x1="576" y1="112" x2="576" y2="486" stroke="{c["grid"]}"/>')
    text(600, 127, "112 of 128 experts: each gate passes", 15, c["ink"], 650)
    text(600, 147, "The results of its other gates", 12.5)
    for n, (label, value, note) in enumerate(TILES):
        top = 168 + n * 80
        add(f'<line x1="600" y1="{top}" x2="848" y2="{top}" stroke="{c["grid"]}"/>')
        text(600, top + 22, label, 12.5, c["muted"])
        text(600, top + 50, value, 24, c["ink"], 650)
        text(600, top + 68, note, 12.5)
    add("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent.parent / "docs" / "img"
    folder.mkdir(parents=True, exist_ok=True)
    for name, colors in THEMES.items():
        (folder / f"probe-full-gate-{name}.svg").write_text(svg(colors))
        print(folder / f"probe-full-gate-{name}.svg")
