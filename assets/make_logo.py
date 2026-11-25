"""Regenerate the Laya icon used by the documentation.

The mark is one ring in two states. Half of it is an unbroken stroke; the other half breaks into
points that shrink and fade, and the smallest point sits next to where the stroke begins again --
so the eye closes the loop on its own.

`laya` (लय) is Sanskrit for dissolution. The mark reads as that cycle: becoming and dissolving,
neither one the end. It carries a second meaning specific to this project -- a continuous stroke
resolving into discrete points is what the model does, turning unstructured state into a typed
decision.

Run python assets/make_logo.py to write the SVG.
"""
import math
from pathlib import Path

ASSETS = Path(__file__).resolve().parent
BLUE = "#2a78d6"

SIZE = 64          # viewBox
CX = CY = 32.0
R = 21.0
STROKE = 5.0

# The stroke covers this sweep; the points cover the rest.
ARC_FROM, ARC_TO = 118.0, 300.0      # degrees, measured counter-clockwise from +x
N_DOTS = 10
DOT_MAX, DOT_MIN = 3.1, 0.85
# The ramp is the mark's whole idea: the points fade as they spiral in, so the eye closes the
# loop. It has to START at full ink, though -- the first point reads as the stroke continuing,
# and at 0.95 the artwork looked washed out on both themes rather than dissolving. The faint
# end is the design; a faint beginning was not.
OP_MAX, OP_MIN = 1.0, 0.42


def pt(deg, r=R):
    a = math.radians(deg)
    return CX + r * math.cos(a), CY - r * math.sin(a)      # SVG y grows downward


def arc_path():
    x0, y0 = pt(ARC_FROM)
    x1, y1 = pt(ARC_TO)
    sweep = (ARC_TO - ARC_FROM) % 360
    large = 1 if sweep > 180 else 0
    # sweep-flag 0 draws counter-clockwise in SVG's y-down space
    return "M %.3f %.3f A %.1f %.1f 0 %d 0 %.3f %.3f" % (x0, y0, R, R, large, x1, y1)


def dots(color):
    """Points leaving the stroke and spiralling inward, shrinking toward a single point.

    Inward rather than around: `laya` is absorption, and a ring that dissolves into its own
    centre says that where a ring of even dots would just read as a loading spinner. It is also
    what the model does -- a spread of options collapsing to one decision.
    """
    span = 268.0                     # how far the spiral travels before reaching the centre
    out = []
    for i in range(N_DOTS):
        f = (i + 1) / float(N_DOTS)
        deg = ARC_TO + span * f
        r = R * (1.0 - 0.93 * (f ** 1.08))          # ease inward, ending near the centre
        x, y = pt(deg, r)
        rr = DOT_MAX + (DOT_MIN - DOT_MAX) * (f ** 0.85)
        op = OP_MAX + (OP_MIN - OP_MAX) * (f ** 1.4)
        out.append('    <circle cx="%.3f" cy="%.3f" r="%.3f" fill="%s" opacity="%.3f"/>'
                   % (x, y, rr, color, op))
    out.append('    <circle cx="%.2f" cy="%.2f" r="2.35" fill="%s"/>' % (CX, CY, color))
    return "\n".join(out)


def mark_svg(color):
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d"
     width="%d" height="%d" role="img" aria-label="Laya">
  <title>Laya</title>
  <desc>An open stroke whose end breaks into points that spiral inward and shrink to a single
  point: becoming, then dissolving into one.</desc>
  <g fill="none" stroke="none">
    <path d="%s" stroke="%s" stroke-width="%.1f" stroke-linecap="round"/>
%s
  </g>
</svg>
''' % (SIZE, SIZE, SIZE, SIZE, arc_path(), color, STROKE, dots(color))


def main():
    path = ASSETS / "logo-mark.svg"
    path.write_text(mark_svg(BLUE), encoding="utf-8")
    print(path.name)


if __name__ == "__main__":
    main()
