"""echostream logo system generator.
All type is converted to outlines so every SVG is font-independent
(imports into Figma as editable vectors).
"""
import math, os
from fontTools.ttLib import TTFont, TTCollection
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.boundsPen import BoundsPen

OUT = "/mnt/user-data/outputs/echostream-logo-final-v3"
os.makedirs(OUT + "/svg", exist_ok=True)

# ---------- tokens ----------
INDIGO = "#1E2A78"
CYAN = "#00C2FF"
VIOLET = "#7B61FF"
CORAL = "#FF5C7A"
NIGHT = "#0B1020"
WHITE = "#FFFFFF"
BLACK = "#000000"
AZURE = "#4597FF"     # OKLCH midpoint cyan -> violet
ORCHID = "#D154CD"    # OKLCH midpoint violet -> coral
MUTED_D = "#A9B3E0"   # secondary text on dark
DIM_D = "#7E88B8"     # tertiary text on dark

FD = "/usr/share/fonts/opentype/inter/"
F_ECHO = TTFont(FD + "InterDisplay-Bold.otf")
F_STREAM = TTFont(FD + "InterDisplay-Light.otf")
F_SEMI = TTFont(FD + "InterDisplay-SemiBold.otf")
F_REG = TTFont(FD + "Inter-Regular.otf")
_ttc = TTCollection("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
F_CJK = next(f for f in _ttc.fonts if "SC" in f["name"].getDebugName(1))
_ttc2 = TTCollection("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
F_CJK_R = next(f for f in _ttc2.fonts if "SC" in f["name"].getDebugName(1))


def P(a, r, cx, cy):
    t = math.radians(a)
    return cx + r * math.cos(t), cy + r * math.sin(t)


def arc(cx, cy, r, a0, a1):
    """SVG arc path, clockwise (screen) from a0 to a1 degrees."""
    x0, y0 = P(a0, r, cx, cy)
    x1, y1 = P(a1, r, cx, cy)
    large = 1 if (a1 - a0) % 360 > 180 else 0
    return f"M{x0:.2f} {y0:.2f}A{r} {r} 0 {large} 1 {x1:.2f} {y1:.2f}"


# ---------- text to outlines ----------
def text_path(font, s, size, x, y, track=0.0):
    """Return (d, width, (xmin,ymin,xmax,ymax)) for string s, baseline at y."""
    gs = font.getGlyphSet()
    cmap = font.getBestCmap()
    upm = font["head"].unitsPerEm
    sc = size / upm
    pen = SVGPathPen(gs)
    cur = 0.0
    bx = [1e9, 1e9, -1e9, -1e9]
    for ch in s:
        gn = cmap.get(ord(ch))
        if gn is None:
            continue
        g = gs[gn]
        tp = TransformPen(pen, (sc, 0, 0, -sc, x + cur, y))
        g.draw(tp)
        bp = BoundsPen(gs)
        g.draw(TransformPen(bp, (sc, 0, 0, -sc, x + cur, y)))
        if bp.bounds:
            b = bp.bounds
            bx = [min(bx[0], b[0]), min(bx[1], b[1]), max(bx[2], b[2]), max(bx[3], b[3])]
        cur += g.width * sc + track * size
    width = cur - track * size
    return pen.getCommands(), width, tuple(bx)


# ---------- the mark (512 artboard) ----------
# "Funnel": the request path drawn as dots.
# row 1  4 small dots  = four recall channels (two-tower, hot, ItemCF, cold-start)
# row 2  3 dots        = LightGBM ranking
# row 3  2 dots        = MMR re-ranking
# row 4  1 coral dot   = the next screen the user sees
# Fewer survivors per stage, each one heavier: hundreds of candidates in, one answer out.


PHI = (1 + 5 ** 0.5) / 2


GROW = PHI ** (2 / 3)   # smallest -> largest radius over all rows
GAP = PHI ** -0.5      # neighbor gap, in units of u


def golden(target, rows=4, cx=256, cy=256, optical=0.5):
    """Golden-ratio funnel, laid out in units of u (the smallest radius).
    radius of row k  = u * sqrt(phi)^(k/(rows-1))  largest / smallest = sqrt(phi),
                       so the coral dot's AREA is exactly phi x a top-row dot
    every pair of touching-neighbor circles (in-row and diagonal) is separated by
    exactly u, so the optical density is even across the whole mark;
    pitch is constant, so each dot sits exactly midway below the two above it.
    The shape is scaled so its larger side = target, then nudged down so its
    area-weighted center sits halfway between the bbox center and the canvas center."""
    r = [GROW ** (k / (rows - 1)) for k in range(rows)]
    g = GAP
    p = 2 * r[-2] + g
    ys = [0.0]
    for a, b in zip(r, r[1:]):
        ys.append(ys[-1] + math.sqrt((a + b + g) ** 2 - (p / 2) ** 2))
    pts = []
    for i, rr in enumerate(r):
        n = rows - i
        for k in range(n):
            pts.append(((k - (n - 1) / 2) * p, ys[i], rr, i))
    x0 = min(x - q for x, y, q, i in pts); x1 = max(x + q for x, y, q, i in pts)
    y0 = min(y - q for x, y, q, i in pts); y1 = max(y + q for x, y, q, i in pts)
    u = target / max(x1 - x0, y1 - y0)
    w = lambda i: (0.25 if (i == 0 and rows >= 5) else 1.0)   # faded pool row weighs less
    A = sum(q * q * w(i) for x, y, q, i in pts)
    cyc = sum(y * q * q * w(i) for x, y, q, i in pts) / A
    bc = (y0 + y1) / 2
    shift = (bc - cyc) * optical
    return [(cx + x * u, cy + (y - bc + shift) * u, q * u, i) for x, y, q, i in pts]


ROWS = 5
DOTS = golden(400, rows=ROWS)
# top row = the whole video pool, held back at 1/phi^2 opacity
FADE = (1 / PHI ** 2, 1, 1, 1, 1)
ART_BBOX = (min(x - r for x, y, r, i in DOTS), min(y - (0.3 * r if i == 0 else r) for x, y, r, i in DOTS),
            max(x + r for x, y, r, i in DOTS), max(y + r for x, y, r, i in DOTS))


def mark_defs(uid):
    return ""


def dots_svg(dots, cols, ops=None):
    out = ""
    for x, y, r, i in dots:
        op = "" if not ops or ops[i] >= 1 else f' fill-opacity="{ops[i]:.3f}"'
        out += f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{cols[i]}"{op}/>'
    return out


def mark_body(cols, uid="", grad=True):
    """Top row = the video pool. Screen versions fade each of its circles from
    0 at the top edge to 1/phi^2 at the bottom edge, so the pool dissolves upward.
    Print/mono versions use a flat 1/phi^2 tint instead (no gradients)."""
    if not grad:
        return dots_svg(DOTS, cols, FADE)
    x0, y0, r0, _ = next(d for d in DOTS if d[3] == 0)
    gid = f"pool{uid}"
    out = (f'<defs><linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="0" y1="{y0-r0:.2f}" '
           f'x2="0" y2="{y0+r0:.2f}"><stop offset="0" stop-color="{cols[0]}" stop-opacity="0"/>'
           f'<stop offset="1" stop-color="{cols[0]}" stop-opacity="{1/PHI**2:.3f}"/></linearGradient></defs>')
    for x, y, r, i in DOTS:
        fill = f'url(#{gid})' if i == 0 else cols[i]
        out += f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}"/>'
    return out


VARIANTS = {
    # one continuous stream: the brand gradient cyan -> violet, carried on to coral,
    # in equal OKLCH steps. Works unchanged on light and dark backgrounds.
    "color":   (CYAN, AZURE, VIOLET, ORCHID, CORAL),
    "on-dark": (CYAN, AZURE, VIOLET, ORCHID, CORAL),
    "black":   (BLACK,) * 5,
    "white":   (WHITE,) * 5,
}


def mark_g(variant, tx=0, ty=0, s=1.0, uid=""):
    return (f'<g transform="translate({tx:.2f} {ty:.2f}) scale({s:.5f})">'
            + mark_body(VARIANTS[variant], uid, grad=variant in ("color", "on-dark")) + "</g>")


def svg(w, h, body, defs="", bg=None, rx=0):
    bgr = f'<rect width="{w}" height="{h}" rx="{rx}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}"><defs>{defs}</defs>{bgr}{body}</svg>')


files = {}

# ---------- 1. icon 512 ----------
for name in VARIANTS:
    files[f"icon-512-{name}.svg"] = svg(512, 512, mark_g(name, uid="i"), mark_defs("i"))
# app tile (dark rounded square) – handy for GitHub org avatar / app icon
tile_s = 0.84
off = 256 * (1 - tile_s)
files["icon-512-tile-dark.svg"] = svg(
    512, 512, mark_g("on-dark", off, off + 4, tile_s, "t"), mark_defs("t"), bg=NIGHT, rx=112)


# ---------- 2. favicons (3-2-1 rows so dots stay >= 1px) ----------
FAV = {32: dict(target=27, rows=3, cx=16, cy=16),
       16: dict(target=13.6, rows=3, cx=8, cy=8)}
for v, cols in (("color", (CYAN, VIOLET, CORAL)), ("on-dark", (CYAN, VIOLET, CORAL))):
    for size, kw in FAV.items():
        files[f"favicon-{size}-{v}.svg"] = svg(size, size, dots_svg(golden(**kw), cols))


# ---------- 3. wordmark helpers ----------
def wordmark(x, y, fs, col_echo, col_stream):
    d1, w1, b1 = text_path(F_ECHO, "echo", fs, x, y, track=-0.012)
    d2, w2, b2 = text_path(F_STREAM, "stream", fs, x + w1 + fs * 0.01, y, track=-0.004)
    body = f'<path d="{d1}" fill="{col_echo}"/><path d="{d2}" fill="{col_stream}"/>'
    bbox = (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))
    return body, bbox


ax0, ay0, ax1, ay1 = ART_BBOX
art_w, art_h = ax1 - ax0, ay1 - ay0


def place_mark(variant, left, top, art_height, uid):
    s = art_height / art_h
    return mark_g(variant, left - ax0 * s, top - ay0 * s, s, uid), art_w * s


# horizontal lockup
def lockup_h(variant, text_col):
    PAD = 40
    H = 160                      # artwork height
    xh_em = F_ECHO["OS/2"].sxHeight / F_ECHO["head"].unitsPerEm
    fs = (H / PHI ** 2) / xh_em              # x-height = H / phi^2
    _, bb = wordmark(0, 0, fs, text_col, text_col)
    asc = -bb[1]
    mk, mw = place_mark(variant, PAD, PAD, H, "h")
    base = PAD + H / 2 + asc / 2             # optically centered on the mark
    tx = PAD + mw + H / PHI ** 2             # gap = H / phi^2
    wm, wb = wordmark(tx, base, fs, text_col, text_col)
    W = math.ceil(wb[2] + PAD)
    Hh = math.ceil(max(PAD + H, wb[3]) + PAD)
    return svg(W, Hh, mk + wm, mark_defs("h"))


files["lockup-horizontal-color.svg"] = lockup_h("color", INDIGO)
files["lockup-horizontal-on-dark.svg"] = lockup_h("on-dark", WHITE)
files["lockup-horizontal-black.svg"] = lockup_h("black", BLACK)
files["lockup-horizontal-white.svg"] = lockup_h("white", WHITE)


# bilingual lockup: mark + echostream over 回响流转 (justified to the same width)
def lockup_cn(variant, col_en, col_cn):
    PAD = 40
    fs_en = 112
    fs_cn = fs_en / PHI                      # EN : CN = phi : 1
    _, bb_en = wordmark(0, 0, fs_en, col_en, col_en)
    en_asc = -bb_en[1]                       # height above baseline
    # CN: left-aligned with the wordmark, gentle tracking
    track = 0.14
    _, cw, cb = text_path(F_CJK, "回响流转", fs_cn, 0, 0, track=track)
    gap = fs_cn / PHI ** 2
    cn_top_from_base = -cb[1]
    block_h = en_asc + gap + cn_top_from_base + cb[3]
    H = block_h                              # artwork height == text block
    mk, mw = place_mark(variant, PAD, PAD, H, "c")
    tx = PAD + mw + H / PHI ** 2
    base_en = PAD + en_asc
    wm, wb = wordmark(tx - bb_en[0], base_en, fs_en, col_en, col_en)
    base_cn = base_en + gap + cn_top_from_base
    dcn, _, cbb = text_path(F_CJK, "回响流转", fs_cn, tx - cb[0], base_cn, track=track)
    W = math.ceil(max(wb[2], cbb[2]) + PAD)
    Hh = math.ceil(PAD * 2 + H)
    return svg(W, Hh, mk + wm + f'<path d="{dcn}" fill="{col_cn}"/>', mark_defs("c"))


files["lockup-bilingual-color.svg"] = lockup_cn("color", INDIGO, INDIGO)
files["lockup-bilingual-on-dark.svg"] = lockup_cn("on-dark", WHITE, MUTED_D)
files["lockup-bilingual-black.svg"] = lockup_cn("black", BLACK, BLACK)
files["lockup-bilingual-white.svg"] = lockup_cn("white", WHITE, WHITE)


# ---------- 4. GitHub social preview 1280x640 ----------
def social():
    W, H = 1280, 640
    L = 104
    defs = mark_defs("s")
    # candidate field: many tiny dots narrowing into one coral dot
    bg = f'<rect width="{W}" height="{H}" fill="{NIGHT}"/>'
    rings = ""
    apex = (1060, 520)
    rows = 11
    for j in range(rows):                    # j=0 is the coral dot
        n = j + 1
        R0 = 50
        rad = lambda t: R0 * PHI ** (-t / 2.5)
        r = rad(j)
        sp = 2 * R0 * PHI ** (-j / 8) * 1.18
        y = apex[1] - sum(rad(t) + rad(t + 1) + rad(t + 1) / PHI for t in range(j))
        ramp_ = [CORAL, ORCHID, VIOLET, AZURE, CYAN]
        col = ramp_[min(j, 4)]
        op = 1 if j <= 4 else max(0.14, PHI ** (-(j - 4) * 0.8))
        for k in range(n):
            x = apex[0] + (k - (n - 1) / 2) * sp * 1.25
            rings += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{col}" opacity="{op:.2f}"/>'

    # lockup (on dark)
    Hm = 112
    xh_em = F_ECHO["OS/2"].sxHeight / F_ECHO["head"].unitsPerEm
    fs = (Hm / PHI ** 2) / xh_em
    _, bb = wordmark(0, 0, fs, WHITE, WHITE)
    mk, mw = place_mark("on-dark", L, 120, Hm, "s")
    base = 120 + Hm / 2 - bb[1] / 2
    wm, _ = wordmark(L + mw + Hm / PHI ** 2, base, fs, WHITE, WHITE)

    # slogan + one-liner
    y = 360
    # type scale in phi steps: 60 -> 37 -> 23 (-> 18 = 23/sqrt(phi))
    d1, _, _ = text_path(F_SEMI, "Every swipe echoes.", 60, L, y, track=-0.01)
    d2, _, _ = text_path(F_CJK, "每一次划动，都有回响。", 37, L, y + 60, track=0.04)
    d3, _, _ = text_path(F_CJK_R, "基于实时行为反馈的分布式短视频推荐系统", 23, L, 522, track=0.03)
    d4, _, _ = text_path(F_REG, "A distributed short-video recommender driven by real-time behavioral feedback",
                         18, L, 556, track=0.0)
    rule = f'<circle cx="{L+6}" cy="476" r="6" fill="{CORAL}"/>'
    text = (f'<path d="{d1}" fill="{WHITE}"/><path d="{d2}" fill="{MUTED_D}"/>{rule}'
            f'<path d="{d3}" fill="{MUTED_D}"/><path d="{d4}" fill="{DIM_D}"/>')
    return svg(W, H, bg + rings + mk + wm + text, defs)


files["github-social-preview-1280x640.svg"] = social()


# ---------- 5. meaning + construction boards ----------
def meaning():
    W, H = 1240, 700
    body = f'<rect width="{W}" height="{H}" fill="#F4F6FB"/>'
    K, OX, OY = 1.3, 20, 17
    body += f'<g transform="translate({OX} {OY}) scale({K})">{mark_body(VARIANTS["color"])}</g>'
    rows = {}
    for x, y, r, i in DOTS:
        rows.setdefault(i, []).append((x * K + OX, y * K + OY, r * K))
    labels = [("视频池", "Pool: every video in KuaiRec"),
              ("召回 · 4 路", "Recall: two-tower, hot, ItemCF, cold-start"),
              ("排序", "Rank: LightGBM scores hundreds of candidates"),
              ("重排", "Re-rank: MMR keeps the screen diverse"),
              ("下一屏", "Serve: the next video, in tens of ms")]
    for i, (zh, en) in enumerate(labels):
        xr = max(x + r for x, y, r in rows[i]) + 40
        y = rows[i][0][1]
        body += f'<rect x="{xr:.0f}" y="{y-1:.0f}" width="{680-xr:.0f}" height="2" fill="#CDD3E6"/>'
        dz, _, _ = text_path(F_CJK, zh, 30, 700, y + 4)
        de, _, _ = text_path(F_REG, en, 20, 700, y + 36)
        body += f'<path d="{dz}" fill="{INDIGO}"/><path d="{de}" fill="#5B6390"/>'
    return svg(W, H, body)


def construction():
    W, H = 1320, 760
    K, OX, OY = 1.4, 10, 22
    G = "#C3CAE0"
    body = f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>'
    rows = {}
    for x, y, r, i in DOTS:
        rows.setdefault(i, []).append((x * K + OX, y * K + OY, r * K))
    x0 = min(x - r for v in rows.values() for x, y, r in v) - 30
    x1 = max(x + r for v in rows.values() for x, y, r in v) + 30
    body += f'<g transform="translate({OX} {OY}) scale({K})">{mark_body(VARIANTS["color"])}</g>'
    # every neighboring gap = u: mark each one with a short coral bar
    allc = [(x, y, r) for v in rows.values() for x, y, r in v]
    for i, (xa, ya, ra) in enumerate(allc):
        for xb, yb, rb in allc[i + 1:]:
            d = math.hypot(xb - xa, yb - ya)
            umin = min(q for _, _, q in allc)
            if abs(d - ra - rb - umin * GAP) < umin * 0.02:
                ux, uy = (xb - xa) / d, (yb - ya) / d
                p0 = (xa + ux * ra, ya + uy * ra); p1 = (xb - ux * rb, yb - uy * rb)
                body += (f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" '
                         f'stroke="{CORAL}" stroke-width="3" stroke-linecap="round"/>')
    # column guides through centers (lattice alignment)
    pitch = (rows[0][1][0] - rows[0][0][0]) / 2
    cx0 = rows[0][0][0]
    for k in range(7):
        xx = cx0 + k * pitch
        body += f'<line x1="{xx:.1f}" y1="{OY+40}" x2="{xx:.1f}" y2="{H-60}" stroke="{G}" stroke-width="1" stroke-dasharray="4 6"/>'
    # outline the circles so the geometry reads
    for v in rows.values():
        for x, y, r in v:
            body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="none" stroke="#FFFFFF" stroke-opacity=".0"/>'
    tx = 760
    lines = [
        ("黄金分割构成", "Golden-ratio construction", 34, 26),
        ("半径逐行 × φ 的六次方根", "Radius × φ^(1/6) per row: u, 1.08u, 1.17u, 1.27u, 1.38u", 26, 20),
        ("顶层由 0 渐显至 1/φ²", "Pool row fades in from 0 to 1/φ²; mono/print uses a flat 1/φ² tint", 26, 20),
        ("下一屏 : 召回点 面积恰好 φ", "Coral dot area : recall-row dot area = φ : 1", 26, 20),
        ("相邻圆间隙都等于 u/√φ", "Every neighboring gap (coral bars) = u/√φ, u = smallest radius", 26, 20),
        ("每个点居中于上方两点之间", "Constant pitch keeps every column on one grid", 26, 20),
        ("视觉重心居中", "Nudged down so the optical center sits at the middle", 26, 20),
    ]
    y = 120
    for i, (zh, en, fz, fe) in enumerate(lines):
        dz, _, _ = text_path(F_CJK, zh, fz, tx, y)
        de, _, _ = text_path(F_REG, en, 17 if i else 20, tx, y + (30 if i else 36))
        body += f'<path d="{dz}" fill="{INDIGO}"/><path d="{de}" fill="#5B6390"/>'
        y += 100 if i == 0 else 86
    return svg(W, H, body)


files["logo-meaning.svg"] = meaning()
files["logo-construction.svg"] = construction()

for n, s in files.items():
    with open(f"{OUT}/svg/{n}", "w") as f:
        f.write(s)
print("\n".join(sorted(files)))
