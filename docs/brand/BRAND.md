# echostream · 流光回响 — logo

## Mark
A 5-4-3-2-1 dot funnel: video pool → 4-way recall → LightGBM rank → MMR re-rank → next screen.

| Rule | Value |
|---|---|
| Radius per row | × φ^(1/6): u, 1.08u, 1.17u, 1.27u, 1.38u |
| Coral : recall-row dot | area ratio = φ |
| Neighbor gap | u/√φ (u = smallest radius) |
| Alignment | constant pitch, every dot centered below two |
| Top row (video pool) | screen: each circle fades from 0 (top) to 1/φ² (bottom); mono/print: flat 1/φ² tint |
| Optical centering | nudged toward the area-weighted center (faded row weighted 1/4) |

## Colors
Mark (top → bottom, equal OKLCH steps, same on light and dark):
Cyan #00C2FF → Azure #4597FF → Violet #7B61FF → Orchid #D154CD → Coral #FF5C7A
Azure and Orchid are the OKLCH midpoints between the brand colors.
Wordmark: Indigo #1E2A78 on light, white on dark. Dark background: Night #0B1020.
Favicons (3 rows): Cyan → Violet → Coral.

## Type
English: Inter Display (echo Bold, stream Light), x-height = mark height / φ².
Chinese: Source Han Sans / Noto Sans CJK SC Medium, EN : CN size = φ : 1.
All type is outlined in the SVGs.

## Files
svg/ and png/ — icon-512 (color, on-dark, black, white, tile-dark), favicon-32/16 (color, on-dark),
lockup-horizontal and lockup-bilingual (4 color versions, PNG @1x/@2x),
github-social-preview-1280x640, logo-meaning, logo-construction.
Regenerate everything with `python3 gen.py` (needs fontTools + Inter Display + Noto Sans CJK SC).
