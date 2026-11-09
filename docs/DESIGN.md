# Design notes for the site

Written before the CSS, so later changes reuse the same decisions instead of drifting back to defaults.

## Idea
The page is a **proof sheet**: you paste an article, and the tool marks it up like an editor with a red/green pen.
The distinctive element is the annotated article itself, not a dashboard of cards.

## Colour (six named roles)
| Role | Light | Dark |
|---|---|---|
| paper (background) | #f6f7f5 | #15181a |
| ink (text) | #1c2024 | #e6e8e3 |
| rule (hairlines) | #cfd3cc | #343a3d |
| fake (stamp / marks) | #a8281f | #ee8a80 |
| real | #1d6b43 | #7fcf9d |
| unsure | #8a5b00 | #e0b252 |

One accent only (ink itself). No gradients, no glow, no glass. Avoid the two "second wave" defaults:
cream + terracotta, and black + neon.

## Type
- Display and reading text: a serif stack (Charter, Iowan Old Style, Palatino, Georgia). System fonts only, so the page makes **no external requests** (it claims privacy, so it must be true).
- UI labels: the system sans at small size. Numbers and the verdict stamp: monospace.
- Body 18px / 1.6, headings tight (-0.01em), small caps-style labels are sentence case, not tracked all-caps.

## Layout
- Asymmetric two columns on wide screens: a narrow left column for the explanation, a wide right column for the tool.
- No hero, no pill badges, no card grid. Hairline rules and whitespace separate sections.
- Radius 3px, no shadows. Hierarchy comes from rules, weight and size.

## Motion
One transition: the position marker on the number line. Nothing fades in.

## Copy
Say what a thing does. No invented numbers: every figure on the page comes from `model.json`.
