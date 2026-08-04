---
name: Artha Systematic Editorial
colors:
  surface: '#17130b'
  surface-dim: '#17130b'
  surface-bright: '#3e382f'
  surface-container-lowest: '#120e06'
  surface-container-low: '#201b12'
  surface-container: '#241f16'
  surface-container-high: '#2e2920'
  surface-container-highest: '#3a342a'
  on-surface: '#ebe1d3'
  on-surface-variant: '#d1c5b6'
  inverse-surface: '#ebe1d3'
  inverse-on-surface: '#353026'
  outline: '#9a8f81'
  outline-variant: '#4e453a'
  surface-tint: '#e8c086'
  primary: '#e8c086'
  on-primary: '#432c00'
  primary-container: '#b08d57'
  on-primary-container: '#3d2700'
  inverse-primary: '#775928'
  secondary: '#a2d0c0'
  on-secondary: '#06372c'
  secondary-container: '#255144'
  on-secondary-container: '#94c2b2'
  tertiary: '#cec5bd'
  on-tertiary: '#34302a'
  tertiary-container: '#99918a'
  on-tertiary-container: '#2f2b25'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#ffdeae'
  primary-fixed-dim: '#e8c086'
  on-primary-fixed: '#281800'
  on-primary-fixed-variant: '#5d4213'
  secondary-fixed: '#beecdc'
  secondary-fixed-dim: '#a2d0c0'
  on-secondary-fixed: '#002019'
  on-secondary-fixed-variant: '#234e42'
  tertiary-fixed: '#eae1d8'
  tertiary-fixed-dim: '#cec5bd'
  on-tertiary-fixed: '#1f1b16'
  on-tertiary-fixed-variant: '#4b4640'
  background: '#17130b'
  on-background: '#ebe1d3'
  surface-variant: '#3a342a'
typography:
  display-lg:
    fontFamily: Libre Caslon Text
    fontSize: 48px
    fontWeight: '400'
    lineHeight: 56px
    letterSpacing: -0.01em
  display-lg-mobile:
    fontFamily: Libre Caslon Text
    fontSize: 32px
    fontWeight: '400'
    lineHeight: 40px
  headline-md:
    fontFamily: Libre Caslon Text
    fontSize: 24px
    fontWeight: '400'
    lineHeight: 32px
  body-md:
    fontFamily: Source Serif 4
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  ui-label:
    fontFamily: Jost
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0.05em
  data-table:
    fontFamily: IBM Plex Mono
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 16px
spacing:
  unit: 4px
  gutter: 24px
  margin-atelier: 64px
  margin-instrument: 16px
  column-gap: 24px
---

## Brand & Style

The design system is built on the intersection of institutional financial rigor and the high-contrast elegance of fashion editorial. It serves professional analysts who require high data density without sacrificing aesthetic sophistication.

The visual language is characterized by **Editorial Minimalist** principles:
- **Atelier Register:** Used for long-form fundamental analysis and company profiles. It emphasizes large typographic scales, generous margins, and an ornamental use of color to guide the narrative flow.
- **Instrument Register:** Used for real-time tickers, balance sheets, and commodity tracking. It strips away all ornamentation in favor of absolute density and legibility, utilizing monospaced data grids and compact UI elements.

The emotional response should be one of "Quiet Authority"—a tool that feels like a bespoke leather-bound ledger digitised for the modern age. No glassmorphism, blurs, or soft shadows are permitted; depth is achieved through color-blocking and structural lines.

## Colors

The palette is rooted in heritage materials: Brass, Emerald, and Ink. 

- **Primary (Brass):** Reserved for active states, primary actions, and brand accents. In the Instrument register, it is used sparingly for highlights.
- **Secondary (Emerald):** Used for growth indicators (positive change) and as a grounding background for high-level navigation.
- **Surface Tones:** The system defaults to "Surface Night" (Lacquer) for the Instrument register to reduce eye strain during deep analysis. "Surface Day" (Parchment) is used for the Atelier register to evoke a printed editorial feel.
- **Semantic Error:** In alignment with the editorial aesthetic, errors are denoted by Burgundy rather than a standard bright red, maintaining high contrast against both Parchment and Lacquer backgrounds.

## Typography

The system employs a quaternary font stack to differentiate between narrative, content, interface, and raw data.

- **Libre Caslon Text:** Used for titles and expressive editorial moments. It provides the "Fashion Editorial" weight.
- **Source Serif 4:** Optimized for readability in fundamental reports and footnotes.
- **Jost:** A geometric sans-serif used for navigation, buttons, and UI labels. The increased letter spacing in uppercase is mandatory for the "Atelier" feel.
- **IBM Plex Mono:** Mandatory for all numerical data, stock tickers, and commodity prices to ensure tabular alignment and vertical scannability.

## Layout & Spacing

This design system uses a dual-grid approach based on the active register:

1.  **Atelier Grid:** A 12-column fixed grid (max-width 1440px) with 64px outer margins. Content is often centered with significant whitespace (3-column offsets) to mimic magazine layouts.
2.  **Instrument Grid:** A fluid, full-width dashboard grid with 16px margins and 8px gutters. This maximizes "pixels-per-insight."

**Breakpoints:**
- **Mobile (<768px):** Single column, margins reduced to 16px. Display sizes shift to mobile-optimized tokens.
- **Tablet (768px - 1024px):** 6-column grid for Instrument; 12-column fluid for Atelier.
- **Desktop (>1024px):** Full architectural layout as defined by the registers.

## Elevation & Depth

The design system rejects shadows in favor of **Tonal Layering** and **Line Work**.

- **Atelier Elevation:** Depth is created by "stacking" colored blocks. A Parchment card sits on an Ink background with 0px offset. Separation is achieved via 1px solid borders in Warm Stone Gray.
- **Instrument Elevation:** Depth is non-existent. The interface is a flat plane of data. Sections are divided by 1px rules (#8C8478 at 30% opacity).
- **Active State:** The only "elevation" cue is a 2px Brass (#B08D57) solid outline for keyboard focus and selected data cells, ensuring AA accessibility compliance.

## Shapes

To maintain the rigor of financial terminals and the sharpness of high-end editorial, the system uses **0px roundedness (Sharp)**.

All buttons, input fields, cards, and data tags must have square corners. This reinforces the architectural nature of the design and ensures that dense data tables in the Instrument register do not appear cluttered by tangential curves.

## Components

### Buttons
- **Primary:** Solid Brass background, Ink text, sharp corners, Jost Medium Uppercase.
- **Secondary:** Transparent background, 1px Brass border.
- **Atelier Variant:** Larger padding (16px 32px) for editorial impact.
- **Instrument Variant:** Compact padding (4px 12px) for density.

### Data Tables (The Core)
- **Header:** Jost, 11px, Uppercase, Stone Gray text, 1px bottom border in Brass.
- **Rows:** IBM Plex Mono, 13px. Alternate row striping using a 5% opacity tint of the secondary color.
- **Positive/Negative:** Text color only (Emerald/Burgundy), no background pills.

### Input Fields
- Underline style only for Atelier (1px Stone Gray).
- Fully boxed 1px border for Instrument. 
- Focus state: 2px Brass outline.

### Cards
- No shadows. 1px border (#8C8478).
- Header areas in Atelier should use a Secondary (Emerald) background with Brass text for high-impact section breaks.

### Tickers & Chips
- Sharp rectangles. 
- Commodities use a small 4px square of Brass as a prefix icon to denote "Physical Asset."
