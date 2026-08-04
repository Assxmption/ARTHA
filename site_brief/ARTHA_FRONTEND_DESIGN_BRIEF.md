# ARTHA Frontend Design Brief — "Atelier & Instrument"

## 1. Yes — Gemini + Stitch MCP is a genuinely good call, not a fallback

I checked this rather than assume from memory, since it's exactly the kind of thing that changes fast. Here's
where things actually stand: **Stitch (stitch.withgoogle.com) is a free Google Labs UI design tool, currently
running on the Gemini 3 model family, with an official MCP server that's also free.** It generates full
screens from natural-language prompts, exports production code (React/Tailwind, Vue, Flutter, SwiftUI, or
plain HTML/CSS), and — the part that matters most for your setup — it exports a portable `DESIGN.md` token
file that an MCP-connected agent can fetch directly, so your Antigravity agent gets your actual color/type/
spacing tokens instead of you copy-pasting hex codes into a prompt by hand. Free tier is roughly 350 standard
generations/month (Gemini 3 Flash-tier) plus 200 higher-fidelity Pro-tier generations/month — comfortably
enough for a full product's worth of screens.

Two things make this a *better* fit for your situation than Claude Design would be right now, not just a
cheaper one:

- **It plugs into the same IDE you're already running the backend build in.** Antigravity has first-class
  Stitch MCP support — install it from Antigravity's MCP marketplace (search "Stitch"), paste your API key,
  and your Opus 4.6 agent can query your Stitch project directly (`List my Stitch projects`, fetch screens,
  fetch the design system) without you ever leaving the IDE or manually transcribing anything.
- **It's free and matches the "not extensive money put in yet" constraint** from earlier in this build,
  exactly the same way the historical-replay simulation strategy did for the quant side.

**One operational detail worth knowing**: in Stitch's model dropdown (it defaults to a Flash-tier model), you
have to explicitly select the Pro-tier model for your highest-stakes screens — the codelab documentation is
explicit that this matters for layout reasoning quality. Spend your ~200 monthly Pro generations on the screens
that carry the most visual weight (landing hero, dashboard shell, the report view), and use the higher-volume
Flash tier for straightforward variants (settings, empty states, minor screens).

## 2. The aesthetic direction: "Atelier & Instrument"

You asked for something that could plausibly be a fashion house's portfolio as much as a stock-insight
platform, without sliding into either generic fintech neon or costume-y pastiche. The way to actually pull
that off — and the way real design teams solve the "serious industry, warm luxury feel" problem — is to split
the product into **two registers that share one token system but wear it differently**:

- **Atelier** — the landing/marketing surface and the narrative report view. This is where you spend the
  design's boldness: editorial layout, art-deco-lineage typography and ornament, generous whitespace, the
  signature motif (below). This is the "fashion portfolio" half of the brief.
- **Instrument** — the live dashboard, the row-level data tables, the signal panels. Same palette and type
  family, but quiet and precise: financial data has to be scannable in a glance, so this register drops the
  ornament and tightens the grid. This is the "the numbers are real" half of the brief, and it's what keeps
  the whole thing from reading as style-over-substance.

This split is a deliberate design decision, not a compromise — over-ornamenting a data table is one of the
most common ways an otherwise beautiful finance product becomes unusable, and under-designing the report view
is how you end up back at generic fintech. Every screen in Section 4 is tagged with which register it belongs
to.

**Real inspiration worth actually looking at** (paraphrased, not reproduced — visit these directly):

- **The Renaissance Edition** (Shopify, Awwwards Site of the Month, Feb 2026) is the closest existing proof
  that this exact idea works: a product-update changelog — about as dry a genre as exists — presented as a
  curated art-gallery experience using an old-master-inspired palette and dramatic composition. If a Shopify
  changelog can feel like walking through a museum, a stock-insight report can too.
- **Aventura Dental Arts** (Awwwards-recognized) solves almost your exact problem in a different industry:
  a clinical, anxiety-inducing category (dentistry) reframed through warm champagne-toned neutrals and
  sophisticated serif type to feel like a boutique spa rather than a clinic. That's the technique — "de-
  medicalize"/de-technicalize a serious category through warmth and typographic confidence — applied to
  finance instead of medicine.
- **Cartier — Watches & Wonders** (Awwwards-recognized) is worth studying for the actual period-accurate art
  deco vocabulary from a house that owns that heritage honestly: restrained geometric linework, metallic
  accents used sparingly, symmetry as a structural device rather than decoration.
- **Jeton** (fintech, Awwwards Site of the Day) is proof a finance product can carry real creative-grade
  interaction and a confident limited palette (black/white plus one signature accent color) without losing
  the "serious finance flow" a user needs.

## 3. Design token system

Following a real token discipline (not just describing a vibe) so both Stitch and your Antigravity agent build
against the same fixed system:

### Color

| Token | Hex | Use |
|---|---|---|
| `--ink` | `#1B1712` | Primary text — warm near-black, not pure `#000` |
| `--parchment` | `#F1E8D6` | Primary ground, Atelier surfaces — warm ivory |
| `--lacquer` | `#0E1A16` | Deep ground, dashboard shell / dark sections — bottle-green-black, not flat near-black |
| `--brass` | `#B08D57` | Metallic accent — hairlines, dividers, deco linework, the signature motif |
| `--emerald-signal` | `#1F4B3F` | Validated signals, positive deltas — a jewel tone, deliberately not neon green |
| `--burgundy-flag` | `#6E2A34` | Risk flags, negative deltas, warnings |

*(Six named tokens, deliberately not the AI-default cream-plus-terracotta or near-black-plus-acid-accent
combinations — those two are the most common templated look right now and would undercut the whole brief.)*

### Type

| Role | Face | Why |
|---|---|---|
| Display | **Libre Caslon Text** (Google Fonts) | Literary serif with historical authority — confirmed direction after the first real Stitch generation, used for hero headlines, report titles, and big numerals treated as typographic art |
| Long-form body | **Source Serif 4** (Google Fonts) | Legible humanist serif for narrative/report body copy |
| Structure / nav / labels | **Jost** (Google Fonts) | A geometric sans genuinely descended from 1920s–30s geometric faces (Kabel/Erbar lineage) — real art-deco pedigree, and far less over-used in AI-generated design right now than Cinzel or Poiret One |
| Data / tables | **IBM Plex Mono** or **Spectral Mono** | Tabular figures for anything numeric — row-level fundamentals, signal tables, backtest metrics — legibility over personality here |

### Layout concept

```
ATELIER — landing hero                    INSTRUMENT — dashboard shell
┌────────────────────────────┐            ┌──────────────────────────────┐
│                             │            │ ARTHA   [regime: calm ●]     │
│      A R T H A              │            ├──────────┬───────────────────┤
│   (Libre Caslon Text,       │            │ Watchlist│ Signal panel      │
│    centered, brass rule     │            │  NSE     │  pair | z | Δ    │
│    above + below)           │            │  MCX     │  ---- table ---- │
│                             │            │          │                   │
│   one-line thesis (Jost)    │            ├──────────┴───────────────────┤
│                             │            │ Risk flags   |  Recent report │
│      ⟡ sunburst motif       │            └──────────────────────────────┘
└────────────────────────────┘
```

### Signature element: "The Sunburst Ledger"

One recurring, deliberately restrained mark: a thin radiating brass line-burst — visually a classic art-deco
sunburst, but drawn so it also reads as a chart's radiating axis lines. Use it once, prominently, on the
landing hero and the top of the main report; echo it small and quiet as a loading-state and section-divider
motif elsewhere. That's the one bold move — everything else in the system stays disciplined around it (per
the "spend your boldness in one place" principle — an over-ornamented product reads as costume, not craft).

One direction worth considering to make this specifically *yours* rather than generic 1920s-Paris pastiche:
fold a subtle Indian ornamental reduction — a geometrically simplified lotus or paisley line — into the
sunburst's negative space. Given the product's name is Sanskrit and its entire scope is NSE/MCX, that fusion
is a genuine "signature," not decoration for its own sake.

## 4. Screen / information-architecture checklist — everything to be added

| Screen | Register | Purpose | Backend source |
|---|---|---|---|
| Landing / marketing page | Atelier | Hero thesis, methodology teaser, signal showcase, footer | static copy |
| Dashboard / home | Instrument | Regime state per watched instrument, top validated signals, risk-flag banner, recent reports | `RegimeState`, `QuantSignal`, `RiskFlag` |
| Fundamentals / row-level view (Screener++) | half-and-half — editorial narrative panel beside an Instrument-register table | Multi-year row-level fundamentals table + narrative explaining deltas, with citation footnotes | `FundamentalRow` |
| Quant signals view | Instrument | Regime timeline, stat-arb pair list with z-score gauge, Sharpe/Sortino/MaxDD per signal, "validated" badge | `QuantSignal`, `CointegratedPair`, `BacktestReport` |
| News & sentiment panel | Instrument | Scoped headline feed, sentiment tag, event-type tag | `NewsSignal` |
| Risk & compliance panel | Instrument | Risk flags with severity, circuit-halt indicator, (Phase 5) Algo-ID/static-IP compliance status | `RiskFlag`, `ComplianceCheck` |
| Report view | Atelier | Full generated narrative report, inline citations linking to source rows, export/download | Writer agent output, full Fact Store trace |
| Simulation / backtest lab | Instrument | Pick a pair/factor, run historical replay, view equity curve | `BacktestReport`, `SimulationRun` |
| Settings / broker connection (Phase 5 only) | Instrument | Broker API status, Algo-ID, static IP whitelist state | `ComplianceCheck` |
| Empty / loading / error states | both | Direction, not mood — explain what happened and what to do next, in the interface's voice | — |
| Auth / onboarding | Atelier-lite | Minimal — this is a personal-use capstone, not a multi-tenant product yet | — |

Don't build a user-facing view of `docs/IMPLEMENTATION_PLAN.md` — that's an internal build-process artifact
for you and the agent, not something end users should see.

## 5. The full sequential prompt set — and the method behind it

I checked Google's own Stitch Prompt Guide rather than just write these from feel, because Stitch actually
responds very differently depending on how you structure a prompt. Three rules from it change what I gave you
before:

1. **Zoom-Out-Zoom-In** (this is Stitch's own documented framework, not a term I'm inventing): for a
   multi-screen product, the *first* prompt should set broad context — the product, its user, its identity —
   and only the *second* half zooms into the specific screen. Individual screen prompts after that stay
   zoomed-in. My earlier three prompts skipped the zoom-out step; this version doesn't.
2. **Micro-prompts beat mega-prompts, and refinements should change one thing at a time.** Asking for several
   changes in one refinement prompt makes Stitch rebuild the whole layout from scratch and often breaks what
   was already working. So below, the two screens that *establish* your design system (the hero and the
   dashboard shell) are built in stages — structure first, then style, then one polish pass — rather than one
   shot. The remaining nine screens lean on **design-DNA matching** ("using the same design system as
   screen X...") instead of restating the whole palette, which is both the efficient move and the one that
   actually keeps 11 screens visually consistent.
3. **State the platform and lean on real-world analogies.** Stitch's output shifts noticeably based on
   whether you say "web app" vs. "mobile app," and referencing a familiar interface ("like a Bloomberg
   terminal," "like a fashion magazine spread") gives it more to work with than adjectives alone. Every prompt
   below states "web app" explicitly and includes at least one analogy. Also worth knowing going in: several
   independent write-ups note Stitch's results are currently a bit more reliable for mobile than for dense web
   dashboards — expect the Instrument-register screens to need an extra refinement pass or two, which is
   already built into the sequence below.

One workflow habit worth adopting from the start: **screenshot the canvas after every generation you're happy
with.** Stitch sessions have been reported to reset unexpectedly, and there's no undo-proof history yet.

---

### Prompt 0 — paste this first, before generating any screen

This sets the zoom-out context for the whole project. Mode: **Ideate** (you're establishing identity, not
executing a known spec yet). Model: **Gemini 3 Pro**.

> **Context:** I'm designing ARTHA, a web app that gives Indian retail investors row-level fundamental
> analysis (NSE equities) and commodity signal tracking (MCX) — think the rigor of a Bloomberg terminal
> crossed with the visual confidence of a fashion editorial magazine, deliberately avoiding typical
> fintech-neon/glassmorphism styling.
> **User:** Someone who already knows markets, checks this daily, and wants to trust the numbers at a glance
> — but who's tired of every finance product looking like the same dark-mode neon template.
> **Design system:** Warm ivory ground (#F1E8D6), warm near-black text (#1B1712), a deep bottle-green-black
> for data-dense screens (#0E1A16), a single brass metallic accent (#B08D57) for hairlines and ornament, a
> deep emerald (#1F4B3F) for positive/validated signals, a deep burgundy (#6E2A34) for risk flags. Display
> type: Libre Caslon Text (literary serif). Long-form body type: Source Serif 4. Structural type: Jost (geometric sans, 1920s lineage).
> Data type: IBM Plex Mono (tabular figures).
> **Goal for this first screen:** Generate the landing page hero — this establishes the ornamental "Atelier"
> half of the system that the marketing/report surfaces will use. (Full spec below.)

---

### Screen 1 — Landing hero (establishes the "Atelier" register)

*Mode: Ideate → Flash for refinements. Model: Gemini 3 Pro throughout.*

**1a — structure first:**
> Design a **web app** landing page hero for ARTHA. Centered, symmetric composition. Massive centered
> headline "ARTHA." A one-line thesis beneath it. A radiating sunburst line motif behind/around the headline.
> Thin horizontal rules above and below the headline block. No nav bar yet, no footer yet — just get the hero
> composition and hierarchy right first.

**1b — style pass, once structure looks right:**
> Apply the design system from Prompt 0 to this hero: warm ivory background, the headline in Libre Caslon Text, the
> thesis line in Jost, the sunburst and hairline rules in the brass accent color. No gradients, no
> glassmorphism, no neon — this should feel closer to a luxury fashion house's site than a typical fintech
> product.

**1c — one polish pass:**
> Increase the whitespace around the headline block by roughly a third, and make the sunburst motif more
> subtle — it should read as a refined detail, not the loudest thing on the page.

---

### Screen 2 — Dashboard shell (establishes the "Instrument" register)

*Mode: Flash (you know what this needs to be now). Model: Gemini 3 Pro.*

**2a — structure + explicit register contrast:**
> Design a **web app** dashboard shell for ARTHA — deliberately the *opposite* register from the landing hero
> in this project: calm, dense, precise, zero ornament, no sunburst motif here. Layout: left sidebar with a
> watchlist split into two sections, "NSE" and "MCX." Top bar: a small colored status dot plus label showing
> the current market regime (e.g. "Regime: Calm"). Main panel: a data table of active signals with columns
> for pair, z-score, Sharpe ratio, and a "validated" badge. Bottom panel: a risk-flags list beside a feed of
> recent reports. Reference point: information density like a Bloomberg terminal, not a consumer app.

**2b — style + data typography pass:**
> Apply the design system from Prompt 0, in dark mode for this screen specifically: deep bottle-green-black
> background (#0E1A16), warm ivory text, thin brass hairlines as the *only* dividers — no cards, no shadows,
> no rounded gradients. Set all numeric data (z-scores, Sharpe values, prices) in IBM Plex Mono with tabular
> figures so columns of numbers align cleanly.

---

### Screens 3–11 — design-DNA matched (one generation prompt each)

For each of these, the pattern is the same: reference the screen whose register it belongs to, state what's
new, and let Stitch propagate the established system. If a result drifts (extra ornament on an Instrument
screen, or a too-quiet Atelier screen), the fix is always the same one-line nudge, given once below rather
than repeated 9 times.

**3. Fundamentals / row-level view** *(hybrid register)*
> Using the same design system as the Dashboard Shell screen for structure but borrowing Libre Caslon Text from the
> Landing Hero for headings, design a **web app** "row-level fundamentals" view: left two-thirds a clean
> multi-year data table (IBM Plex Mono, tabular figures) showing a company's fundamentals row by row across
> years; right third a narrative panel (Libre Caslon Text heading, Source Serif 4 body) explaining what changed
> year over year, with small numbered citation marks linking each claim back to a specific table row. Think
> equity-research terminal crossed with a magazine sidebar.

**4. Quant signals view** *(Instrument)*
> Using the same design system as the Dashboard Shell screen, design a **web app** quant signals view: a
> regime timeline across the top (a horizontal band showing regime changes over time), below it a table of
> statistical-arbitrage pairs with a z-score gauge per row, and Sharpe/Sortino/max-drawdown metrics with a
> "validated" badge shown only when a signal has passed backtesting. No decorative elements.

**5. News & sentiment panel** *(Instrument)*
> Using the same design system as the Dashboard Shell screen, design a **web app** news and sentiment panel:
> a scoped headline feed (ticker-tagged), each headline with a small sentiment tag (positive/neutral/negative
> using the emerald/ivory/burgundy palette) and an event-type tag (earnings, policy, supply shock). Compact,
> scannable, like a terminal news ticker rather than a social feed.

**6. Risk & compliance panel** *(Instrument)*
> Using the same design system as the Dashboard Shell screen, design a **web app** risk and compliance panel:
> a list of risk flags with severity indicators (info/warning/block, using ivory/brass/burgundy), a
> circuit-halt status indicator, and a compliance status widget showing Algo-ID and static-IP status. Should
> feel authoritative and calm, not alarming — this is a status readout, not an error page.

**7. Report view** *(Atelier)*
> Using the same design system as the Landing Hero screen, design a **web app** report view: a long-form
> narrative document with a Libre Caslon Text title, generous margins, body text in Source Serif 4, and small
> numbered citation marks throughout the text that link back to specific data points. Include a brass hairline
> divider between sections and a restrained export/download control. This should read like a well-typeset
> essay, not a dashboard.

**8. Simulation / backtest lab** *(Instrument)*
> Using the same design system as the Dashboard Shell screen, design a **web app** backtest lab: a
> pair/factor selector at the top, a large equity-curve chart in the center (thin brass line on the dark
> ground, no gridlines clutter), and a metrics summary strip below (Sharpe, Sortino, max drawdown, win rate).

**9. Settings / broker connection** *(Instrument, Phase 5)*
> Using the same design system as the Dashboard Shell screen, design a **web app** settings page for broker
> API connection status: connection state (connected/disconnected), Algo-ID display, static IP whitelist
> status, and a clearly separated "danger zone" section for disconnecting — using the burgundy accent only
> for that section.

**10. Empty / loading / error states** *(refinement pass, not a new screen — apply to existing ones)*
> For the Dashboard Shell and Quant Signals screens: add an empty state for when no signals have been
> validated yet (a quiet message plus the sunburst motif rendered very faint, as the one place it's allowed
> to appear in the Instrument register), and a loading state using a slow single-line brass progress rule
> rather than a spinner.

**11. Auth / onboarding** *(Atelier-lite, minimal)*
> Using the same design system as the Landing Hero screen but simplified, design a **web app** sign-in screen:
> centered card, email field, "Continue" button, minimal — this is a personal-use tool, not a consumer signup
> flow, so resist adding marketing copy or social proof here.

**The one-line drift fix**, if any screen 3–11 doesn't match: *"This doesn't match the Dashboard Shell's
design system — refer back to that screen and align the colors, type, and density to it."* Stitch will
re-anchor rather than reinterpret from scratch.


## 6. Exactly how to wire this into the repo — the verified sequence

This is the correct order, confirmed against Google's own Antigravity + Stitch MCP walkthrough — design happens
in Stitch's own canvas first; the MCP bridge in Antigravity is only used afterward, to pull the finished design
into the coding agent's context. It does not generate the design itself, and `DESIGN.md` is not a manual
export button inside Stitch — it's something you ask the Antigravity agent to produce via the MCP bridge.

**Phase 1 — design in Stitch (stitch.withgoogle.com), before touching Antigravity**

1. Start a new design → **Web**. Switch the model dropdown from its Flash-tier default to **Gemini 3 Pro** (or
   the current Pro-tier option) before generating anything you care about.
2. Paste **Prompt 0**, then **Prompt 1a/1b/1c** from Section 5, generating and refining in the chat sidebar
   until the hero is actually right — this is your visual quality gate, do the art-directing here, not in code.
3. Name the project `ARTHA`.
4. For every additional screen, stay **inside that same Stitch project** so they share one design system.
   Prefer prompts like *"Using the same design system as the Dashboard Shell screen in this project, generate
   a Fundamentals/Row-Level screen for: [description]"* over independently re-specifying the palette each
   time — this is how you avoid screen-to-screen drift.
5. Profile picture (top right) → **Stitch settings → API key → Create key**. Copy it somewhere safe.

**Phase 2 — bridge into Antigravity via MCP**

6. Open Antigravity, open the **Agent Manager** (Cmd+E / Ctrl+E toggles between it and the editor).
7. Point a workspace at your **existing ARTHA repo folder** — not a fresh directory, you already have the
   backend build in there.
8. In Agent Manager, open **MCP Servers** (via the **"..."** dropdown at the top of the agent pane), search
   **"Stitch"**, click **Install**, and paste your Stitch API key when prompted.
9. Verify the bridge — in the agent chat, type exactly:
   ```
   List my Stitch projects.
   ```
   It should return `ARTHA`. If it doesn't, the key didn't save — redo step 8.

**Phase 3 — pull the design in, then build**

10. Ask the agent to generate the design-system file — this is the actual DESIGN.md step, done through the
    agent, not a Stitch export button:
    ```
    Use the Stitch MCP to fetch the "ARTHA" project. Extract the color palette and typography,
    then generate a DESIGN.md file in my root directory.
    ```
11. Open the resulting `frontend/DESIGN.md` (move it there if the agent wrote it to the repo root) and check
    it against the token table in Section 3 — this is your chance to catch drift before any code gets written.
12. Give the build prompt below to implement it.
13. To fix drift later, don't hand-edit CSS against memory: *"The [X] doesn't match — refer back to the Stitch
    design and update it."* The agent re-fetches from Stitch via MCP and corrects it against the source of
    truth, rather than guessing.

```bash
# once frontend/DESIGN.md exists and looks right:
git add frontend/
git commit -m "Add Stitch-generated design system (DESIGN.md) for ARTHA frontend"
```

Then, in the same Antigravity workspace, paste this to kick off the actual frontend build:

> Read `AGENTS.md` and `frontend/DESIGN.md` in this repo before doing anything. `frontend/DESIGN.md` is the
> Stitch-exported design system (colors, type, spacing) for ARTHA's frontend — use the Stitch MCP tools to
> fetch the actual screen designs from my Stitch project (list projects, then fetch each screen) rather than
> guessing at layout from the token file alone. Implement the frontend in `frontend/` as a React + Tailwind
> app, wired to the existing FastAPI backend in `app/api/routes.py`. Build screens in this order: dashboard
> shell, fundamentals/row-level view, quant signals view, then the rest of the checklist in
> `docs/ARTHA_FRONTEND_DESIGN_BRIEF.md` Section 4. Respect the Atelier/Instrument register split — don't let
> dashboard/table screens inherit the landing page's ornamentation. Before writing code for each screen,
> confirm you've fetched its actual Stitch design via MCP rather than reconstructing it from memory of the
> prompt. Update `docs/IMPLEMENTATION_PLAN.md` with a new "Frontend" phase and check in with me at that
> checkpoint before starting, per `AGENTS.md` rule 12.

## 7. One last check before you build a lot of screens

Generate just the landing hero and the dashboard shell first, look at both side by side, and ask yourself the
"remove one accessory" question — if either one still feels like it's trying too hard, cut something before
generating the rest of the checklist. It's much cheaper to fix the token system after two screens than after
eleven.
