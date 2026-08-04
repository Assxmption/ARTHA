# ARTHA — Master Stitch Build Script (v3, fresh session, hardened)

This supersedes `ARTHA_STITCH_MASTER_V2.md`. Same design system — that part was already correct — but every
prompt below now leads with an unmistakable generation command instead of scene-setting context, so the
request can't be read as a planning/brainstorming task even if mode selection is unclear. There's also a
troubleshooting section right here, first, since that's what actually blocked the last attempt.

---

## If you're not sure the mode is set right — read this before Prompt 0

**The symptom that caused last attempt's stall — getting a document, personas, or logo/brand concepts back
instead of a screen — means the mode is on Ideate, which is a planning agent, not a screen generator.** It
doesn't matter how the prompt is worded if that mode is active; it will never output a finished UI screen.

- Look for a small selector near the prompt input box — it may show as a labeled pill/dropdown ("Ideate,"
  "Flash," "Thinking," "Redesign") rather than a conventional dropdown menu. Click directly on the current
  mode name to see the other options.
- If you genuinely cannot locate or change it: type the mode name directly into your first message as a
  plain instruction — **"Use Flash mode."** — as the very first line, before anything else. This has a
  reasonable chance of being honored even if the UI toggle itself isn't cooperating.
- **The reliable tell, every time, going forward:** if what comes back isn't a rendered screen — if it's
  prose, a list, a plan, or an image that isn't a UI layout — stop and fix the mode before sending anything
  else. Don't try to prompt your way out of the wrong mode; no wording fix works around it.

---

## Operating rules — read before typing anything into Stitch

1. **One screen, one thread, one change at a time.** Bundled asks ("fix the color AND the copy AND add a
   feature") are what caused content loss and silent no-ops throughout the last build.
2. **Reference-based fixes beat property patches.** "Match this screen's system to [known-good screen]" has
   landed correctly every time. "Change the font to X" on an existing screen has failed or dropped content
   almost every time. Prefer the former whenever a screen already has content worth keeping.
3. **If a screen doesn't respond to two correctly-scoped fix attempts, its session is probably stuck.** Test
   with something trivial ("change this screen's title to TEST123"). If even that doesn't land, delete the
   screen and regenerate fresh from a known-good reference rather than attempting a third patch.
4. **Always give an explicit 1:1 color-role mapping, never a flat hex list.** Stitch's schema wants exactly
   Primary/Secondary/Tertiary/Neutral/Error — an unstructured list of brand colors gets extra colors invented
   to fill any role you didn't explicitly assign. Map every token to a named role, every time, no exceptions.
5. **Verify against the exported code and DESIGN.md, not the screenshot.** A screenshot can look right while
   the underlying Tailwind config is still on stale values — the single most common false-positive last time.
6. **Any prompt touching type or color on a screen that already has content must say so explicitly:** *"keep
   the exact same layout and text content unchanged — only update [the specific thing]."*
7. **Stay inside one project, named "ARTHA," the whole time.** Confirm the project name after every few
   screens — it silently forked into a second, differently-named design system last time and two systems
   quietly disagreed with each other for several rounds before anyone noticed.

---

## The design system (confirmed, final)

### Colors — explicit role mapping, use exactly this wording in Prompt 0
- **Primary** = brass `#B08D57`
- **Secondary** = emerald `#1F4B3F`
- **Tertiary** = ink `#1B1712` *(do not let Stitch invent a Tertiary — this is it)*
- **Neutral** = warm stone gray `#8C8478`
- **Error** = burgundy `#6E2A34`
- **Surface (day mode)** = parchment `#F1E8D6`
- **Surface (night mode)** = lacquer `#0E1A16`

### Typography
- Display: **Libre Caslon Text** (literary serif, historical authority)
- Long-form body: **Source Serif 4**
- Structural (nav/labels/buttons): **Jost** (geometric sans, 1920s lineage)
- Data (all numeric tables): **IBM Plex Mono**, tabular figures

### Register — layout density, fixed per screen, independent of color mode
- **Atelier**: editorial, ornamental, generous whitespace — landing, reports
- **Instrument**: dense, quiet, zero ornament — dashboard, signal tables, settings

### Mode — a real, user-facing toggle, independent of register
A single control (sun/moon icon, top-right of every screen) swaps color only, everywhere, at once: **Day** =
parchment surfaces/ink text/brass accents. **Night** = lacquer surfaces/parchment-colored text/brass accents.
Register stays fixed either way — layout density never changes when the toggle is flipped, only color does.

### Accessibility baseline — required on every screen, not a separate screen
- WCAG AA contrast: 4.5:1 body text, 3:1 large text (24px+) and UI components, in both modes.
- Every interactive element gets a visible 2px brass focus outline, not just a color shift.
- Icon-only controls get explicit ARIA labels described in the prompt itself.
- The Dashboard shell includes a "skip to main content" link, visually hidden until focused.
- Motion (sunburst reveal, chart animations) respects `prefers-reduced-motion` with a described static
  fallback.
- Status/signal colors are always paired with a text label or icon — never color alone.

---

## Prompt 0 — paste first, before any screen

*Confirm Flash mode is active (see troubleshooting above) before sending this.*

> **Generate a UI screen now — this is a design request, not a planning or branding request.** I'm designing
> ARTHA, a web app giving Indian retail investors row-level fundamental analysis (NSE equities) and commodity
> signal tracking (MCX) — the rigor of a Bloomberg terminal crossed with the visual confidence of a fashion
> editorial magazine, deliberately avoiding fintech-neon/glassmorphism styling. User: someone who checks this
> daily, wants to trust the numbers at a glance, and may navigate by keyboard or screen reader.
>
> **Color system — map to your schema's exact roles, do not invent additional colors for any role:** Primary
> = brass `#B08D57`. Secondary = emerald `#1F4B3F`. Tertiary = ink `#1B1712`. Neutral = warm stone gray
> `#8C8478`. Error = burgundy `#6E2A34`. Surface has two modes: day = parchment `#F1E8D6`, night = lacquer
> `#0E1A16` — build both from the start, controlled by one toggle.
>
> **Typography:** Display = Libre Caslon Text. Long-form body = Source Serif 4. Structural (nav/labels/
> buttons) = Jost — must appear in the actual font import. Data/numeric = IBM Plex Mono, tabular figures.
>
> **Accessibility, required on every screen:** WCAG AA contrast in both modes; visible 2px brass focus
> outlines on every interactive element; ARIA labels on icon-only controls; status colors always paired with a
> text label, never color alone.
>
> **Register:** "Atelier" screens (landing, reports) are editorial and ornamental. "Instrument" screens
> (dashboard, signals, settings) are dense and quiet, zero ornament — independent of day/night color mode.
>
> **This screen specifically:** the landing page hero, establishing the Atelier register, with a working
> day/night mode toggle (sun/moon icon, top-right, ARIA label "Toggle dark mode"). Full spec below.

---

## Screen 1 — Landing hero

**1a — structure** *(Flash):*
> Generate a UI screen: a **web app** landing hero for ARTHA. Centered, symmetric. Massive centered headline
> "ARTHA." A one-line thesis beneath it. A radiating sunburst line motif behind the headline. Thin horizontal
> rules above and below. A day/night mode toggle (sun/moon icon) top-right. No nav bar or footer yet.

**1b — style, both modes** *(Flash):*
> Apply the design system from Prompt 0, in day mode: parchment background, Libre Caslon Text headline, Jost
> thesis line, brass sunburst and hairlines. Then show night mode on the same screen: lacquer background,
> parchment-colored text, brass accents unchanged. Same layout — only color mode differs.

**1c — accessibility + polish** *(Thinking, since this is the most visible screen):*
> Add a visible 2px brass focus outline to the mode toggle button, ARIA label "Toggle dark mode." Confirm the
> thesis text meets 4.5:1 contrast against the background in both modes.

---

## Screen 2 — Dashboard shell

**2a — structure** *(Flash):*
> Generate a UI screen: a **web app** dashboard shell for ARTHA, Instrument register — calm, dense, zero
> ornament, no sunburst. Left sidebar: watchlist split "NSE" / "MCX." Top bar: regime status indicator (dot +
> label, e.g. "Regime: Calm"), the day/night toggle, a "skip to main content" link (hidden until
> keyboard-focused). Main panel: active-signals table — pair, z-score, Sharpe, validated badge. Bottom panel:
> risk flags beside a recent-reports feed. Reference: Bloomberg-terminal density, not a consumer app.

**2b — style + accessibility** *(Flash):*
> Apply the design system, night mode as default (lacquer background, parchment text, brass hairline
> dividers only, no cards/shadows) with the same day/night toggle from the Hero working here too. All numeric
> data in IBM Plex Mono, tabular figures. Status badges pair color with a text label, never color alone. Every
> table row and nav item has a visible focus outline.

---

## Screens 3–11 — reference-matched, one prompt each, all lead with "Generate a UI screen"

**3. Fundamentals / row-level view** *(hybrid register, Flash)*
> Generate a UI screen, using the same design system as the Dashboard Shell: a row-level fundamentals view for
> an NSE-listed company (use "Reliance Industries," not a US company). Left two-thirds: multi-year data table
> (IBM Plex Mono, tabular figures). Right third: narrative panel (Libre Caslon Text heading, Source Serif 4
> body) explaining year-over-year changes, numbered citations linking to table rows. Respect the mode toggle.
> Table headers have `scope="col"` for screen readers.

**4. Quant signals view** *(Instrument, Flash)*
> Generate a UI screen, using the same design system as the Dashboard Shell: a quant signals view — regime
> timeline across the top, a table of stat-arb pairs with a z-score gauge per row, Sharpe/Sortino/max-drawdown
> metrics, a "validated" badge paired with a checkmark icon, not color alone. Respect the mode toggle.

**5. News & sentiment panel** *(Instrument, Flash)*
> Generate a UI screen, using the same design system as the Dashboard Shell including its dark/Instrument
> treatment: a news and sentiment panel — ticker-scoped headline feed, each with a sentiment tag (icon + text
> label, e.g. an up/down/flat arrow beside the color) and an event-type tag. Compact, terminal-style.

**6. Risk & compliance panel** *(Instrument, Flash)*
> Generate a UI screen, using the same design system as the Dashboard Shell: a risk and compliance panel —
> risk flags with severity (icon + text + color, never color alone), a circuit-halt status indicator, a
> compliance status widget (Algo-ID, static IP). Calm and authoritative, not alarming.

**7. Report view** *(Atelier, Flash)*
> Generate a UI screen, using the same design system as the Landing Hero: a report view — long-form
> narrative, Libre Caslon Text title, Source Serif 4 body, generous margins, numbered citations linking to
> data points, a brass hairline between sections, a restrained export control. Respect the mode toggle. Reads
> like a well-typeset essay, not a dashboard.

**8. Simulation / backtest lab** *(Instrument, Flash)*
> Generate a UI screen, using the same design system as the Dashboard Shell: a backtest lab — pair/factor
> selector, a large equity-curve chart (thin brass line, no gridline clutter) with a text-based data summary
> alongside it for screen readers, a metrics strip (Sharpe, Sortino, max drawdown, win rate) below.

**9. Settings / broker connection** *(Instrument, Flash)*
> Generate a UI screen, using the same design system as the Dashboard Shell: a settings page — broker
> connection status, Algo-ID, static IP whitelist status, a clearly separated danger-zone section for
> disconnecting using burgundy `#6E2A34` only in that section, with a confirmation step before any destructive
> action.

**10. Empty / loading / error states** *(refinement pass, after 2–9 exist, Flash)*
> On the Dashboard and Quant Signals screens: add an empty state for zero validated signals (quiet message,
> faint sunburst — the one place it's allowed in the Instrument register) and a loading state using a slow
> single-line brass progress rule instead of a spinner, with `aria-live="polite"` so screen readers announce
> when loading completes.

**11. Auth / onboarding** *(Atelier-lite, Flash)*
> Generate a UI screen, using the same design system as the Landing Hero, simplified: a sign-in screen —
> centered card, email field with a visible label (not placeholder-only text), "Continue" button, mode toggle
> available here too. Minimal — personal-use tool, no marketing copy or social proof.

---

## Before generating all eleven

Generate Screens 1 and 2 only, first. Confirm what came back is an actual rendered screen (not a document —
see the troubleshooting note at the top if it isn't). Confirm the mode toggle swaps colors on both, confirm
keyboard Tab visibly moves through focus states, and check the exported code for the exact hex values above.
Only then move through 3–11. Fixing the foundation twice is cheap; fixing eleven screens built on a wrong one
is not.
