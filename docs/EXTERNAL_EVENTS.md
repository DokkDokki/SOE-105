# External Events & Market Context (idea C3)

**Status: documented overlay — no external number enters the forecast pipeline.**
Prepared 2026-08-24, on Paul's go-ahead (idea originally raised by Paul's family,
2026-08-05; logged as C3 in `docs/IDEAS.md`).

This document does three things:

1. **Explains** the two turning points in the supplied history that the models
   could not explain — the −42.8% FY2023 collapse and the +25.1% FY2024 rebound.
2. **Tests** whether an external indicator would have called those turns.
3. **Reads** the external evidence that was publicly available at the FY2025
   forecast origin, and states what it implies for the delivered number.

Nothing here is fitted, and nothing here changes `outputs/`.

**Outcome (decided 2026-08-24, §7a): the headline forecast is unchanged at 174,514
units, and an external-context scenario of 211,000 units / ฿1,766.7m is published
alongside it as a labelled cross-check.** Both sit inside the already-published
range of 118,093 – 215,741.

---

## 0. Provenance rule — restated honestly

`docs/ASSUMPTIONS.md` §0 and `README.md` claim *"no external data is used anywhere
in this project."* That claim is now qualified, and the qualification is stated
wherever the claim appears:

> **The forecast pipeline still uses only the three supplied files.** No external
> series is fitted, joined, or fed to any model; the codebase still contains no
> networking library. **This document** is a separate, clearly-labelled analyst
> overlay built from public sources, kept outside the pipeline by design.

That separation is the whole point: external context is used to *interpret and
challenge* the forecast, never to silently move it.

---

## 1. What needs explaining

Portfolio demand from the supplied extract:

| Fiscal year | Units | YoY |
|---|---:|---:|
| FY2021 (Apr 21 – Mar 22) | 274,156 | — |
| FY2022 (Apr 22 – Mar 23) | 267,765 | −2.3% |
| FY2023 (Apr 23 – Mar 24) | **153,092** | **−42.8%** |
| FY2024 (Apr 24 – Mar 25) | **191,592** | **+25.1%** |

The collapse is not a drift. March 2023 was the largest month in the entire
48-month series (37,280 units); **April 2023 was the smallest (6,386 units,
−68.2% YoY)**. Demand fell off a cliff between two consecutive months, and the
supplied files contain nothing that explains it.

This matters because the back-test error is *the turning points*, not noise:

| Fold | Test year | Actual | Forecast (as delivered) | Total error |
|---|---|---:|---:|---:|
| 1 | FY2023 (crash year) | 142,911 | 208,021 | **+45.6% too high** |
| 2 | FY2024 (rebound year) | 190,100 | 150,869 | **−20.6% too low** |

The model is not biased in one direction. It **carries the recent level forward
and therefore lags every turn** — too high going into a downturn, too low coming
out of one. That is exactly the failure mode external indicators exist to fix.

---

## 2. Finding 1 — the FY2023 crash was the global electronics inventory correction

**What happened in the world.** Through 2021–22 the semiconductor and electronic
component shortage pushed lead times past 52 weeks on some parts, and customers
responded by *double- and triple-ordering* to reserve supply. When lead times
normalised through late 2022 and 2023 (automotive MCUs 52+ → 20–24 weeks, MLCCs
20–30 → 10–14 weeks), those duplicate orders were cancelled and customers ran
down inventory instead of buying. The industry calls this the 2023 inventory
correction; it ran until the inventory overhang cleared in late Q3/Q4 2023.

**The match to this data is close to exact.** Mitsubishi Electric's own Factory
Automation Systems segment reported **orders down 43% year-on-year in the
June-2023 quarter**, worsening from −25% in the March-2023 quarter, with the
company citing weak investment in semiconductor/digital and machine-tool
industries and a weak Chinese economy, and expecting inventory adjustment to
continue through its fiscal 2024 (Apr 2023 – Mar 2024).

That window — Apr 2023 to Mar 2024 — **is FY2023 in this project**, and the
group's −43% FA order decline sits alongside this portfolio's **−42.8%** unit
decline.

**Reading.** The FY2023 collapse was not a Mitsubishi Electric Thailand problem
and not a customer-churn problem. It was this portfolio's share of a global
FA/electronics inventory correction, amplified locally by the pull-forward
visible in the record March 2023 month. The apparent "loss of demand" was largely
*demand that had already been pulled into FY2022*.

---

## 3. Finding 2 — the FY2024 rebound was FA/electronics, **not** Thai automotive

This is the finding that a plausible-sounding story would have got wrong, so it
is worth stating plainly.

The customer base is commonly described as automotive/electronics, which makes
"Thailand's car industry recovered" the tempting explanation for +25.1%. **It is
false.** Thai vehicle production **fell 20% in calendar 2024** to 1.47m units, a
four-year low; domestic sales fell 26.2% to a 15-year low as auto-loan rejection
rates reached about 70% against household debt near 90% of GDP. Thailand's
overall Manufacturing Production Index fell 1.79% in 2024.

The rebound therefore came from the other side of the customer base:

- **Mitsubishi Electric group FA orders grew every quarter of Apr 2024 – Mar 2025:
  +8%, +15%, +33%, +28% YoY** — the inventory correction ending and capital
  expenditure resuming.
- Thai **PCB exports returned to growth** from June 2024 (+10.4% YoY over
  Jun–Aug, with Japan +63.2% and China +16.4%), on the cyclical upswing in global
  electronics.
- Electronics is Thailand's largest export category (27.4% of 2024 export value),
  and nearly 60 PCB manufacturers from China and Taiwan built new Thai factories
  inside three years — new FA-consuming capacity landing in Thailand.

**Reading.** FY2024's recovery is an electronics/FA capital-goods recovery that
happened *despite* a severe Thai automotive recession. Any overlay built on Thai
auto volumes would have pointed the wrong way. The right external axis for this
portfolio is **the FA / electronics capital-expenditure cycle**, not Thai vehicle
output.

---

## 4. Does an external indicator actually call the turns?

Retrospective test, using Mitsubishi Electric group FA order growth as the
indicator — the series a planner could have read at each forecast origin:

| Forecast origin | Indicator reading available at origin | Indicator says | Model did | Actual |
|---|---|---|---|---|
| Apr 2023 (→FY2023) | FA orders −25% YoY (Mar-23 qtr), worsening | **cut hard** | +45.6% too high | −42.8% |
| Apr 2024 (→FY2024) | FA orders +8% YoY (Mar-24 qtr), inventory correction ending | **raise** | −20.6% too low | +25.1% |

**Two out of two on direction, both times against the model.** With n = 2 this is
an anecdote, not a validated feature — it cannot be scored through the back-test
harness the way A1–A4 were, and §6 explains why. But it is the only signal
examined on this project that gets both turning points right, and both are turns
the models got badly wrong.

---

## 5. What the same indicator said at the FY2025 origin

The forecast origin is **April 2025** (the extract ends 2025-03-31). Restricting
strictly to what was public by then:

| Signal, as known in Apr 2025 | Reading |
|---|---|
| Mitsubishi Electric FA orders, Jan–Mar 2025 quarter | **+28% YoY** (fourth consecutive rising quarter: +8 / +15 / +33 / +28) |
| Global electronics cycle | inventory correction cleared; upcycle running |
| Thai electronics exports | largest export category; PCB capacity boom continuing |
| Thai automotive | still weak — a drag, but demonstrably not this portfolio's driver |
| Thai domestic economy | soft: GDP ~1.8–2.1% for 2025, household debt ~90% of GDP |

**The delivered model forecasts 174,514 units — an 8.9% decline against FY2024 —
in a year whose only leading indicator was pointing up.** That is the same
turning-point lag documented in §1, arriving for the third time.

### What has since been published (hindsight — validation only, not an input)

FY2025 (Apr 2025 – Mar 2026) has now closed and Mitsubishi Electric has reported
it (28 April 2026):

- **FA Systems revenue ¥798.2bn, up ¥72.6bn (+10.0%) YoY**, with orders up
  **+23% / +15% / +17% / +30%** across the four quarters — and the company guides
  FA up a further **+8.4% to ¥865.0bn** for Apr 2026 – Mar 2027.
- Thailand's own MPI, by contrast, **fell 0.78% in 2025** (a strong baht eroding
  export competitiveness) — but **electronic components and boards rose 10.52%**
  and automotive recovered +5.02%.

Both reads point the same way as the April-2025 indicator did: the FA/electronics
demand this portfolio serves expanded through FY2025 while the broad Thai
industrial economy stagnated.

---

## 6. Why this cannot be fitted as a model feature

For completeness, because the obvious next question is "so add it as a regressor":

- The overlay axis is **annual**. There are **four** fiscal years of history and
  **two** back-test folds. A macro regressor would be fitted on effectively 2–4
  observations — unscoreable through the existing harness, and guaranteed to look
  brilliant in-sample.
- The project rule is that nothing is adopted on plausibility; every candidate is
  scored on a withheld year. **This candidate cannot be scored.** Adopting it as a
  fitted feature would break the rule that has governed every other decision here
  — the same rule that correctly rejected A1, A2, hierarchical reconciliation and
  quantile stocking.
- Monthly external series exist (Thai MPI, PMI) and would give ~48 points, but
  they measure the *Thai* industrial economy — which §3 shows moved opposite to
  this portfolio in FY2024. Fitting to the wrong axis is worse than not fitting.

**Therefore the honest use is judgemental and documented — a stated scenario
weight on the portfolio total, not a hidden coefficient.** That is what §7 offers.

---

## 7. What this implies for the delivered number — options, not a decision

The published FY2025 point forecast and range are unchanged by this document.
For reference, against the portfolio FY2024 actual of 191,592 units
(net-of-VIP realised price ≈ ฿8,373/unit):

| Route | Basis | FY2025 units | Revenue (net) | vs FY2024 |
|---|---|---:|---:|---:|
| **Model as delivered** | pipeline | **174,514** | **฿1,461.1m** | −8.9% |
| Judgmental cross-check | `JUDGMENTAL_FORECAST.md` | 183,000 | ฿1,532m | −4.5% |
| Flat | no growth | 191,592 | ฿1,604m | 0.0% |
| External-context scenario | FA market +10% | 210,751 | ฿1,765m | +10.0% |
| Model bias-corrected | undo the −20.6% FY2024 fold lag | 219,902 | ฿1,841m | +14.8% |

**Published range: 118,093 – 215,741 units.**

Three observations worth presenting exactly as they stand:

1. **The range holds.** Every external scenario except the extreme
   bias-correction lands inside the published interval. The interval was built
   from internal evidence alone and it survives contact with external reality —
   that is a genuine result, and the strongest thing this analysis says about the
   deliverable.
2. **The point forecast sits low in its own range**, and every independent route
   — judgmental, flat, external, bias-corrected — sits *above* it. Four routes
   pointing one way is a pattern, not noise.
3. **Two independent routes converge on ~210–220k**: the external FA market read
   (+10% → 210,751) and simply undoing the model's measured FY2024 turning-point
   lag (219,902). They share no inputs.

### The three options

| Option | What it does | Cost |
|---|---|---|
| **A. Explain only** | Headline stays 174,514; no second number. | None. |
| **B. Publish a scenario band** | Headline stays 174,514, and a named, sourced external-context scenario is published beside it. | The deck must carry two numbers and explain the difference. |
| **C. Move the headline** | Apply an explicit uplift to the published figure. | Judgemental override of a measured pipeline; cannot be back-tested; forfeits the "nothing adopted on plausibility" rule. |

---

## 7a. DECIDED 2026-08-24 — Option B. The scenario band

**Paul's decision: publish the scenario alongside the forecast.** Both numbers go
on the record; the model's number remains the deliverable.

### The two published numbers

| | **Model forecast** *(the deliverable)* | **External-context scenario** *(the cross-check)* |
|---|---:|---:|
| FY2025 demand | **174,514 units** | **211,000 units** |
| Revenue, net of VIP 37% | **฿1,461.1m** | **฿1,766.7m** |
| Revenue, Base-Price terms | ฿1,903.8m | ฿2,301.8m |
| vs FY2024 actual (191,592) | −8.9% | +10.1% |
| Position in published range | lower half | 97.8% of the upper bound |

**Published range 118,093 – 215,741 units is unchanged and covers both.**

### How the scenario is constructed — one line, fully auditable

```
FY2024 actual portfolio demand  191,592 units
  x  FA market growth in the same window  +10.0%
  =  210,751  ->  rounded to 211,000 units
```

`+10.0%` is Mitsubishi Electric group Factory Automation Systems revenue growth for
Apr 2025 – Mar 2026 (¥725.6bn → ¥798.2bn), reported 28 April 2026. Revenue is the
scenario quantity multiplied by this project's own realised prices — ฿8,373/unit net
of the VIP discount, ฿10,909/unit in Base-Price terms — the same prices used for the
model headline, so the two columns differ only in quantity.

### The rules this scenario is published under

1. **It does not touch the pipeline.** No file in `src/`, `config.yaml` or
   `outputs/` changes. The scenario exists in this document only. `outputs/` remains
   derived exclusively from the three supplied files, exactly as §0 states.
2. **The model number stays the deliverable.** The scenario is labelled a
   cross-check, never a forecast. If asked "what is your number", the answer is
   174,514.
3. **Its information basis is stated openly.** The +10.0% is a *published outturn*
   for the forecast window, i.e. it uses information that post-dates the April-2025
   forecast origin. It is a market-direction check, not a prediction, and the deck
   must say so. The origin-only read (FA orders +28% YoY in the Jan–Mar 2025
   quarter, §5) pointed the same direction with the same order of magnitude — that
   is what makes the scenario a fair check rather than hindsight decoration.
4. **It is not a substitute for the grading actuals.** Group FA revenue in yen is
   not Mitsubishi Electric Thailand unit sales. See §8.

### Why this is the right call

- If the actuals land near 174,514, the model forecast is on the record and correct,
  and the scenario is a documented alternative that was honestly labelled.
- If the actuals land high, the project demonstrably saw it coming and said so in
  advance, with sources.
- Either way the **published range holds**, which is the claim the project actually
  made.

### The Shock Simulator (dashboard page, added 2026-08-25)

The scenarios above are wired into an interactive **Shock Simulator** page on
the dashboard. It applies a chosen demand shock (the FY2023-style −43% crash,
the FY2024-style +25% rebound, the FA +10% read, or a custom size, with a
selectable onset month) to the FY2025 plan and runs the same order-up-to stock
simulation used by the Inventory Planning page — stock held to the *plan*,
demand arriving *shocked*. At stock factor 1.0x it shows the business problem
in one screen: a +25% rebound costs ~43,600 missed units (~฿365m); a −43%
crash leaves ~6,400 units of average surplus. It is a what-if view only —
nothing on it feeds the pipeline or the published numbers.

### For the semi-final deck (2026-09-07)

One slide, three beats:

1. **"We could not explain the two turning points from the data — so we went and
    found out."** FY2023 −42.8% next to Mitsubishi Electric's own FA orders −43%.
    The audience recognises their own company's numbers.
2. **"The obvious answer was wrong."** Thai car production fell 20% in the year our
    demand rose 25%. The driver is the FA/electronics capex cycle, not automotive.
    This is the credibility beat — it shows the analysis was tested, not assumed.
3. **"So here is our number, and here is what the market says."** 174,514 and
    211,000, both inside the range we published. State plainly that the model lags
    turning points by construction (+45.6% / −20.6% on the two folds) and that this
    is why the second number exists.

---

## 8. What is explicitly NOT claimed

- Not claimed: that Mitsubishi Electric group FA revenue predicts Mitsubishi
  Electric **Thailand's** unit sales. The group figure is global, in yen, and
  includes price and exchange-rate effects. It is a **directional** indicator for
  the demand cycle this portfolio sits inside — nothing more.
- Not claimed: that any of this is a fitted, validated, or back-tested result.
  §4 rests on n = 2. §6 states why it cannot be more.
- Not claimed: any knowledge of the FY2025 actuals used for grading. Those are
  Mitsubishi Electric Thailand's own order lines and appear in no public source.
  The published figures in §5 are the *group's* segment results, a different
  quantity entirely.
- Not used: any external number inside `src/`, `config.yaml`, or `outputs/`.

---

## 9. Sources

**Mitsubishi Electric**
- Consolidated Financial Results Briefing for Fiscal 2026 (28 Apr 2026) — FA
  Systems revenue ¥798.2bn (+¥72.6bn), FY27 forecast ¥865.0bn, quarterly FA order
  trends: https://www.mitsubishielectric.com/en/pr/2026/pdf/0428_co2.pdf
- Consolidated Financial Results Briefing for Fiscal 2025 (28 Apr 2025):
  https://www.mitsubishielectric.com/en/pr/2025/pdf/0428-a2.pdf
- Fiscal 2024 H1 results (30 Oct 2023) — FA weakness, inventory adjustment:
  https://www.businesswire.com/news/home/20231030409620/en/
- Morningstar, "Weak FA Systems Sales Hit Operating Income" — FA orders −43% YoY
  in the June-2023 quarter:
  https://www.morningstar.com/stocks/mitsubishi-electric-earnings-weak-fa-systems-sales-hit-operating-income-midterm-outlook-intact

**Electronics cycle**
- Sourcengine, Q4 2023 lead-time report — inventory correction, double-ordering:
  https://www.sourcengine.com/blog/q4-2023-electronic-component-lead-time-report-highlights
- Sourcengine, Q1 2024 lead-time report — normalisation:
  https://www.sourcengine.com/blog/q1-2024-electronic-components-lead-time-report-highlights

**Thailand**
- Bangkok Post, "Manufacturing dips 1.79% in 2024":
  https://www.bangkokpost.com/business/general/2953025/manufacturing-dips-1-79-in-2024
- Bangkok Post, "Thailand car production down 20% in 2024":
  https://www.bangkokpost.com/business/motoring/2949236/cars
- Bangkok Post, "Auto loan rejections could hit 40%":
  https://www.bangkokpost.com/business/general/2803036/auto-loan-rejections-could-hit-40-
- Krungsri Research, Industry Outlook: Printed Circuit Boards — PCB export
  recovery from Jun 2024:
  https://www.krungsri.com/en/research/industry/industry-outlook/hi-tech-industries/printed-pircuit-boards/io/pcb-2024
- Nation Thailand — PCB investment wave, ~60 new factories:
  https://www.nationthailand.com/business/tech/40054942
- OIE (Office of Industrial Economics) — MPI releases:
  https://www.oie.go.th/view/1/industrial_indices/EN-US
- OECD Economic Survey: Thailand 2025 — GDP ~1.8–2.1%, household debt ~90% of GDP:
  https://www.oecd.org/en/publications/oecd-economic-surveys-thailand-2025_426b9bc0-en.html

*All external figures are public company reports, government statistics, or trade
press. None of Mitsubishi Electric Thailand's supplied data appears in any of them.*
