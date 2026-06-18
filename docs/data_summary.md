# Frostfire data — catalog and reference

Definitive reference to the data the Frostfire StatsPlus API exposes. Built by inspecting every file directly. Companion to `docs/model_spec.md`. Self-contained for fresh conversations.

---

## TL;DR

- **League:** Frostfire, OOTP Baseball on StatsPlus at `https://atl-02.statsplus.net/frostfire/`
- **API base:** `https://atl-02.statsplus.net/frostfire/api/`
- **Puller:** `src/data/puller.py` writes to `frostfire_data/`
- **League size:** 22 major-league teams (custom, not MLB-equivalent), each with 4 minor-league affiliates. Two of the 22 are custom teams: Montreal Spinners (`team_id=301`) and Pallet Town Pikachus (`team_id=302`).
- **User's team:** Washington Nationals, `team_id=30`, `park_id=32`
- **Current league date:** 2035 (mid-late season at last pull)
- **FA eligibility:** 5 years MLB service (Frostfire rule — NOT MLB's 6). Arbitration after 3 years. No qualifying offers. No compensation picks for unsigned FAs. Rule 5 Draft: max 5 rounds.
- **Historical depth:** 21 years of stat files (2015–2035), platoon splits for batting/pitching
- **Contracts depth:** Only currently active contracts. Effective meaningful data starts from the 2031 offseason (since 5-year max means anything signed 2031+ that hasn't expired is still in the file)
- **Active ML players:** 643. FA-eligible (**5+ service years** — Frostfire rule, `mlb_service_years >= 5` on the current snapshot): **212** as of the last pull — verified directly against `frostfire_data/players.csv` (2026-06-17) and matches the 212-player evaluation population in `intermediate/recommendations.csv`. (An earlier draft of this doc left a stale placeholder of "~141+" computed with the wrong `>= 6` threshold; corrected here.)

---

## League structure

### Real major-league teams (22 with parks)

From `ballparks.json` and `teams.csv` cross-reference:

| team_id | park_id | Team | Abbr |
|---|---|---|---|
| 1 | 7 | Arizona Diamondbacks | AZ |
| 3 | 8 | Baltimore Orioles | BAL |
| 4 | 9 | Boston Red Sox | BOS |
| 6 | 10 | Chicago Cubs | CHC |
| 7 | 11 | Cincinnati Reds | CIN |
| 8 | 12 | Cleveland Guardians | CLE |
| 9 | 13 | Colorado Rockies | COL |
| 13 | 16 | Kansas City Royals | KC |
| 16 | 19 | Milwaukee Brewers | MIL |
| 17 | 20 | Minnesota Twins | MIN |
| 18 | 21 | New York Yankees | NYY |
| 20 | 72 | Las Vegas Aces | LV |
| 21 | 23 | Philadelphia Phillies | PHI |
| 23 | 25 | San Diego Padres | SD |
| 24 | 26 | Seattle Mariners | SEA |
| 25 | 27 | San Francisco Giants | SF |
| 26 | 28 | St. Louis Cardinals | STL |
| 27 | 29 | Tampa Bay Rays | TB |
| 29 | 31 | Pittsburgh Pirates | PIT |
| 30 | 32 | **Washington Nationals (user's team)** | WSH |
| 301 | 240 | Montreal Spinners (custom) | MTL |
| 302 | 241 | Pallet Town Pikachus (custom) | PTP |

`team_id` is non-sequential (no 2, 5, 10–12, 14–15, etc.). Do not assume consecutive IDs.

### Other "teams" in `teams.csv`

The file has 123 rows total. Beyond the 22 ML teams, you'll find:
- ~88 minor-league affiliates (4 per ML parent)
- ~12 All-Star game teams (IDs 31, 32, 120, 121, 184, 185, 225, 226, 254, 255 etc.) — these have `Parent Team ID = 0` but are not real teams. They show up when stats are aggregated across All-Star events.

**When filtering player stats to "real ML players," exclude `team_id` values that are All-Star teams.** The clean filter is: join to `teams.csv`, keep only rows where the team is in the 22-team list above. Alternatively, only keep stat rows where `team_id` corresponds to a `league_id=203, level_id=1` team that also appears in `ballparks.json`.

### League/level conventions

| Field | Value | Meaning |
|---|---|---|
| `league_id=203` | majors | The only league you can pull stats for via the public API |
| `level_id=1` | majors | The only level exposed |
| `level_id=2,3,4...` | minors | These exist in the schema but the API ignores `level_id` as a filter — no minor-league stats are accessible |

---

## Player bio (`players.csv`)

**Size:** 136,291 rows (every player ever — alive, retired, draftees). 45 columns. ~13 MB uncompressed.

### Full schema

```
ID, First Name, Last Name, Team ID, Parent Team ID, Level, Pos, Role, Age,
Retired, Organization ID, League ID,
date_of_birth, height, weight, bats, throws,
draft_year, draft_round, draft_supplemental, draft_pick, draft_overall_pick,
hall_of_fame, inducted, uniform_number,
is_active, is_on_secondary, is_on_waivers, designated_for_assignment,
is_on_dl, is_on_dl60, dl_days_this_year,
mlb_service_years, mlb_service_days, mlb_service_days_this_year,
pro_service_years, pro_service_days, pro_service_days_this_year,
secondary_service_years, secondary_service_days, secondary_service_days_this_year,
days_on_waivers, days_on_waivers_left,
has_received_arbitration, was_traded
```

### Codes verified from data

**`bats` (batting handedness):**
- `1` = Right (986 active ML players)
- `2` = Left (~459)
- `3` = Switch (~137)

**`throws` (throwing handedness):**
- `1` = Right (~1249 active ML players)
- `2` = Left (~375)

**`Pos` (primary position):**
- `1` = Pitcher (892 active ML)
- `2` = Catcher (126)
- `3` = 1B (66)
- `4` = 2B (87)
- `5` = 3B (77)
- `6` = SS (110)
- `7` = LF (77)
- `8` = CF (121)
- `9` = RF (63)
- `10` = DH (5)

**`Role`:**
- `0` = position player (731 active)
- `11` = starting pitcher (576)
- `12` = middle reliever (292)
- `13` = closer (25)

**`Retired`, `is_active`:** Booleans (0/1). Use `is_active=1 AND Level=1 AND Retired=0` to get current ML roster.

### Key columns and their use

- `ID` — joins to all stat files as `player_id`
- `Team ID` — current team. `0` means free agent / no team (verify this with a known FA)
- `Level` — `1` = majors, higher = minor levels
- `Age` — current age as of the league date; for historical seasons compute from `date_of_birth`
- `date_of_birth` — YYYY-MM-DD format (e.g., `1983-03-06`)
- `mlb_service_years` — fractional years of MLB service. **FA-eligible if ≥ 5 (Frostfire rule).** Arbitration-eligible after 3 years.
- `mlb_service_days_this_year` — days accrued this season (use to project end-of-season service time)
- `has_received_arbitration` — 1 if the player has hit arbitration (3–5 service years range in Frostfire)
- `was_traded` — has this player ever been traded
- `is_on_dl`, `is_on_dl60`, `dl_days_this_year` — **point-in-time only.** There is no historical injury log.
- `hall_of_fame`, `inducted` — HOF status if applicable
- `draft_year`, `draft_overall_pick` — pedigree features (high picks may signal upside)

### Important nuances

- The 643 active ML players is much smaller than `players.csv` (136K rows) because the file includes every player who ever existed. Always filter.
- For computing age in a historical season: `season_year - year(date_of_birth)` (approximate; off by 1 for players born late in the year).
- A player with `Team ID = 0` and `is_active = 1` is likely a current free agent.

---

## Per-player batting (`player_batting_YYYY*.csv`)

**One row per player-stint per season** (a player traded mid-season has multiple rows distinguished by `stint`).

### Schema (34 columns)

```
id, player_id, year, team_id, game_id, league_id, level_id, split_id,
position, ab, h, k, pa, pitches_seen, g, gs,
d, t, hr, r, rbi, sb, cs, bb, ibb, gdp, sh, sf, hp, ci,
wpa, stint, ubr, war
```

### Row counts (representative)

- `player_batting_2035.csv` (overall): 1,197 rows
- `player_batting_2035_vsLHP.csv`: 395 rows (only qualifying PAs vs LHP)
- `player_batting_2035_vsRHP.csv`: 401 rows

### Key columns

- `player_id`, `team_id`, `year` — primary identifiers
- `split_id` — `1`=overall, `2`=vs LHP, `3`=vs RHP. **All three splits are embedded in `player_batting_YYYY.csv`** — the `_vsLHP` and `_vsRHP` files are redundant subsets; load only the main file. There are also `split_id=21` rows (undocumented, dropped by the foundation loader).
- `stint` — increments when a player changes teams mid-season. Aggregate by `(player_id, year)` summing counting stats. **Batting grain: `(player_id, year, stint, split_id)`.**
- `position` — primary offensive position for this stint (uses same numbering as `Pos` above)
- `war` — OOTP's WAR computation (offense + baserunning included)
- `ubr` — Ultimate baserunning runs (a sub-component of `war`)
- `wpa` — Win Probability Added (situational stat; not used in v1 model)

### Component decomposition (for the model)

| Bucket | Columns |
|---|---|
| Power | `d`, `t`, `hr` (and `pa` as denominator for rates) |
| Contact / eye | `bb`, `k`, `ibb`, `hp`, `h`, `pa` |
| Baserunning | `sb`, `cs`, `ubr` |
| Plate appearances volume | `pa`, `ab`, `g`, `gs` |

---

## Per-player pitching (`player_pitching_YYYY*.csv`)

**One row per pitcher-stint per season.**

### Schema (59 columns)

```
id, player_id, year, team_id, game_id, league_id, level_id, split_id,
ip, ab, tb, ha, k, bf, rs, bb, r, er, gb, fb, pi, ipf, g, gs,
w, l, s, sa, da, sh, sf, ta, hra, bk, ci, iw, wp, hp, gf, dp,
qs, svo, bs, ra, cg, sho, sb, cs, hld, ir, irs, wpa, li, stint,
outs, sd, md, war, ra9war
```

### Row counts (representative)

- `player_pitching_2035.csv` (overall): 1,140 rows
- `player_pitching_2035_vsLHB.csv`: 380 rows
- `player_pitching_2035_vsRHB.csv`: 380 rows

### Key columns

- **Grain: `(player_id, year, team_id, split_id)`** — the `stint` column is always 0 in pitching files and does NOT increment on mid-season trades. Use `team_id` to differentiate stints (same as fielding). All three splits (overall + vs LHB + vs RHB) are embedded in `player_pitching_YYYY.csv`; the `_vsLHB`/`_vsRHB` files are redundant. There are also `split_id=21` rows (undocumented, dropped by the foundation loader).
- `bf` — batters faced (use as denominator for component rates)
- `ip` — innings pitched (decimal: `8.2` means 8⅔ innings, not 8.2)
- `outs` — innings × 3 as a clean integer (more reliable for math than `ip`)
- `gs`, `g` — games started vs total games. **Use `gs/g` ratio to classify starter vs reliever.**
- `qs`, `svo`, `s`, `bs`, `hld` — quality starts, save opportunities, saves, blown saves, holds (role classification + leverage)
- `war` — FIP-based WAR (skill-based)
- `ra9war` — RA9-based WAR (results-based; includes defense behind the pitcher)
- `li` — leverage index (average game state when pitcher entered)

### Component decomposition (for the model)

| Bucket | Columns |
|---|---|
| Strikeout rate | `k` / `bf` |
| Walk/HBP rate | `(bb + hp)` / `bf` |
| HR suppression | `hra` / `bf` |
| Batted-ball profile | `gb`, `fb` (groundball/flyball ratio for splash) |
| Workload | `ip`, `outs`, `g`, `gs` |
| Role classification | `gs/g`, `svo`, `hld`, `Role` from players.csv |

---

## Per-player fielding (`player_fielding_YYYY.csv`)

**One row per player-position per season.** A player who logged innings at 2B and SS gets two rows that year. **No splits exist for fielding.**

### Schema (40 columns)

```
id, player_id, year, team_id, league_id, level_id, split_id,
position, tc, a, po, er, ip, g, gs, e, dp, tp, pb, sba, rto, ipf,
plays, plays_base, roe,
opps_0, opps_made_0, opps_1, opps_made_1, opps_2, opps_made_2,
opps_3, opps_made_3, opps_4, opps_made_4, opps_5, opps_made_5,
framing, arm, zr
```

### Row counts (representative)

- `player_fielding_2035.csv`: 1,238 rows (more than 1,197 batting rows because multi-position players generate multiple rows)

### Key columns

- `position` — fielding position number (1=P, 2=C, etc.)
- `ip` — innings at this position (the weight for aggregation)
- `zr` — zone rating, the headline fielding value
- `framing` — catcher framing runs; **zero for non-catchers** (this is how to identify a framing-relevant row)
- `arm` — catcher arm + outfielder arm runs
- `pb`, `sba`, `rto` — passed balls, stolen base attempts, runners thrown out (catcher-specific defensive stats)
- `tc`, `a`, `po`, `e`, `dp`, `tp` — total chances, assists, putouts, errors, double plays, triple plays
- `opps_N` / `opps_made_N` — defensive opportunities by difficulty tier (granular fielding context)
- `plays`, `plays_base`, `roe` — plays made, baseline expected plays, reached on error

### Aggregation rule (per the model spec)

For a player's defense bucket: weight `zr` (and `arm` where applicable) by `ip` across all positions, sum, then apply standard positional adjustments (catcher and SS positive, 1B/DH negative). Treat `framing` as a separate fifth component, fit on catcher-seasons only, zero elsewhere.

---

## Team-level batting (`team_batting_YYYY.csv`)

**One row per team per year.** 34 columns. Useful for league-average baselines and team-context features.

### Schema

```
name, tid, abbr,
pa, ab, h, k, tb, s, d, t, hr, sb, cs, rbi, r, bb, ibb, hp, sh, sf, ci, gidp, xbh,
avg, obp, slg, ops, iso, k_pct, bb_pct, babip, woba,
split_id
```

`tid` joins to `teams.csv ID`. Includes pre-computed rate stats: `avg`, `obp`, `slg`, `ops`, `iso`, `babip`, `woba`. **Includes minor-league teams too** — filter by `league_id` (not exposed in this file directly; use the teams.csv join to filter to majors).

---

## Team-level pitching (`team_pitching_YYYY.csv`)

**One row per team per year.** 46 columns.

### Schema

```
name, tid, abbr,
ip, ab, tb, ha, k, bf, bb, r, er, gb, fb, pi, ipf, sa, d, sh, sf, t, hra, bk, ci, iw, wp, hp,
s, bs, cg, outs,
era, lob, k_pct, bb_pct, k_bb_pct, fip, x_fip, e_f, babip, gbfb, hrfb, hr_pct, avg, obp,
split_id
```

Pre-computed rate stats include `era`, `fip`, `x_fip`, `e_f` (ERA-FIP differential), `babip`, `gbfb` (GB/FB ratio).

---

## Contracts (`contracts.csv`) — important corrected understanding

**This is the file most likely to be misunderstood. Read this section carefully.**

### Schema (40 columns)

```
player_id, team_id, league_id, is_major, no_trade,
last_year_team_option, last_year_player_option, last_year_vesting_option,
next_last_year_team_option, next_last_year_player_option, next_last_year_vesting_option,
contract_team_id, contract_league_id, season_year,
salary0, salary1, salary2, salary3, salary4, salary5, salary6, salary7,
salary8, salary9, salary10, salary11, salary12, salary13, salary14,
years, current_year,
minimum_pa, minimum_pa_bonus, minimum_ip, minimum_ip_bonus,
mvp_bonus, cyyoung_bonus, allstar_bonus,
next_last_year_option_buyout, last_year_option_buyout
```

### Structure (the gotchas)

`contracts.csv` has 7,381 rows. **NOT** one row per active contract — the structure is:

1. **6,448 rows are placeholders** (`season_year=0`, `current_year=0`, mostly minor-league entries with no salary).
2. **The remaining ~933 rows are actual active contracts**, with one row per contract.
3. `is_major=1` filters to ML contracts (~2,539 rows, but only ~925 with `season_year > 0` are real contracts).

**Each real contract row represents the whole contract**, not one year of it. The fields work like this:

- `season_year` = the year the contract was **signed** (the start year)
- `current_year` = years elapsed since signing (`0` = signed this year, `1` = second year, etc.)
- `years` = total contract length
- `salary0` = first-year salary (year `season_year`), `salary1` = second year, etc.

So `season_year + current_year` always equals the current league season (2035 in the last snapshot). A contract from 2031 (`season_year=2031`) that's currently 4 years into a 5-year deal has `current_year=4` and `years=5`.

### Implication for the contract regression (#3)

**The correct filter for "FA signing events" is:**

```
is_major = 1
AND season_year > 0
AND player_id NOT IN contract_extensions.csv
AND mlb_service_years_at_signing >= 5  (Frostfire FA threshold; computed from players.csv minus elapsed years)
```

The `salary0` column is your AAV target (or use the mean of non-zero `salary0..salary{years-1}` for true AAV).

**Note:** A previous draft of the model spec incorrectly suggested filtering to `current_year=1` for signing events. That's wrong — each contract appears exactly once regardless of current_year, and the signing year is `season_year`.

### Empirical signing-event counts

Actual active contracts by signing year (verified):

| Signing year | Contracts visible | Likely interpretation |
|---|---|---|
| 2031 | 1 | One 5-year deal still in force |
| 2032 | 9 | 4-year and 5-year deals |
| 2033 | 20 | 3-year+ deals |
| 2034 | 44 | 2-year+ deals |
| 2035 | 852 | All deals signed this year (most are 1-year) |

**Breakdown of all active multi-year deals (2+ years):**
- 2-year: 33
- 3-year: 28
- 4-year: 20
- 5-year: 44
- **Total multi-year deals across all 5 cohorts: 125**

The 801 one-year deals signed in 2035 include many league-minimum / replacement-level signings (586 are under $1M).

### The training-set reality

After joining service time and filtering to true FAs (`mlb_service_years >= 5` — Frostfire rule), the actual #3 training set will be far smaller than the raw row count suggests. **Expect the order of 100–200 high-quality training rows.** This is the model's primary fragility — use ridge regularization aggressively and validate with leave-one-season-out CV.

### ✅ Retired (2026-06-17): `contracts.csv` is survivorship-biased — transaction log now used instead

**Confirmed empirically:** the `contract` API endpoint ignores `?year=`, `?season=`, `?season_year=`, and `?current_year=` entirely — every query returns the byte-identical live snapshot (see `probe_results/probe_report.txt`). There is no way to query `contracts.csv` for a past state. This means a contract row only exists if it hasn't expired yet as of the 2035 pull date, which severely survivorship-biases early signing cohorts toward long deals only (2031: only 1 of however many were actually signed is visible, because only a 5-year deal would still be active; 2032: only 4+ year deals are visible; etc. — see the table in `step8_transaction_log_rework.md`).

**The fix (implemented):** `src/data/transactions_parser.py` parses OOTP's own in-game transaction log (rendered to static HTML and saved at
`C:\Users\Felto\Documents\Out of the Park Developments\OOTP Baseball 26\saved_games\Frostfire Baseball League.lg\news\html\leagues\league_203_all_transactions_{M}_{YYYY}.html`)
into `intermediate/fa_signings_log.csv`, capturing every signing event with no survivorship bias (player_id and team_id recoverable from hrefs, contract years + total $ in the text). Steps 3 and 5 have been rebuilt on this source, 6 and 7 re-run, and step 8 (nested k-fold validation) has been built and run. **See `step8_transaction_log_rework.md` for the original rebuild plan and `CLAUDE.md`'s build-progress section for results.**

`fa_signings_log.csv` columns: `player_id`, `player_name`, `team_id`, `team_name`, `position` (text abbreviation, e.g. `SP`/`2B`/`CL`, not the numeric position codes used elsewhere), `file_year`, `file_month` (calendar month/year the transaction was logged), `signing_year` (the season the contract is *for* — see mapping below), `years`, `total_value`, `aav` (`total_value/years`, used as the `salary0` proxy since the log has no year-by-year salary breakdown), `is_extension`, `human_era` (`signing_year >= 2031`; pre-2031 was AI-controlled, not human market behavior — confirmed with owner). The parser writes every row it finds and does NOT drop any of them; the `human_era` flag lets downstream steps decide what to exclude.

**✅ Full history now available (confirmed 2026-06-17):** the OOTP saved-game news folder now has a transaction-log HTML file for every month from **March 2015 through October 2035 with no gaps** (the one apparent gap, September 2027, was confirmed present on re-check — the file exists). Re-running `src/data/transactions_parser.py` against the full set picks up **251 files → 4,130 signing rows (1,521 FA signings + 2,609 extensions)**, spanning `signing_year` 2015–2036 (a handful of rows are pre-signed for the 2036 season from late-2035 files). Of the 1,521 FA signings, 734 are `human_era=True` (signing_year ≥ 2031, the same set used by steps 3/5/8 today) and 787 are `human_era=False` (the pre-2031 "AI days" — previously parsed but never used downstream). This is a large jump from the previous working set (72 files / 798 rows, 2030–2035 only) — the AI-era rows (2015–2030) were always being written by the parser, just never consumed by steps 3/5/8 because of the `human_era` exclusion.

**✅ Resolved (2026-06-17, see CLAUDE.md step 10 for the full investigation): `human_era=False` data is NOT incorporated into training, by decision, not by default.** The owner's proposal (weight AI-era rows less heavily, validate only on the human-era window) was implemented and swept (`research/step10_ai_era_weight_sweep.py`), then diagnosed further (`research/step10_ai_era_diagnose.py`) after the first sweep was unexpectedly negative. Two real mechanisms were found and partially/fully fixed along the way — an eligibility-filter bug (see below) and an alpha-tuning scope bug — but even after fixing both and giving the model an `is_ai_era` dummy + slope term to absorb a possible AI/human price-level difference, blending in AI-era rows at any weight never beat leaving them out. **Decision: AI-era signings are excluded from training, confirmed via exhaustive testing rather than left untried.**

**Side discovery from this investigation — a real, separate bug, now fixed in production:** the FA-eligibility filter (`service_at_signing >= 6`, used in steps 3/5/8) computed service time as the player's *current* (2035 snapshot) `mlb_service_years` minus years elapsed since the signing. That formula assumes continuous service accrual all the way to 2035, which silently breaks for any player who has since retired — their `mlb_service_years` counter freezes, so subtracting elapsed time overcounts and pushes `service_at_signing` deeply negative, dropping the row. This was misclassifying human-era rows too, not just AI-era ones (only 37 of 787 AI-era signings were surviving the filter before the fix, but the human-era pool also grew from 210 → 280 once corrected). **Fixed by deriving `service_at_signing` from each player's actual MLB appearance history** (distinct years with a qualifying row in `batting_raw.csv`/`pitching_raw.csv`/`fielding_raw.csv`) instead of the snapshot — robust to retirement since it never depends on present-day state. Propagated into `src/pipeline/step3_dollar_curve.py`, `src/pipeline/step5_market_regression.py`, `src/pipeline/step8_validation.py`. New step 8 baseline: 18.6% within±15% (was 16.7%), R²=0.568 (was 0.503).

**`signing_year` mapping (derived, not directly in the HTML):** a contract signed in month ≤ 6 of calendar year Y is mapped to `signing_year = Y` (spring signing, for the Y season); month > 6 of Y maps to `signing_year = Y+1` (post-season offseason signing, for the Y+1 season). This matches `contracts.csv`'s `season_year` semantics and was confirmed empirically — Dec/Jan/Feb show a sharp signing-volume spike (the real offseason), while Mar–Nov are near zero, and the spike's mid-year split point lines up with this rule.

### Bonus structure

- `mvp_bonus`, `cyyoung_bonus`, `allstar_bonus` — performance escalators baked into the contract. Useful as proxies for star-status indicators in #3.
- `minimum_pa` + `minimum_pa_bonus` — playing-time guarantee (PA threshold + bonus dollars)
- `minimum_ip` + `minimum_ip_bonus` — same for pitchers
- `*_option` columns — team/player/vesting options on the final years (rare in this league)
- `*_buyout` — buyout amounts if an option isn't exercised

---

## Contract extensions (`contract_extensions.csv`)

**Same 40-column schema as `contracts.csv`.** Currently contains 16 active extensions:
- 1 signed in 2035
- 15 signed in 2036 (the upcoming season — extensions can be pre-signed)

**Strictly excluded from the #3 training set.** Extensions carry a pre-FA discount that biases AAV predictions downward if pooled. Keep them in a separate model or analytical track.

---

## Ballparks (`ballparks.json`)

**JSON-formatted** (~7 KB). One entry per ballpark for each of the 22 ML teams. Park factors are simple multipliers (1.04 means 4% above league average for that event).

### Per-park fields

```
team_id, league_id, park_id, name, nickname, display_name, abbr,
avg_r, avg_l, avg,
d, t, hr_r, hr_l, hr,
capacity, stadium_type, surface
```

- `avg_r`, `avg_l` — overall batting-average factor vs RHP / LHP
- `avg` — blended overall factor (weighted by handedness mix)
- `d`, `t` — doubles, triples factors (not handedness-split)
- `hr_r`, `hr_l` — home run factor vs RHP / LHP
- `hr` — blended HR factor
- `stadium_type` — `Outdoor`, `Dome`, `Retractable Roof`
- `surface` — `Grass`, `Artificial Turf`

### Nationals Stadium (`park_id=32`) — the user's home park

| Factor | Value | Interpretation |
|---|---|---|
| `avg_r` | 0.97 | Slightly hit-suppressing vs RHP |
| `avg_l` | 1.02 | Slightly hit-boosting vs LHP |
| `avg` | 0.9875 | Slightly pitcher-friendly overall |
| `d` | 1.00 | Neutral for doubles |
| `t` | 0.92 | Suppresses triples |
| `hr_r` | 1.02 | Slight HR boost vs RHP |
| `hr_l` | 1.00 | Neutral HR vs LHP |
| `hr` | 1.013 | Slight HR boost overall |
| `stadium_type` | Outdoor | |
| `surface` | Grass | |

Nationals Stadium is **mildly pitcher-friendly for batting average, mildly hitter-friendly for HRs, suppresses triples.**

### Notable park extremes

- **Colorado Rockies (park_id=13):** Extreme hitter's park. `avg=1.07`, `t=1.35`, `hr=1.07`. Inflates everything.
- **Cincinnati Reds (park_id=11):** Big HR park. `hr=1.18`, especially vs LHP (`hr_l=1.24`).
- **Pallet Town Pikachus (park_id=241):** Custom park, hitter-friendly (`hr=1.115`).
- **Seattle Mariners (park_id=26):** Pitcher's park. `avg=0.96`, `t=0.80`.
- **Boston Red Sox (park_id=9):** Doubles haven. `d=1.12`, `avg=1.05`.
- **Arizona Diamondbacks (park_id=7):** Triples park. `t=1.17`, suppresses HRs (`hr=0.91`).

### Park-adjustment approach

A player plays roughly half their games at home and half on the road. The standard approximation:

```
effective_park_factor = (home_park_factor + 1.0) / 2
```

(assumes road games average to neutral, which is approximately true across a balanced schedule). To park-neutralize a player's stats, divide by `effective_park_factor`. To re-apply Nationals Stadium for the value calculation, multiply by `(home_NS_factor + 1.0) / 2`.

For handedness-aware adjustment: use `hr_r` for RHB and `hr_l` for LHB; use the blended `hr` for switch hitters or when batter handedness is unknown.

---

## Draft (`draft.json`)

**CSV-formatted despite the `.json` extension.** Contains the current draft year's picks. ~300+ rows.

### Schema (13 columns)

```
ID, Round, Pick In Round, Supp, Overall, Player Name, Team, Team ID,
Position, Age, College, Auto Pick, Time (UTC)
```

`ID` joins to `players.csv ID`. The `Position` field uses position abbreviations here (`SP`, `RP`, `SS`, etc.) — different from the numeric codes in stat files. `College` field is an integer flag (`1` = college, `0` = high school / international).

**Usefulness for the model:** High-round / high-overall picks may signal scouting upside the projection misses. Could add `draft_overall_pick` as a feature in #3, but minimum priority for v1.

---

## Auxiliary snapshot files

### `teams.csv`

123 rows, 4 columns: `ID, Name, Nickname, Parent Team ID`. Joins to all `team_id` references. Use `Parent Team ID = 0` to filter to majors, then cross-reference with `ballparks.json` to exclude All-Star teams.

### `date.json`

10 bytes. Just a date string (e.g., `2035-05-24`). The current league date.

### `exports.json`

~800 bytes JSON. History of past data exports keyed by date, with team_id lists. Useful for understanding the league's pull cadence but not used by the model.

### `game_history.csv` and `ratings`

Both **unavailable** without authentication. `game_history` returns HTTP 401. Ratings would require a token from the StatsPlus Preferences page; the user does not have one. The puller skips both.

---

## Identifiers cheat sheet

| Field | Value | Meaning |
|---|---|---|
| `team_id = 30` | Washington Nationals | The user's team |
| `park_id = 32` | Nationals Stadium | The user's home park |
| `league_id = 203` | Majors | The only league exposed via the API |
| `level_id = 1` | Majors | The only level exposed |
| `split_id = 1, 2, 3` | Overall / vs L / vs R | Hitter splits are vs LHP/RHP; pitcher splits are vs LHB/RHB |
| `is_major = 1` | ML contract | Filter for FA modeling |
| `bats: 1, 2, 3` | R, L, Switch | |
| `throws: 1, 2` | R, L | |
| `position` (in stat files) | 1=P, 2=C, 3=1B, 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF, 10=DH | |
| `Pos` (in players.csv) | Same numbering | |
| `Role` (in players.csv) | 0=position player, 11=SP, 12=MR, 13=CL | |
| `Level` (in players.csv) | 1=majors, higher=minors | |
| `mlb_service_years ≥ 5` | FA-eligible (Frostfire rule) | |
| Active ML player filter | `is_active=1 AND Level=1 AND Retired=0` | 643 players in last pull |

---

## Confirmed unavailable

Empirically dead via the public API (verified by `probes/api_probe_round1.py` and `probes/api_probe_round2.py`):

- **Award histories** — no MVP, Cy Young, All-Star, HOF endpoints. The only award proxy is `allstar_bonus` encoded in contracts.
- **Transaction logs** — no precise FA signing date, no trade history, no waiver claims. Must reconstruct from contract structure + service time.
- **Injury logs** — only point-in-time `dl_days_this_year` in `players.csv`. No historical injury timing.
- **Minor-league stats** — `level_id` parameter is silently ignored. No prospect priors substituting for ratings.
- **Game-by-game data** — `gamehistory` endpoint requires login (HTTP 401). Season totals only.
- **Situational splits (4+)** — home/away, RISP, by month, vs starter/reliever require login. Only platoon splits 1/2/3 are publicly accessible.
- **Ratings** — accessible via `ratings` endpoint with a StatsPlus Preferences token, but the user doesn't have one. Talent estimation is stats-only.
- **Historical contracts** — only currently-active deals are in `contracts.csv`. Expired contracts are gone.

---

## Sample sizes — what's normal

For 2035 (a representative recent season):

| File | Rows | Notes |
|---|---|---|
| `player_batting_2035.csv` (overall) | 1,197 | All ML batters |
| `player_batting_2035_vsLHP.csv` | 395 | Only qualifying PAs |
| `player_batting_2035_vsRHP.csv` | 401 | Only qualifying PAs |
| `player_pitching_2035.csv` (overall) | 1,140 | All ML pitchers |
| `player_pitching_2035_vsLHB.csv` | 380 | |
| `player_pitching_2035_vsRHB.csv` | 380 | |
| `player_fielding_2035.csv` | 1,238 | More than batting because multi-position players |
| `team_batting_2035.csv` | 66 | All teams (majors + minors mixed) |
| `team_pitching_2035.csv` | 66 | |

| Snapshot file | Rows | Notes |
|---|---|---|
| `players.csv` | 136,291 | Every player ever |
| Active ML players within | 643 | `is_active=1 AND Level=1` |
| FA-eligible within active | 212 | `mlb_service_years >= 5` (Frostfire rule); verified 2026-06-17, matches `intermediate/recommendations.csv` |
| `contracts.csv` | 7,380 | Most are placeholders (off-by-one from an earlier pull's 7,381 — negligible, snapshot drifts row-by-row each pull) |
| Real ML contracts within | 926 | `is_major=1 AND season_year > 0` |
| `contract_extensions.csv` | 7,380 | Same schema, mostly placeholders |
| Real ML extensions within | 16 | Very small set |
| `teams.csv` | 123 | 22 ML + 88 affiliates + 13 All-Star "teams" |
| `ballparks.json` | 22 | One per real ML team |
| `draft.json` | ~300 | Current draft year |

---

## Join reference

```
player stats (bat/pitch/field) .player_id  ←→  players.csv .ID
                                team_id    ←→  teams.csv .ID
                                year       ←→  contracts.csv .season_year (for signing-year alignment)
contracts.csv .player_id        ←→  players.csv .ID
team_batting/pitching .tid      ←→  teams.csv .ID
draft.json .ID                  ←→  players.csv .ID
ballparks.json .team_id         ←→  teams.csv .ID
ballparks.json .park_id         →  applied at foundation layer based on team's park
```

### Specifically for the contract regression (#3) training set

**Note: the pseudocode below predates both the transaction-log rework (it still queries `contracts.csv` directly) and the step 10 service-time fix.** It's kept for illustrative join structure only — the actual implementation (`src/pipeline/step5_market_regression.py` etc.) sources from `fa_signings_log.csv` and derives `service_at_signing` from performance-panel appearance history, not the `mlb_service_years - elapsed` formula shown here. See CLAUDE.md step 10 for why that formula was replaced.

```sql
-- pseudocode
SELECT
  c.player_id, c.season_year, c.salary0, c.years,
  p.date_of_birth, p.bats, p.throws, p.Pos, p.Role,
  p.draft_overall_pick,
  p.mlb_service_years - (CURRENT_LEAGUE_YEAR - c.season_year) AS service_at_signing,
  c.mvp_bonus, c.cyyoung_bonus, c.allstar_bonus
FROM contracts c
JOIN players p ON c.player_id = p.ID
WHERE c.is_major = 1
  AND c.season_year > 0
  AND c.player_id NOT IN (
    SELECT player_id FROM contract_extensions
    WHERE is_major = 1 AND season_year > 0
  )
  AND (p.mlb_service_years - (CURRENT_LEAGUE_YEAR - c.season_year)) >= 6
```

Then join in pre-signing trailing stats from `player_batting_YYYY.csv` and `player_pitching_YYYY.csv` for years `season_year-3, season_year-2, season_year-1`.

---

## Refresh cadence

- Run `src/data/puller.py` after each Frostfire season finishes (or sim period of interest)
- `SKIP_EXISTING = True` makes refresh fast — only new year files and snapshots get re-pulled
- Snapshot files (contracts, players, ballparks, teams) always re-pull since they're point-in-time
- Historical year files (2015–2034) don't change once a season ends; they get cached
- Only the current year's stat files (2035) get re-pulled mid-season
- Full fresh pull (`SKIP_EXISTING = False`): ~3.5 minutes for 21 years of data
