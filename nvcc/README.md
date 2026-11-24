# nvcc: GPU hand-class equity

Offline tools that compute the 169×169 starting-hand equity matrix on an NVIDIA GPU. The app never runs
them; it only reads the results.

## Requirements

- An NVIDIA GPU. The commands below use `-arch=sm_89` (RTX 40 series, Ada). For another card change it:
  `nvidia-smi --query-gpu=compute_cap --format=csv` gives the value (8.6 → `sm_86`, 9.0 → `sm_90`, …).
- A CUDA toolkit with `nvcc` and a matching driver. Tested with CUDA 13.1, driver 590.
  `nvcc` is often not on `PATH`; the toolkit usually lives in `/usr/local/cuda/bin`.
- A host C++ compiler that `nvcc` accepts (`g++`).

## Files

Each tool is a `.cu` program; code shared by tools lives in a `.cuh`; the checks for a tool are in its own
`test_<tool>.cu`. New offline tools follow the same layout.

| File             | What it is                                                                         |
| ---------------- | ---------------------------------------------------------------------------------- |
| `poker.cuh`      | Shared: card encoding, the 7-card evaluator `eval7`, the RNG, the 169 hand classes |
| `equity.cu`      | The 169×169 equity matrix: sampled or exact (no checks inside)                     |
| `test_equity.cu` | Checks for it: GPU evaluator vs a slow host reference, and the class tables        |
| `ranges.cuh`     | Shared (host): JSON reader and the hand-list grammar of `spots.ts` (`charts.json`, `spots.json`) |
| `realization.cuh`| Realisation factors: the GPU kernels, the betting tree, drivers, the results file  |
| `realization.cu` | Command line: `run`, `show`, `compare`, `sens`                                      |
| `test_realization.cu` | Checks for it (see the section below)                                         |

## Build

```bash
NVCC=/usr/local/cuda/bin/nvcc
$NVCC -O3 -arch=sm_89 -o equity equity.cu
$NVCC -O3 -arch=sm_89 -o test_equity test_equity.cu
```

## Use

```bash
./equity pair AKs QQ 8000000          # one matchup by sampling
./equity matrix 1000000 out.bin       # all 14,196 matchups by sampling, 1,000,000 samples each
./equity exact out.bin                # all matchups by full enumeration (no noise)
./equity exact out.bin AKs QQ         # one matchup, exact
```

`out.bin` is 169×169 float64, row-major. `out[i*169 + j]` is the share of the pot class `i` wins against
class `j` (a tie counts half). The class order is `ALL_HANDS` in `src/lib/poker/ranges.ts`
(AA, AKs, AKo, AQs, …, 22), so the file lines up with the TypeScript tools. The diagonal is 0.5.

Read it in Python with `numpy.fromfile('out.bin', dtype=numpy.float64).reshape(169, 169)`.

## Check

```bash
./test_equity [deals]                 # default 200,000 deals per batch
```

1. **Evaluator.** It deals random showdowns, decides the winner on the GPU and with a slow, independent host
   reference (best of the 21 five-card subsets, ranked by sorting rank groups), and counts mismatches. The
   second batch draws from a 32-card deck (A and 2 to 8) so that quads, full houses and straight flushes are
   common.
2. **Class tables.** 169 classes in `ranges.ts` order, 1326 distinct combos, 6 / 4 / 12 per pair / suited /
   offsuit class.
3. **Exact enumeration.** Reference values for AA v KK and AKs v QQ, a class against itself is exactly 50%,
   and swapping the two classes gives 1 minus the equity.
4. **Sampling vs exact.** Six matchups at 4,000,000 samples stay within 5 standard errors of the exact value.

It prints `PASS` and exits 0, otherwise `FAIL` and exits 1. Run it after any change to `poker.cuh`.

## How it works

- Cards are `0..51`, `(rank index) * 4 + suit`, rank index 0 = deuce, the same as `ranges.ts`.
- The evaluator scores a 7-card hand with bit masks only (no lookup tables), so a whole sample stays in
  registers.
- **Sampling:** a random combo of each class (skipped if they share a card), a random board from the other
  48 cards. Seeds depend only on the pair and the thread, so a run is reproducible.
- **Exact:** every non-conflicting pair of combos is a job, and each job runs all C(48,5) = 1,712,304
  boards, split over 512 threads. Ties count half.

## Measured (RTX 4060 Ti 16GB)

|                                                                       | 14,196 matchups  |
| --------------------------------------------------------------------- | ---------------- |
| CPU (`tools/make-equity.ts`, 4 cores, 1,000,000 samples)              | about 50 minutes |
| PyTorch, 1,000,000 samples (an earlier version, not kept in the repo) | about 12 minutes |
| `equity matrix 1000000`                                               | 1.8 seconds      |
| `equity exact` (808,392 jobs, 1.38×10¹² boards, no noise)             | 5 min 18 s       |

The sampled matrix agrees with the exact one to within sampling error: over the 14,196 cells the
difference has an RMS of 0.048% (expected from sampling alone: 0.047%), and the largest is 0.19%. The exact
matrix is antisymmetric to the last bit (`m[i][j] + m[j][i] = 1`).

# Realisation factors (`realization`)

How much of its raw equity does a hand turn into chips once the flop is dealt? The preflop solver
(`tools/solve-preflop.ts`) uses one formula for every hand, `realisedIp(eq) = clamp(0.5 + 1.08 (eq - 0.5) + 0.025)`.
`realization` replaces it with a table per hand class (169), side (in position / out of position), pot type
(single-raised, 3-bet, 4-bet) and depth (20, 40, 80, 100bb), made by simulating heads-up play on the GPU.
Result: `data/realization.json` (200,000 boards per pot type). **It is an approximation**; the site should
label it so. Limits are listed at the end of this section.

```bash
$NVCC -O3 -arch=sm_89 -o realization realization.cu
$NVCC -O3 -arch=sm_89 -o test_realization test_realization.cu
./realization run 200000 data/realization.json     # the table; about 20 minutes, resumable per pot type
./realization show srp 100 20000 [all]             # one case on screen
./realization compare data/realization.json        # how far the table is from the old formula
./realization sens 20000                           # how much it moves when the policy constants change
./test_realization [boards]                        # the checks, about 2 minutes
```

The ranges are read from `src/lib/poker/data/charts.json` and `spots.json` (folder `../src/lib/poker/data`,
or the environment variable `POKER_DATA`) with the hand-list grammar of `spots.ts` (`ranges.cuh`). The run
resumes: after each pot type it writes `<out>.<pot>.state`, and a rerun with the same boards and seed skips the
finished ones.

## What is simulated

Heads-up, the small blind (button, **IP**) against the big blind (**OOP**), the sizes of `solve-preflop.ts`
(open 2.5, 3-bet 9, 4-bet 22, nothing else), both sides having put in `c`, so the flop pot is `2c`:

| pot type | `c` | IP range                                  | OOP range                                         | depths (flop pot, SPR at 100bb) |
| -------- | --- | ----------------------------------------- | ------------------------------------------------- | ------------------------------- |
| `srp`    | 2.5 | chart `SB6` (22+, A2s+, K4s+, … 438 combos) | `vs_open` BB vs SB, call (456 combos)             | 20, 40, 80, 100 (5bb, 19.5)     |
| `3bet`   | 9   | `vs_3bet` SB vs BB, call (63 combos)      | `vs_open` BB vs SB, bluff + value (190 combos)    | 20, 40, 80, 100 (18bb, 5.1)     |
| `4bet`   | 22  | `vs_3bet` SB vs BB, bluff + value (91 combos) | `vs_open` BB vs SB, value (42 combos)         | 40, 80, 100 (44bb, 1.8)         |

The 4-bet pot does not reach the flop at 20bb (a 22bb raise is all-in there, as in the solver). Depth is the
effective stack at the start of the hand, so the stack behind on the flop is `depth - c`.

The board is the only random thing: five distinct cards per sample. Both players then follow a **fixed policy**
(`default_pol()` in `realization.cuh`, written into the file):

- A hand is ranked by its equity against the opponent's *current* range on the board shown (exact card removal,
  every runout to come: 1081 turn-and-river pairs on the flop, 46 rivers on the turn). `q` is the share of the
  player's own current range that is stronger, 0 = best. Hands outside the range are ranked too (as if added).
- Bets: one size per street (33%, 66%, 75% of the pot). OOP leads with its top 10% / 14% / 18% for value (flop /
  turn / river), IP bets after a check with its top 32% / 30% / 30%. Bluffs make up `b/(1+b)` of the value
  bets (the balanced ratio): the best hands that would fold on the flop and turn (draws), the worst on the river.
- Facing a bet of `b` pots: the top `1/(1+b)` of the range continues (the minimum defence frequency), the rest
  folds. The top 5% raises all-in when the stack left after a call is at most 2 pots; an all-in raise is met with
  the pot-odds share. A bet of at least 60% of the remaining stack is an all-in.
- After each action both ranges are narrowed by the exact probability that a hand takes that action, so later
  streets see the ranges that really arrive. No actions are sampled: for one board the result is the exact
  expectation over all hands of both players.

Output per hand class `h`, side, pot type and depth, against the opponent's range, averaged over boards and
hands with card removal (a board counts with the weight of opponent hands it leaves):

| field          | meaning                                                                                       |
| -------------- | --------------------------------------------------------------------------------------------- |
| `raw`          | all-in equity before the flop (five cards to come)                                            |
| `real`         | chips won / final pot, folds count as 0 (ratio of the two averages)                           |
| `factor`       | `real / raw`, the usual "equity realisation"                                                  |
| `se`           | standard error of `factor` (boards are the independent units; delta method)                   |
| `ev_bb`        | average chips won minus chips put in after the flop                                           |
| `final_pot_bb` | average final pot                                                                             |

Use in the solver (a suggestion; the solver is a later piece of work): the value of calling preflop and seeing the
flop with class `h` against the opponent's range is `ev_bb - c` (`c` = `preflop_each_bb`, paid before the flop). Against
one opponent class `j`, with raw equity `eq_hj`, move along the line through that point:
`ev_bb + factor * final_pot_bb * (eq_hj - raw) - c`. Every chip is accounted for: `real * final_pot_bb` is the chips won
and `real * final_pot_bb - ev_bb` the chips put in after the flop.

## JSON schema (`schema: "realization/v1"`)

```
{ "schema": "realization/v1", "kind": "approximation", "description": "...",
  "classes": [169 names in ALL_HANDS order],            // AA, AKs, AKo, AQs, ... 22
  "boards_per_pot_type": 200000, "seed": 20261124,
  "policy": { bet_pot_fraction[3], oop_lead_value_share[3], ip_bet_value_share[3], raise_share, ... },
  "pots": [ { "pot": "srp" | "3bet" | "4bet", "preflop_each_bb": 2.5,
              "ip_range_from": "...", "oop_range_from": "...",
              "ip_range": [169 shares 0..1], "oop_range": [169],   // the range each side played
              "identity_errors": { path_probability, chips, invested, raw_equity },
              "depths": [ { "depth_bb": 100, "pot_bb": 5, "stack_bb": 97.5, "spr": 19.5,
                            "ip":  { "range_summary": { combos, raw, factor, ev_bb, old_formula_factor,
                                                        mean_share_change_vs_old, rms_share_change_vs_old },
                                     "raw": [169], "real": [169], "factor": [169], "se": [169],
                                     "ev_bb": [169], "final_pot_bb": [169] },
                            "oop": { ... same ... } } ] } ] }
```

All 169 classes are filled for both sides, in range or not (`*_range` says which); a class outside the range
is valued as if it were added to it, and plays by its rank against that range.

## Checks (`test_realization`)

1. **Ranges.** The parser reproduces the combo count every one of the 30 spots states, the percentage of every
   chart (within 0.5, the sources round), and single combos / spans / class names as `spots.ts` does.
2. **Scan.** The GPU sort-and-prefix-sum equity (win weight and opponent weight per hand, with card removal) against a naive
   host loop on 25 boards, including a paired board and a board where every hand ties (3.4 million tied pairs);
   percentile ranks against a naive loop, with many ties.
3. **Zero-sum and bookkeeping**, every board, pot type and depth: the action probabilities of every hand pair add up to
   1 over all terminals; chips won by IP + chips won by OOP = final pot (pair-weighted); chips put in = final pot - flop
   pot; raw equities add up to 1. Worst relative error 5×10⁻⁸ (float arithmetic).
4. **Raw equity vs the exact matrix** (`data/equity-exact.bin`): 1014 class cells, largest |z| 3.3, mean z² 1.00.
5. **Determinism**: same seed gives bit-identical results, a different seed does not.
6. **Sanity**: in every pot type and depth the IP range's factor is above the OOP range's; every in-range factor is
   in 0.5 to 1.6 (found: 0.79 to 1.40) and every class's in 0.3 to 2.0 (found: 0.79 to 1.71); suited connectors and
   one-gappers realise about 0.12 more than offsuit unconnected hands of the same raw equity (both sides, many
   standard errors); at SPR 0.02 every factor is within 2% of 1 (mean |factor - 1| 0.004; ranges 1.003 and 0.996) and the
   distance from 1 grows with the stack; identical ranges give each side raw equity 1/2 and IP still realises more.
7. **The file**: schema, 169 numbers per column, `factor = real / raw`, ranges equal to today's parsing, identity errors,
   at least 50,000 boards per pot type, standard errors below 0.03 (largest 0.0016).

It also prints how the table differs from the old formula (below).

## Results (RTX 4060 Ti 16GB)

Speed: about 480 boards per second per pot type with all depths (4 scenarios), 200,000 boards in 7 minutes
(`srp` 418s, `3bet` 406s, `4bet` 328s). The flop equity (1081 runouts) is computed once per board and shared by
all depths.

Equity-weighted factor of each whole range (realised share of the pot / raw equity), 100bb: single-raised IP 1.043 / OOP 0.923;
3-bet pot 1.015 / 0.924; 4-bet pot 1.176 / 0.846. The gap between IP and OOP grows with depth (single-raised, IP
minus OOP: 0.08 at 20bb, 0.13 at 40bb, 0.12 at 80 and 100bb; 3-bet pot 0.05, 0.08, 0.09, 0.09; 4-bet pot 0.22, 0.30, 0.33).
By class (single-raised, 100bb, hands in the range): pocket pairs TT+ IP 0.95 / OOP 0.83, low pairs 22-66 1.12 / 0.90,
offsuit broadways 1.02 / 0.88, suited aces 1.02 / 0.88, suited connectors 1.18 / 1.05.

Against the old formula (`./realization compare data/realization.json`, realised share of the pot per class,
new minus old, over the side's own range): the average over the range is close, because the old constants were
fitted to it (single-raised IP -0.005, OOP -0.008 at 100bb; 3-bet pot IP -0.022, OOP -0.005), but the class-by-class
difference is large: at 100bb the **RMS over the range is 0.046 / 0.051 (single-raised IP / OOP), 0.070 / 0.048 (3-bet pot),
0.071 / 0.083 (4-bet pot) of the pot, and the worst class is off by 0.11 to 0.16**. The 4-bet pot differs most on average
(IP +0.05, OOP -0.08 at 100bb). Over all 22 rows the RMS runs from 0.015 (3-bet pot, 20bb, OOP) to 0.083.

**How much the policy constants matter** (`./realization sens 20000`, 100bb; each variant changes one thing; the
last two columns are the RMS change of the realised share per class against the default, weighted by the range):

| pot type | variant                       | IP factor | OOP factor | IP rms | OOP rms |
| -------- | ----------------------------- | --------- | ---------- | ------ | ------- |
| srp      | default                       | 1.043     | 0.923      |        |         |
| srp      | defend 0.85x / 1.15x          | 1.051 / 1.033 | 0.914 / 0.934 | 0.007 / 0.009 | 0.007 / 0.006 |
| srp      | value shares 0.6x / 1.5x      | 1.043 / 1.053 | 0.928 / 0.909 | 0.006 / 0.011 | 0.014 / 0.010 |
| srp      | bets 0.7x / 1.4x              | 1.033 / 1.053 | 0.939 / 0.910 | 0.011 / 0.012 | 0.014 / 0.011 |
| srp      | no raises / raises 3x as often | 1.040 / 1.059 | 0.927 / 0.905 | 0.002 / 0.016 | 0.003 / 0.012 |
| 3bet     | default                       | 1.016     | 0.923      |        |         |
| 3bet     | worst of the above            | 0.968 (raises 3x) | 0.906 (value 1.5x) | 0.029 | 0.023 |
| 4bet     | default                       | 1.177     | 0.845      |        |         |
| 4bet     | worst of the above            | 1.153 (value 0.6x) | 0.866 (bets 0.7x) | 0.015 | 0.018 |

Moving a constant by 15 to 50% changes the range factors by up to 0.03 and the class-by-class realised share by
about 0.01 RMS (the exception: making raises three times as common in the 3-bet pot moves the factors by 0.05 and the
shares by 0.03 RMS). That is about a quarter of the distance to the old
formula (0.046 to 0.083), so the difference from the old formula is not an artefact of one constant, but the
table is only as good as the family of policies it came from. The full output is in the log of the run.


## Limits (read before using the table)

- **A fixed policy, not an equilibrium.** Nobody best-responds. The strategies are balanced by construction
  (bluff : value, minimum defence), not solved; the numbers are what this policy realises against itself.
- **Equity ranks hands, nothing more.** A hand's rank comes from its equity against the opponent's range at that node
  (not narrowed by the action just taken), so the model has no idea of blockers, board texture beyond equity, or traps.
  Draws are only seen through their equity.
- **Few actions.** One bet size per street, no re-raises except an all-in shove when the stack is small, so very deep
  stacks (SPR above 5) rarely get all in and strong hands under-realise relative to real play. Expect the
  deep-stack numbers of strong hands to be low and weak hands that bluff to be high.
- **Ranges are the 6-max charts at 100bb** (`SB6`, BB vs SB open, SB vs BB 3-bet), used at every depth; the 20bb ranges are really
  tighter. They are the button/small blind against the big blind only; other seats have other ranges.
- **Classes outside the range** are valued as if added to it, and realise too much (up to 1.7×): they check back
  against a range that has already shown strength elsewhere, and their weak equity keeps them out of big pots.
  Do not read their factors as "what that hand does if it plays"; use them only as a continuation of the curve.
- **Pot types:** no limped pots, no 5-bet pots. Ranges for the 3- and 4-bet pots are the lists of the source pages, a
  hand with several actions counted with its share.
- Accuracy: standard errors of the factors are at most 0.002 (sampling of boards only); the model error is
  much larger and is not measured against a solver.
