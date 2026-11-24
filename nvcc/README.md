# GPU offline table creator

CUDA tools that build the tables the poker app reads: the 169×169 hand-class equity matrix, solved heads-up and
push/fold charts, three-way equity, realisation factors (and, in progress, 6-max charts). They run offline on an
NVIDIA GPU; the app never runs them, it only reads the results.

How every tool runs: all GPU work is launched through `dispatch()` (batches, a temperature check before each
one), and a long task is a relay: it saves its state as it goes and an interrupted run carries on from the last
saved batch instead of starting again (see `dispatch.cuh`).

## Requirements

- An NVIDIA GPU. The commands below use `-arch=sm_89` (RTX 40 series, Ada). For another card change it:
  `nvidia-smi --query-gpu=compute_cap --format=csv` gives the value (8.6 → `sm_86`, 9.0 → `sm_90`, …).
- A CUDA toolkit with `nvcc` and a matching driver. Tested with CUDA 13.1, driver 590.
  `nvcc` is often not on `PATH`; the toolkit usually lives in `/usr/local/cuda/bin`.
- A host C++ compiler that `nvcc` accepts (`g++`).
- Heat: long runs load the card fully. Cap the clocks on the machine that has the GPU, for example
  `sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 210,1800` (undo: `sudo nvidia-smi -rgc`). Inside a Docker
  container this needs `cap_add: [SYS_ADMIN]`. Measured on an RTX 4060 Ti: uncapped 2760 MHz, 132 W, 75 °C after
  30 s and still rising; capped at 2300 MHz about 70 W, 64 °C after 30 s (the machine in use is now capped at 1800 MHz). Without that permission,
  `gpu-cooler.sh [pause_c] [resume_c]` pauses every CUDA process of the user above a temperature and resumes it
  after it has cooled down.

## Files

Each tool is a `.cu` program; code shared by tools lives in a `.cuh`; the checks for a tool are in its own
`test_<tool>.cu`. New offline tools follow the same layout.

| File                                                                     | What it is                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `poker.cuh`                                                              | Shared: card encoding, the 7-card evaluator `eval7`, the RNG, the 169 hand classes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `thermal.cuh`                                                            | Thermal gate. A task starts only when the GPU is at or below 52 °C (`GPU_RESUME_C`); a running task goes on until 60 °C (`GPU_PAUSE_C`), then waits until 52 °C again; `GPU_POLL_S` is the polling interval and `GPU_PAUSE_C=0` turns the gate off. Reads `nvidia-smi`, no root                                                                                                                                                                                                                                                                                                                                                                                                      |
| `dispatch.cuh`                                                           | `dispatch(total, per_batch, launch, label, verbose, &relay)`: **the** way to launch GPU work. The first batch waits until the GPU is cool enough to start, every later batch waits while it is too hot, then the batch runs, is waited for, and progress is printed. With a `Relay` it also saves the task's state (every 60 s by default) and restores it when the same problem is started again; a file for another problem is ignored, a finished task keeps its file (delete it to recompute). It can only stop **between** batches: a batch that is already running is interrupted by `gpu-cooler.sh` (see Requirements), which also covers programs that do not use `dispatch` |
| `test_thermal.cu`, `test_dispatch.cu`                                    | Checks for the two (fake thermometer and clock, so nothing waits for a real GPU to cool; the relay is checked with a simulated interruption)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ranges.cuh`                                                             | Shared (host): JSON reader and the hand-list grammar of `spots.ts` (`charts.json`, `spots.json`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `equity.cuh`                                                             | The equity computation: kernels and host drivers (`run_pairs` sampled, `exact_pairs` exact)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `equity.cu`                                                              | Command line for it: `pair`, `matrix`, `exact` (no checks inside)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `test_equity.cu`                                                         | Checks: GPU evaluator vs a slow host reference, class tables, exact reference values, sampled vs exact                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `data/equity-exact.bin`                                                  | The exact 169×169 matrix (float64, row-major, 228,488 bytes), the input of the solvers. Regenerate: `./equity exact data/equity-exact.bin` (5 min on an RTX 4060 Ti)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `solve.cuh`, `solve.cu`, `test_solve.cu`, `test_solve_external.cu`       | Heads-up preflop CFR+ solver, its command line and its checks (see "Heads-up preflop solver")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `data/out/headsup-<bb>.json`                                             | Solved heads-up trees, 5 to 150bb                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `solve_realization.cuh`, `test_solve_realization.cu`, `solve_compare.cu` | The solver reading the realisation table (`--real`), its checks, a comparison of two result sets (see "The solver with the realisation table")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `data/out/headsup-<bb>-real.json`, `-real-cap.json`                      | The same trees solved with the realisation table (default rules / out-of-range classes capped)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `equity3.cuh`, `equity3.cu`, `test_equity3.cu`                           | Three-way equity: kernels, table format and lookup, range function; command line; checks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `data/equity3.bin`, `data/equity3-se.bin`                                | The generated three-way table (6.5 MB) and its standard errors (0.8 MB)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `realization.cuh`, `realization.cu`, `test_realization.cu`               | Realisation factors: kernels and betting tree; command line (`run`, `show`, `compare`, `sens`); checks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `data/realization.json`                                                  | The realisation-factor table (182 KB)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

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

# Heads-up preflop solver

`solve.cu` is the CUDA port of `tools/solve-preflop.ts`: CFR+ over the 169 hand classes for the small blind
(button) against the big blind, with raise ladders, limp, all-in and the realised-equity heuristic for calls
that see a flop. It reads the exact equity matrix and writes `headsup-<stack>.json` in the schema of
`src/lib/poker/data/headsup-*.json`. Same requirements as the equity tool (nvcc, CUDA, `-arch`).

| File                     | What it is                                                                                    |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| `solve.cuh`              | Game trees, kernels, the `Solver` class (iterate, exploitability, value, checkpoint, JSON)    |
| `solve.cu`               | Command line                                                                                  |
| `test_solve.cu`          | Checks (below)                                                                                |
| `test_solve_external.cu` | Reads the output of the external solver poker-cfr and compares                                |
| `data/out/`              | Solved trees, exact matrix, 30,000 iterations: 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150 bb |

Build with **`-fmad=false`**. Then every sum runs in the same order and with the same rounding as the
TypeScript tool, and the output is identical to it (see Check).

```bash
NVCC=/usr/local/cuda/bin/nvcc
$NVCC -O3 -fmad=false -arch=sm_89 -o solve solve.cu
$NVCC -O3 -fmad=false -arch=sm_89 -o test_solve test_solve.cu
$NVCC -O3 -fmad=false -arch=sm_89 -o test_solve_external test_solve_external.cu
```

## Use

```bash
./solve --stacks 10,20,40,80 --iters 30000 --out out        # exact matrix from data/equity-exact.bin
./solve --stacks 5,10,15,20,30,40,50,60,80,100,150 --iters 30000 --out out --ckpt out/prod
./solve --raw ...                      # no realisation: every call is a showdown on raw equity
./solve --open 2.5,9,22 --limp 4,12    # the raise ladders (default shown); "/" gives alternatives: 2/2.5,9
./solve --tree mult ...                # the tree of poker-cfr: raises of 2.5x, 3x, 3.5x, 4x the bet faced
```

It prints the exploitability (best response from both seats, in bb) every `--report` iterations. With
`--ckpt PREFIX` it saves `PREFIX.ckpt` every `--ckpt-every` iterations and resumes from it when started
again. `samples_per_matchup` is written as 0 (`--samples` changes it): 0 means the exact matrix.
The exploitability field in the JSON is rounded to 3 decimals as in the TypeScript tool, so it reads 0; the
console shows the real value.

## Check

```bash
./test_solve [matrix.bin]
```

1. **Two-hand game** solved by hand (H beats L, stack 2): H always shoves and calls, L never calls, value 0.
   The best-response value of a made-up profile equals an enumeration written separately in the test.
2. **Host reference.** A recursive host CFR+ (a plain port of the TypeScript) on the 20bb tree, 150
   iterations: regrets and strategy sums agree (measured: exactly, difference 0), and so does the
   exploitability.
3. **Solved run**, 10/20/40/80 bb, 3000 iterations: each hand's frequencies sum to 1 at every node (error
   4e-16), exploitability below 0.01 bb (measured 0.00004 to 0.0005), zero-sum, and a best response is worth
   at least the profile's value.
4. **TypeScript numbers.** `npm run solve -- 0 3000` with the same matrix (`tools/.cache/equity-0.json` is
   the matrix as JSON; 0 samples means exact) gives exploitability 0.0000396 bb at 10bb and 0.0001309 bb at
   20bb; the test requires the same to 1e-6, the same node keys in the same order, and the same reach and
   root frequencies. Outside the test, the whole JSON for 10, 20, 40 and 80 bb was compared with the
   TypeScript output: 4,557 frequencies, 0 differences, identical node lists.

```bash
./test_solve_external 20 75000 ../.tmp/poker-cfr/output/preflop-20-75000.bin
```

Reads the bincode file of poker-cfr (BSD-2, heads-up preflop with checks to showdown, so raw equity and no
realisation), solves the same tree (`--tree mult`) and compares every node. Not a pass/fail test.
Measured, our 45,000 to 180,000 iterations against its files, raw equity, same tree:

| Stack | Nodes | Mean total-variation distance of per-hand strategies | SB value (ours / poker-cfr) |
| ----- | ----- | ---------------------------------------------------- | --------------------------- |
| 10bb  | 44    | 0.0024                                               | -0.0280 / -0.0280 bb        |
| 15bb  | 80    | 0.0005                                               | -0.0570 / -0.0570 bb        |
| 20bb  | 100   | 0.0045                                               | -0.0706 / -0.0706 bb        |
| 30bb  | 164   | 0.0011                                               | -0.0623 / -0.0623 bb        |

The distance is weighted by how likely each node is to be reached; a node nobody plays into has no
determined strategy, so it is left out of that weighting. The agreement is that close because the trees are
made identical on purpose. With our own ladders (`2.5, 9, 22`, not 2.5x to 4x of the bet faced) the ranges
differ, as they should.

## How it works

- A tree is a list of decisions, folds and showdowns. All stacks of one run are packed together. One CFR+
  iteration is a top-down kernel per depth (regret matching, reach of the children), one kernel for all
  showdowns (one block per showdown, one thread per hand class: 169×169 matrix-vector products), and a
  bottom-up kernel per depth (values, regret floor at 0, strategy sums weighted by the iteration number).
  The kernels of an iteration are recorded in a CUDA graph and replayed.
- Exploitability and the value of the averaged profile reuse the same kernels (best response: the hero's
  nodes take the best action per hand, the other player's reach carries his averaged strategy).
- The TypeScript tool is a single-threaded reference; the host reference in `test_solve.cu` is a second one.

## Measured (RTX 4060 Ti 16GB)

|                                                        | TypeScript (1 core) | CUDA  |
| ------------------------------------------------------ | ------------------- | ----- |
| 10, 20, 40, 80 bb, 3000 iterations                     | 3 min 49 s          | 1.3 s |
| 11 stacks (5 to 150 bb), 30,000 iterations             | not run             | 39 s  |
| poker-cfr tree at 20bb (100 nodes), 75,000 iterations  |                     | 54 s  |
| poker-cfr tree at 30bb (164 nodes), 180,000 iterations |                     | 224 s |

Exploitability of the 30,000-iteration run (bb, from both seats): 5 to 50 bb below 0.000005, 60 to 150 bb
about 0.00001. SB value in bb: 5: +0.0562, 10: -0.0168, 20: -0.0266, 40: +0.0133, 80: +0.0186, 150: +0.0154.

## The solver with the realisation table (`--real`)

By default a call that sees a flop is valued with the old formula. `--real data/realization.json` values it with
the per-class factors of the `realization` tool instead. Off by default, so the old output, the TypeScript
cross-check and every test above are untouched (the formula run reproduces the committed `headsup-<bb>.json`
byte for byte). Code: `solve_realization.cuh` (table reader, matrix builder; the design is written at the top of
the file), checks: `test_solve_realization.cu`, comparison of two result sets: `solve_compare.cu`.

```bash
$NVCC -O3 -fmad=false -arch=sm_89 -o solve solve.cu
./solve --stacks 5,10,15,20,30,40,50,60,80,100,150 --iters 30000 --out out --suffix -real --real data/realization.json
#   --real-out formula|cap|none   classes outside the table's ranges (default formula)
#   --real-limp formula|srp       limped pots (default formula)
$NVCC -O3 -fmad=false -arch=sm_89 -o test_solve_realization test_solve_realization.cu && ./test_solve_realization
$NVCC -O3 -arch=sm_89 -o solve_compare solve_compare.cu
./solve_compare data/out headsup-%s.json data/out headsup-%s-real.json 5,10,20,40,100 20,40,100   # stacks, hand lists
```

**Design decisions**

1. **Pot type** from the number of raises before the call: open = single-raised (table `srp`), 3-bet = `3bet`, 4-bet
   = `4bet`; an all-in call is still raw equity. **Limped pots** (check after a limp, call of a raise after a limp)
   have no data and keep the old formula (`--real-limp srp` borrows the single-raised factors instead; sensitivity below).
2. **Depth.** What counts is the stack behind on the flop, `x = stack - c` (`c` = chips each has in). Between the
   table's depths (20, 40, 80, 100; 4-bet pot 40, 80, 100) the factor is linear in `x`; above the deepest it is the
   deepest (150 bb uses 100 bb: no extrapolation); below the shallowest it goes linearly to 1 at `x = 0` (nothing
   behind: no realisation effect). So 5 to 15 bb get a fraction of the 20 bb deviation. At 5 and 10 bb nobody calls an
   open and sees a flop with a hand that matters, so those two solutions barely move (5 bb: BB raise-vs-limp only).
3. **Zero-sum.** The table gives a factor per class and side, not per matchup. For SB class `i`, BB class `j`, SB raw
   equity `e`: `a = f_ip(i) e`, `b = f_oop(j) (1 - e)`, SB gets `a / (a + b)`, BB `b / (a + b)`. The two add to the pot
   in every matchup (checked: largest error 1.4e-14 on pot weights up to about 1100), and factors of 1 give the raw
   matrices bit for bit. **Approximation:** a class's table factor is its average against the table's whole opponent
   range, not against that hand; the normalisation then spreads the gap between the sides in ratio; the table was made with
   the 100 bb 6-max ranges and fixed policies (see Limits of the realisation tool), and the solver's ranges differ.
4. **Classes outside the table's ranges** (factors up to 1.7) decide the result, because the solver plays exactly those
   hands at the edge of its ranges. Default `formula`: they keep the old formula (the table speaks only where it was
   measured). `cap`: their factor is limited to their side's equity-weighted range factor. `none`: the table as is.

**Result.** 30,000 iterations, exact matrix, exploitability at most 0.00001 bb printed (test: below 0.0009 at 3000
iterations for every rule). Files: `data/out/headsup-<bb>-real.json` (default rules) and
`headsup-<bb>-real-cap.json` (cap). Same schema as `headsup-<bb>.json` plus a top-level string field
`"realisation"` that names the model (`table:realization/v1,out-of-range=formula,limped=formula`); a reader that
ignores unknown fields is unaffected. SB value, old formula -> table (bb): 20: -0.0266 -> -0.0311, 30: -0.0012 -> -0.0054,
40: +0.0133 -> +0.0110, 100: +0.0166 -> +0.0162, 150: +0.0154 -> +0.0171.

Frequencies in % (combo weighted), old formula -> table, default rules (`solve_compare`). The "defends" rows only where SB
actually opens 2.5 (otherwise the node is not reached). 5 and 10 bb: no change in SB's play, BB-vs-limp at 5 bb 59.9 -> 58.7.

| Depth  | SB fold      | SB limp      | SB raise 2.5 | SB shove     | BB defends 2.5 open | BB 3-bets 2.5 open | BB calls shove | BB raises a limp |
| ------ | ------------ | ------------ | ------------ | ------------ | ------------------- | ------------------ | -------------- | ---------------- |
| 15 bb  | 13.6 -> 13.6 | 61.4 -> 63.2 | 4.6 -> 0.0   | 20.4 -> 23.2 | -                   | -                  | 27.2 -> 26.7   | 40.5 -> 40.0     |
| 20 bb  | 7.8 -> 5.7   | 71.5 -> 76.1 | 17.1 -> 11.6 | 3.6 -> 6.6   | 69.0 -> 72.8        | 24.2 -> 24.0       | 21.3 -> 20.6   | 36.1 -> 34.9     |
| 30 bb  | 7.2 -> 5.7   | 63.7 -> 65.6 | 29.1 -> 28.7 | 0 -> 0       | 72.8 -> 77.6        | 20.5 -> 20.6       | 15.8 -> 15.0   | 35.9 -> 35.2     |
| 40 bb  | 9.3 -> 8.1   | 52.5 -> 55.2 | 38.2 -> 36.7 | 0 -> 0       | 76.9 -> 79.1        | 18.7 -> 19.1       | 13.7 -> 13.1   | 38.9 -> 38.0     |
| 50 bb  | 12.7 -> 12.4 | 41.1 -> 43.0 | 46.3 -> 44.6 | 0 -> 0       | 81.1 -> 83.7        | 17.3 -> 17.8       | 10.1 -> 8.9    | 42.2 -> 41.2     |
| 60 bb  | 16.0 -> 13.4 | 32.8 -> 38.6 | 51.3 -> 48.0 | 0 -> 0       | 82.8 -> 83.7        | 16.9 -> 17.2       | 8.1 -> 8.0     | 44.4 -> 43.2     |
| 80 bb  | 16.0 -> 13.3 | 33.1 -> 36.9 | 50.9 -> 49.8 | 0 -> 0       | 84.0 -> 83.7        | 17.8 -> 18.0       | 6.4 -> 6.3     | 44.8 -> 43.1     |
| 100 bb | 13.0 -> 13.0 | 37.2 -> 37.7 | 49.9 -> 49.4 | 0 -> 0       | 82.3 -> 83.7        | 18.9 -> 18.9       | 5.2 -> 5.3     | 43.2 -> 42.5     |
| 150 bb | 12.2 -> 11.5 | 42.3 -> 43.0 | 45.5 -> 45.5 | 0 -> 0       | 81.0 -> 83.7        | 19.6 -> 18.9       | 3.6 -> 3.7     | 41.6 -> 40.1     |

(The full per-depth table, every row, is the output of `solve_compare`.) The changes are small, at most a few points
and 6 at the worst (SB limps at 60 bb), and the same size as the solver's own slack between neighbouring depths.

**Which hands change** (default rules). SB plays more: 20 bb 42s, 53o, 63o (limp/shove, previously folded); 40 bb 62s, 73o,
43s; at 100 bb none. BB defends more: 20 bb Q5o, Q6o, 85o, J5o, Q7o; 40 bb 85o, J3o; 100 bb T4o, T5o (53o drops out). Nobody plays
less. BB 3-bets: with the table more A5o/A2o/A8o/A6s/A7s/66 and KTo/QTs, fewer suited connectors (20 bb: T8s, 97s, 75s, 64s, 54s;
40 bb: J8s, 98s, 87s, 65s, 54s; 100 bb: JTs 100 -> 49%). Mixed 3-bet bluffs are the least stable part of any
equilibrium, so single hands there should not be over-read.

**Plausibility and warnings**

- Direction is plausible for the SB: it plays slightly wider (low suited hands, a few offsuit gappers), limps a bit more
  at 20 to 60 bb, raises less at 20 bb, and BB defends a little wider (+1 to +5 points). The 3-bet range moves towards
  unsuited ace blockers and away from suited connectors, which the table explains (out-of-position suited connectors
  realise well when they call, 1.05 in the single-raised pot, but less as 3-bet bluffs). The expected "fewer weak offsuit
  hands" did **not** appear: BB adds offsuit junk (T4o, T5o, Q5o at the edge) because the table gives out-of-position hands
  at the weak end a factor around 1 (T8o 1.02, Q8o 0.96 at 100 bb) against 0.89 from the old formula. That is the least reliable part.
- **The out-of-range rule matters more than the table.** Big blind defence of the 2.5 open, old -> table, at 20 / 40 / 100 bb:
  default (formula) 69 -> 73 / 77 -> 79 / 82 -> 84; `cap` 69 -> 80 / 77 -> 86 / 82 -> 90; `none` (table as is) 100% defend at 40
  and 100 bb, which is plainly wrong (T2o called). A published number from the table alone would be an artefact of
  inflated out-of-range factors. SB limp at 100 bb, same three rules: 37.7 / 38.2 / 23.9 (old 37.2). `cap` files are kept (`-real-cap`) for comparison.
- **Limped pots** use the old formula; borrowing single-raised factors (`--real-limp srp`) moves SB limp frequency (change against the default rule)
  by -3.5 / +0.5 / +5.6 points at 20 / 40 / 100 bb (100 bb: 43.3 against 37.7), so the limp frequency is uncertain by about that much.
- SB limp frequencies at 20 to 60 bb (50 to 75%) were already high with the old formula; the table does not change that
  picture. They come from the tree (one raise ladder, no limp-raise sizes between) as much as from realisation.
- The 3-bet and 4-bet pots rest on 13/26 and 14/7 in-range classes; with the default rule almost every hand there stays on
  the old formula, so those nodes change little. The solution is "old formula, corrected where the table has data", not a
  postflop-accurate solution. Nothing here has been checked against a postflop solver.

# equity3: three-way equity

6-max nodes with a caller (squeeze, cold call, 3-bet after a flat) need the all-in equity of three hands, which
the 169×169 matrix cannot give. `equity3` stores, for every unordered triple of hand classes, the share of the pot
each player wins when all five board cards are dealt (ties split among the tied players, a three-way tie is 1/3).

## Build, use

```bash
NVCC=/usr/local/cuda/bin/nvcc
$NVCC -O3 -arch=sm_89 -o equity3 equity3.cu
$NVCC -O3 -arch=sm_89 -o test_equity3 test_equity3.cu
./equity3 gen run                       # whole table into run/equity3.bin and run/equity3-se.bin, resumable
./equity3 triple AKs QQ 72o 8000000     # one triple by sampling (players in the order given)
./equity3 exact AKs QQ 72o              # one triple, every combo triple on every board (about a second)
./equity3 query data/equity3.bin AA KK QQ
./equity3 ranges data/equity3.bin "TT+,AQs+,AKo" "22+,A2s+,KTs+,ATo+" "55+,A9s+,KJs+,AJo+"
```

`gen [target_se] [n0] [nmax]` writes `run/ckpt.bin` after every round (and every two minutes); start the same
command again and it carries on. Run it in tmux. Defaults: target standard error 0.03%, 100,000 samples in the
first round, at most 16,000,000 per triple.

## The table

- **Cell.** Classes (a, b, c) with a ≤ b ≤ c in `ALL_HANDS` order (AA = 0). The value is the average over every
  ordered choice (combo of a, combo of b, combo of c) with six different cards, and all boards. Classes may
  repeat when the cards allow it (AA AA KK: yes; AA AA AA: no). 818,805 cells (169·170·171/6), of which 325 are
  impossible (not enough cards); those have N = 0 and all zeros.
- **Index.** `idx = C(k+2,3) + C(j+1,2) + i` for sorted i ≤ j ≤ k. `e3_lookup(table, c0, c1, c2, eq)` takes the
  classes in any player order, sorts them (ties by player number) and returns each player's share, plus N. Players
  holding the same class have the same value (the table stores the average of their samples).
- **File `equity3.bin`.** 16-byte header (`EQ3\0`, version 1, 818805, 169; uint32 each), then 818,805 records of
  four uint16: N (number of combo triples, 0 to 1728), and the shares of sorted slots 0, 1, 2 in 1/65535. The
  shares sum to 65535 ± 2. 6,550,456 bytes in total, small enough to commit (`data/equity3.bin`). Read it in
  Python with `numpy.fromfile(path, dtype='<u2', offset=16).reshape(-1, 4)`.
- **File `equity3-se.bin`.** One byte per triple: the largest of its three standard errors, in units of 0.002%.

## Range against range (for the solvers)

`e3_ranges_cpu` / `e3_ranges_gpu` (or `e3_lookup` inside your own kernel). `w[p][c]` is player p's weight on class
c, every combo of a class equally likely. For player p, with q and r the other two, and each class i:

    mass[p][i] = sum over j, k of  w[q][j] * w[r][k] * N(i,j,k)
    eq[p][i]   = sum over j, k of  w[q][j] * w[r][k] * N(i,j,k) * E_p(i,j,k)  /  mass[p][i]

`N` is the number of combo triples without a shared card, `E_p` the table's share for player p. **This is exact for
class-uniform ranges**, not an approximation: card removal is the factor N (the probability of a combo triple is
proportional to w[p][i]·w[q][j]·w[r][k] among the non-conflicting ones), and all combo triples of a class triple have the
same average equity because a suit permutation maps one to another and leaves class-uniform ranges unchanged. The
class-level model only loses something if a range weights combos of one class differently (for example, suited
hands by suit) or the board is partly known; neither is expressed by a 169-vector. The only error is the table's
sampling noise (at most 0.03% per cell, and it averages down over the many cells of a range). `mass` tells how many
weighted combo triples exist, and 0 means the class cannot occur; weights need not be normalised. A whole range's
equity is `sum_i w[p][i] mass[p][i] eq[p][i] / sum_i w[p][i] mass[p][i]`; the three players' whole-range equities sum
to 1.

## Check

```bash
./test_equity3 data/equity3.bin data/equity3-se.bin [host_deals_per_triple]
```

1. **Structure.** For all 818,805 cells: N equals an independent count (and is 0 exactly for the 325 impossible
   triples), the three shares sum to 1, equal classes have equal shares, largest standard error ≤ 0.1%.
2. **Lookup.** All six player orders of 20,000 random class triples return the permuted answer.
3. **Exact enumeration.** The GPU enumerates every combo triple on all C(46,5) = 1,370,754 boards for 12 chosen
   triples and for 150 random triples (random player order); the table agrees within the standard error (worst
   difference 3.1 SE, mean z 0.00, rms z 0.93).
4. **Independent host reference.** 30 random triples, 20,000 deals each, dealt by separate host code and scored
   by the slow 21-subset evaluator: mean z -0.03, rms z 1.08.
5. **Sampler symmetry.** The sampling kernel on all six player orders of three class triples.
6. **Range function.** GPU = CPU, permuting the ranges permutes the answers, whole-range equities sum to 1. A
   provable identity: with all combos equally likely for all three players, every player averages exactly 1/3
   (exchangeable, shares sum to 1); the table gives 0.333333 for each. Known value: AA against two random hands 73.44%.
   Against a combo-level Monte Carlo (about 8×10⁸ deals per setup, own sampling code: each combo drawn with its class
   weight, conflicts rejected) on three realistic setups (a squeeze, a cold call, mixed weights): worst difference
   0.047% (3.2 Monte Carlo SE), frequency-weighted mean difference 0.006 to 0.009%.

There is no reduction of the three-way table to the 169×169 matrix: whichever hand the third player holds, it can win
and it removes cards, so no limiting case is exact except the exchangeability identity above.

## How it works

- **Sampling.** A random combo for each player (retry on a shared card, which keeps the draw uniform over the valid
  combo triples), a random board from the other 46 cards, three `eval7`. The per-triple accumulators are integers
  (sum of 6×share, and of its square), so the standard error is exact.
- **Adaptive.** Round 0 gives each possible triple 100,000 samples. A later round adds, for triples whose largest
  standard error is above 0.03%, the samples the estimate says are missing (+10%, at most doubling). It stops when
  every triple is below the target. Seeds depend on (triple, round, thread), so a chunk redone after a crash gives the
  same samples. Stopping at a threshold biases a cell by far less than its standard error at these sample sizes.
- **Exact.** Same job layout as `equity exact`: one job per combo triple, 512 slices of the boards per job.

## Measured (RTX 4060 Ti 16GB)

| `equity3 gen run` (default target 0.03%)       |                                                     |
| ---------------------------------------------- | --------------------------------------------------- |
| Samples                                        | 2.35×10¹² (about 2.9 million per triple on average) |
| Time                                           | 749 s (6 rounds), about 3 to 4×10⁹ samples/s        |
| Standard error per cell (largest of 3 players) | 0.0286% typical, 0.0300% worst                      |
| 150 random triples exact                       | 26 s                                                |
| `test_equity3`                                 | about 1 minute (mostly the slow host reference)     |

The target is a command-line argument: 0.02% would take about 2.25 times as long (about 28 minutes), 0.01% about 9 times.

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

| pot type | `c` | IP range                                      | OOP range                                      | depths (flop pot, SPR at 100bb) |
| -------- | --- | --------------------------------------------- | ---------------------------------------------- | ------------------------------- |
| `srp`    | 2.5 | chart `SB6` (22+, A2s+, K4s+, … 438 combos)   | `vs_open` BB vs SB, call (456 combos)          | 20, 40, 80, 100 (5bb, 19.5)     |
| `3bet`   | 9   | `vs_3bet` SB vs BB, call (63 combos)          | `vs_open` BB vs SB, bluff + value (190 combos) | 20, 40, 80, 100 (18bb, 5.1)     |
| `4bet`   | 22  | `vs_3bet` SB vs BB, bluff + value (91 combos) | `vs_open` BB vs SB, value (42 combos)          | 40, 80, 100 (44bb, 1.8)         |

The 4-bet pot does not reach the flop at 20bb (a 22bb raise is all-in there, as in the solver). Depth is the
effective stack at the start of the hand, so the stack behind on the flop is `depth - c`.

The board is the only random thing: five distinct cards per sample. Both players then follow a **fixed policy**
(`default_pol()` in `realization.cuh`, written into the file):

- A hand is ranked by its equity against the opponent's _current_ range on the board shown (exact card removal,
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

| field          | meaning                                                                     |
| -------------- | --------------------------------------------------------------------------- |
| `raw`          | all-in equity before the flop (five cards to come)                          |
| `real`         | chips won / final pot, folds count as 0 (ratio of the two averages)         |
| `factor`       | `real / raw`, the usual "equity realisation"                                |
| `se`           | standard error of `factor` (boards are the independent units; delta method) |
| `ev_bb`        | average chips won minus chips put in after the flop                         |
| `final_pot_bb` | average final pot                                                           |

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

| pot type | variant                        | IP factor          | OOP factor         | IP rms        | OOP rms       |
| -------- | ------------------------------ | ------------------ | ------------------ | ------------- | ------------- |
| srp      | default                        | 1.043              | 0.923              |               |               |
| srp      | defend 0.85x / 1.15x           | 1.051 / 1.033      | 0.914 / 0.934      | 0.007 / 0.009 | 0.007 / 0.006 |
| srp      | value shares 0.6x / 1.5x       | 1.043 / 1.053      | 0.928 / 0.909      | 0.006 / 0.011 | 0.014 / 0.010 |
| srp      | bets 0.7x / 1.4x               | 1.033 / 1.053      | 0.939 / 0.910      | 0.011 / 0.012 | 0.014 / 0.011 |
| srp      | no raises / raises 3x as often | 1.040 / 1.059      | 0.927 / 0.905      | 0.002 / 0.016 | 0.003 / 0.012 |
| 3bet     | default                        | 1.016              | 0.923              |               |               |
| 3bet     | worst of the above             | 0.968 (raises 3x)  | 0.906 (value 1.5x) | 0.029         | 0.023         |
| 4bet     | default                        | 1.177              | 0.845              |               |               |
| 4bet     | worst of the above             | 1.153 (value 0.6x) | 0.866 (bets 0.7x)  | 0.015         | 0.018         |

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

# Heads-up push/fold tables (2-20bb)

`pushfold.cu` solves the push/fold game (small blind folds or shoves, big blind folds or calls; cEV, no ICM)
with the CFR+ solver of `solve.cuh` on its push/fold tree (`build_pushfold_tree`, also `solve --tree pushfold`),
on the exact matrix, and writes the data the app reads. Every call is an all-in showdown, so no realisation
heuristic enters. Build with `-fmad=false` like the solver.

| File                     | What it is                                                                                                 |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `pushfold.cuh`           | Tables from the averaged strategies, an independent exploitability evaluator (`PfEval`), the JSON          |
| `pushfold.cu`            | Command line                                                                                               |
| `test_pushfold.cu`       | Checks                                                                                                     |
| `data/out/pushfold.json` | Result: `equity_vs_random`, `pushfold`, `mix` (schemas of `tables.json` and `pushfold-mix.json`), `solver` |

```bash
$NVCC -O3 -fmad=false -arch=sm_89 -o pushfold pushfold.cu
$NVCC -O3 -fmad=false -arch=sm_89 -o test_pushfold test_pushfold.cu
./pushfold --iters 300000 --out data/out/pushfold.json      # 70 s on an RTX 4060 Ti (19 stacks at once)
./test_pushfold                                             # 20,000 iterations, 2 s
npm run apply-pushfold && npx prettier --write src/lib/poker/data   # from the repo root: copy into the app's data
```

- The pure ranges are the classes whose averaged shove / call frequency is above one half (as before); the
  frequencies of at least 2% go to `pushfold-mix.json`.
- `exploitability_bb` is what a best response gains against the **pure** ranges, both seats added, in bb per hand
  dealt (the big blind's gain counts only the hands where he is shoved at). The averaged strategy itself is
  below 1e-7 bb at every stack (`solver.stacks.*.exploitability_mixed_bb`); thresholding the edge hands costs up to
  1.3e-4 bb at 8bb. The old tables, from fictitious play on a 4000-sample matrix, were measured on that same noisy matrix.
- `tables.json` keeps `samples_per_matchup` above zero (1,712,304 = every board of each combo pair), since the field means
  samples per matchup and the exact matrix enumerates all of them.
- **Check:** the evaluator against a two-hand game worked out by hand; the solver's exploitability and value equal the
  evaluator's on the same strategy (3e-15); zero-sum; exploitability bounds; ranges shrink with depth; strong hands
  always in; ranges at 2, 5, 10, 20bb in published regions; JSON lists, percentages and frequencies agree;
  `equity_vs_random` known values and mean exactly 0.5.

# 6-max preflop solver (`sixmax`)

Six-seat preflop charts (UTG, MP, CO, BTN, SB, BB) solved on the GPU: first in, facing an open, facing a 3-bet, with
mixed frequencies, in the shapes the app displays. **It is an approximation, not an equilibrium of poker**: six players
(no Nash guarantee), a small action set, realisation factors from a fixed-policy table, and a few rules that remove
multiway branches (below). The app does not read its output yet.

| File                        | What it is                                                                                                       |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `sixmax.cuh`                | The tree builder, the CFR+ kernels, the `Solver6` class (iterate, best-response gap, checkpoint)                 |
| `sixmax_data.cuh`           | Inputs: pot-share matrices from `realization.json`, the three-way tensors, the source charts, the start strategy |
| `sixmax_run.cuh`            | The two stages, reading the result in the app's shapes, the JSON                                                 |
| `sixmax.cu`                 | Command line                                                                                                     |
| `test_sixmax.cu`            | Checks (below)                                                                                                   |
| `data/out/sixmax-<bb>.json` | Result per depth (schema `sixmax/v1`), see "Status of the runs"                                                  |

```bash
NVCC=/usr/local/cuda/bin/nvcc
$NVCC -O3 -arch=sm_89 -o sixmax sixmax.cu
$NVCC -O3 -arch=sm_89 -o test_sixmax test_sixmax.cu
./sixmax tree 100                                   # nodes of the tree
./sixmax solve --stack 100 --iters1 3000 --iters 20000 --report 1000 --ckpt data/out/c100 --out data/out
./test_sixmax                                       # about 2 minutes
```

Run from `nvcc/`; it reads `data/equity-exact.bin`, `data/equity3.bin`, `data/realization.json` and the source charts
(`../src/lib/poker/data`, or `POKER_DATA`). `--ckpt PREFIX` writes `PREFIX.ckpt` at every report and resumes stage 2 from
it (stage 1 is deterministic and cheap, it is simply redone). About 650 nodes, 20 of them three-way showdowns; 1000
iterations take about 25 s on the RTX 4060 Ti, the setup (the 169^3 tensors) about 20 s.

## The game that is solved

Equal stacks, blinds 0.5 / 1, one tree for the whole hand, all six seats trained at once by vector-form CFR+ (regrets
floored at 0, strategy sums weighted by the iteration number, as in the heads-up solver) over the 169 hand classes.

| Situation      | Actions                                                                                                                                                                                         |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| first in       | UTG, MP, CO, BTN: fold or raise to **2.5**. SB: fold, limp, or raise to **3** (the source chart's size). Folded to BB: BB wins                                                                  |
| BB vs SB limp  | check or raise to 4; SB then folds or calls                                                                                                                                                     |
| facing an open | each seat behind, in turn: fold, call, or 3-bet. 3-bet = **3x** the open when the 3-bettor is in position after the flop, **4x** out of position, plus one open per caller already in (squeeze) |
| after a 3-bet  | everyone but opener and 3-bettor folds, except that the players who called the open may call a squeeze if the opener folds. Opener: fold, call, 4-bet to **2.2x** the 3-bet                     |
| after a 4-bet  | 3-bettor: fold, call, all-in. Opener: fold or call the all-in. A raise to 70% of the stack or more is an all-in (at 20bb the 4-bet is)                                                          |
| limits         | at most two players call an open (a third caller is not allowed); no cold 4-bets, no cold calls of a 3-bet, no 5-bet other than all-in                                                          |

Every decision is its own information set (the strategy after a caller differs from the one without): 257 decisions at
100bb. The ones the app shows are the "clean" nodes: first in (5), facing an open with nobody in between (15), facing a
3-bet with no callers (15). `bluff` and `value` of the app are derived: of the hands that 3-bet, the share that folds to a
4-bet is `bluff`, the rest `value` (same for 4-bets and the all-in).

**Showdowns.** Two players: a pot-share matrix per pot type (limp: the old `realised_ip` heuristic; single-raised, 3-bet,
4-bet: from `realization.json`; all-in: raw equity). For a table pot type at depth D, IP's expected chips from the flop pot
are `ev_bb + factor * final_pot * (eq(i,j) - raw)` (the line the realisation section suggests), the same for OOP, and the
shares used are the average of IP's number and `pot - OOP's number`, so that the pot is always shared out exactly
(zero-sum); linear between the table's depths (60bb), clamped to [0, 1], applied to the real pot (dead money included).
Three players (an open called twice): the exact three-way table `equity3`, each player's share multiplied by the
realisation factor of his class in the single-raised table (the OOP column for the player who acts first after the flop,
the IP column for the last, the mean for the middle one) and renormalised to 1. This multiway rule is a guess; 4-way pots
do not exist.

**Card removal.** A value is the expected chips of a hero holding class h, given h. The chance that another player holds
class j is `Wn[h][j]` (non-clashing combo pairs over combos of h x 1225). Folded players enter by the probability that
they take the path (`sum_j Wn[h][j] reach[j]`), two players in a showdown by the pair weights, three by the exact triple
counts of `equity3`. **Approximated**: removal among the _other_ players. The game is therefore zero-sum only up to that;
the sum of the six players' values is 0.014 bb/hand (stage 1) to 0.047 bb/hand (solved), a few bb/100, and it is exactly 0
in a game without removal (checked). Treat values to about that precision.

**Starting strategy and stages.** The strategy starts at the source charts (`charts.json` for first in, `spots.json` for
facing an open / a 3-bet; squeeze nodes copy the facing-open numbers; the BB's raise over a limp and the SB's reply are
made up), with 2% noise; the regrets are seeded with 1 x that strategy. **Stage 1** trains only the first-in decisions
with every reply frozen at the source (first-in ranges against replies as in the charts); **stage 2** unfreezes all
decisions and continues from the stage-1 averages, until the fixed point is reached.

**Convergence number.** After every report the best-response gap of each seat: the value of the best response to the other
five seats' averaged strategies (all of its decisions at once, hand by hand) minus the value of the averaged strategy, in
bb per hand (x100 = bb/100). The total is the sum over the six seats (NashConv). In stage 1 frozen decisions play their
strategy inside the best response, so the number says how well the first-in ranges answer the frozen replies.

## Checks (`test_sixmax`)

1. **Three-player game** (two hand classes, no removal) solved on paper: the value of the profile "continue with H only" is
   0.75 for H and -1 for L (paper); the engine equals an enumeration of every deal on 20 random profiles for every player
   and class (difference 2e-16) and a separate best-response search (4e-16); the game is exactly zero-sum; CFR+ drives its
   gap to 6e-7 with the values still summing to 0.
2. **Tables**: each hand's chances sum to 1 and are symmetric, AA against AA 1/1225, KK 6/1225, AKs 2/1225, AKo 6/1225;
   three-way counts sum to combos x 1225 x 1128 for every class, the three layouts hold the same cells and the shares sum
   to 1; pot shares in [0, 1], AA takes more than 0.8 of a single-raised pot against 72o.
3. **Independent host evaluator** (plain recursion, direct `equity3` lookups, no tensors) on the real 100bb tree, random
   profile, 8 classes: values of all six players and best-response values of UTG, SB and BB agree to 7e-15 (largest value
   7.4 bb). This is the best-response gap checked against a host reference on the full tree.
4. **Short real run**: frozen decisions never move, first-in decisions do; two runs of 150 iterations are bit-identical;
   each hand's frequencies sum to 1 (3e-16) and none is negative; best-response gaps are non-negative; zero-sum defect
   below 0.1 bb; the JSON parses, has 5 + 15 + 15 spots and agrees with the strategies.

## Status of the runs (read this)

The GPU was stopped on request (thermal) while the **final** 100bb run was at about 16,000 of 20,000 iterations, and the
other depths (20, 40, 60, 80bb) have not been run. So `data/out/sixmax-*.json` is **not committed**; nothing here is
final. What exists:

- A complete 100bb run of the previous tree version (callers could not answer a squeeze), 3,000 + 20,000 iterations:
  best-response gap in bb/100 UTG 0.001, MP 0.001, CO 0.001, BTN 0.001, SB 0.003, BB 0.004 (total 0.011); values
  in bb/100 UTG 11.4, MP 14.3, CO 18.8, BTN 26.9, SB -24.5, BB -42.2. First-in open % solved / source: UTG 14.6 / 14.3, MP
  18.0 / 18.9, CO 25.2 / 25.5, BTN 39.5 / 43.3, SB raise 40.9 / 33.0 (plus 10.3 limp). 5 to 6.5% of the combos of UTG, MP,
  CO and BTN differ from their chart (mostly the weakest suited hands dropped, a few offsuit aces added), 20.7% for SB.
- The run with callers answering a squeeze, same settings, reached a total gap of 0.023 bb/100 at 15,000 iterations
  (UTG 0.003, MP 0.002, CO 0.001, BTN 0.002, SB 0.006, BB 0.009) when it was stopped. Checkpoint on the box:
  `/home/ubuntu/app/Github/LatticeMage/Poker/sixmax/data/out/c100b.ckpt`.

To finish, on the box in `/home/ubuntu/app/Github/LatticeMage/Poker/sixmax`:
`./sixmax solve --stack 100 --iters1 3000 --iters 20000 --report 1000 --ckpt data/out/c100b --out data/out`
(resumes stage 2 from the checkpoint, about 4,000 iterations left), then `--stack 20`, `40`, `60`, `80` with
`--iters 10000` (about 5 minutes of GPU each, new `--ckpt` prefix per depth), then copy `data/out/sixmax-<bb>.json` here.

## Things that look implausible (judge them yourself)

From the preliminary 100bb run; the squeeze-answer version may move them.

- **Cold calling is almost absent in position**: BTN calls UTG's open with 1.9%, CO 0.1%, and 3-bets 9% instead of the
  usual roughly 12% call and 6-7% 3-bet. The model may undervalue a flat (the realisation factors come from a fixed-policy
  simulation; the squeeze threat is modelled but cold calls of a 3-bet are not). BB vs BTN calls 26.8% against 44.6% in
  the source chart (3-bet 11.2% against 13.1%); BB defends only 9.7% + 4.9% 3-bet against UTG.
- **4-bets are wide**: UTG against an MP 3-bet 4-bets 24% (19% as bluffs), against CO 14%, BTN 12% (real solutions are
  nearer 8 to 10%). The 3-bettor's only answers to a 4-bet are fold, call and all-in.
- **BTN opens 39.5% against the chart's 43.3%** (too tight by the chart's measure): SB and BB 3-bet it more (SB 16.1%,
  BB 11.2%) and no cold callers exist. **SB raises 41%** and limps 10%, wider than the 33% chart.
- Open sizes are fixed (2.5; 3 for SB), so sizing never adapts.

## Limits

- Six players: no equilibrium guarantee; the gap is measured against the others' _averaged_ strategies, and CFR's average
  profile has no multiplayer theory behind it (it converged smoothly here).
- Removed branches (cold 4-bets, cold calls of a 3-bet, callers after a 4-bet, a third caller, 4-way pots) are removed for
  everybody, which biases the model against flatting and for 3-bet bluffs. Their reach is small where measured: the third
  caller was forbidden in 0.06 to 0.13% of the reach of facing-open decisions.
- Realisation factors come from `data/realization.json` (read its Limits): fixed policy, one bet size, the 100bb ranges at
  every depth, IP/OOP by position only (the table was made for SB against BB), no limped pots (old formula) and no
  multiway pots (the rule above).
- Stack depth enters only through the sizes (all-in thresholds) and the realisation depth; 20bb is not a push/fold solution.
