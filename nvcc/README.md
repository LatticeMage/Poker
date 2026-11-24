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
