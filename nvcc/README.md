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
| `equity3.cuh`    | Three-way equity: kernels, table format and lookup, range function, generator      |
| `equity3.cu`     | Command line for it: build the table, one triple, exact, query, ranges              |
| `test_equity3.cu`| Checks for it: structure, exact enumeration, host reference, ranges vs Monte Carlo |
| `data/equity3.bin`, `data/equity3-se.bin` | The generated three-way table (6.5 MB) and its standard errors (0.8 MB) |

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

| `equity3 gen run` (default target 0.03%) | |
| --- | --- |
| Samples | 2.35×10¹² (about 2.9 million per triple on average) |
| Time | 749 s (6 rounds), about 3 to 4×10⁹ samples/s |
| Standard error per cell (largest of 3 players) | 0.0286% typical, 0.0300% worst |
| 150 random triples exact | 26 s |
| `test_equity3` | about 1 minute (mostly the slow host reference) |

The target is a command-line argument: 0.02% would take about 2.25 times as long (about 28 minutes), 0.01% about 9 times.
