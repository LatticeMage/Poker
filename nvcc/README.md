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
