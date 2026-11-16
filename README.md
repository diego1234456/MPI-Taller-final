# Parallel Word Counting in a Text Corpus with MPI (LAB)

## Problem Description

The goal of this lab is to count how many times each word of a query list (`consulta.txt`, 200 words) appears in a corpus of text files and to report the 10 most frequent query words.

A sequential program has to open and tokenize the 3,000 files one after another, so it spends a lot of time on file I/O and string processing. I parallelized the task with MPI (`mpi4py`) and compared two ways of splitting the files among the processes: a static split (version 1) and a dynamic master-worker split (version 2).

## Environment and Execution Instructions

All experiments ran inside Docker on Windows, using the course image `augustosalazar/slim-mpi:2` (OpenMPI and Python 3). The commands below are run from the `word_lab` directory in PowerShell.

Generate the dataset:

```powershell
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/generator.py
```

Run the sequential baseline:

```powershell
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py
```

Run MPI version 1 (static partitioning, p = 4):

```powershell
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 mpiexec -n 4 --allow-run-as-root --use-hwthread-cpus --oversubscribe python /app/mpi1.py
```

Run MPI version 2 (dynamic master-worker, p = 4):

```powershell
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 mpiexec -n 4 --allow-run-as-root --use-hwthread-cpus --oversubscribe python /app/mpi2.py
```

## Experimental Plan

I compare three implementations on the same dataset (3,000 files, 200 query words):

1. Sequential baseline (p = 1), which gives the reference time T_seq.
2. MPI version 1 (p = 4): static, equal-size blocks of files, using `bcast` and `gather`.
3. MPI version 2 (p = 4): files handed out on demand by a master process, using `send` and `recv`.

The value p = 4 is the number of processes reported by `detectar_procesos_mpi.py`. Each MPI version was run 3 times. For each version I compute:

- T_p: the average total execution time over the 3 runs.
- Speedup: S_p = T_seq / T_p
- Efficiency: E_p = S_p / p
- Load imbalance: LI = max(t_i) / mean(t_i), where t_i is the local processing time of rank i. LI = 1.0 means perfect balance, and higher values mean more imbalance. I also report the relative spread (max(t_i) - min(t_i)) / mean(t_i).

### Sequential baseline (what it does and how)

`baseline_secuencial.py` does everything in one process. It reads the words in `dataset/consulta.txt` into a set (so each lookup is fast), then goes through the 3,000 files in order. For each file it reads the lines, splits them by whitespace and adds one to a global counter every time a token is in the query set. At the end it sorts the counters, prints the top 10 and prints the total time.

### MPI version 1 (what it does and how)

`mpi1.py` splits the files statically:

1. Rank 0 reads `consulta.txt` and builds the sorted list of the 3,000 file paths. Both are sent to every rank with `comm.bcast`.
2. The list is cut into equal contiguous blocks. With 4 processes that is 750 files per rank, and each rank computes its own start and end index.
3. Each rank counts the query words in its 750 files and measures that work with `MPI.Wtime()`. This is its local time.
4. With `comm.gather`, every rank sends its partial counts, its local time and its number of files to rank 0. Rank 0 adds the partial counts, prints the top 10 and prints the per-process report.

### Test procedure

- The dataset was generated once and used for every run, so all versions read exactly the same files.
- Each MPI version was run 3 times with the same command, one run right after the other. I recorded the total time (measured with `MPI.Wtime()` from start to end of the program) and the local time of every rank.
- The top 10 words and counts printed by `mpi1.py` and `mpi2.py` were compared with the baseline output and matched in every run.
- p was fixed to 4, the value reported by `detectar_procesos_mpi.py`. I did not try other values of p.

## Experimental Plan Execution

### Sequential baseline timing

T_seq = 37.560000 s (one run).

Top 10 printed by the baseline: `a` (785,774), `para` (392,156), `sus` (228,913), `otros` (105,530), `ante` (99,832), `unos` (88,794), `otra` (83,901), `vosotros` (61,617), `mios` (58,420), `tuya` (56,635).

### MPI version 1 timing results

Total time and local time per rank (seconds). In every run each rank got 750 files. LI is computed per run over the 4 ranks.

| Run | Total time | Rank 0 | Rank 1 | Rank 2 | Rank 3 | LI |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | 13.373651 | 12.398582 | 13.338711 | 12.433259 | 12.575934 | 1.051 |
| 2 | 13.416253 | 12.407499 | 13.274184 | 12.477423 | 12.583566 | 1.046 |
| 3 | 10.125624 | 9.445507 | 10.068173 | 9.310057 | 9.612649 | 1.048 |
| Average | 12.305176 | | | | | 1.049 |

With the average time: S_4 = 37.56 / 12.305 = 3.05 and E_4 = 3.05 / 4 = 76.3 %.
Per run: S_4 = 2.81, 2.80 and 3.71, and E_4 = 70.2 %, 70.0 % and 92.7 %.

### Load imbalance evidence (per-process local times and LI)

In run 1 every rank got 750 files, but the local times were not equal:

| Rank | Assigned files | Local time (s) |
| :-: | :-: | :-: |
| 0 | 750 | 12.398582 |
| 1 | 750 | 13.338711 |
| 2 | 750 | 12.433259 |
| 3 | 750 | 12.575934 |

The maximum was 13.338711 s (rank 1), the minimum 12.398582 s (rank 0) and the mean 12.686622 s. This gives LI = 13.338711 / 12.686622 = 1.051, and a relative spread of (13.338711 - 12.398582) / 12.686622 = 0.074. Ranks 0, 2 and 3 finished earlier and waited about 0.94 s at the `gather` for rank 1.

The same thing happened in the other runs: rank 1 was the slowest every time, LI was 1.046 and 1.048, and the other ranks waited 0.87 s and 0.76 s. The average LI over the 3 runs was 1.049 and the average relative spread was 0.074.

### MPI version 2: how it corrects the imbalance, and its results

`mpi2.py` replaces the static blocks with a master-worker scheme using `comm.send` and `comm.recv`. Rank 0 does not count words. It only hands out files: each time a worker (ranks 1, 2 and 3) finishes a file it asks for the next one, and rank 0 sends one file path. When the files run out, rank 0 sends the stop tag. A worker that gets slower files simply asks for fewer files, so the workers should end at about the same time. The price is that only 3 of the 4 processes do the counting.

Run 1, with 4 processes:

| Rank | Role | Processed files | Local time (s) |
| :-: | :-- | :-: | :-: |
| 0 | Master | 0 | 12.904536 |
| 1 | Worker | 1,052 | 12.871123 |
| 2 | Worker | 1,003 | 12.873030 |
| 3 | Worker | 945 | 12.904505 |

Rank 0 processed no files, so its local time is just the time it stayed in the dispatch loop, and it stays within about 0.03 s of the slowest worker in every run. I do not count it as work. LI is therefore computed over the 3 workers: in run 1, max = 12.904505 s, min = 12.871123 s, mean = 12.882886 s, so LI = 1.0017 and the relative spread is 0.0026. If I include rank 0 as well, LI is 1.0013.

All runs (LI over the 3 workers):

| Run | Total time | Rank 0 | Rank 1 | Rank 2 | Rank 3 | LI |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | 12.936121 | 12.904536 | 12.871123 | 12.873030 | 12.904505 | 1.002 |
| 2 | 17.098873 | 17.046913 | 17.046896 | 17.002610 | 17.005992 | 1.002 |
| 3 | 18.038531 | 17.838616 | 17.795259 | 17.871217 | 17.800571 | 1.003 |
| Average | 16.024508 | | | | | 1.002 |

Files processed by each worker (rank 1 / rank 2 / rank 3): 1,052 / 1,003 / 945 in run 1, 987 / 970 / 1,043 in run 2 and 1,018 / 1,007 / 975 in run 3. In every run the three numbers add up to 3,000.

With the average time: S_4 = 37.56 / 16.025 = 2.34 and E_4 = 2.34 / 4 = 58.6 %.
Per run: S_4 = 2.90, 2.20 and 2.08, and E_4 = 72.6 %, 54.9 % and 52.1 %.

### Comparison

| | T_p (s) | S_p | E_p | LI |
| :-- | :-: | :-: | :-: | :-: |
| Sequential | 37.56 | 1.00 | 1.00 | - |
| MPI v1 (p = 4) | 12.31 | 3.05 | 76.3 % | 1.049 |
| MPI v2 (p = 4) | 16.02 | 2.34 | 58.6 % | 1.002 |

## Analysis

### Parallel performance: did MPI version 1 improve on the sequential baseline, and how close was its speedup to the ideal (S_p = p)? What explains the gap?

Yes. Version 1 went from 37.56 s to an average of 12.31 s, a speedup of 3.05 against the ideal of 4 (efficiency 76.3 %). The three runs were very different from each other: 2.81, 2.80 and 3.71. Run 3 reached an efficiency of 92.7 %, which is close to ideal.

To see where the gap comes from I split the time of each run. The ideal time with 4 processes is 37.56 / 4 = 9.39 s. In run 1 the real time was 13.37 s, so 3.98 s were missing:

- About 3.30 s because each process was slower than in the sequential run. The mean local time was 12.69 s instead of 9.39 s. Adding up the 4 local times gives 50.7 s of work, 35 % more than the 37.56 s of the sequential program.
- About 0.65 s because of the imbalance (the slowest rank, 13.34 s, versus the mean, 12.69 s).
- About 0.035 s for everything else (listing files, `bcast`, `gather`).

Run 2 looks the same (50.7 s of work). Run 3 is different: the 4 local times add up to 38.4 s, only 2 % more than sequential, so in that run almost all the missing time came from the imbalance (0.46 s) and a small extra per-file cost (0.22 s). So the serial part and the communication are small in all runs, below 0.15 s. The main reason for the gap in runs 1 and 2 is that the same job took about 35 % more total CPU work, and in run 3 it did not. I did not find out why. My guess is I/O or background load on the computer, but I did not test it.

### Load balance: what evidence of load imbalance did version 1 show (per-process local times and LI), and did version 2 reduce it enough to produce a real performance improvement?

Version 1 gave every rank 750 files, yet the local times were different in all three runs (for example 12.40 s to 13.34 s in run 1). LI was 1.051, 1.046 and 1.048 (average 1.049), and three ranks waited between 0.76 s and 0.94 s for the slowest one. Rank 1 was the slowest every time, so this looks like a real property of its block of files and not just noise.

Version 2 reduced LI to 1.002 on average, so the imbalance almost disappeared. But it did not give a real performance improvement: the average time went from 12.31 s (v1) to 16.02 s (v2), and v2 was slower in runs 2 and 3. It was faster only in run 1 (12.94 s against 13.37 s), and I do not think that run shows a real improvement. The raw times are hard to compare because the total work changed between runs. The table adds up the local times of the processes that count words, and compares the total time with the time a perfectly balanced run would take with the same work (sum divided by number of workers):

| Run | v1: sum of local times (s) | v1: total / (sum / 4) | v2: sum of worker times (s) | v2: total / (sum / 3) |
| :-: | :-: | :-: | :-: | :-: |
| 1 | 50.75 | 1.054 | 38.65 | 1.004 |
| 2 | 50.74 | 1.058 | 51.06 | 1.005 |
| 3 | 38.44 | 1.054 | 53.47 | 1.012 |

The sums go from 38 s to 53 s for the same job, so the machine was faster or slower depending on the run (in run 1, v2 was lucky and v1 was not). The last column of each version does not depend on that. Version 1 ends about 5.5 % later than a perfectly balanced run, and version 2 only 0.4 % to 1.2 % later. So v2 recovers about 5 % of the time. But the master does not count words, so v2 has 3 workers instead of 4 and each worker has 4/3 times the work. With the same amount of work, I expect v2 to be around 25 % slower than v1 (1.33 × 1.01 / 1.055 ≈ 1.27). That agrees with the direction of the averages and with runs 2 and 3. The speedup ceiling of v2 is 3, and its average speedup of 2.34 is 78 % of that. In run 1 it reached 97 %.

### Limitations: what limitations affected your experiment, and how much do they weaken your conclusions?

- Only 3 runs per version and a single p. The total times vary a lot from run to run: from 10.13 s to 13.42 s for v1 and from 12.94 s to 18.04 s for v2 (standard deviations of 1.9 s and 2.7 s). That is larger than any effect of the load balancing, which is at most about 0.65 s in one run.
- The work per run changed between 38 s and 53 s. I do not know why. The dataset is in a OneDrive-synced Windows folder mounted into a Linux container, and the computer was not used only for the lab, so I suspect file access or background load, but I did not check it. Because I ran the runs of each version one after the other, a change in the machine can end up on only one version.
- T_seq was measured only once (37.56 s). If the machine was in a faster or slower state during that run, all the speedups and efficiencies move with it.
- Only p = 4 was tested, and the dataset is synthetic (made by `generator.py`), so the imbalance with real text could be different.
- Version 2 sends one message per file (3,000 round trips). At p = 4 this cost is small (the total time was at most 0.17 s above the slowest worker), but I did not test it with larger p.

These limitations do not change the main result: parallel is clearly faster than sequential in every run. They do weaken any claim based on raw time differences between v1 and v2, and I avoided those. The LI result is consistent in the 3 runs (about 1.05 against 1.00), and the argument for the final comparison (3 workers against 4) does not need the raw times, but a test with more runs and with other values of p would make it stronger.

## Conclusions

For this problem and p = 4, I would recommend version 1 (static partitioning). Over 3 runs it reached an average speedup of 3.05 with an efficiency of 76.3 % and LI = 1.049, while version 2 reached 2.34 with an efficiency of 58.6 % and LI = 1.002. The dynamic scheme did almost remove the imbalance, which cost version 1 about 5 % of its time, but it needed one of the four processes as coordinator, and losing a quarter of the workers costs about 33 % more time per unit of work. The raw times of the runs changed too much to compare them directly, but when I normalize by the work done in each run, v2 is expected to be about 25 % slower than v1 at this size. I would expect v2 to pay off with a larger number of processes, where one coordinator is a small part of the total, or in a version where the master also counts words, but I did not test either of those.
