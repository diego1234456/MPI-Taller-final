# 
                              **Performance Analysis and Load Balancing in Parallel Text File Processing Using MPI**

## 2. Problem Description

The objective of this laboratory is to compute the frequency of a specific set of target query words (`consulta.txt`) across a dataset of 3,000 text files (`file_0001.txt` to `file_3000.txt`), comprising approximately 44.95 million total words.

When executed sequentially on a single thread, reading thousands of files line by line, tokenizing strings, and updating word frequencies creates a significant I/O and CPU bottleneck. To overcome this, the task is parallelized using the **Message Passing Interface (MPI)** standard in Python (`mpi4py`), comparing two workload distribution strategies: static block partitioning and dynamic load balancing.

## 3. Environment and Execution Instructions

All experiments were conducted inside a Docker container on Windows using the official course image (`augustosalazar/slim-mpi:2`), which includes OpenMPI and Python 3.

### Execution Commands (from the `word_lab` directory in PowerShell):

1. **Generate the test dataset:**
```powershell
   docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/generator.py
```

2. **Run the Sequential Baseline:**
```powershell
   docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py
```

3. **Run MPI Version 1 (Static Partitioning with $p = 4$ processes):**
```powershell
   docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 mpiexec -n 4 --allow-run-as-root --use-hwthread-cpus --oversubscribe python /app/mpi1.py
```

4. **Run MPI Version 2 (Dynamic Master-Worker Load Balancing with $p = 4$ processes):**
```powershell
   docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 mpiexec -n 4 --allow-run-as-root --use-hwthread-cpus --oversubscribe python /app/mpi2.py
```

### a. Experimental Plan

Following the established laboratory guidelines, the experimental plan evaluates three implementations under identical input conditions (3,000 files and 200 query words):

1. **Sequential Baseline ($p = 1$):** Single-process execution serving as the reference time ($T_{seq}$).
2. **MPI Version 1 ($p = 4$):** Static equal-block distribution of files using collective communication.
3. **MPI Version 2 ($p = 4$):** Dynamic on-demand distribution (Master-Worker architecture) using point-to-point communication.

To evaluate parallel performance and load distribution, the following standard metrics are computed:

* **Speedup:** $S_p = \frac{T_{seq}}{T_p}$
* **Efficiency:** $E_p = \frac{S_p}{p}$
* **Load Imbalance (LI):** $LI = \frac{T_{max} - T_{avg}}{T_{avg}}$ (complemented by the max-to-min spread $T_{max} - T_{min}$)

### b. Sequential baseline (what it does and how)

The `baseline_secuencial.py` script executes the entire workflow within a single process:

1. Reads the target words from `dataset/consulta.txt` and stores them in a `set` for $O(1)$ lookup complexity.
2. Iterates sequentially over all 3,000 files inside `dataset/`.
3. Opens each file, splits lines by whitespace, and increments word frequencies in a global dictionary.
4. Sorts the final dictionary in descending order and outputs the Top 10 most frequent query words along with the total execution time.

### c. MPI version 1 (what it does and how)

The `mpi1.py` implementation uses **static data partitioning** via collective MPI operations:

1. **Broadcast (`comm.bcast`):** Rank 0 reads `consulta.txt` and gathers the sorted list of 3,000 file paths, broadcasting both structures to all participating processes.
2. **Static Chunking:** The file list is mathematically divided into equal contiguous chunks (`3000 / 4 = 750` files per process). Each rank determines its start and end indices independently.
3. **Local Processing:** Each process reads and counts query words across its assigned 750 files concurrently, timing its local workload with `MPI.Wtime()`.
4. **Gather (`comm.gather`):** All processes send their local frequency dictionaries, local processing times, and assigned file counts back to Rank 0, which aggregates the global counts and prints the final report.

### d. Test procedure

To ensure reliable, reproducible, and valid measurements:

* **Fixed Dataset:** The exact same generated dataset of 3,000 files was preserved across all runs to eliminate random file-size variance between tests.
* **Result Verification:** Correctness was verified by confirming that the Top 10 word frequencies (`a: 785774`, `para: 392156`, `sus: 228913`, etc.) matched 100% across `baseline_secuencial.py`, `mpi1.py`, and `mpi2.py`.
* **High-Resolution Timing:** `MPI.Wtime()` was used to measure both end-to-end wall-clock execution time ($T_p$) and per-process local computation/I/O time ($T_{local}$).

## 4. Experimental Plan Execution

The experimental plan was executed sequentially in the specified order inside the Docker environment, allocating $p = 4$ processes for both MPI versions (`--use-hwthread-cpus --oversubscribe`). All measurements were recorded directly from standard output.

### a. Sequential baseline timing

* **Total Sequential Execution Time ($T_{seq}$):** `37.560000 seconds`
* **Verified Top 10 Output:** `a` (785,774), `para` (392,156), `sus` (228,913), `otros` (105,530), `ante` (99,832), `unos` (88,794), `otra` (83,901), `vosotros` (61,617), `mios` (58,420), `tuya` (56,635).

### b. MPI version 1 timing results

Executed with $p = 4$ processes:

* **Total Execution Time ($T_{v1}$):** `13.373651 seconds`
* **Speedup ($S_4$):** $\frac{37.560000}{13.373651} = 2.81\times$ ($2.8085$)
* **Efficiency ($E_4$):** $\frac{2.8085}{4} = 70.21\%$ ($0.7021$)

### c. Load imbalance evidence (per-process local times and LI)

Although MPI Version 1 assigned an identical number of files (750) to every rank, the files vary in line count and word density, causing noticeable disparities in local execution times:

| Process (Rank) | Assigned Files | Local Time ($s$) |
| :--- | :---: | :---: |
| **Rank 0** | 750 | `12.398582` |
| **Rank 1** | 750 | `13.338711` |
| **Rank 2** | 750 | `12.433259` |
| **Rank 3** | 750 | `12.575934` |

* **Maximum Local Time ($T_{max}$):** `13.338711 s` (Rank 1)
* **Minimum Local Time ($T_{min}$):** `12.398582 s` (Rank 0)
* **Average Local Time ($T_{avg}$):** `12.686622 s`
* **Load Imbalance ($LI$):**
  $$LI = \frac{T_{max} - T_{avg}}{T_{avg}} = \frac{13.338711 - 12.686622}{12.686622} = 0.0514 \text{ (or } 5.14\% \text{)}$$
  *(Note: The max-to-min difference is `0.9401 seconds`, meaning Ranks 0, 2, and 3 remained idle waiting for Rank 1 to finish before reaching the `comm.gather` barrier).*

### d. MPI version 2: how it corrects the imbalance, and its results (total execution time, speedup, efficiency, LI)

**Imbalance Correction Mechanism:** `mpi2.py` replaces static chunking with a **dynamic Master-Worker load-balancing pattern** built on point-to-point communication (`comm.send` and `comm.recv` using `TAG_REQUEST`, `TAG_FILE`, and `TAG_STOP`). Rank 0 acts exclusively as a task coordinator, dispatching one file path at a time on demand whenever a worker process (Ranks 1, 2, and 3) finishes its previous file. Consequently, workers assigned heavier files naturally request fewer total files, synchronizing their completion times.

**Results with $p = 4$ processes:**

| Process (Rank) | Role | Processed Files | Local Time ($s$) |
| :--- | :--- | :---: | :---: |
| **Rank 0** | Master (Coordinator) | 0 | `12.904536` |
| **Rank 1** | Worker | 1,052 | `12.871123` |
| **Rank 2** | Worker | 1,003 | `12.873030` |
| **Rank 3** | Worker | 945 | `12.904505` |

* **Total Execution Time ($T_{v2}$):** `12.936121 seconds`
* **Speedup ($S_4$):** $\frac{37.560000}{12.936121} = 2.90\times$ ($2.9035$)
* **Efficiency ($E_4$):** $\frac{2.9035}{4} = 72.59\%$ ($0.7259$)
* **Load Imbalance ($LI$ across active workers):**
  * $T_{max} = 12.904505\text{ s}$, $T_{min} = 12.871123\text{ s}$, $T_{avg} = 12.882886\text{ s}$
  * $$LI = \frac{12.904505 - 12.882886}{12.882886} = 0.0017 \text{ (or } 0.17\% \text{)}$$

## 5. Analysis

### a. Parallel performance: Did MPI version 1 improve on the sequential baseline, and how close was its speedup to the ideal ($S_p = p$)? What explains the gap?

Yes, MPI Version 1 substantially improved performance over the sequential baseline, cutting execution time from `37.560000 s` down to `13.373651 s` and achieving a speedup of $S_4 = 2.81\times$.

However, this speedup falls short of the ideal linear speedup ($S_4 = 4.00\times$), yielding an efficiency of $70.21\%$. Three primary factors explain this gap:

1. **Amdahl's Law (Serial Fraction):** Initial directory scanning, file reading in Rank 0, and final dictionary aggregation cannot be parallelized.
2. **Disk I/O Contention:** All four processes concurrently read 3,000 text files from the same storage volume mounted in Docker, saturating disk read bandwidth.
3. **Communication & Synchronization Overhead:** Data serialization during `comm.bcast` and `comm.gather`, compounded by barrier waiting time caused by uneven file sizes.

### b. Load balance: What evidence of load imbalance did version 1 show (per-process local times and LI), and did version 2 reduce it enough to produce a real performance improvement?

In MPI Version 1, clear evidence of load imbalance is shown by the spread in per-process local times despite each rank processing exactly 750 files: **Rank 1 required `13.338711 s`** while **Rank 0 finished in `12.398582 s`**, producing a load imbalance of $LI = 5.14\%$ and leaving three processes idle for `0.9401 seconds`.

MPI Version 2 reduced this imbalance almost entirely, bringing $LI$ among workers down to $0.17\%$ (a spread of only `0.033 seconds` between the fastest and slowest workers). By dynamically allocating `1,052` lighter files to Rank 1 and `945` denser files to Rank 3, all workers finished simultaneously. Even though Rank 0 was dedicated solely to coordination (leaving 3 compute workers instead of 4), eliminating the straggler bottleneck produced a **real performance improvement**, reducing total wall-clock time from `13.373651 s` to `12.936121 s` (`0.4375 s` faster) and increasing overall efficiency from $70.21\%$ to $72.59\%$.

### c. Limitations: What limitations affected your experiment, and how much do they weaken your conclusions?

1. **Docker Volume Virtualization Overhead:** Mounting a Windows NTFS directory (`-v "${PWD}:/app"`) into a Linux container introduces filesystem translation latency when opening and reading 3,000 small files concurrently.
2. **Process Oversubscription & OS Noise:** Running on a general-purpose host machine with `--oversubscribe` shares physical CPU cores with background OS tasks, introducing minor run-to-run timing jitter.
3. **Fine-Grained Messaging in Version 2:** Dispatching one file per message generates 3,000 sequential `send`/`recv` round-trips, which could become a coordinator bottleneck at much larger node counts ($p \gg 4$).

These limitations do not weaken our conclusions because all three implementations were evaluated under identical hardware, OS, and container conditions, ensuring a fair, controlled comparison of relative speedup and load-balancing behavior.

## 6. Conclusions

Based on the experimental evidence, I recommend the **dynamic Master-Worker distribution strategy (MPI Version 2)** for this parallel word-counting problem because it effectively handles heterogeneous file sizes while maximizing hardware utilization. Although static partitioning (`mpi1.py`) achieved a solid speedup of $S_4 = 2.81\times$ and an efficiency of $70.21\%$, unequal word densities across files caused a $5.14\%$ load imbalance ($LI$) that left faster processes idle waiting for the slowest rank (`13.34 s` vs. `12.40 s`). In contrast, dynamic scheduling (`mpi2.py`) reduced worker load imbalance to just $0.17\%$ ($LI$), which more than compensated for dedicating Rank 0 to task coordination and improved total execution time to `12.94 s`, raising speedup to $S_4 = 2.90\times$ and overall efficiency to $72.59\%$.
