# OS Tutor - FAQ answers

Retrieval mode: hybrid

## 1. What is the convoy effect?

**Convoy effect** – In a non‑preemptive scheduling algorithm (e.g., FCFS), when a long‑running CPU‑bound process gets the CPU, all other ready processes—including short, I/O‑bound ones—must wait until that process finishes. The waiting processes form a “convoy” behind the long job, causing the CPU and I/O devices to be under‑utilized while the short jobs sit idle [S1][S2].

**Common mistake to avoid:** Assuming FCFS always yields good response times; in fact, the convoy effect can make response times for short processes very poor [S2].

**Sources**

- [S1] Teacher slides (Module 3), slide 21
- [S2] Silberschatz 10e, 5.3.1 First-Come, First-Served Scheduling, pp. 206-207
- [S3] Silberschatz 10e, 5.3.2 Shortest-Job-First Scheduling, p. 208
- [S4] Silberschatz 10e, Chapter 5 Exercises, p. EX-14
- [S5] Teacher slides (Module 3), slides 155-156
- [S6] Silberschatz 10e, 5.5.4 Processor Affinity, p. 225

**Figures**

- Gantt chart (p. 206) #1 (Silberschatz 10e)
- Gantt chart (p. 206) #2 (Silberschatz 10e)
- Figure 5.4 (Silberschatz 10e)

## 2. Explain SRTF with an example

**Shortest‑Remaining‑Time‑First (SRTF)** is the pre‑emptive version of Shortest‑Job‑First: at every instant the scheduler picks the **ready** process whose **remaining CPU‑burst time is smallest**; if a new process arrives (or returns from I/O) with a shorter remaining time, the current process is pre‑empted [S1].  

Below is the complete walk‑through of the SRTF example that appears in the lecture slides (processes with an I/O burst).

---

### 1. Process data  

| PID | Arrival AT | CPU‑burst BT₁ | I/O time | CPU‑burst BT₂ |
|-----|------------|---------------|----------|---------------|
| P1  | 2          | 4             | –        | – |
| P2  | 3          | 2             | 3        | 2 |
| P3  | 5          | 1             | –        | – |  

(From slide 101) [S5]

---

### 2. Decision points & Gantt chart  

| Time | Event (decision point) | Ready‑queue (remaining BT) before the decision | CPU given to |
|------|------------------------|-----------------------------------------------|--------------|
| 0‑2  | No process has arrived → CPU idle | – | **Idle** |
| 2‑3  | P1 arrives (AT = 2) | P1 (4) | **P1** (runs 1 unit) |
| 3‑4  | P2 arrives (AT = 3) – compare remaining times: P2 (2) < P1 (3) → pre‑empt | P1 (3), P2 (2) | **P2** |
| 4‑5  | P2 continues (still shortest) | P1 (3), P2 (1) | **P2** |
| 5‑6  | P2 finishes its first CPU burst and goes to I/O (3 units) → P3 arrives (AT = 5). Ready‑queue now: P1 (3), P3 (1). Shortest is P3 → pre‑empt | P1 (3), P3 (1) | **P3** |
| 6‑7  | P3 finishes (its only burst). Ready‑queue: P1 (3) | P1 (3) | **P1** |
| 7‑8  | P1 continues (still only ready process) | P1 (2) | **P1** |
| 8‑9  | P1 continues | P1 (1) | **P1** |
| 9‑11 | P1 terminates at 9. P2 returns from I/O at 8 (remaining 2) → only P2 in ready‑queue, so it runs to completion. | P2 (2) | **P2** |

These steps are illustrated in slides 112 – 119 and the detailed “compare remaining‑time” pictures of slides 103‑111 [S1][S5][S6].

**Resulting Gantt chart**

```
| Idle 0‑2 | P1 2‑3 | P2 3‑5 | P3 5‑6 | P1 6‑9 | P2 9‑11 |
```

---

### 3. Performance metrics  

| PID | Completion CT | Turn‑around TAT = CT‑AT | Waiting WT = TAT‑BT₁‑BT₂ | Response RT = First‑CPU‑Start‑AT |
|-----|---------------|------------------------|--------------------------|---------------------------------|
| P1  | 9  | 9‑2 = **7** | 7‑4 = **3** | 2‑2 = **0** |
| P2  | 11 | 11‑3 = **8**| 8‑(2+2) = **4** | 3‑3 = **0** |
| P3  | 6  | 6‑5 = **1** | 1‑1 = **0** | 5‑5 = **0** |

(From slide 113) [S1]

**Averages**

* Avg TAT = (7 + 8 + 1) / 3 = **5.3 ms**  
* Avg WT  = (3 + 4 + 0) / 3 = **2.3 ms**  
* Avg RT  = (0 + 0 + 0) / 3 = **0 ms**  

(Computed directly from the table; the slide already lists the same values) [S1].

**CPU utilisation & throughput**

* Total CPU‑burst time = 9 ms, total elapsed time = 11 ms → utilisation = 9/11 × 100 ≈ **81.8 %**.  
* Throughput = 3 processes / 11 ms = **0.27 processes/ms** (rounded to 0.3 in the slide).  
* Number of context‑switches = **4** (each pre‑emption counted) [S1][S4].

---

### 4. How the schedule is derived (step‑by‑step)  

1. **Time 0‑2** – No arrivals → CPU idle (slide 112).  
2. **Time 2** – P1 arrives; it is the only ready process, so it starts (slide 112).  
3. **Time 3** – P2 arrives. The scheduler compares remaining times (P2 = 2 < P1 = 3) and pre‑empts P1 (slide 103‑104).  
4. **Time 5** – P2 finishes its first CPU burst and goes to I/O; simultaneously P3 arrives. The ready‑queue now holds P1 (3) and P3 (1); the shorter remaining time (P3) gets the CPU (slide 107).  
5. **Time 6‑9** – P3 finishes; P1 is the only ready process, so it runs to completion (slides 108‑110).  
6. **Time 9‑11** – P1 terminates; P2 returns from I/O at 8 with 2 units left, so it runs to finish (slide 111).  

Each of these decisions follows the SRTF rule of “pick the ready process with the smallest remaining CPU time” and treats **arrival**, **completion of a CPU burst**, and **return from I/O** as decision points (course convention).  

---

### 5. Common mistake to avoid  

A frequent error is **ignoring the I/O‑completion event as a decision point**; if the scheduler does not re‑evaluate the ready‑queue when a process returns from I/O, it may keep a longer‑remaining process on the CPU and violate the SRTF rule, leading to an incorrect Gantt chart and wrong waiting‑time calculations [S6].  

---  

**Key take‑aways**  

* SRTF always runs the process with the **shortest remaining CPU time**.  
* Every **arrival** or **I/O return** triggers a re‑evaluation of the ready‑queue.  
* Pre‑emptions are counted separately from ordinary context switches.  

(These points follow the course conventions for pre‑emptive scheduling.)

**Sources**

- [S1] Teacher slides (Module 3), slides 112-114
- [S2] Teacher slides (Module 3), slides 44-46
- [S3] Teacher slides (Module 3), slides 47-49
- [S4] Teacher slides (Module 3), slides 41-43
- [S5] Teacher slides (Module 3), slides 101-105
- [S6] Teacher slides (Module 3), slides 106-111

**Figures**

- Slide 112 (Teacher slides (Module 3))
- Slide 44 (Teacher slides (Module 3))
- Slide 45 (Teacher slides (Module 3))
- Slide 47 (Teacher slides (Module 3))

## 3. Solve using Round Robin with time quantum 3. P1: AT 0, BT 5. P2: AT 1, BT 4. P3: AT 3, BT 2. P4: AT 6, BT 3. Give the Gantt chart, CT, TAT, WT and the averages.

Algorithm: Round Robin, time quantum = 3

Gantt chart:
| P1 0-3 | P2 3-6 | P3 6-8 | P1 8-10 | P4 10-13 | P2 13-14 |

| Process | AT | BT | CT | TAT | WT | RT |
|---|---|---|---|---|---|---|
| P1 | 0 | 5 | 10 | 10 | 5 | 0 |
| P2 | 1 | 4 | 14 | 13 | 9 | 2 |
| P3 | 3 | 2 | 8 | 5 | 3 | 3 |
| P4 | 6 | 3 | 13 | 7 | 4 | 4 |

Average TAT = 35/4 = 8.75
Average WT = 21/4 = 5.25
Average RT = 9/4 = 2.25
CPU utilisation = 14/14 = 100%    Throughput = 4/14 = 0.29 processes per time unit

Formulas: TAT = CT - AT, WT = TAT - BT, RT = first time on CPU - AT.

Decisions:
  t=0: P1 arrives. Ready queue (head first): P1. -> P1 runs.
  t=1: P2 arrives. Running: P1. Ready: P2. -> P1 continues.
  t=3: P3 arrives; P1's quantum expires (2 left), it goes to the tail. Ready queue (head first): P2, P3, P1. -> P2 runs.
  t=6: P4 arrives; P2's quantum expires (1 left), it goes to the tail. Ready queue (head first): P3, P1, P4, P2. -> P3 runs.
  t=8: P3 completes. Ready queue (head first): P1, P4, P2. -> P1 runs.
  t=10: P1 completes. Ready queue (head first): P4, P2. -> P4 runs.
  t=13: P4 completes. Ready queue (head first): P2. -> P2 runs.
  t=14: P2 completes. All processes have finished.

Explanation
-----------
**Round‑Robin (time‑quantum = 3)**  

The simulator’s exact schedule is:

**Gantt chart** (as given in the computed solution)  

| P1 0‑3 | P2 3‑6 | P3 6‑8 | P1 8‑10 | P4 10‑13 | P2 13‑14 |

**Step‑by‑step decisions** (quoted verbatim from the computed solution; the same reasoning follows the Round‑Robin method shown in the slides)  

1. **t = 0** – P1 arrives. Ready queue: **P1** → *P1 runs*.  
2. **t = 1** – P2 arrives. Ready queue: **P2** (P1 still running) → *P1 continues* until its quantum ends.  
3. **t = 3** – P3 arrives; P1’s quantum expires with 2 units left, so P1 is placed at the **tail** of the queue. Ready queue (head‑first): **P2, P3, P1** → *P2 runs*.  
4. **t = 6** – P4 arrives; P2’s quantum expires with 1 unit left, so P2 goes to the tail. Ready queue: **P3, P1, P4, P2** → *P3 runs*.  
5. **t = 8** – P3 finishes. Ready queue: **P1, P4, P2** → *P1 runs*.  
6. **t = 10** – P1 finishes. Ready queue: **P4, P2** → *P4 runs*.  
7. **t = 13** – P4 finishes. Ready queue: **P2** → *P2 runs*.  
8. **t = 14** – P2 finishes. All processes are done.

These decisions use the **Round‑Robin queueing rule** that a newly arriving process is placed **before** the pre‑empted process when they occur at the same instant (see the course convention). The ordering of the ready queue at each decision point matches the step‑by‑step illustration in the slides for RR examples [**S3**] and the general RR description in the slides [**S1**].

**Completion times (CT) and derived metrics** (computed by the simulator; formulas from the slides)  

- **P1**: CT = 10 → TAT = CT − AT = 10 − 0 = 10, WT = TAT − BT = 10 − 5 = 5.  
- **P2**: CT = 14 → TAT = 14 − 1 = 13, WT = 13 − 4 = 9.  
- **P3**: CT = 8  → TAT = 8 − 3 = 5,  WT = 5 − 2 = 3.  
- **P4**: CT = 13 → TAT = 13 − 6 = 7,  WT = 7 − 3 = 4.  

Formulas used: **TAT = CT − AT** and **WT = TAT − BT** (Slide 76 [**S2**]).

**Averages** (taken directly from the simulator output)  

- Average Turn‑Around Time = 8.75 units.  
- Average Waiting Time = 5.25 units.  
- Average Response Time = 2.25 units (RT = first‑CPU‑time − AT, also from Slide 76 [**S2**]).

---

**Common mistake to avoid** – When a quantum expires at the same instant a new process arrives, many students incorrectly place the pre‑empted process ahead of the newcomer. The correct rule (course convention) is to enqueue the newly arriving process **before** the pre‑empted one; otherwise the Gantt chart and the waiting‑time calculations will be off.

