"""OSmosis - the OS tutor's chat interface.

    streamlit run app.py

Three pages over rag.ask, rag.numericals and rag.quiz:
  Chat              questions answered from the knowledge base, with citations and figures
  Numerical solver  CPU scheduling, Banker's algorithm, page replacement and disk scheduling,
                    solved exactly from the numbers entered
  Quiz              MCQ, numerical and viva questions with marking and adaptive difficulty
Set LLM_API_KEY before starting to get written answers; without it the app shows the
retrieved passages (and still solves numericals).
"""
import os
import random
import re
import zlib
from pathlib import Path

import pandas as pd
import streamlit as st

from rag import banker, disk, numericals, paging
from rag.ask import Reply, answer, explain_mistake
from rag.config import DATA_DIR, LLM_MODEL_DEFAULT
from rag.diagnosis import Finding, diagnose, similar_problem
from rag.diagnosis import rule as rule_text
from rag import learning
from rag.misconceptions import CONCEPTS, MISCONCEPTIONS
from rag.trace import Session
from rag.trace import event_for
from rag.trace import events as lab_events
from rag.numericals import LABELS, kind
from rag.quiz import TOPICS, Progress, Question, check_numeric, grade_viva, next_question, offline_topics
from rag.retriever import Retriever
from rag.solver import Problem, ProblemError, Proc, Solution, title

EXAMPLES = [
    "What is the convoy effect?",
    "Explain the Banker's algorithm",
    "Difference between paging and segmentation",
    "How can the dining philosophers deadlock?",
    "Compare the SCAN and C-SCAN disk scheduling algorithms",
    "What happens when a process calls fork()?",
    "How do I change file permissions with chmod?",
    "How do you write a while loop in a shell script?",
]
ALGO_LABELS = {"fcfs": "FCFS", "sjf": "SJF", "srtf": "SRTF", "priority": "Priority",
               "priority_p": "Priority (preemptive)", "rr": "Round Robin"}
DEFAULT_ROWS = [{"AT": 0, "BT": 5, "Priority": 2}, {"AT": 1, "BT": 3, "Priority": 1},
                {"AT": 2, "BT": 1, "Priority": 4}, {"AT": 3, "BT": 2, "Priority": 3}]
NEW_ROW = {"AT": 0, "BT": 1, "Priority": 1}
PALETTE = ["#8ecae6", "#ffb703", "#b7e4c7", "#f4a6a6", "#cdb4db", "#fdd49e", "#a8dadc", "#e9c46a"]
IDLE = "repeating-linear-gradient(45deg,#d9d9d9,#d9d9d9 6px,#f2f2f2 6px,#f2f2f2 12px)"
PROGRESS_FILE = DATA_DIR / "quiz_progress.json"   # quiz results, kept between sessions

ASSETS = Path(__file__).parent / "assets"
st.set_page_config(page_title="OSmosis", page_icon=str(ASSETS / "logo_icon.svg"), layout="centered")
HAS_KEY = bool(os.environ.get("LLM_API_KEY"))


@st.cache_resource(show_spinner="Loading the knowledge base ...")
def retriever() -> Retriever:
    return Retriever()


def logo_file() -> Path:
    """The wordmark that reads on the current theme."""
    return ASSETS / ("logo_dark.svg" if st.get_option("theme.base") == "dark" else "logo_light.svg")


# ---------------------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------------------
def gantt_html(s: Solution) -> str:
    """The Gantt chart as proportional coloured bars with the time under each boundary."""
    return gantt_bars(s.gantt, s.end)


def gantt_bars(segments, end) -> str:
    bars, ticks = [], []
    for label, a, b in segments:
        colour = IDLE if label == "Idle" else PALETTE[(int(label[1:]) - 1) % len(PALETTE)]
        bars.append(
            f'<div title="{label}: {a} to {b}" style="flex:{b - a} 1 0;min-width:0;background:{colour};'
            f'color:#1b1b1b;text-align:center;padding:14px 0;font-weight:600;font-size:15px;'
            f'border-right:2px solid rgba(255,255,255,.85);overflow:hidden;white-space:nowrap">{label}</div>')
        ticks.append(f'<div style="flex:{b - a} 1 0;min-width:0;font-size:12px">{a}</div>')
    ticks.append(f'<div style="flex:0 0 0;font-size:12px;white-space:nowrap">{end}</div>')
    return ('<div style="padding-right:24px;margin:4px 0 14px">'
            f'<div style="display:flex;border-radius:8px;overflow:hidden">{"".join(bars)}</div>'
            f'<div style="display:flex;opacity:.75;margin-top:3px">{"".join(ticks)}</div></div>')


def show_solution(s, open_steps: bool = False):
    """Any solver's solution: tables, chart and the steps."""
    {"scheduling": show_schedule, "banker": show_banker, "paging": show_paging, "disk": show_disk}[kind(s)](
        s, open_steps)


def show_steps(lines, label="Step-by-step decisions", open_steps=False):
    with st.expander(label, expanded=open_steps):
        for line in lines:
            st.markdown(f"- {line}")


def show_schedule(s: Solution, open_steps: bool = False):
    st.markdown(f"**{title(s.problem)}**")
    st.markdown(gantt_html(s), unsafe_allow_html=True)
    cols = {"Process": [f"P{r['pid']}" for r in s.rows], "AT": [r["at"] for r in s.rows],
            "BT": [r["bt"] for r in s.rows]}
    if s.problem.algo.startswith("priority"):
        cols["Priority"] = [r["prio"] for r in s.rows]
    for name, key in (("CT", "ct"), ("TAT", "tat"), ("WT", "wt"), ("RT", "rt")):
        cols[name] = [r[key] for r in s.rows]
    st.dataframe(pd.DataFrame(cols), hide_index=True, width="stretch")
    n, busy = len(s.rows), sum(r["bt"] for r in s.rows)
    for col, (name, value) in zip(st.columns(4), (
            ("Average TAT", f"{sum(r['tat'] for r in s.rows) / n:.2f}"),
            ("Average WT", f"{sum(r['wt'] for r in s.rows) / n:.2f}"),
            ("Average RT", f"{sum(r['rt'] for r in s.rows) / n:.2f}"),
            ("CPU utilisation", f"{100 * busy / s.end:.1f}%"))):
        col.metric(name, value, border=True)
    st.caption("CT = completion time · TAT = CT − AT · WT = TAT − BT · RT = first time on CPU − AT")
    show_steps(s.trace, open_steps=open_steps)


def _vec(v) -> str:
    return " ".join(map(str, v))


def sequence_html(p: banker.Problem, check: banker.Check, start) -> str:
    """The safe sequence as coloured boxes, with Work after each process finishes."""
    cells = [f'<div style="padding:8px 12px;opacity:.8;font-size:13px">Work<br><b>{_vec(start)}</b></div>']
    for i, work in zip(check.order, check.works):
        colour = PALETTE[i % len(PALETTE)]
        cells.append('<div style="align-self:center;opacity:.6">→</div>')
        cells.append(f'<div style="background:{colour};color:#1b1b1b;border-radius:8px;padding:8px 12px;'
                     f'text-align:center;font-size:13px"><b style="font-size:15px">{p.prefix}{p.pids[i]}</b><br>'
                     f'{_vec(work)}</div>')
    return f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 12px">{"".join(cells)}</div>'


def show_banker(s: banker.Solution, open_steps: bool = False):
    p = s.problem
    res = " ".join(p.resources)
    st.markdown(f"**{banker.title(p)}**")
    cols = {"Process": [f"{p.prefix}{x}" for x in p.pids], f"Allocation ({res})": [_vec(r) for r in p.alloc]}
    if p.detect:
        cols[f"Request ({res})"] = [_vec(r) for r in p.requests]
    else:
        cols[f"Max ({res})"] = [_vec(r) for r in p.maxm]
        cols[f"Need = Max − Allocation ({res})"] = [_vec(r) for r in s.need]
    st.dataframe(pd.DataFrame(cols), hide_index=True, width="stretch")
    left, right = st.columns(2)
    left.metric(f"Available ({res})", _vec(p.available), border=True)
    right.metric(f"Total instances ({res})", _vec(s.total), border=True)
    st.markdown("**Order in which the processes finish**" if s.check.order else "**No process can finish**")
    st.markdown(sequence_html(p, s.check, p.available), unsafe_allow_html=True)
    (st.success if s.check.safe else st.error)(banker.state(p, s.check))
    if not p.detect and s.check.safe:
        st.caption("Other safe sequences may exist: any order in which each process's Need fits in Work at its "
                   "turn is correct. This one checks the processes in table order, pass after pass, as the slides do.")
    show_steps(s.check.steps, "Step-by-step (detection algorithm)" if p.detect else "Step-by-step (safety algorithm)",
               open_steps)
    if s.request:
        r = s.request
        pid, req = p.request
        st.markdown(f"**Resource request: {p.prefix}{pid} asks for ({_vec(req)})**")
        for line in r["steps"]:
            st.markdown(f"- {line}")
        if r["check"] and r["check"].order:
            st.markdown(sequence_html(p, r["check"], r["available"]), unsafe_allow_html=True)
        box = {"granted": st.success, "wait": st.warning}.get(r["verdict"], st.error)
        box(r["reason"])
        if r["check"]:
            show_steps(r["check"].steps, "Step-by-step (safety algorithm on the new state)", open_steps)


def frames_html(refs, run: paging.Run) -> str:
    """The frame table: one column per reference, the page brought in on a fault highlighted."""
    border = "border:1px solid rgba(128,128,128,.35);padding:4px 7px;text-align:center"
    head = "".join(f'<th style="{border}">{x}</th>' for x in refs)
    rows = []
    for k in range(run.frames):
        cells = []
        for t, col in enumerate(run.table):
            page = col[k]
            new = run.faults[t] and page == refs[t]
            style = f"{border};background:#ffb703;color:#1b1b1b;font-weight:700" if new else border
            cells.append(f'<td style="{style}">{"" if page is None else page}</td>')
        rows.append(f'<tr><th style="{border};white-space:nowrap">Frame {k + 1}</th>{"".join(cells)}</tr>')
    marks = "".join(f'<td style="{border};font-weight:600">{"F" if f else "✓"}</td>' for f in run.faults)
    rows.append(f'<tr><th style="{border}">Fault</th>{marks}</tr>')
    return (f'<div style="overflow-x:auto;margin:4px 0 12px"><table style="border-collapse:collapse;font-size:14px">'
            f'<tr><th style="{border}">Ref</th>{head}</tr>{"".join(rows)}</table></div>')


def show_paging(s: paging.Solution, open_steps: bool = False):
    p = s.problem
    st.markdown(f"**{paging.title(p)}**")
    n = len(p.refs)
    for r in s.runs:
        if len(s.runs) > 1:
            st.markdown(f"##### {paging.ALGOS[r.algo]}, {r.frames} frames")
        st.markdown(frames_html(p.refs, r), unsafe_allow_html=True)
        for col, (name, value) in zip(st.columns(4), (
                ("Page faults", r.n_faults), ("Hits", n - r.n_faults),
                ("Fault rate", f"{100 * r.n_faults / n:.1f}%"), ("Hit ratio", f"{100 * (n - r.n_faults) / n:.1f}%"))):
            col.metric(name, value, border=True)
        show_steps(r.steps, open_steps=open_steps and len(s.runs) == 1)
    st.caption("Highlighted: the page brought in on a fault. F = page fault, ✓ = hit.")
    if len(s.runs) > 1:
        st.info(paging.summary(s))


def seek_svg(p: disk.Problem, run: disk.Run) -> str:
    """Head movement chart in the textbook's style: cylinder across, one row per move."""
    w, pad, row = 640, 28, 26
    h = pad * 2 + row * (len(run.path) - 1)

    def x(c):
        return pad + c * (w - 2 * pad) / (p.cylinders - 1)

    parts = [f'<line x1="{pad}" y1="{pad - 10}" x2="{w - pad}" y2="{pad - 10}" stroke="currentColor" '
             'stroke-opacity=".35"/>']
    for c in (0, p.cylinders - 1):
        parts.append(f'<text x="{x(c)}" y="{pad - 15}" font-size="11" text-anchor="middle" fill="currentColor" '
                     f'fill-opacity=".7">{c}</text>')
    pts = [(x(c), pad + i * row) for i, c in enumerate(run.path)]
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts, pts[1:])):
        dash = ' stroke-dasharray="5 4" stroke-opacity=".55"' if i in run.jumps else ""
        parts.append(f'<line x1="{x1:.1f}" y1="{y1}" x2="{x2:.1f}" y2="{y2}" stroke="#2f6fed" stroke-width="2"{dash}/>')
    served = set(run.served)
    for i, ((px, py), c) in enumerate(zip(pts, run.path)):
        fill = "#2f6fed" if c in served or i == 0 else "none"
        parts.append(f'<circle cx="{px:.1f}" cy="{py}" r="4" fill="{fill}" stroke="#2f6fed" stroke-width="1.5"/>')
        anchor, dx = ("end", -8) if px > w - 90 else ("start", 8)
        parts.append(f'<text x="{px + dx:.1f}" y="{py + 4}" font-size="12" text-anchor="{anchor}" '
                     f'fill="currentColor">{c}{" (start)" if i == 0 else ""}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px;margin:4px 0 8px" role="img" '
            f'aria-label="Head movement for {disk.SHORT[run.algo]}">{"".join(parts)}</svg>')


def show_disk(s: disk.Solution, open_steps: bool = False):
    p = s.problem
    st.markdown(f"**{disk.title(p)}**")
    if p.assumed:
        st.caption("Assumed, since the question does not say: " + "; ".join(p.assumed) + ".")
    for r in s.runs:
        if len(s.runs) > 1:
            st.markdown(f"##### {disk.ALGOS[r.algo]}")
        st.markdown(seek_svg(p, r), unsafe_allow_html=True)
        st.markdown("Head path: " + " → ".join(map(str, r.path)))
        values = [("Total movement (cyl.)", s.total(r)),
                  ("Average seek", f"{s.total(r) / len(p.requests):.2f}")]
        if r.jumps:
            values.append((f"Return seek ({'counted' if p.count_return else 'not counted'})", r.return_seek))
        for col, (name, value) in zip(st.columns(3), values):
            col.metric(name, value, border=True)
        show_steps(r.steps, open_steps=open_steps and len(s.runs) == 1)
    if len(s.runs) > 1:
        best = min(s.runs, key=s.total)
        st.info(disk.summary(s) + f" Least head movement: {disk.SHORT[best.algo]}.")


def plain(text: str) -> str:
    """A passage as literal text in Markdown: shell comments (#), backticks and $ stay as typed,
    and line breaks and indentation are kept."""
    lines = []
    for line in re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$])", r"\\\1", text).split("\n"):
        body = line.lstrip(" ")
        lines.append("&nbsp;" * (len(line) - len(body)) + body)
    return "  \n".join(lines)


def show_reply(reply: Reply, with_solution: bool = True):
    if reply.solution and with_solution:
        if reply.example:
            e = reply.example
            st.caption(f"Worked example: {e.name}, {e.where} (solved exactly in code)")
            st.markdown(f"> {e.question}")
        show_solution(reply.solution)
    if reply.llm_used:
        if reply.solution:
            st.markdown("##### Explanation")
        st.markdown(reply.body)
    else:
        if reply.error:
            st.warning(f"The LLM could not be reached ({reply.error}).")
        if reply.hits:
            st.caption("No LLM answer - these are the passages the tutor would answer from.")
        for i, h in enumerate(reply.hits, 1):
            with st.expander(f"[S{i}] {h.cite}", expanded=i == 1 and not reply.solution):
                st.markdown(plain(h.item["text"]))
    if reply.figs:
        with st.expander(f"Figures ({len(reply.figs)})", expanded=reply.solution is None):
            for i, col in enumerate(st.columns(2)):
                for f in reply.figs[i::2]:
                    col.image(str(DATA_DIR / f["image"]), width="stretch",
                              caption=f"{f['label']} - {f['source_short']}")
    if reply.llm_used and reply.hits:
        with st.expander("Sources"):
            for i, h in enumerate(reply.hits, 1):
                st.markdown(f"**[S{i}]** {h.cite}")


# ---------------------------------------------------------------------------------------
# Page: chat
# ---------------------------------------------------------------------------------------
def conversation(messages):
    """Earlier (question, answer) pairs and the last numerical solved, for follow-up questions."""
    history, previous, asked = [], None, None
    for m in messages:
        if m["role"] == "user":
            asked = m["content"]
        elif isinstance(m["content"], Reply):
            history.append((asked, m["content"].text))
            if m["content"].solution:
                previous = m["content"].solution.problem
    return history, previous


def chat_page():
    messages = st.session_state.setdefault("messages", [])
    question = st.chat_input("Ask about operating systems")
    if not messages and not question:
        welcome = st.empty()
        with welcome.container():
            st.image(str(logo_file()), width=250)
            st.markdown("Your Operating Systems tutor. Ask about **processes, threads, CPU scheduling, "
                        "synchronization, deadlocks, memory, storage, file systems, Linux commands and shell "
                        "programming**: answers come from the Silberschatz textbook, your teacher's slides "
                        "(Modules 1 to 5) and the lab notes, with the page or slide cited.")
            st.caption("Try one of these")
            for i, col in enumerate(st.columns(2)):
                for q in EXAMPLES[i::2]:
                    if col.button(q, width="stretch"):
                        question = q
            st.caption("For a numerical, open **Numerical solver** in the sidebar.")
        if question:
            welcome.empty()
    for m in messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            elif isinstance(m["content"], Reply):
                show_reply(m["content"])
            else:
                st.warning(m["content"])
    if not question:
        return
    history, previous = conversation(messages)
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Working on it ..."):
                content = answer(retriever(), question, use_llm=st.session_state.use_llm,
                                 history=history, previous=previous)
            show_reply(content)
        except ProblemError as e:
            content = (f"This looks like a numerical, but it cannot be solved as written: {e} "
                       "The **Numerical solver** page in the sidebar has a form for it.")
            st.warning(content)
    messages.append({"role": "assistant", "content": content})


# ---------------------------------------------------------------------------------------
# Page: numerical solver
# ---------------------------------------------------------------------------------------
def process_table(n: int, priority: bool, with_io: bool) -> pd.DataFrame:
    """Editable table with one row per process; values survive a change in the number of rows."""
    if st.session_state.get("proc_n") != n:
        old = st.session_state.get("proc_rows", DEFAULT_ROWS)
        st.session_state.proc_base = [dict(old[i]) if i < len(old) else dict(NEW_ROW) for i in range(n)]
        st.session_state.proc_n = n
    base = pd.DataFrame(st.session_state.proc_base, columns=["AT", "BT", "Priority", "IO", "BT2"])
    base.insert(0, "Process", [f"P{i + 1}" for i in range(n)])
    whole = dict(min_value=0, step=1, format="%d")
    edited = st.data_editor(
        base, key=f"processes_{n}", hide_index=True, num_rows="fixed", width="stretch", disabled=["Process"],
        column_order=["Process", "AT", "BT"] + (["Priority"] if priority else []) + (["IO", "BT2"] if with_io else []),
        column_config={
            "AT": st.column_config.NumberColumn("Arrival time", required=True, **whole),
            "BT": st.column_config.NumberColumn("Burst time", required=True, **{**whole, "min_value": 1}),
            "Priority": st.column_config.NumberColumn("Priority", **whole),
            "IO": st.column_config.NumberColumn("I/O time", help="Leave empty if the process has no I/O", **whole),
            "BT2": st.column_config.NumberColumn("Burst after I/O", help="CPU burst after the I/O", **whole),
        })
    st.session_state.proc_rows = edited.drop(columns="Process").to_dict("records")
    return edited


def read_processes(table: pd.DataFrame, priority: bool, with_io: bool) -> list[Proc]:
    procs = []
    for i, r in enumerate(table.itertuples(index=False), 1):
        if pd.isna(r.AT) or pd.isna(r.BT) or r.BT < 1:
            raise ProblemError(f"P{i} needs an arrival time and a burst time of at least 1.")
        cpu, io = [int(r.BT)], []
        if with_io and (pd.notna(r.IO) or pd.notna(r.BT2)):
            if pd.isna(r.IO) or pd.isna(r.BT2) or r.BT2 < 1:
                raise ProblemError(f"P{i}: give both the I/O time and the burst after it, or leave both empty.")
            cpu, io = cpu + [int(r.BT2)], [int(r.IO)]
        if priority and pd.isna(r.Priority):
            raise ProblemError(f"P{i} needs a priority.")
        procs.append(Proc(i, int(r.AT), cpu, io, int(r.Priority) if priority else None))
    return procs


def scheduling_form():
    """CPU-scheduling inputs -> Problem when Solve is pressed, else None."""
    st.caption("The schedule is computed exactly, following the course conventions: ties go to the earlier "
               "arrival, then the lower process number.")
    algo = st.radio("Scheduling algorithm", list(ALGO_LABELS), format_func=ALGO_LABELS.get, horizontal=True)
    priority = algo.startswith("priority")
    left, right = st.columns(2)
    n = left.number_input("Number of processes", min_value=2, max_value=12, value=4, step=1)
    quantum, high = None, False
    if algo == "rr":
        quantum = right.number_input("Time quantum", min_value=1, value=2, step=1)
    elif priority:
        high = right.selectbox("Highest priority is the", ["smallest number", "largest number"]) == "largest number"
    with_io = st.toggle("Processes have an I/O burst between two CPU bursts")
    table = process_table(int(n), priority, with_io)
    if st.button("Solve", type="primary"):
        return Problem(algo, read_processes(table, priority, with_io), int(quantum) if quantum else None, high)
    return None


# the teacher's Banker's Example 2 (slides 32-57) and deadlock-detection example (slide 61), as defaults
BANKER_DEFAULTS = {
    "Allocation": [[2, 0, 0, 1], [3, 1, 2, 1], [2, 1, 0, 3], [1, 3, 1, 2], [1, 4, 3, 2]],
    "Max": [[4, 2, 1, 2], [5, 2, 5, 2], [2, 3, 1, 6], [1, 4, 2, 4], [3, 6, 6, 5]],
    "Need": [[2, 2, 1, 1], [2, 1, 3, 1], [0, 2, 1, 3], [0, 1, 1, 2], [2, 2, 3, 3]],
    "Available": [3, 3, 2, 1], "Total instances": [12, 12, 8, 10], "Request": [1, 1, 0, 0],
}
DETECT_DEFAULTS = {
    "Allocation": [[0, 1, 0], [2, 0, 0], [3, 0, 3], [2, 1, 1], [0, 0, 2]],
    "Request": [[0, 0, 0], [2, 0, 2], [0, 0, 0], [1, 0, 0], [0, 0, 2]],
    "Available": [0, 0, 0], "Total instances": [7, 2, 6],
}


def matrix_editor(where, label: str, n: int, m: int, default, key: str) -> list[list[int]]:
    """An editable n x m table of whole numbers, one row per process."""
    names = [chr(65 + j) for j in range(m)]
    fits = default is not None and len(default) == n and all(len(r) == m for r in default)
    rows = default if fits else [[0] * m for _ in range(n)]
    base = pd.DataFrame(rows, columns=names)
    base.insert(0, "Process", [f"P{i}" for i in range(n)])
    where.markdown(f"**{label}**")
    edited = where.data_editor(base, key=key, hide_index=True, num_rows="fixed", width="stretch",
                               disabled=["Process"], column_config={
                                   c: st.column_config.NumberColumn(c, min_value=0, step=1, format="%d")
                                   for c in names})
    if edited[names].isna().any().any():
        raise ProblemError(f"fill in every cell of the {label} table.")
    return edited[names].astype(int).values.tolist()


def vector_editor(label: str, m: int, default, key: str) -> list[int]:
    names = [chr(65 + j) for j in range(m)]
    row = default if default is not None and len(default) == m else [0] * m
    st.markdown(f"**{label}**")
    edited = st.data_editor(pd.DataFrame([row], columns=names), key=key, hide_index=True, num_rows="fixed",
                            width="stretch", column_config={
                                c: st.column_config.NumberColumn(c, min_value=0, step=1, format="%d") for c in names})
    if edited.isna().any().any():
        raise ProblemError(f"fill in every value of {label}.")
    return edited.iloc[0].astype(int).tolist()


def banker_form():
    st.caption("Need, the safety algorithm and the resource-request algorithm follow the teacher's slides: each "
               "pass checks the processes in order and lets every process whose Need fits in Work finish.")
    mode = st.radio("What to check", ["safety", "request", "detect"], horizontal=True, format_func={
        "safety": "Safe state", "request": "Resource request", "detect": "Deadlock detection"}.get)
    detect = mode == "detect"
    defaults = DETECT_DEFAULTS if detect else BANKER_DEFAULTS
    left, right = st.columns(2)
    n = int(left.number_input("Number of processes", min_value=1, max_value=10, value=5, step=1))
    m = int(right.number_input("Number of resource types", min_value=1, max_value=6, value=3 if detect else 4,
                               step=1, key=f"bk_m_{detect}"))
    left, right = st.columns(2)
    second = "Request" if detect else left.radio("Second table", ["Max", "Need"], horizontal=True)
    vector = right.radio("Free resources given as", ["Available", "Total instances"], horizontal=True)
    key = f"bk_{mode}_{n}x{m}"
    left, right = st.columns(2)
    alloc = matrix_editor(left, "Allocation", n, m, defaults["Allocation"], f"{key}_alloc")
    other = matrix_editor(right, second, n, m, defaults[second], f"{key}_{second}")
    free = vector_editor(vector, m, defaults[vector], f"{key}_{vector}")
    request = None
    if mode == "request":
        who = st.selectbox("Requesting process", [f"P{i}" for i in range(n)], index=min(1, n - 1))
        request = (int(who[1:]), vector_editor(f"Request from {who}", m, defaults["Request"], f"{key}_req"))
    if not st.button("Solve", type="primary"):
        return None
    if vector == "Total instances":
        free = [t - sum(r[j] for r in alloc) for j, t in enumerate(free)]
        if min(free) < 0:
            raise ProblemError("the total is smaller than what is already allocated.")
    pids = list(range(n))
    if detect:
        return banker.Problem(pids, alloc, None, free, None, other)
    maxm = other if second == "Max" else [[a + x for a, x in zip(ar, nr)] for ar, nr in zip(alloc, other)]
    return banker.Problem(pids, alloc, maxm, free, request)


def _numbers(text: str, what: str) -> list[int]:
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if not nums:
        raise ProblemError(f"enter the {what} as whole numbers separated by commas or spaces.")
    return nums


def paging_form():
    st.caption("Frames start empty. FIFO replaces the page loaded first, LRU the page used least recently, "
               "Optimal the page used farthest in the future.")
    refs = st.text_input("Reference string", "7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1")
    left, right = st.columns([1, 2])
    frames = left.text_input("Number of frames", "3", help="Several counts, e.g. 3, 4, are solved side by side "
                                                           "(this shows Belady's anomaly for FIFO).")
    algos = right.pills("Algorithms", list(paging.ALGOS), selection_mode="multi", default=list(paging.ALGOS),
                        format_func=paging.SHORT.get)
    if not st.button("Solve", type="primary"):
        return None
    if not algos:
        raise ProblemError("pick at least one algorithm.")
    return paging.Problem(_numbers(refs, "reference string"), sorted(set(_numbers(frames, "number of frames"))),
                          [a for a in paging.ALGOS if a in algos])


def disk_form():
    st.caption("Total head movement in cylinders. SCAN and C-SCAN go to the end of the disk; LOOK and C-LOOK turn "
               "at the last request.")
    requests = st.text_input("Request queue (in arrival order)", "98, 183, 37, 122, 14, 124, 65, 67")
    a, b, c = st.columns(3)
    head = int(a.number_input("Head starts at", min_value=0, value=53, step=1))
    cylinders = int(b.number_input("Cylinders", min_value=2, value=200, step=1, help="Numbered 0 to cylinders − 1"))
    up = c.selectbox("Head moving", ["towards larger numbers", "towards 0"]) == "towards larger numbers"
    algos = st.pills("Algorithms", list(disk.ALGOS), selection_mode="multi", default=list(disk.ALGOS),
                     format_func=disk.SHORT.get)
    count_return = st.toggle("Count the return seek of C-SCAN and C-LOOK", value=True)
    if not st.button("Solve", type="primary"):
        return None
    if not algos:
        raise ProblemError("pick at least one algorithm.")
    return disk.Problem(_numbers(requests, "request queue"), head, [a for a in disk.ALGOS if a in algos], up,
                        cylinders, count_return)


SOLVER_FORMS = {"scheduling": scheduling_form, "banker": banker_form, "paging": paging_form, "disk": disk_form}


def solver_page():
    st.title("Numerical solver")
    what = st.segmented_control("Problem", list(SOLVER_FORMS), format_func=LABELS.get, default="scheduling",
                                key="solver_kind") or "scheduling"
    with st.container(border=True):
        try:
            problem = SOLVER_FORMS[what]()
            if problem is not None:
                st.session_state.solved = {"kind": what, "solution": numericals.solve(problem),
                                           "text": numericals.to_text(problem), "reply": None}
        except ProblemError as e:
            st.session_state.pop("solved", None)
            st.error(str(e))

    solved = st.session_state.get("solved")
    if not solved or solved["kind"] != what:
        return
    st.subheader("Solution")
    show_solution(solved["solution"], open_steps=True)
    if solved["reply"] is None:
        if st.button("Explain this step by step", disabled=not st.session_state.use_llm,
                     help=None if st.session_state.use_llm else "Needs the LLM (set LLM_API_KEY)"):
            with st.spinner("Writing the explanation ..."):
                solved["reply"] = answer(retriever(), solved["text"], use_llm=True)
    if solved["reply"] is not None:
        st.subheader("Explanation")
        show_reply(solved["reply"], with_solution=False)


# ---------------------------------------------------------------------------------------
# Page: Reasoning Debugger ("Check my working")
# ---------------------------------------------------------------------------------------
DEBUG_EXAMPLES = {   # slide / textbook examples, as text numericals.parse_problem reads
    "scheduling": "SRTF. P1: AT 0, BT 8. P2: AT 1, BT 4. P3: AT 2, BT 9. P4: AT 3, BT 5.",
    "banker": "Banker's algorithm. Allocation: P0 0 1 0, P1 2 0 0, P2 3 0 2, P3 2 1 1, P4 0 0 2. Max: P0 7 5 3, "
              "P1 3 2 2, P2 9 0 2, P3 2 2 2, P4 4 3 3. Available: 3 3 2. Request from P1: (1 0 2).",
    "paging": "Page replacement with LRU. Reference string: 7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3. Number of frames: 3.",
    "disk": "Disk scheduling with SSTF. The disk has 200 cylinders (0-199). Head at 53, moving towards larger "
            "cylinders. Request queue: 98, 183, 37, 122, 14, 124, 65, 67.",
}
STAGES = ("Hint", "Why is it wrong?", "Show the correction")


def debug_problem(kind: str):
    """The problem being worked on -> Problem, or None (with the reason shown)."""
    key = f"dbg_text_{kind}"
    st.session_state.setdefault(key, DEBUG_EXAMPLES[kind])
    if f"dbg_pending_{kind}" in st.session_state:          # set by "Try a similar problem" below the box
        st.session_state[key] = st.session_state.pop(f"dbg_pending_{kind}")
    left, right = st.columns(2)
    if left.button("New practice problem", width="stretch", key=f"dbg_new_{kind}"):
        try:
            current = numericals.parse_problem(st.session_state[key])
            st.session_state[key] = numericals.to_text(similar_problem(current, st.session_state.setdefault(
                "rng", random.Random())))
        except (ProblemError, TypeError):
            st.session_state[key] = DEBUG_EXAMPLES[kind]
        st.session_state.pop(f"dbg_result_{kind}", None)
    solved = st.session_state.get("solved")
    if right.button("Use the Numerical solver's problem", width="stretch", key=f"dbg_use_{kind}",
                    disabled=not solved or solved["kind"] != kind):
        st.session_state[key] = solved["text"]
        st.session_state.pop(f"dbg_result_{kind}", None)
    text = st.text_area("Problem", key=key, height=90)
    try:
        problem = numericals.parse_problem(text)
    except ProblemError as e:
        st.error(f"This problem cannot be read: {e}")
        return None
    if problem is None or numericals.kind(problem) != kind:
        st.error(f"Write a {LABELS[kind].lower()} problem here (see the example).")
        return None
    return problem


def debug_answer(kind: str, problem, tag: str) -> dict:
    """Inputs for the student's own working -> the answer dict rag.diagnosis reads."""
    if kind == "scheduling":
        st.text_input("Your Gantt chart", key="dbg_gantt", placeholder="P1 0-1, P2 1-5, P4 5-10, ...",
                      help="One slice per process run: P1 0-3, P2 3-7, Idle 7-8, ...")
        cols = ["CT", "TAT", "WT", "RT"]
        base = pd.DataFrame([[f"P{q.pid}"] + [None] * 4 for q in problem.procs], columns=["Process"] + cols)
        st.markdown("**Your table** (leave cells empty to skip them)")
        table = st.data_editor(base, key=f"dbg_table_{tag}", hide_index=True, num_rows="fixed",
                               width="stretch", disabled=["Process"], column_config={
                                   c: st.column_config.NumberColumn(c, step=1.0) for c in cols})
        a, b = st.columns(2)
        avg_tat = a.number_input("Your average TAT", value=None, step=0.01, format="%.2f")
        avg_wt = b.number_input("Your average WT", value=None, step=0.01, format="%.2f")
        rows = {int(r.Process[1:]): {c.lower(): (None if pd.isna(getattr(r, c)) else float(getattr(r, c)))
                                     for c in cols} for r in table.itertuples(index=False)}
        return {"gantt": st.session_state.dbg_gantt.strip(), "table": rows,
                "averages": {"tat": avg_tat, "wt": avg_wt}}
    if kind == "banker":
        a, b = st.columns([2, 1])
        seq = a.text_input("Your safe sequence", placeholder="P1, P3, P4, P0, P2")
        safe = b.radio("Is the state safe?", ["Not answered", "Safe", "Unsafe"], horizontal=True)
        req = None
        if problem.request:
            req = st.radio(f"Can P{problem.request[0]}'s request be granted immediately?",
                           ["Not answered", "granted", "unsafe", "wait", "error"], horizontal=True, format_func={
                               "Not answered": "Not answered", "granted": "Yes, granted",
                               "unsafe": "No: unsafe afterwards", "wait": "No: not available",
                               "error": "No: exceeds its claim"}.get)
        need = None
        if not problem.detect and st.toggle("Also check my Need matrix"):
            names = problem.resources
            base = pd.DataFrame([[f"{problem.prefix}{x}"] + [None] * len(names) for x in problem.pids],
                                columns=["Process"] + names)
            ed = st.data_editor(base, key=f"dbg_need_{tag}", hide_index=True, num_rows="fixed", width="stretch",
                                disabled=["Process"])
            need = [[None if pd.isna(v) else int(v) for v in row[1:]] for row in ed.itertuples(index=False)]
        return {"sequence": seq.strip() or None, "safe": None if safe == "Not answered" else safe == "Safe",
                "request": None if req in (None, "Not answered") else req, "need": need}
    if kind == "paging":
        marks = st.text_input(f"Your hit / fault marks ({len(problem.refs)} references)",
                              placeholder="F F F F H F H F ...", help="One letter per reference: F fault, H hit.")
        faults = st.number_input("Your number of page faults", value=None, step=1, min_value=0)
        return {"marks": marks.strip() or None, "faults": faults}
    order = st.text_input("Your order of service (cylinders)", placeholder="65, 67, 37, 14, 98, ...")
    total = st.number_input("Your total head movement", value=None, step=1, min_value=0)
    return {"order": order.strip() or None, "total": total}


def show_finding(f: Finding, key: str):
    """Staged feedback: the observation first, then each stage on request."""
    if f.misconception:
        m = MISCONCEPTIONS[f.misconception]
        st.caption(f"{'Cause' if f.certain else 'Possible cause'}: **{m.name}**")
    st.markdown(f"**{f.where}** - {f.observation}")
    shown = st.session_state.setdefault(key, 0)
    texts = (f.hint, f.explanation, f.correction)
    for i in range(shown):
        st.markdown(f"**{STAGES[i]}** {texts[i]}")
    if shown == 3 and f.consequences:
        st.markdown("**This also changes:** " + "; ".join(f.consequences) + ".")
    if shown < 3 and st.button(STAGES[shown], key=f"{key}_next"):
        st.session_state[key] = shown + 1
        st.rerun()
    return shown


def debugger_page():
    st.title("Check my working")
    st.caption("Enter your own working for a problem. OSmosis compares it step by step with the exact solver, "
               "finds the first step that goes wrong, and helps you fix it one hint at a time.")
    what = st.segmented_control("Problem type", list(DEBUG_EXAMPLES), format_func=LABELS.get, default="scheduling",
                                key="dbg_kind") or "scheduling"
    problem = debug_problem(what)
    if problem is None:
        return
    text = numericals.to_text(problem)
    tag = str(zlib.crc32(text.encode()))                    # widgets reset when the problem changes
    with st.container(border=True):
        st.markdown(f"**{numericals.title(problem)}**")
        answer_given = debug_answer(what, problem, tag)
        if st.button("Check my working", type="primary"):
            d = diagnose(problem, answer_given)
            st.session_state[f"dbg_result_{what}"] = (text, d)
            st.session_state[f"dbg_stage_{what}"] = 0
            save_evidence(learning.record_diagnosis(load_progress(), d, text))
    text_d = st.session_state.get(f"dbg_result_{what}")
    if text_d is None or text_d[0] != text:
        return
    d = text_d[1]
    if d.verdict == "correct":
        st.success("All correct: " + ", ".join(d.correct) + ".")
    elif d.verdict == "unclear":
        st.info("Nothing could be checked yet. " + " ".join(d.unclear))
    else:
        st.warning(f"{len(d.findings)} thing{'s' * (len(d.findings) != 1)} to fix. Start with the first one.")
    if d.verdict != "unclear":
        st.caption("Checked step by step against the exact solver (deterministic, no LLM).")
    if d.correct and d.verdict != "correct":
        st.markdown("**Right so far:** " + ", ".join(d.correct) + ".")
    for u in d.unclear if d.verdict != "unclear" else []:
        st.caption(u)
    if d.first:
        with st.container(border=True):
            shown = show_finding(d.first, f"dbg_stage_{what}")
        rest = d.findings[1:]
        if rest:
            with st.expander(f"Other differences ({len(rest)})"):
                for f in rest:
                    st.markdown(f"**{f.where}** - {f.observation} {f.explanation}")
        if shown == 3:
            with st.expander("Full solution", expanded=False):
                show_solution(d.solution, open_steps=True)
        explain_with_notes(d, what)
    cols = st.columns(2)
    if d.first and cols[0].button("See it run in OS Lab", width="stretch",
                                  help="Step through the real algorithm from where your working went wrong"):
        lab = lab_events(d.problem)
        session = Session(lab)
        session.jump(event_for(lab, d.first.where))
        st.session_state[f"lab_{what}"] = session
        st.session_state[f"lab_text_{what}"] = numericals.to_text(d.problem)
        st.session_state["lab_kind"] = what
        st.switch_page(PAGES["lab"])
    if d.verdict != "unclear" and cols[1].button("Try a similar problem", width="stretch"):
        st.session_state[f"dbg_pending_{what}"] = numericals.to_text(similar_problem(
            d.problem, st.session_state.setdefault("rng", random.Random())))
        st.session_state.pop(f"dbg_result_{what}", None)
        st.rerun()


def explain_with_notes(d, what: str):
    """An LLM explanation of the first mistake, from retrieved course passages (cited); the solver's
    result and its description of the mistake are passed in as fixed facts (rag.ask.explain_mistake)."""
    if not st.session_state.use_llm:
        return
    key = f"dbg_notes_{what}"
    f = d.first
    if st.session_state.get(key, (None,))[0] != (f.where, f.explanation):
        if not st.button("Explain with the course notes"):
            return
        with st.spinner("Reading the course notes ..."):
            st.session_state[key] = ((f.where, f.explanation),
                                     explain_mistake(retriever(), d.problem, f.where, f.hint, f.explanation))
    reply = st.session_state[key][1]
    st.caption("Written by the LLM from the course passages cited below; the numbers come from the exact solver.")
    show_reply(reply, with_solution=False)


def save_evidence(added):
    """Write the progress file when new learning evidence was recorded."""
    if added:
        PROGRESS_FILE.write_text(load_progress().to_json(), encoding="utf-8")


# ---------------------------------------------------------------------------------------
# Page: OS Lab - Predict, Run, Explain
# ---------------------------------------------------------------------------------------
LAB_EXAMPLES = {
    "scheduling": "Round Robin, time quantum 2. P1: AT 0, BT 5. P2: AT 1, BT 3. P3: AT 2, BT 1. P4: AT 3, BT 2. "
                  "P5: AT 4, BT 3.",
    "banker": DEBUG_EXAMPLES["banker"].replace(" Request from P1: (1 0 2).", ""),
    "paging": DEBUG_EXAMPLES["paging"],
    "disk": DEBUG_EXAMPLES["disk"],
}


def chips(items, colour_of=None) -> str:
    """Process ids (or pages) as coloured boxes in a row; '(empty)' when there are none."""
    if not items:
        return '<span style="opacity:.6">(empty)</span>'
    out = []
    for x in items:
        n = int(re.sub(r"\D", "", str(x)) or 0)
        colour = colour_of(x) if colour_of else PALETTE[(n - 1) % len(PALETTE)]
        out.append(f'<span style="display:inline-block;background:{colour};color:#1b1b1b;border-radius:6px;'
                   f'padding:4px 10px;margin:2px 4px 2px 0;font-weight:600">{x}</span>')
    return "".join(out)


def lab_state(lab, state: dict, final: bool):
    """The system state before (or after) an event, drawn for the lab's kind."""
    if lab.kind == "scheduling":
        cols = st.columns(3)
        cols[0].metric("Time", state["time"], border=True)
        cols[1].metric("On the CPU" if final else "Was running", state["running"], border=True)
        if "quantum" in state and not final and state["running"] != "CPU idle":
            cols[2].metric("Quantum used", f"{state['used']} / {state['quantum']}", border=True)
        elif final and state.get("preempted"):
            cols[2].metric("Preempted", state["preempted"], border=True)
        if not final and state["events"]:
            st.markdown("**Now:** " + "; ".join(state["events"]) + ".")
        st.markdown(f"**Ready queue{' (head first)' if lab.problem.algo in ('rr', 'fcfs') else ''}:** "
                    + chips(state["ready"]), unsafe_allow_html=True)
        if not final:
            rem = state["remaining"]
            if rem:
                st.dataframe(pd.DataFrame([rem], index=["CPU time left"]), width="stretch")
            if state["gantt"]:
                st.markdown(gantt_bars(state["gantt"], state["time"]), unsafe_allow_html=True)
    elif lab.kind == "banker":
        cols = st.columns(3)
        cols[0].metric("Checking", state["process"], border=True)
        cols[1].metric("Work", state["work"], border=True)
        key = "need" if "need" in state else "request"
        cols[2].metric(key.title(), state[key], border=True)
        st.markdown("**Finished:** " + chips(state["finished"]), unsafe_allow_html=True)
    elif lab.kind == "paging":
        cols = st.columns(2)
        cols[0].metric("Reference", state["reference"], border=True)
        cols[1].metric("Page", state["page"], border=True)
        frames = state["frames"].split()
        st.markdown("**Frames:** " + chips(frames, lambda x: "#e6e6e6" if x == "-" else "#8ecae6"),
                    unsafe_allow_html=True)
    else:
        cols = st.columns(2)
        cols[0].metric("Head at", state["head"], border=True)
        cols[1].metric("Moved so far", state["moved so far"], border=True)
        st.markdown("**Pending requests:** " + (state["pending"] or "none"))


def lab_page():
    st.title("OS Lab: Predict, Run, Explain")
    st.caption("Step through a real run of an algorithm. Before each decision, predict what the operating system "
               "will do, then run it and see why. Every step comes from the exact solver.")
    what = st.segmented_control("Experiment", list(LAB_EXAMPLES), format_func=LABELS.get, default="scheduling",
                                key="lab_kind") or "scheduling"
    key = f"lab_text_{what}"
    st.session_state.setdefault(key, LAB_EXAMPLES[what])
    with st.expander("Problem", expanded=f"lab_{what}" not in st.session_state):
        st.text_area("Problem", key=key, height=80, label_visibility="collapsed")
        if st.button("Start experiment", type="primary"):
            try:
                problem = numericals.parse_problem(st.session_state[key])
                if problem is None or numericals.kind(problem) != what:
                    raise ProblemError(f"write a {LABELS[what].lower()} problem (see the example).")
                st.session_state[f"lab_{what}"] = Session(lab_events(problem))
                st.rerun()
            except ProblemError as e:
                st.error(f"This problem cannot be run: {e}")
    s: Session | None = st.session_state.get(f"lab_{what}")
    if s is None:
        st.info("Press **Start experiment** to begin.")
        return
    lab, n = s.lab, len(s.lab.events)
    st.markdown(f"**{numericals.title(lab.problem)}**")
    right, made = s.score()
    st.progress(min(s.pos, n) / n, text=f"Step {min(s.pos + 1, n)} of {n} · predictions right: {right} / {made}")
    if s.done:
        st.success(f"Run complete. You predicted {right} of {made} decisions correctly.")
        show_solution(lab.solution, open_steps=False)
        st.markdown(f"**Think about it:** {lab.reflection}")
        if st.button("Now solve a similar problem yourself", type="primary",
                     help="Opens Check my working, which checks your own working step by step"):
            fresh = similar_problem(lab.problem, st.session_state.setdefault("rng", random.Random()))
            st.session_state["dbg_kind"] = what
            st.session_state[f"dbg_pending_{what}"] = numericals.to_text(fresh)
            st.session_state.pop(f"dbg_result_{what}", None)
            st.switch_page(PAGES["check"])
    else:
        e = s.event
        with st.container(border=True):
            st.subheader(e.title)
            lab_state(lab, e.before, final=False)
            ran = s.pos in s.ran
            choice = st.radio(e.question, e.options, index=s.predictions.get(s.pos), key=f"lab_pick_{what}_{s.pos}",
                              disabled=ran)
            if choice is not None and not ran:
                s.predict(e.options.index(choice))
            if not ran:
                if st.button("Run", type="primary", help="Execute this step of the real algorithm"):
                    s.run()
                    save_evidence(learning.record_prediction(load_progress(), lab, s.pos, s.predictions.get(s.pos),
                                                             numericals.to_text(lab.problem)))
                    st.rerun()
            else:
                guess = s.predictions.get(s.pos)
                actual = e.options[e.answer]
                if guess is None:
                    st.info(f"Actual: **{actual}**")
                elif guess == e.answer:
                    st.success(f"Right: **{actual}**")
                else:
                    st.error(f"You predicted **{e.options[guess]}**; actually: **{actual}**")
                st.markdown(f"**Why:** {e.explanation}")
                st.markdown("**After this step**")
                lab_state(lab, e.after, final=True)
    cols = st.columns(4)
    if cols[0].button("Previous", disabled=s.pos == 0, width="stretch"):
        s.previous()
        st.rerun()
    if cols[1].button("Next step", disabled=s.done or s.pos not in s.ran, width="stretch"):
        s.next()
        st.rerun()
    if cols[2].button("Replay", disabled=s.pos == 0, width="stretch", help="Back to the start, keeping your answers"):
        s.replay()
        st.rerun()
    if cols[3].button("Reset", width="stretch", help="Start again with no answers"):
        s.reset()
        st.rerun()
    if not s.done:
        with st.expander("Explain this event"):
            if lab.kind == "scheduling":
                st.markdown(rule_text(lab.problem))
            st.markdown(s.event.explanation if s.pos in s.ran else "Run the step to see the solver's explanation.")


# ---------------------------------------------------------------------------------------
# Page: My learning map (Misconception Graph)
# ---------------------------------------------------------------------------------------
# a problem to practise each concept on (a fresh variant is generated from it)
PRACTICE = {
    "sched.gantt": ("scheduling", "FCFS. P1: AT 4, BT 5. P2: AT 6, BT 4. P3: AT 0, BT 3. P4: AT 6, BT 2."),
    "sched.metrics": ("scheduling", DEBUG_EXAMPLES["scheduling"]),
    "sched.fcfs": ("scheduling", "FCFS. P1: AT 4, BT 5. P2: AT 6, BT 4. P3: AT 0, BT 3. P4: AT 6, BT 2."),
    "sched.sjf": ("scheduling", "SJF (non-preemptive). P1: AT 0, BT 10. P2: AT 1, BT 3. P3: AT 2, BT 6. P4: AT 3, BT 4."),
    "sched.preemption": ("scheduling", DEBUG_EXAMPLES["scheduling"]),
    "sched.srtf": ("scheduling", DEBUG_EXAMPLES["scheduling"]),
    "sched.priority": ("scheduling", "Priority scheduling (preemptive). P1: AT 0, BT 11, priority 2. P2: AT 5, BT 28, "
                                     "priority 0. P3: AT 12, BT 2, priority 3. P4: AT 2, BT 10, priority 1."),
    "sched.rr": ("scheduling", LAB_EXAMPLES["scheduling"]),
    "sched.io": ("scheduling", "SRTF. P1: AT 2, BT 4. P2: AT 3, BT 2, I/O 3, BT 2. P3: AT 5, BT 1."),
    "banker.need": ("banker", DEBUG_EXAMPLES["banker"]),
    "banker.safety": ("banker", LAB_EXAMPLES["banker"]),
    "banker.request": ("banker", DEBUG_EXAMPLES["banker"]),
    "paging.faults": ("paging", DEBUG_EXAMPLES["paging"]),
    "paging.fifo": ("paging", DEBUG_EXAMPLES["paging"].replace("LRU", "FIFO")),
    "paging.lru": ("paging", DEBUG_EXAMPLES["paging"]),
    "paging.optimal": ("paging", DEBUG_EXAMPLES["paging"].replace("LRU", "Optimal")),
    "disk.movement": ("disk", DEBUG_EXAMPLES["disk"].replace("SSTF", "FCFS")),
    "disk.sstf": ("disk", DEBUG_EXAMPLES["disk"]),
    "disk.scan": ("disk", DEBUG_EXAMPLES["disk"].replace("SSTF", "SCAN")),
    "disk.cscan": ("disk", DEBUG_EXAMPLES["disk"].replace("SSTF", "C-SCAN")),
}
STATUS_HELP = {
    "recurring": "seen with certainty in two or more different problems, and not yet followed by two correct attempts",
    "tentative": "seen once, or only as a likely cause; more evidence is needed before calling it a pattern",
    "resolved": "followed by at least two correct attempts on the same concepts",
}


def practise(concept: str):
    """Open Check my working on a fresh problem for this concept."""
    kind_name, template = PRACTICE[concept]
    fresh = similar_problem(numericals.parse_problem(template), st.session_state.setdefault("rng", random.Random()))
    st.session_state["dbg_kind"] = kind_name
    st.session_state[f"dbg_pending_{kind_name}"] = numericals.to_text(fresh)
    st.session_state.pop(f"dbg_result_{kind_name}", None)
    st.switch_page(PAGES["check"])


def learning_page():
    st.title("My learning map")
    st.caption("Built only from your own attempts in Check my working, OS Lab and the Quiz. A mistake seen once "
               "is shown as tentative; it becomes a pattern only when it repeats in different problems.")
    evidence = load_progress().evidence
    sample = False
    if not evidence:
        st.info("No attempts recorded yet. Check a solution in **Check my working**, predict steps in **OS Lab**, "
                "or answer numerical questions in the **Quiz**, and your map will appear here.")
        sample = st.toggle("Show a sample map (made-up data)")
        if not sample:
            return
        evidence = learning.sample_evidence()
        st.warning("SAMPLE DATA - an example of what the map looks like. These are not your results.")
    a = learning.assess(evidence)
    concepts = a["concepts"]
    counts = {k: sum(v["status"].startswith(k) for v in concepts.values() if v["attempts"] or v["active"])
              for k in ("understood", "needs more evidence", "weak")}
    for col, (name, value) in zip(st.columns(4), (("Attempts recorded", len(evidence)),
                                                  ("Understood", counts["understood"]),
                                                  ("Need more evidence", counts["needs more evidence"]),
                                                  ("Weak spots", counts["weak"]))):
        col.metric(name, value, border=True)
    st.graphviz_chart(learning.dot(a, only_seen=not st.toggle("Show every concept")), width="stretch")
    st.caption("Boxes are concepts (green understood, grey needs more evidence, orange tentative weakness, red "
               "recurring problem); arrows point from a prerequisite to what builds on it; ellipses are your "
               "misconceptions, joined to the concepts where they were seen.")

    recs = learning.recommendations(a)
    st.subheader("What to practise next")
    if not recs:
        st.success("No open misconceptions. Keep practising to turn grey concepts green.")
    for i, r in enumerate(recs[:3]):
        m = MISCONCEPTIONS[r["misconception"]]
        with st.container(border=True):
            st.markdown(f"**{m.name}** ({r['status']}) - {r['do']}")
            st.caption("Revise: " + "; ".join(r["refs"]) + (" · first check: " + ", ".join(
                CONCEPTS[p].name for p in r["prerequisites"]) if r["prerequisites"] else ""))
            left, right = st.columns(2)
            if left.button("Practise now", key=f"rec_practise_{i}", disabled=sample, width="stretch"):
                practise(r["concept"])
            if right.button("Quiz on this topic", key=f"rec_quiz_{i}", disabled=sample, width="stretch"):
                st.session_state["quiz_topic"] = r["topic"]
                st.switch_page(PAGES["quiz"])

    if a["misconceptions"]:
        st.subheader("Look at a misconception")
        pick = st.selectbox("Misconception", list(a["misconceptions"]), label_visibility="collapsed",
                            format_func=lambda m: f"{MISCONCEPTIONS[m].name} ({a['misconceptions'][m]['status']})")
        v, m = a["misconceptions"][pick], MISCONCEPTIONS[pick]
        st.markdown(f"**What it means.** {m.description}")
        st.markdown(f"**Status: {v['status']}** - {STATUS_HELP[v['status']]}. Evidence: {v['certain']} certain, "
                    f"{v['uncertain']} uncertain, in {v['problems']} problem{'s' * (v['problems'] != 1)}; "
                    f"{v['correct_since']} correct attempt{'s' * (v['correct_since'] != 1)} since the last one.")
        rows = [{"When": e.get("time", ""), "Where": e.get("source", ""), "Step": e.get("where", ""),
                 "Diagnosis": "certain" if e["certain"] else "uncertain"} for e in v["evidence"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        concept = CONCEPTS[m.concept]
        st.markdown(f"**How to fix it.** {m.remediation} Revise {'; '.join(concept.refs)}.")
        with st.expander("From the course material"):
            hits = retriever().search(f"{concept.name}: {m.description}", k=2)
            for h in hits:
                st.caption(h.cite)
                st.markdown(plain(h.item["text"][:900] + ("..." if len(h.item["text"]) > 900 else "")))
        st.markdown(f"**Check it is fixed.** Solve a fresh {m.practice} problem correctly in Check my working; "
                    "two correct attempts on these concepts mark it resolved.")
        if st.button("Practise this misconception", disabled=sample):
            practise(v["concepts"][0] if v["concepts"][0] in PRACTICE else m.concept)


# ---------------------------------------------------------------------------------------
# Page: quiz
# ---------------------------------------------------------------------------------------
KIND_LABELS = {"mcq": "Multiple choice", "numeric": "Numerical", "viva": "Viva (typed answer)"}


def load_progress() -> Progress:
    if "progress" not in st.session_state:
        try:
            st.session_state.progress = Progress.from_json(PROGRESS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st.session_state.progress = Progress()
    return st.session_state.progress


def new_question(quiz: dict, progress: Progress, kinds, topic):
    use_llm = st.session_state.use_llm
    last = quiz["q"].topic if quiz["q"] else None
    try:
        with st.spinner("Preparing a question ..."):
            q, note = next_question(retriever() if use_llm else None, progress, st.session_state.rng, kinds,
                                    None if topic == "adaptive" else topic, use_llm, quiz["seen"], last)
    except ValueError as e:  # a topic with no built-in questions while the LLM is off or failing
        quiz.update(q=None, result=None, note=str(e))
        return
    quiz.update(q=q, result=None, note=note, n=quiz["n"] + 1)


def mark(q: Question, given) -> dict:
    """Mark an answer -> {"points": 1 | 0.5 | 0, "feedback": str}."""
    if q.options:      # multiple choice, including numericals with a computed set of options
        return {"points": 1.0 if given == q.answer else 0.0, "feedback": ""}
    if q.kind == "numeric":
        return {"points": 1.0 if check_numeric(q, given) else 0.0, "feedback": ""}
    with st.spinner("Marking your answer ..."):
        score, feedback = grade_viva(q, given)
    return {"points": score / 2, "feedback": feedback}


def show_result(q: Question, result: dict):
    points = result["points"]
    if q.kind == "viva":
        box = st.success if points == 1 else st.warning if points else st.error
        box(f"Score: {int(points * 2)} / 2. {result['feedback']}")
        st.markdown("**A full answer covers:**\n" + "\n".join(f"- {p}" for p in q.key_points))
    elif points:
        st.success("Correct.")
    elif q.options:
        st.error(f"Not quite. The answer is: **{q.options[q.answer]}**")
    else:
        st.error(f"Not quite. The answer is **{q.answer:g}**.")
    if q.explanation and q.kind != "numeric":
        st.markdown(q.explanation)
    if q.cite:
        st.caption(f"Source: {q.cite}")
    if q.solution:
        show_solution(q.solution, open_steps=True)


def quiz_page():
    st.title("Quiz")
    st.caption("Questions get harder after two correct answers in a row and easier after a wrong one, and "
               "your weaker topics come up more often.")
    progress = load_progress()
    st.session_state.setdefault("rng", random.Random())
    quiz = st.session_state.setdefault("quiz", {"q": None, "result": None, "note": "", "seen": [], "n": 0})
    use_llm = st.session_state.use_llm

    for col, (name, value) in zip(st.columns(3), (
            ("Answered", progress.asked), ("Score", f"{progress.points:g} / {progress.asked}"),
            ("Accuracy", f"{100 * progress.points / progress.asked:.0f}%" if progress.asked else "–"))):
        col.metric(name, value, border=True)

    left, right = st.columns([3, 2])
    available = ["mcq", "numeric"] + (["viva"] if use_llm else [])
    kinds = left.pills("Question types", available, selection_mode="multi", default=available,
                       format_func=KIND_LABELS.get) or ["mcq"]
    topics = ["adaptive"] + (list(TOPICS) if use_llm else offline_topics(kinds))
    if st.session_state.get("quiz_topic") not in topics:      # e.g. set by "Quiz on this topic" on the learning map
        st.session_state.pop("quiz_topic", None)
    topic = right.selectbox("Topic", topics, key="quiz_topic",
                            format_func=lambda t: "Adaptive (recommended)" if t == "adaptive" else TOPICS[t][0])
    if not use_llm:
        st.caption("The LLM is off: multiple-choice questions come from a small built-in bank that covers the "
                   "scheduling, process, Linux and shell topics only, and viva questions are unavailable. Numerical questions "
                   "(CPU scheduling, Banker's algorithm, page replacement, disk scheduling) are unaffected.")

    q = quiz["q"]
    if q is None:
        if quiz["note"]:
            st.warning(quiz["note"])
        if st.button("Start quiz", type="primary"):
            new_question(quiz, progress, kinds, topic)
            st.rerun()
    else:
        with st.container(border=True):
            st.caption(f"Question {quiz['n']} · {TOPICS[q.topic][0]} · Level {q.level} · {KIND_LABELS[q.kind]}")
            if quiz["note"]:
                st.caption(quiz["note"])
            st.markdown(q.prompt)
            if quiz["result"] is None:
                with st.form(f"answer_{quiz['n']}", border=False):
                    if q.options:
                        choice = st.radio("Your answer", q.options, index=None)
                        given = q.options.index(choice) if choice is not None else None
                    elif q.kind == "numeric":
                        given = st.number_input("Your answer", value=None, step=0.01, format="%.2f")
                    else:
                        given = st.text_area("Your answer", height=120).strip() or None
                    submit, skip = st.columns([1, 5])
                    submitted = submit.form_submit_button("Submit", type="primary")
                    skipped = skip.form_submit_button("Skip")
                if skipped:
                    new_question(quiz, progress, kinds, topic)
                    st.rerun()
                if submitted and given is None:
                    st.warning("Give an answer first, or press Skip.")
                elif submitted:
                    try:
                        result = mark(q, given)
                    except Exception as e:
                        result = None
                        st.error(f"Could not mark this answer ({e}). Try again, or press Skip.")
                    if result:
                        level_before = progress.level(q.topic)
                        progress.record(q.topic, result["points"])
                        learning.record_quiz(progress, q, result["points"])
                        result["level_change"] = progress.level(q.topic) - level_before
                        quiz["result"] = result
                        quiz["seen"].append(q.prompt)
                        PROGRESS_FILE.write_text(progress.to_json(), encoding="utf-8")
                        st.rerun()
            else:
                show_result(q, quiz["result"])
                change = quiz["result"].get("level_change", 0)
                if change:
                    st.info(f"{TOPICS[q.topic][0]} moves {'up' if change > 0 else 'down'} to level "
                            f"{progress.level(q.topic)}.")
                if st.button("Next question", type="primary"):
                    new_question(quiz, progress, kinds, topic)
                    st.rerun()

    with st.expander("Your progress by topic", expanded=q is None and progress.asked > 0):
        rows = [{"Topic": TOPICS[t][0], "Answered": progress.stats.get(t, {}).get("asked", 0),
                 "Score": progress.stats.get(t, {}).get("points", 0.0),
                 "Accuracy": None if progress.accuracy(t) is None else round(100 * progress.accuracy(t)),
                 "Level": progress.level(t)} for t in TOPICS]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=100, format="%d%%")})
        weak = progress.weak_topics()
        if weak:
            st.markdown("**Revise:** " + ", ".join(TOPICS[t][0] for t in weak) + ".")
        if (progress.asked or progress.evidence) and st.button(
                "Reset progress", help="Clears quiz progress and the learning map's evidence"):
            st.session_state.progress = Progress()
            PROGRESS_FILE.unlink(missing_ok=True)
            st.session_state.pop("quiz", None)
            st.rerun()


# ---------------------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------------------
def apply_theme():
    """Follow the Dark mode switch. The theme is a server-side option, so set it and redraw."""
    base = "dark" if st.session_state.dark else "light"
    if st.get_option("theme.base") != base:
        st._config.set_option("theme.base", base)
        st.rerun()


PAGES = {
    "chat": st.Page(chat_page, title="Chat", icon=":material/chat:", default=True),
    "solver": st.Page(solver_page, title="Numerical solver", icon=":material/calculate:", url_path="solver"),
    "check": st.Page(debugger_page, title="Check my working", icon=":material/fact_check:", url_path="check"),
    "lab": st.Page(lab_page, title="OS Lab", icon=":material/science:", url_path="lab"),
    "quiz": st.Page(quiz_page, title="Quiz", icon=":material/quiz:", url_path="quiz"),
    "map": st.Page(learning_page, title="My learning map", icon=":material/hub:", url_path="map"),
}
page = st.navigation(list(PAGES.values()))
with st.sidebar:
    st.caption("Operating Systems · BCSE303L")
    st.toggle("Dark mode", value=st.get_option("theme.base") == "dark", key="dark")
    apply_theme()
st.logo(str(logo_file()), size="large", icon_image=str(ASSETS / "logo_icon.svg"))
with st.sidebar:
    st.toggle("Written answers (LLM)", value=HAS_KEY, disabled=not HAS_KEY, key="use_llm")
    if HAS_KEY:
        st.caption(f"Model: {os.environ.get('LLM_MODEL', LLM_MODEL_DEFAULT)}")
    else:
        st.caption("LLM_API_KEY is not set, so answers show the retrieved passages only. Numericals still work.")
    if st.session_state.get("messages") and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
page.run()
