"""Central configuration for the OSmosis (OS tutor) knowledge base.

Edit SOURCES to add/remove material. Paths are relative to the project root
(the folder that contains the `rag/` package).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"          # chunks.jsonl, figures.jsonl
FIGURES_DIR = DATA_DIR / "figures"              # extracted figure / slide images
CHROMA_DIR = DATA_DIR / "chroma"                # vector database
# Hand transcriptions of image-only slides: data/annotations/<source_id>.json
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Drop extra teacher material (PDF or PPTX) in this folder; ingest picks it up.
TEACHER_NOTES_DIR = ROOT / "teacher_notes"

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
# pdf_pages are 1-based PDF page numbers (what a PDF viewer shows in its page
# box), NOT the printed page numbers. Citations use the printed numbers.
SOURCES = [
    {
        "id": "silberschatz10",
        "kind": "textbook",
        "title": "Silberschatz, Galvin & Gagne - Operating System Concepts, 10th ed.",
        "short": "Silberschatz 10e",
        "file": "Abraham-Silberschatz-Operating-System-Concepts-10th-2018.pdf",
        "pdf_pages": [
            (85, 140),    # Ch 2: Operating-System Structures (system calls, shell / CLI)
            (143, 215),   # Ch 3: Processes (fork / exec, IPC, producer-consumer)
            (216, 265),   # Ch 4: Threads & Concurrency
            (266, 329),   # Ch 5: CPU Scheduling, incl. exercises
            (332, 374),   # Ch 6: Synchronization Tools
            (375, 414),   # Ch 7: Synchronization Examples (bounded buffer, dining philosophers)
            (415, 452),   # Ch 8: Deadlocks (Banker's algorithm)
            (455, 500),   # Ch 9: Main Memory (paging, segmentation)
            (501, 570),   # Ch 10: Virtual Memory (page replacement)
            (573, 616),   # Ch 11: Mass-Storage Structure (disk scheduling)
            (659, 752),   # Ch 13-15: File-System Interface, Implementation, Internals
            (915, 962),   # Ch 20: The Linux System
        ],
    },
    {
        "id": "keerthika_m3",
        "kind": "teacher_notes",
        "title": "Teacher slides - Module 3: Scheduling (Dr. P. Keerthika, BCSE303L)",
        "short": "Teacher slides (Module 3)",
        "file": "FALLSEM2026-27_VL_BCSE303L_00100_TH_2026-08-04_Module-3---Scheduling (1).pdf",
        "skip_pages": [1],  # title slide
    },
    {
        "id": "keerthika_m1",
        "kind": "teacher_notes",
        "title": "Teacher slides - Module 1: Introduction (Dr. P. Keerthika, BCSE303L)",
        "short": "Teacher slides (Module 1)",
        "file": "FALLSEM2026-27_VL_BCSE303L_00100_TH_2026-07-09_Module-1.pdf",
        "skip_pages": [1],
    },
    {
        "id": "keerthika_m2",
        "kind": "teacher_notes",
        "title": "Teacher slides - Module 2: OS Principles (Dr. P. Keerthika, BCSE303L)",
        "short": "Teacher slides (Module 2)",
        "file": "FALLSEM2026-27_VL_BCSE303L_00100_TH_2026-07-20_Module-2.pdf",
        "skip_pages": [1],
    },
    {
        "id": "keerthika_m3_deadlock",
        "kind": "teacher_notes",
        "title": "Teacher slides - Module 3: Deadlocks (Dr. P. Keerthika, BCSE303L)",
        "short": "Teacher slides (Module 3, deadlocks)",
        "file": "OS Module 3.pdf",
        "skip_pages": [1],
    },
    {
        "id": "keerthika_m4",
        "kind": "teacher_notes",
        "title": "Teacher slides - Module 4: Concurrency (Dr. P. Keerthika, BCSE303L)",
        "short": "Teacher slides (Module 4)",
        "file": "OS module 4.pdf",
        "skip_pages": [1],
    },
    {
        "id": "keerthika_m5a",
        "kind": "teacher_notes",
        "title": "Teacher slides - Module 5, part 1: Memory allocation (Dr. P. Keerthika, BCSE303L)",
        "short": "Teacher slides (Module 5, part 1)",
        "file": "OS Module 5-1.pdf",
        "skip_pages": [1],
    },
    {
        "id": "notes_m5b",
        "kind": "teacher_notes",
        "title": "Teacher slides - Module 5, part 2: Memory management",
        "short": "Teacher slides (Module 5, part 2)",
        "file": "OS module 5-2.pdf",
        "skip_pages": [1],
    },
    # Lab material. "format": "notes" = a continuous document split at its section headings
    # (rag/parse_notes.py) instead of one chunk group per slide.
    {
        "id": "lab_linux_commands",
        "kind": "teacher_notes",
        "title": "Lab notes - Basic Unix commands",
        "short": "Linux commands notes",
        "file": "Linux commands.pdf",
        "format": "notes",
        "headings": "numbered",
    },
    {
        "id": "lab_shell_slides",
        "kind": "teacher_notes",
        "title": "Lab slides - Linux shell programming",
        "short": "Shell programming lab slides",
        "file": "Shell programming.pptx",
    },
    {
        "id": "lab_shell_exercises",
        "kind": "teacher_notes",
        "title": "Lab exercises - Basic shell scripting using bash (after H.-W. Loidl, Heriot-Watt)",
        "short": "Shell scripting exercises",
        "file": "Shell programming - sample  excersice.doc",
        "format": "notes",
        "headings": "lines",
    },
]

# ---------------------------------------------------------------------------
# Course conventions (teacher's slides win over the textbook)
# ---------------------------------------------------------------------------
CONVENTIONS = """\
- Priority scheduling: smallest integer = highest priority (teacher's worked examples).
- Ties: earlier arrival first, then lower process number.
- Round Robin: a process arriving at the same instant a quantum expires joins the
  ready queue BEFORE the preempted process.
- Response time = first time on CPU - arrival time. Waiting time = TAT - burst.
- TAT = completion time - arrival time.
"""

# ---------------------------------------------------------------------------
# Embeddings / retrieval
# ---------------------------------------------------------------------------
# bge-base is a good fit for an 8 GB laptop. Switch to "BAAI/bge-large-en-v1.5"
# for slightly better retrieval if your machine handles it (rebuild the index).
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_TARGET_CHARS = 1100
CHUNK_MAX_CHARS = 1600

TOP_K = 6
# How the keyword (BM25) and embedding (BGE) rankings are combined: weighted Reciprocal Rank
# Fusion, score = sum of weight / (rrf_k + rank), then
#   teacher_boost          x for teacher material (slides, lab notes) - a small preference
#   example_teacher_boost  x once more for example / numerical questions (rag/intent.py): the
#                          course's worked examples and scripts are in the teacher material
#   back_matter_weight     x for textbook exercises, projects and reading lists, which name
#                          topics without explaining them
#   max_per_section        how many passages of one section may reach the top k (None = no cap)
#   kind_slot              move the best textbook and best teacher passage up to this place (None = off)
# Chosen by `python eval_retrieval.py --tune` (see docs/retrieval_evaluation.md). The settings used
# before: teacher_boost 1.15, no example boost, back_matter_weight 1.0.
FUSION = {"bm25_weight": 1.0, "dense_weight": 1.0, "rrf_k": 60, "teacher_boost": 1.05,
          "example_teacher_boost": 1.15, "back_matter_weight": 0.5, "max_per_section": None, "kind_slot": None,
          "depth": 50}
FIGURE_HITS = 2        # only the top passages contribute their linked figures
FIGURE_POOL = 10       # a searched figure must rank this high in every ranker to be shown

# ---------------------------------------------------------------------------
# LLM (any OpenAI-compatible endpoint: Groq, Ollama, OpenRouter, ...)
# ---------------------------------------------------------------------------
# Groq (free tier):   LLM_BASE_URL=https://api.groq.com/openai/v1  LLM_API_KEY=gsk_...
# Ollama (offline):   LLM_BASE_URL=http://localhost:11434/v1       LLM_API_KEY=ollama
LLM_BASE_URL_DEFAULT = "https://api.groq.com/openai/v1"
LLM_MODEL_DEFAULT = "openai/gpt-oss-120b"
