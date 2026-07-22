"""Generate the architecture diagram for the GCP serverless forecasting project."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ---- palette ---------------------------------------------------------------
C_CORE = "#E8F0FE"  # local package
C_CORE_EDGE = "#4285F4"
C_PIPE = "#FEF7E0"  # pipeline/KFP
C_PIPE_EDGE = "#F9AB00"
C_VERTEX = "#E6F4EA"  # vertex serverless
C_VERTEX_EDGE = "#34A853"
C_STORE = "#FCE8E6"  # gcp storage services
C_STORE_EDGE = "#EA4335"
C_CICD = "#F3E8FD"  # ci/cd
C_CICD_EDGE = "#A142F4"
C_TEXT = "#202124"

fig, ax = plt.subplots(figsize=(15, 11))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=10, bold=False, r=0.02):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.3,rounding_size={r * 100}",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=C_TEXT,
        fontweight="bold" if bold else "normal",
        zorder=3,
        wrap=True,
    )
    return (x + w / 2, y, x + w / 2, y + h, x, y + h / 2, x + w, y + h / 2)


def group(x, y, w, h, label, ec):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.4,rounding_size=1.5",
        linewidth=2.0,
        edgecolor=ec,
        facecolor="none",
        linestyle=(0, (6, 3)),
        zorder=1,
    )
    ax.add_patch(p)
    ax.text(
        x + 1.5,
        y + h - 2.2,
        label,
        ha="left",
        va="center",
        fontsize=11,
        color=ec,
        fontweight="bold",
        zorder=3,
    )


def arrow(x1, y1, x2, y2, color="#5F6368", style="-|>", lw=1.8, ls="-"):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=16,
        color=color,
        linewidth=lw,
        linestyle=ls,
        zorder=4,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(a)


# ---- Title -----------------------------------------------------------------
ax.text(
    50,
    97.5,
    "Serverless Demand Forecasting on GCP — Architecture",
    ha="center",
    fontsize=17,
    fontweight="bold",
    color=C_TEXT,
)
ax.text(
    50,
    94.3,
    "Author with KFP SDK  →  compile to YAML  →  run serverlessly on Vertex AI  (Free-Tier)",
    ha="center",
    fontsize=10.5,
    color="#5F6368",
    style="italic",
)

# ============================================================================
# LEFT: local package (src/forecasting)
# ============================================================================
group(2, 34, 30, 52, "src/forecasting  (installable core)", C_CORE_EDGE)

box(
    4,
    76,
    26,
    7,
    "config/settings.py\nENV vars + free-tier guardrails\n(machine size, BQ byte cap)",
    C_CORE,
    C_CORE_EDGE,
    fs=8.5,
    bold=True,
)
_pure = box(
    4,
    62,
    26,
    11,
    "models/  •  utils/  •  data/\n\nPURE ML logic (no cloud)\nfeatures · train · ensemble\nmetrics · gcs · generator",
    C_CORE,
    C_CORE_EDGE,
    fs=8.5,
    bold=True,
)
_local = box(
    4,
    50,
    26,
    8,
    "local_runner.py\ncloud-free mirror of both DAGs\n(notebook + tests parity)",
    "#D2E3FC",
    C_CORE_EDGE,
    fs=8.5,
    bold=True,
)
_comp = box(
    4,
    44,
    26,
    5,
    "components/  (thin KFP @component wrappers)",
    C_PIPE,
    C_PIPE_EDGE,
    fs=8.5,
    bold=True,
)
_pipe = box(
    4,
    37,
    26,
    5,
    "pipelines/  (KFP @pipeline DAGs + registry)",
    C_PIPE,
    C_PIPE_EDGE,
    fs=8.5,
    bold=True,
)

# internal arrows
arrow(17, 76, 17, 73, C_CORE_EDGE)  # settings -> pure
arrow(10, 62, 10, 58, C_CORE_EDGE)  # pure -> local
arrow(24, 62, 24, 49, C_PIPE_EDGE)  # pure -> components
arrow(17, 44, 17, 42, C_PIPE_EDGE)  # components -> pipelines

ax.text(11.5, 59.6, "notebook\n& tests", ha="center", fontsize=7, color=C_CORE_EDGE)

# ============================================================================
# MIDDLE: deploy step
# ============================================================================
_deploy = box(
    37,
    55,
    24,
    9,
    "deployment/\ndeploy_pipeline.py\n\ncompile → YAML → submit / schedule",
    "#FFF8E1",
    C_PIPE_EDGE,
    fs=9,
    bold=True,
)
arrow(30, 39.5, 37, 57, C_PIPE_EDGE)  # pipelines -> deploy
ax.text(34, 49.5, "compile", ha="center", fontsize=7.5, color=C_PIPE_EDGE, rotation=18)

# ============================================================================
# RIGHT-TOP: Vertex AI serverless
# ============================================================================
group(
    64, 40, 34, 46, "Vertex AI Pipelines  (serverless — no cluster/GPU)", C_VERTEX_EDGE
)

# Pipeline 1
box(
    66,
    71,
    30,
    5,
    "Pipeline 1 — Data / Validation / Drift",
    "#CEEAD6",
    C_VERTEX_EDGE,
    fs=9,
    bold=True,
)
box(
    66,
    63,
    30,
    6.5,
    "run_dbt_transform → extract_ref/current → detect_drift (Evidently)",
    C_VERTEX,
    C_VERTEX_EDGE,
    fs=8,
)

# Pipeline 2
box(
    66,
    54,
    30,
    5,
    "Pipeline 2 — Parallel Train / Ensemble",
    "#CEEAD6",
    C_VERTEX_EDGE,
    fs=9,
    bold=True,
)
box(
    66,
    42.5,
    30,
    9.5,
    "load_data\n  ├─► train ridge  ─────┐\n  └─► train rand_forest ┤→ build_ensemble\n         (parallel)     (inverse-RMSE fan-in)",
    C_VERTEX,
    C_VERTEX_EDGE,
    fs=8,
)

arrow(61, 62, 66, 68, C_VERTEX_EDGE)  # deploy -> P1
arrow(61, 58, 66, 50, C_VERTEX_EDGE)  # deploy -> P2
ax.text(63.5, 66, "submit", ha="center", fontsize=7.5, color=C_VERTEX_EDGE, rotation=20)

# ============================================================================
# RIGHT-BOTTOM: GCP data/storage services
# ============================================================================
group(64, 6, 34, 30, "GCP Data & Storage", C_STORE_EDGE)

_bq = box(
    66,
    22,
    30,
    10,
    "BigQuery (Sandbox)\nfeature mart via dbt\nqueries capped @ 1 GB",
    C_STORE,
    C_STORE_EDGE,
    fs=8.5,
    bold=True,
)
_gcs = box(
    66,
    8.5,
    30,
    10,
    "Cloud Storage (GCS)\npipeline root · drift reports\npickled model bundles",
    C_STORE,
    C_STORE_EDGE,
    fs=8.5,
    bold=True,
)

arrow(75, 63, 74, 32, C_STORE_EDGE, ls="--")  # P1 -> BQ
arrow(88, 42.5, 88, 18.5, C_STORE_EDGE, ls="--")  # P2 -> GCS
arrow(82, 42.5, 82, 32, C_STORE_EDGE, ls="--")  # P2 <- BQ

# ============================================================================
# BOTTOM-LEFT: CI/CD + scheduling
# ============================================================================
group(
    2,
    6,
    56,
    24,
    "CI/CD & Scheduling  (keyless via Workload Identity Federation)",
    C_CICD_EDGE,
)

box(
    4,
    18,
    25,
    8,
    "GitHub Actions — ci.yml\nlint + unit tests + COMPILE pipelines\n(zero cloud cost)",
    C_CICD,
    C_CICD_EDGE,
    fs=8.5,
)
box(
    31,
    18,
    25,
    8,
    "GitHub Actions — deploy.yml\ngated: bootstrap / submit / schedule",
    C_CICD,
    C_CICD_EDGE,
    fs=8.5,
)
box(4, 9, 25, 7, "Vertex-native cron schedule", "#EFDBFB", C_CICD_EDGE, fs=8.5)
box(31, 9, 25, 7, "e2-micro VM cron (submit-only)", "#EFDBFB", C_CICD_EDGE, fs=8.5)

arrow(43.5, 26, 45, 55, C_CICD_EDGE, ls=":")  # deploy.yml -> deploy_pipeline
ax.text(47, 40, "invoke", ha="center", fontsize=7.5, color=C_CICD_EDGE, rotation=88)

# ---- Legend ----------------------------------------------------------------
legend_elems = [
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=C_CORE,
        markeredgecolor=C_CORE_EDGE,
        markersize=13,
        label="Local package (pure logic)",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=C_PIPE,
        markeredgecolor=C_PIPE_EDGE,
        markersize=13,
        label="KFP components / DAGs / deploy",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=C_VERTEX,
        markeredgecolor=C_VERTEX_EDGE,
        markersize=13,
        label="Vertex AI (serverless run)",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=C_STORE,
        markeredgecolor=C_STORE_EDGE,
        markersize=13,
        label="GCP data & storage",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=C_CICD,
        markeredgecolor=C_CICD_EDGE,
        markersize=13,
        label="CI/CD & scheduling",
    ),
]
ax.legend(
    handles=legend_elems,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.055),
    fontsize=10,
    frameon=True,
    framealpha=0.95,
    ncol=5,
)

plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved docs/architecture.png")
