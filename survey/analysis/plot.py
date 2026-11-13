"""Generate charts from processed stats JSON -> output/figures/*.png."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "output", "figures")

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "font.size": 9,
})


def _load(name):
    with open(os.path.join(PROC, name), encoding="utf-8") as f:
        return json.load(f)


def barh(data, title, fname, top=12, color="#2c7fb8"):
    items = list(data.items())[:top][::-1]
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(items) + 1.2))
    ax.barh(labels, vals, color=color)
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def year_line(stats, fname):
    yt = stats["year_trend"]
    years = sorted(yt)
    vals = [yt[y] for y in years]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(years, vals, marker="o", color="#2c7fb8")
    for x, y in zip(years, vals):
        ax.text(x, y, str(y), ha="center", va="bottom", fontsize=8)
    ax.set_title("Retrieved sample by v1 year (latest year may be partial)")
    ax.set_xlabel("Year")
    ax.set_ylabel("# papers")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def module_trend_share(stats, fname, top=8):
    myt = stats["module_year_trend"]
    yt = stats["year_trend"]
    years = sorted(yt)
    # pick top modules by total count
    totals = {m: sum(v.values()) for m, v in myt.items()}
    top_mods = sorted(totals, key=lambda m: -totals[m])[:top]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for m in top_mods:
        share = [myt[m].get(str(y), 0) / yt[str(y)] * 100 for y in years]
        ax.plot(years, share, marker=".", label=m)
    ax.set_title("Keyword share in retrieved sample (not population trends)")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of papers")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    s = _load("arxiv_stats.json")
    so = _load("openalex_stats.json")

    year_line(s, "year_trend.png")
    barh(s["venue_dist"], "arXiv TOD papers by venue (from comment/journal_ref)",
         "venue_arxiv.png")
    barh(so["venue_dist"], "OpenAlex top-cited TOD works by venue",
         "venue_openalex.png", color="#d95f02")
    barh(s["module_dist"], "Mentioned modules / techniques", "modules.png")
    module_trend_share(s, "module_trend.png")
    barh(s["detector_dist"], "Mentioned detectors / baselines", "detectors.png")
    barh(s["domain_dist"], "Application domains", "domains.png")
    barh(s["dataset_dist"], "Benchmark datasets", "datasets.png")

    print(f"charts written to {FIG}")


if __name__ == "__main__":
    main()
