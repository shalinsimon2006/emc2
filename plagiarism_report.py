#!/usr/bin/env python3
import argparse
import html
import math
import os
from pathlib import Path
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from difflib import HtmlDiff
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


TOKEN_REGEX = re.compile(
    r"""
    (?:[A-Za-z_][A-Za-z_0-9]*)   # identifiers/words
    |(?:\d+(?:\.\d+)?)        # numbers (ints/floats)
    |(?:==|!=|<=|>=|->|=>)       # multi-char operators
    |(?:.)                       # any single char (punctuation)
    """,
    re.VERBOSE,
)

DEFAULT_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".rb", ".php", ".scala", ".swift", ".sql", ".html", ".css", ".scss", ".md", ".txt", ".json", ".yml", ".yaml"
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", "node_modules", "dist", "build", "out", ".next", "coverage", "target", "bin", "obj",
    "__pycache__", ".venv", "venv", ".idea", ".vscode"
}


@dataclass(frozen=True)
class Document:
    path: Path
    text: str
    tokens: List[str]
    shingles: Tuple[str, ...]  # tuple for deterministic iteration
    shingle_set: Set[str]
    shingle_counts: Counter


@dataclass
class PairSimilarity:
    path_a: Path
    path_b: Path
    similarity: float
    overlap: int
    size_a: int
    size_b: int
    metric: str


def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for match in TOKEN_REGEX.finditer(text):
        tok = match.group(0)
        if tok.strip() == "":
            continue
        if tok[0].isalpha() or tok[0] == "_":
            tokens.append(tok.lower())
        elif tok[0].isdigit():
            tokens.append("0")
        else:
            tokens.append(tok)
    return tokens


def generate_shingles(tokens: Sequence[str], k: int) -> List[str]:
    if k <= 0:
        raise ValueError("k must be >= 1")
    if len(tokens) < k:
        return []
    shingles: List[str] = []
    # Use an unambiguous separator that does not occur in tokens
    sep = "\u0001"
    for i in range(len(tokens) - k + 1):
        shingle = sep.join(tokens[i : i + k])
        shingles.append(shingle)
    return shingles


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> Tuple[float, int]:
    if not set_a and not set_b:
        return (1.0, 0)
    if not set_a or not set_b:
        return (0.0, 0)
    inter = set_a & set_b
    union = set_a | set_b
    return (len(inter) / len(union), len(inter))


def cosine_similarity(counts_a: Counter, counts_b: Counter) -> Tuple[float, int]:
    if not counts_a and not counts_b:
        return (1.0, 0)
    if not counts_a or not counts_b:
        return (0.0, 0)
    # dot product
    dot = 0
    overlap = 0
    if len(counts_a) < len(counts_b):
        smaller, larger = counts_a, counts_b
    else:
        smaller, larger = counts_b, counts_a
    for key, val in smaller.items():
        if key in larger:
            dot += val * larger[key]
            overlap += 1
    # norms
    norm_a = math.sqrt(sum(v * v for v in counts_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counts_b.values()))
    if norm_a == 0 or norm_b == 0:
        return (0.0, overlap)
    return (dot / (norm_a * norm_b), overlap)


def discover_files(root: Path, include_extensions: Set[str], exclude_dirs: Set[str], max_files: int, min_bytes: int) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_dir():
            # skip excluded directories early
            if path.name in exclude_dirs:
                # skip walking into excluded directories
                # Path.rglob doesn't allow pruning; we filter files later too
                pass
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in include_extensions:
            continue
        try:
            size = path.stat().st_size
        except Exception:
            continue
        if size < min_bytes:
            continue
        # ensure it's not inside an excluded directory
        parts = set(p.name for p in path.parents)
        if parts & exclude_dirs:
            continue
        files.append(path)
        if len(files) >= max_files:
            break
    files.sort(key=lambda p: str(p))
    return files


def build_documents(paths: Sequence[Path], k: int) -> List[Document]:
    documents: List[Document] = []
    for p in paths:
        text = read_text(p)
        if text is None:
            continue
        tokens = tokenize(text)
        shingles_list = generate_shingles(tokens, k)
        shingle_set = set(shingles_list)
        shingle_counts = Counter(shingles_list)
        documents.append(
            Document(
                path=p,
                text=text,
                tokens=tokens,
                shingles=tuple(shingles_list),
                shingle_set=shingle_set,
                shingle_counts=shingle_counts,
            )
        )
    return documents


def compute_pairwise(documents: Sequence[Document], metric: str, min_similarity: float, top_k: int) -> List[PairSimilarity]:
    results: List[PairSimilarity] = []
    n = len(documents)
    for i in range(n):
        a = documents[i]
        for j in range(i + 1, n):
            b = documents[j]
            if metric == "jaccard":
                sim, overlap = jaccard_similarity(a.shingle_set, b.shingle_set)
            elif metric == "cosine":
                sim, overlap = cosine_similarity(a.shingle_counts, b.shingle_counts)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            if sim >= min_similarity:
                results.append(
                    PairSimilarity(
                        path_a=a.path,
                        path_b=b.path,
                        similarity=sim,
                        overlap=overlap,
                        size_a=len(a.shingle_set) if metric == "jaccard" else sum(a.shingle_counts.values()),
                        size_b=len(b.shingle_set) if metric == "jaccard" else sum(b.shingle_counts.values()),
                        metric=metric,
                    )
                )
    results.sort(key=lambda r: r.similarity, reverse=True)
    if top_k > 0 and len(results) > top_k:
        results = results[:top_k]
    return results


def escape(s: str) -> str:
    return html.escape(s, quote=True)


def format_percent(x: float) -> str:
    return f"{x*100:.1f}%"


def generate_diff_html(text_a: str, text_b: str, fromdesc: str, todesc: str) -> str:
    # Using difflib HtmlDiff; the output includes its own minimal styling
    try:
        differ = HtmlDiff(tabsize=2, wrapcolumn=80)
        return differ.make_table(
            text_a.splitlines(), text_b.splitlines(), fromdesc=fromdesc, todesc=todesc
        )
    except Exception:
        return "<em>Diff generation failed.</em>"


def generate_report_html(
    root: Path,
    documents: Sequence[Document],
    pairs: Sequence[PairSimilarity],
    metric: str,
    k: int,
    min_similarity: float,
    generated_at: float,
    include_diffs_for_top: int = 20,
) -> str:
    total_files = len(documents)
    compared_pairs = (total_files * (total_files - 1)) // 2
    title = "Plagiarism Similarity Report"

    # Basic CSS and JS for sorting/filtering
    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
    h1 { margin-bottom: 0; }
    .meta { color: #555; margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; margin-top: 16px; }
    th, td { border: 1px solid #e5e7eb; padding: 8px 10px; font-size: 14px; }
    th { background: #f8fafc; cursor: pointer; position: sticky; top: 0; }
    tr:nth-child(even) { background: #fafafa; }
    .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; background: #eef2ff; color: #3730a3; font-size: 12px; }
    .muted { color: #666; }
    .nowrap { white-space: nowrap; }
    details { margin-top: 8px; }
    summary { cursor: pointer; }
    .note { background: #fffbeb; border: 1px solid #f59e0b33; padding: 8px 10px; border-radius: 6px; }
    .ok { color: #16a34a; }
    .warn { color: #b45309; }
    .danger { color: #dc2626; }
    """

    sort_js = """
    function sortTable(tableId, colIndex, numeric=false) {
      const table = document.getElementById(tableId);
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);
      const dirAttr = table.getAttribute('data-sort-dir') || 'desc';
      const newDir = dirAttr === 'asc' ? 'desc' : 'asc';
      rows.sort((a,b) => {
        let av = a.cells[colIndex].dataset.sort || a.cells[colIndex].innerText;
        let bv = b.cells[colIndex].dataset.sort || b.cells[colIndex].innerText;
        if (numeric) { av = parseFloat(av) || 0; bv = parseFloat(bv) || 0; }
        if (av < bv) return newDir === 'asc' ? -1 : 1;
        if (av > bv) return newDir === 'asc' ? 1 : -1;
        return 0;
      });
      rows.forEach(r => tbody.appendChild(r));
      table.setAttribute('data-sort-dir', newDir);
    }
    """

    header = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{escape(title)}</title>
  <style>{css}</style>
  <script>{sort_js}</script>
</head>
<body>
  <h1>{escape(title)}</h1>
  <div class=\"meta\">Root: <code>{escape(str(root))}</code></div>
  <div class=\"meta\">Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(generated_at))}</div>
  <div class=\"meta\">Metric: <span class=\"badge\">{escape(metric)}</span> · n-gram k=<code>{k}</code> · min similarity ≥ {format_percent(min_similarity)}</div>
  <div class=\"meta\">Files: <strong>{total_files}</strong> · Compared pairs: <strong>{compared_pairs}</strong> · Reported pairs: <strong>{len(pairs)}</strong></div>
  <div class=\"note\">Higher similarity suggests potential duplication. Manually review context and intent before drawing conclusions.</div>
"""

    table_header = (
        "<table id=\"pairs\"><thead><tr>"
        "<th onclick=\"sortTable('pairs',0,true)\">Rank</th>"
        "<th onclick=\"sortTable('pairs',1,true)\">Similarity</th>"
        "<th onclick=\"sortTable('pairs',2)\">File A</th>"
        "<th onclick=\"sortTable('pairs',3)\">File B</th>"
        "<th onclick=\"sortTable('pairs',4,true)\">Overlap</th>"
        "<th onclick=\"sortTable('pairs',5,true)\">Size A</th>"
        "<th onclick=\"sortTable('pairs',6,true)\">Size B</th>"
        "</tr></thead><tbody>"
    )

    rows_html: List[str] = []
    for idx, p in enumerate(pairs, start=1):
        severity_cls = "ok"
        if p.similarity >= 0.8:
            severity_cls = "danger"
        elif p.similarity >= 0.5:
            severity_cls = "warn"
        sim_pct = format_percent(p.similarity)
        row = (
            f"<tr>"
            f"<td class=\"nowrap\" data-sort=\"{idx}\">{idx}</td>"
            f"<td class=\"{severity_cls}\" data-sort=\"{p.similarity:.6f}\">{sim_pct}</td>"
            f"<td data-sort=\"{escape(str(p.path_a))}\"><code>{escape(str(p.path_a))}</code></td>"
            f"<td data-sort=\"{escape(str(p.path_b))}\"><code>{escape(str(p.path_b))}</code></td>"
            f"<td data-sort=\"{p.overlap}\">{p.overlap}</td>"
            f"<td data-sort=\"{p.size_a}\">{p.size_a}</td>"
            f"<td data-sort=\"{p.size_b}\">{p.size_b}</td>"
            f"</tr>"
        )
        rows_html.append(row)
    table_footer = "</tbody></table>"

    details_html: List[str] = []
    if pairs:
        details_html.append("<h2>Top pair details</h2>")
        for idx, p in enumerate(pairs[:include_diffs_for_top], start=1):
            details_html.append("<details>")
            details_html.append(
                f"<summary>#{idx} — {escape(str(p.path_a))} ↔ {escape(str(p.path_b))} · similarity {format_percent(p.similarity)}</summary>"
            )
            # Find documents to pull original text
            text_a = next((d.text for d in documents if d.path == p.path_a), "")
            text_b = next((d.text for d in documents if d.path == p.path_b), "")
            details_html.append(generate_diff_html(text_a, text_b, str(p.path_a), str(p.path_b)))
            details_html.append("</details>")

    footer = """
</body>
</html>
"""

    return header + table_header + "".join(rows_html) + table_footer + "".join(details_html) + footer


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a plagiarism similarity report over a directory of files.")
    parser.add_argument("--root", default=".", help="Root directory to scan (default: .)")
    parser.add_argument("--output", default="plagiarism_report.html", help="Output HTML file path")
    parser.add_argument("--extensions", default=",".join(sorted(DEFAULT_EXTENSIONS)), help="Comma-separated list of file extensions to include (with leading dots)")
    parser.add_argument("--exclude-dirs", default=",".join(sorted(DEFAULT_EXCLUDE_DIRS)), help="Comma-separated list of directory names to exclude")
    parser.add_argument("--k", type=int, default=5, help="Shingle (n-gram) size")
    parser.add_argument("--metric", choices=["jaccard", "cosine"], default="jaccard", help="Similarity metric")
    parser.add_argument("--min-sim", type=float, default=0.3, help="Minimum similarity threshold to include a pair")
    parser.add_argument("--top-k", type=int, default=200, help="Report at most top K pairs")
    parser.add_argument("--max-files", type=int, default=1000, help="Maximum number of files to scan")
    parser.add_argument("--min-bytes", type=int, default=100, help="Minimum file size in bytes to include")

    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    output = Path(args.output)
    include_extensions = {ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()}
    exclude_dirs = {d.strip() for d in args.exclude_dirs.split(",") if d.strip()}

    if not root.exists() or not root.is_dir():
        print(f"Root directory does not exist or is not a directory: {root}", file=sys.stderr)
        return 2

    started = time.time()
    files = discover_files(root, include_extensions, exclude_dirs, args.max_files, args.min_bytes)
    if not files:
        print("No files found matching criteria.")
    documents = build_documents(files, args.k)

    pairs = compute_pairwise(documents, args.metric, args.min_sim, args.top_k)

    report_html = generate_report_html(
        root=root,
        documents=documents,
        pairs=pairs,
        metric=args.metric,
        k=args.k,
        min_similarity=args.min_sim,
        generated_at=time.time(),
    )

    try:
        output.write_text(report_html, encoding="utf-8")
    except Exception as exc:
        print(f"Failed to write report to {output}: {exc}", file=sys.stderr)
        return 3

    elapsed = time.time() - started
    print(
        f"Report written to {output} — files: {len(documents)} — pairs: {len(pairs)} — elapsed {elapsed:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
