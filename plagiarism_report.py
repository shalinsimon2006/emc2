#!/usr/bin/env python3
"""
Plagiarism Report Generator (dependency-free)
- Scans a directory of text files
- Computes pairwise similarity using:
  - Cosine similarity on token frequency vectors
  - Jaccard similarity on token n-gram (shingle) sets
- Outputs:
  - JSON summary of all pairs and per-file stats
  - HTML report sorted by highest similarity pairs

Usage:
  python3 plagiarism_report.py --input samples --output report --threshold 0.3 --ngram 5

Notes:
- Suitable for small-to-medium corpora (O(n^2) pairwise comparisons)
- Uses only Python standard library
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import html
import itertools
import json
import math
import os
import pathlib
import re
import sys
from typing import Dict, List, Tuple, Iterable, Set


@dataclasses.dataclass
class Document:
    path: str
    name: str
    text: str
    tokens: List[str]
    term_freq: collections.Counter
    ngrams: Set[Tuple[str, ...]]


@dataclasses.dataclass
class PairResult:
    file_a: str
    file_b: str
    cosine: float
    jaccard: float
    overlap_ngrams: List[str]
    overlap_tokens: List[Tuple[str, int]]

    @property
    def combined_score(self) -> float:
        # Use max to be conservative (flag if either metric is high)
        return max(self.cosine, self.jaccard)


def find_text_files(input_dir: str, exts: Tuple[str, ...]) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            lower = fname.lower()
            if any(lower.endswith(ext) for ext in exts):
                paths.append(os.path.join(root, fname))
    paths.sort()
    return paths


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as exc:
        return f"[ERROR READING FILE: {exc}]"


def normalize(text: str) -> str:
    # Lowercase and replace non-alphanumeric with spaces, then collapse whitespace.
    lowered = text.lower()
    normalized = ''.join(ch if ch.isalnum() else ' ' for ch in lowered)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return text.split()


def build_term_frequency(tokens: List[str]) -> collections.Counter:
    return collections.Counter(tokens)


def build_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    if n <= 0:
        return set()
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)}


def cosine_similarity(a: collections.Counter, b: collections.Counter) -> float:
    if not a or not b:
        return 0.0
    # Dot product
    intersection = set(a.keys()) & set(b.keys())
    dot = sum(a[t] * b[t] for t in intersection)
    if dot == 0:
        return 0.0
    # Norms
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard_similarity(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def top_common_ngrams(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]], top_k: int) -> List[str]:
    common = list(a & b)
    common.sort()
    phrases = [" ".join(ng) for ng in common]
    # Prefer longer sort by frequency is not available (sets). Keep lexical order and slice
    return phrases[:top_k]


def top_common_tokens(a: collections.Counter, b: collections.Counter, top_k: int) -> List[Tuple[str, int]]:
    common_tokens = []
    for tok in set(a.keys()) & set(b.keys()):
        common_tokens.append((tok, min(a[tok], b[tok])))
    # Sort by descending overlap count then token
    common_tokens.sort(key=lambda x: (-x[1], x[0]))
    return common_tokens[:top_k]


def analyze_documents(paths: List[str], ngram: int) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        raw = read_text(p)
        norm = normalize(raw)
        tokens = tokenize(norm)
        tf = build_term_frequency(tokens)
        ngs = build_ngrams(tokens, ngram)
        docs.append(
            Document(
                path=p,
                name=os.path.relpath(p, start=os.path.commonpath([os.path.dirname(p)] + [os.path.dirname(paths[0])])),
                text=raw,
                tokens=tokens,
                term_freq=tf,
                ngrams=ngs,
            )
        )
    return docs


def pairwise_results(docs: List[Document], top_k_overlap_ngrams: int, top_k_overlap_tokens: int) -> List[PairResult]:
    results: List[PairResult] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            da = docs[i]
            db = docs[j]
            cos = cosine_similarity(da.term_freq, db.term_freq)
            jac = jaccard_similarity(da.ngrams, db.ngrams)
            overlaps_ng = top_common_ngrams(da.ngrams, db.ngrams, top_k_overlap_ngrams)
            overlaps_tok = top_common_tokens(da.term_freq, db.term_freq, top_k_overlap_tokens)
            results.append(
                PairResult(
                    file_a=da.path,
                    file_b=db.path,
                    cosine=cos,
                    jaccard=jac,
                    overlap_ngrams=overlaps_ng,
                    overlap_tokens=overlaps_tok,
                )
            )
    # Sort descending by combined score then secondary by cosine
    results.sort(key=lambda r: (-r.combined_score, -r.cosine, -r.jaccard, r.file_a, r.file_b))
    return results


def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def write_json_report(output_dir: str, docs: List[Document], pairs: List[PairResult]) -> str:
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, "report.json")
    payload = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "num_files": len(docs),
        "files": [
            {
                "path": d.path,
                "name": d.name,
                "num_tokens": len(d.tokens),
                "num_ngrams": len(d.ngrams),
            }
            for d in docs
        ],
        "pairs": [
            {
                "file_a": p.file_a,
                "file_b": p.file_b,
                "cosine": round(p.cosine, 6),
                "jaccard": round(p.jaccard, 6),
                "combined": round(p.combined_score, 6),
                "overlap_ngrams": p.overlap_ngrams,
                "overlap_tokens": p.overlap_tokens,
            }
            for p in pairs
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _html_escape(s: str) -> str:
    return html.escape(s, quote=True)


def write_html_report(output_dir: str, docs: List[Document], pairs: List[PairResult], threshold: float) -> str:
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, "index.html")

    def fmt_pct(x: float) -> str:
        return f"{x*100:.1f}%"

    # Simple, self-contained styles and table-based layout
    style = """
    :root { --bg: #0f172a; --panel: #111827; --text: #e5e7eb; --muted:#9ca3af; --accent:#60a5fa; --warn:#f59e0b; --danger:#ef4444; }
    * { box-sizing: border-box; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial, "Apple Color Emoji", "Segoe UI Emoji"; margin: 0; background: var(--bg); color: var(--text); }
    header { padding: 24px 20px; background: linear-gradient(180deg, #0b1220, #0f172a); border-bottom: 1px solid #1f2937; }
    h1 { margin: 0 0 6px; font-size: 20px; }
    .meta { color: var(--muted); font-size: 12px; }
    main { padding: 20px; }
    .panel { background: var(--panel); border: 1px solid #1f2937; border-radius: 10px; padding: 16px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px; border-bottom: 1px solid #1f2937; vertical-align: top; }
    th { text-align: left; color: var(--muted); font-weight: 600; font-size: 12px; letter-spacing: .02em; }
    td { font-size: 14px; }
    .score { font-variant-numeric: tabular-nums; font-weight: 700; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
    .risk-high { background: color-mix(in srgb, var(--danger) 20%, transparent); color: var(--danger); border: 1px solid color-mix(in srgb, var(--danger) 50%, transparent); }
    .risk-med { background: color-mix(in srgb, var(--warn) 20%, transparent); color: var(--warn); border: 1px solid color-mix(in srgb, var(--warn) 50%, transparent); }
    .risk-low { background: color-mix(in srgb, var(--accent) 20%, transparent); color: var(--accent); border: 1px solid color-mix(in srgb, var(--accent) 50%, transparent); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; }
    .muted { color: var(--muted); }
    .nowrap { white-space: nowrap; }
    details { margin-top: 8px; }
    summary { cursor: pointer; color: var(--accent); }
    .pill { display: inline-block; border: 1px dashed #374151; border-radius: 8px; padding: 4px 8px; margin: 4px 6px 0 0; font-size: 12px; }
    footer { color: var(--muted); font-size: 12px; padding: 16px 20px; border-top: 1px solid #1f2937; }
    @media (max-width: 800px) { .nowrap { white-space: normal; } }
    """

    # Risk badge based on combined score
    def risk_badge(score: float) -> str:
        if score >= 0.7:
            klass = "risk-high"
            label = "High"
        elif score >= 0.4:
            klass = "risk-med"
            label = "Medium"
        else:
            klass = "risk-low"
            label = "Low"
        return f'<span class="badge {klass}">{label}</span>'

    rows_html: List[str] = []
    for p in pairs:
        if p.combined_score < threshold:
            continue
        name_a = _html_escape(os.path.relpath(p.file_a, start=os.path.commonpath([d.path for d in docs])))
        name_b = _html_escape(os.path.relpath(p.file_b, start=os.path.commonpath([d.path for d in docs])))
        overlaps_html = ""
        if p.overlap_ngrams:
            overlaps_html += "<div class=\"muted\">Top overlapping n-grams:</div>"
            for ph in p.overlap_ngrams:
                overlaps_html += f"<span class=\"pill mono\">{_html_escape(ph)}</span>"
        if p.overlap_tokens:
            overlaps_html += "<div class=\"muted\" style=\"margin-top:6px\">Top common tokens:</div>"
            for tok, cnt in p.overlap_tokens:
                overlaps_html += f"<span class=\"pill mono\">{_html_escape(tok)} ×{cnt}</span>"
        rows_html.append(
            """
            <tr>
              <td class="nowrap"><span class="mono">{name_a}</span><br><span class="muted mono">{name_b}</span></td>
              <td class="score">{cos}</td>
              <td class="score">{jac}</td>
              <td class="score">{comb} {badge}</td>
              <td>{overlaps}</td>
            </tr>
            """.format(
                name_a=name_a,
                name_b=name_b,
                cos=fmt_pct(p.cosine),
                jac=fmt_pct(p.jaccard),
                comb=fmt_pct(p.combined_score),
                badge=risk_badge(p.combined_score),
                overlaps=overlaps_html or "<span class=\"muted\">(none)</span>",
            )
        )

    # Files table
    files_rows = []
    for d in docs:
        files_rows.append(
            f"<tr><td class=\"mono\">{_html_escape(os.path.relpath(d.path, start=os.path.commonpath([x.path for x in docs])) )}</td>"
            f"<td>{len(d.tokens):,}</td><td>{len(d.ngrams):,}</td></tr>"
        )

    html_doc = f"""
<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Plagiarism Report</title>
<style>{style}</style>
<body>
  <header>
    <h1>Plagiarism Report</h1>
    <div class="meta">Generated at {html.escape(dt.datetime.utcnow().isoformat() + 'Z')} · {len(docs)} files analyzed · threshold {threshold:.2f}</div>
  </header>
  <main>
    <section class="panel" style="margin-bottom:16px">
      <h2 style="margin:0 0 10px; font-size:16px">Files</h2>
      <div style="overflow:auto">
        <table>
          <thead><tr><th>File</th><th>Tokens</th><th>n-grams</th></tr></thead>
          <tbody>
            {''.join(files_rows) if files_rows else '<tr><td colspan=3 class=\'muted\'>(no files)</td></tr>'}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2 style="margin:0 0 10px; font-size:16px">Similarity Pairs (≥ threshold)</h2>
      <div class="muted" style="margin-bottom:8px">Sorted by highest combined score (max of cosine, jaccard)</div>
      <div style="overflow:auto">
        <table>
          <thead>
            <tr>
              <th>Pair</th>
              <th>Cosine</th>
              <th>Jaccard</th>
              <th>Combined</th>
              <th>Overlaps</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html) if rows_html else '<tr><td colspan=5 class=\'muted\'>(no pairs ≥ threshold)</td></tr>'}
          </tbody>
        </table>
      </div>
    </section>
  </main>
  <footer>
    Generated by a dependency-free Python tool. Metrics are heuristic indicators and require human judgment.
  </footer>
</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return out_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a plagiarism similarity report for text files")
    parser.add_argument("--input", required=False, default="samples", help="Input directory containing text files")
    parser.add_argument("--output", required=False, default="report", help="Output directory for reports")
    parser.add_argument("--extensions", required=False, default=".txt,.md,.text", help="Comma-separated file extensions to include")
    parser.add_argument("--ngram", type=int, default=5, help="n for n-gram shingles (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum combined score to include in HTML table")
    parser.add_argument("--top_k_ngrams", type=int, default=8, help="Number of top overlapping n-grams to show")
    parser.add_argument("--top_k_tokens", type=int, default=8, help="Number of top common tokens to show")
    parser.add_argument("--no_json", action="store_true", help="Do not write JSON report")
    parser.add_argument("--no_html", action="store_true", help="Do not write HTML report")

    args = parser.parse_args(argv)

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    exts = tuple([e.strip() if e.strip().startswith(".") else "." + e.strip() for e in args.extensions.split(",") if e.strip()])

    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    files = find_text_files(input_dir, exts)
    if not files:
        print(f"No text files found in {input_dir} with extensions {exts}", file=sys.stderr)
        return 3

    docs = analyze_documents(files, args.ngram)
    pairs = pairwise_results(docs, args.top_k_ngrams, args.top_k_tokens)

    if not args.no_json:
        json_path = write_json_report(output_dir, docs, pairs)
        print(f"Wrote JSON report: {json_path}")
    if not args.no_html:
        html_path = write_html_report(output_dir, docs, pairs, threshold=args.threshold)
        print(f"Wrote HTML report: {html_path}")

    print(f"Analyzed {len(docs)} files, produced {len(pairs)} pair results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
