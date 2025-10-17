#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

try:
    from pypdf import PdfReader
except Exception as e:  # pragma: no cover
    PdfReader = None


@dataclass
class DocumentStats:
    page_count: int
    word_count: int
    unique_word_count: int
    type_token_ratio: float
    sentence_count: int
    average_words_per_sentence: float


@dataclass
class CitationStats:
    total_citations_detected: int
    bracket_number_citations: int
    author_year_citations: int
    url_like_citations: int
    references_section_present: bool


@dataclass
class NgramItem:
    ngram: str
    count: int


@dataclass
class FlaggedSentence:
    sentence: str
    word_count: int
    reason: str


@dataclass
class LongQuote:
    quote: str
    word_count: int


@dataclass
class AnalysisResult:
    document_title: str
    file_name: str
    stats: DocumentStats
    citations: CitationStats
    top_trigrams: List[NgramItem]
    top_fivegrams: List[NgramItem]
    flagged_sentences: List[FlaggedSentence]
    long_quotes: List[LongQuote]
    internal_similarity_risk_0_100: int
    notes: List[str]


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    if PdfReader is None:
        raise RuntimeError("pypdf is not available; install it to extract text")
    reader = PdfReader(pdf_path)
    pages_text: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        pages_text.append(page_text)
    page_count = len(reader.pages)
    full_text = "\n".join(pages_text)
    normalized = re.sub(r"\s+", " ", full_text).strip()
    return normalized, page_count


def split_into_sentences(text: str) -> List[str]:
    # Simple, conservative sentence splitter
    # Avoid splitting on common abbreviations
    abbreviations = set([
        "e.g.", "i.e.", "Fig.", "Eq.", "Dr.", "Mr.", "Ms.", "Prof.",
        "et al.", "vs.", "No.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."
    ])

    # Temporary token to protect abbreviations during split
    placeholder_map: Dict[str, str] = {}
    protected_text = text
    for abbr in abbreviations:
        token = f"<ABBR_{hash(abbr) & 0xffff}>"
        placeholder_map[token] = abbr
        protected_text = protected_text.replace(abbr, token)

    rough_sentences = re.split(r"(?<=[.!?])\s+", protected_text)

    sentences: List[str] = []
    for s in rough_sentences:
        for token, abbr in placeholder_map.items():
            s = s.replace(token, abbr)
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences


def tokenize_words(text: str) -> List[str]:
    # Letters, digits, apostrophes and hyphens inside tokens
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", text)
    return [t.lower() for t in tokens]


def compute_stats(text: str, page_count: int) -> Tuple[DocumentStats, List[str], List[str]]:
    sentences = split_into_sentences(text)
    words = tokenize_words(text)
    unique_words = set(words)

    ttr = (len(unique_words) / len(words)) if words else 0.0
    avg_words_per_sentence = (sum(len(tokenize_words(s)) for s in sentences) / len(sentences)) if sentences else 0.0

    # Find possible title candidate: text before 'Abstract' or first 12 words capitalized chunk
    title_candidate = None
    m = re.search(r"\bAbstract\b", text, flags=re.IGNORECASE)
    if m:
        prefix = text[: m.start()].strip()
        # Take the last line-ish
        pieces = re.split(r"[\n\r]", prefix)
        pieces = [p.strip() for p in pieces if p.strip()]
        if pieces:
            title_candidate = pieces[-1]
    if not title_candidate:
        # First sentence up to ~12 words
        first_words = " ".join(tokenize_words(text)[:12]).title()
        title_candidate = first_words if first_words else "Untitled"

    return (
        DocumentStats(
            page_count=page_count,
            word_count=len(words),
            unique_word_count=len(unique_words),
            type_token_ratio=round(ttr, 4),
            sentence_count=len(sentences),
            average_words_per_sentence=round(avg_words_per_sentence, 2),
        ),
        sentences,
        words,
    )


def detect_citations(text: str, sentences: List[str]) -> CitationStats:
    bracket_number = len(re.findall(r"\[(?:\s*\d+\s*(?:,\s*)?)+\]", text))
    author_year = len(re.findall(r"\([A-Z][A-Za-z\-]+(?:\s+et al\.)?,\s*\d{4}[a-z]?\)", text))
    url_like = len(re.findall(r"https?://|doi\.org/|arxiv\.org/", text, flags=re.IGNORECASE))

    # Very rough references section detection
    has_refs = bool(re.search(r"\n\s*References\s*\n|\n\s*Bibliography\s*\n", text, flags=re.IGNORECASE))

    total = bracket_number + author_year + url_like
    return CitationStats(
        total_citations_detected=total,
        bracket_number_citations=bracket_number,
        author_year_citations=author_year,
        url_like_citations=url_like,
        references_section_present=has_refs,
    )


def top_ngrams(tokens: List[str], n: int, min_count: int = 2, limit: int = 20) -> List[NgramItem]:
    if len(tokens) < n:
        return []
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    items = [NgramItem(ngram=g, count=c) for g, c in counts.items() if c >= min_count]
    items.sort(key=lambda x: (-x.count, x.ngram))
    return items[:limit]


def has_citation(sentence: str) -> bool:
    if re.search(r"\[(?:\s*\d+\s*(?:,\s*)?)+\]", sentence):
        return True
    if re.search(r"\([A-Z][A-Za-z\-]+(?:\s+et al\.)?,\s*\d{4}[a-z]?\)", sentence):
        return True
    if re.search(r"https?://|doi\.org/|arxiv\.org/", sentence, flags=re.IGNORECASE):
        return True
    return False


def detect_long_quotes(text: str, min_words: int = 40) -> List[LongQuote]:
    results: List[LongQuote] = []
    for m in re.finditer(r'"([^"\n]{20,})"', text):
        quote = m.group(1).strip()
        wc = len(tokenize_words(quote))
        if wc >= min_words:
            results.append(LongQuote(quote=quote, word_count=wc))
    return results


def flag_risky_sentences(sentences: List[str], long_threshold: int = 35) -> List[FlaggedSentence]:
    flagged: List[FlaggedSentence] = []
    for s in sentences:
        wc = len(tokenize_words(s))
        if wc >= long_threshold and not has_citation(s):
            flagged.append(FlaggedSentence(sentence=s, word_count=wc, reason="Long sentence without explicit citation"))
    return flagged


def compute_internal_risk(stats: DocumentStats, citations: CitationStats, flagged: List[FlaggedSentence]) -> int:
    if stats.sentence_count == 0:
        return 0
    ratio_flagged = len(flagged) / stats.sentence_count
    citation_factor = 0.0 if citations.total_citations_detected > 0 else 0.15
    risk = min(1.0, 0.6 * ratio_flagged + citation_factor)
    return int(round(risk * 100))


def render_html(result: AnalysisResult, stylesheet_href: Optional[str]) -> str:
    css_link = f'<link rel="stylesheet" href="{stylesheet_href}">' if stylesheet_href else ''
    def esc(s: str) -> str:
        return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))

    def render_ngrams(items: List[NgramItem]) -> str:
        if not items:
            return '<p>No repeated n-grams above threshold.</p>'
        rows = ''.join(
            f"<tr><td>{esc(it.ngram)}</td><td class=right>{it.count}</td></tr>" for it in items
        )
        return f"<table class=compact><thead><tr><th>N-gram</th><th>Count</th></tr></thead><tbody>{rows}</tbody></table>"

    def render_flagged(items: List[FlaggedSentence]) -> str:
        if not items:
            return '<p>No long uncited sentences detected.</p>'
        lis = ''.join(
            f"<li><span class=badge>{it.word_count} words</span> {esc(it.sentence)}</li>" for it in items[:50]
        )
        more = '' if len(items) <= 50 else f"<p>+{len(items)-50} more not shown...</p>"
        return f"<ol class=flagged>{lis}</ol>{more}"

    def render_quotes(items: List[LongQuote]) -> str:
        if not items:
            return '<p>No long quotes (>= 40 words) detected.</p>'
        lis = ''.join(
            f"<li><span class=badge>{it.word_count} words</span> \"{esc(it.quote)}\"</li>" for it in items[:20]
        )
        more = '' if len(items) <= 20 else f"<p>+{len(items)-20} more not shown...</p>"
        return f"<ol class=flagged>{lis}</ol>{more}"

    s = result.stats
    c = result.citations
    notes = ''.join(f"<li>{esc(n)}</li>" for n in result.notes)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Plagiarism Report · {esc(result.document_title)}</title>
  {css_link}
  <style>
    /* minimal inline fallback if stylesheet missing */
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; margin: 24px; color: #0f172a; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    header h1 {{ margin: 0; font-size: 1.6rem; }}
    header .meta {{ color: #475569; font-size: .9rem; }}
    .kpi {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0 24px; }}
    .card {{ border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; background: #fff; }}
    .card h3 {{ margin: 0 0 8px; font-size: 1rem; color: #0f172a; }}
    .highlight {{ background: #fff7ed; border-left: 4px solid #f59e0b; padding: 10px 12px; border-radius: 6px; }}
    .risk {{ font-size: 2rem; font-weight: 700; }}
    section {{ margin: 28px 0; }}
    table.compact {{ width: 100%; border-collapse: collapse; }}
    table.compact th, table.compact td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 6px; text-align: left; }}
    .right {{ text-align: right; }}
    ol.flagged {{ padding-left: 18px; }}
    .badge {{ display: inline-block; background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe; border-radius: 999px; padding: 2px 8px; margin-right: 6px; font-size: .8rem; }}
    footer {{ margin-top: 40px; color: #64748b; font-size: .9rem; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Plagiarism-style Heuristic Report</h1>
      <div class="meta">File: {esc(result.file_name)} · Title guess: {esc(result.document_title)}</div>
    </header>

    <section class="highlight card">
      <h3>Overall heuristic risk</h3>
      <div class="risk">{result.internal_similarity_risk_0_100}/100</div>
      <p>This is an internal similarity risk score based on long uncited sentences and citation presence. It is <strong>not</strong> a web similarity check.</p>
    </section>

    <div class="kpi">
      <div class="card"><h3>Words</h3><div>{s.word_count:,}</div></div>
      <div class="card"><h3>Unique words</h3><div>{s.unique_word_count:,} (TTR {s.type_token_ratio:.2f})</div></div>
      <div class="card"><h3>Sentences</h3><div>{s.sentence_count:,} (avg {s.average_words_per_sentence:.1f}/sent.)</div></div>
    </div>

    <section class="card">
      <h3>Citation signals</h3>
      <table class="compact">
        <thead><tr><th>Type</th><th class=right>Count</th></tr></thead>
        <tbody>
          <tr><td>Bracket-number citations [1], [2,3]</td><td class=right>{c.bracket_number_citations}</td></tr>
          <tr><td>Author-year citations (Smith, 2020; Doe et al., 2021)</td><td class=right>{c.author_year_citations}</td></tr>
          <tr><td>URL/DOI citations</td><td class=right>{c.url_like_citations}</td></tr>
          <tr><td><strong>References/Bibliography section present</strong></td><td class=right>{'Yes' if c.references_section_present else 'No'}</td></tr>
        </tbody>
      </table>
    </section>

    <section class="card">
      <h3>Repeated phrases (trigrams)</h3>
      {render_ngrams(result.top_trigrams)}
    </section>

    <section class="card">
      <h3>Repeated phrases (five-grams)</h3>
      {render_ngrams(result.top_fivegrams)}
    </section>

    <section class="card">
      <h3>Long uncited sentences</h3>
      {render_flagged(result.flagged_sentences)}
    </section>

    <section class="card">
      <h3>Long quotes (>= 40 words)</h3>
      {render_quotes(result.long_quotes)}
    </section>

    <section class="card">
      <h3>Notes</h3>
      <ul>{notes if notes else '<li>No special notes.</li>'}</ul>
    </section>

    <footer>
      Generated locally without external source comparison. Use a web plagiarism checker for definitive results.
    </footer>
  </div>
</body>
</html>
"""


def analyze(pdf_path: str, stylesheet_href: Optional[str]) -> AnalysisResult:
    text, page_count = extract_text_from_pdf(pdf_path)
    stats, sentences, words = compute_stats(text, page_count)

    citations = detect_citations(text, sentences)
    trigrams = top_ngrams(words, 3, min_count=2, limit=20)
    fivegrams = top_ngrams(words, 5, min_count=2, limit=20)
    flagged = flag_risky_sentences(sentences, long_threshold=35)
    quotes = detect_long_quotes(text, min_words=40)

    risk = compute_internal_risk(stats, citations, flagged)

    # Title guess: prefer explicit title if found earlier; we packed it in stats function
    title_guess = None
    m = re.search(r"\bAbstract\b", text, flags=re.IGNORECASE)
    if m:
        prefix = text[: m.start()].strip()
        pieces = re.split(r"[\n\r]", prefix)
        pieces = [p.strip() for p in pieces if p.strip()]
        title_guess = pieces[-1] if pieces else None
    if not title_guess:
        title_guess = (" ".join(words[:12])).title() if words else "Untitled"

    notes: List[str] = []
    if stats.average_words_per_sentence > 30:
        notes.append("Unusually long average sentence length; may warrant closer review.")
    if not citations.references_section_present:
        notes.append("No References/Bibliography section detected.")
    if citations.total_citations_detected == 0:
        notes.append("No inline citation patterns detected.")

    return AnalysisResult(
        document_title=title_guess,
        file_name=os.path.basename(pdf_path),
        stats=stats,
        citations=citations,
        top_trigrams=trigrams,
        top_fivegrams=fivegrams,
        flagged_sentences=flagged,
        long_quotes=quotes,
        internal_similarity_risk_0_100=risk,
        notes=notes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a heuristic plagiarism-style report for a PDF.")
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument("--html-output", required=True, help="Path to output HTML report")
    parser.add_argument("--json-output", required=True, help="Path to output JSON data")
    parser.add_argument("--stylesheet", default=None, help="Stylesheet href to embed in HTML (e.g., ../style.css)")
    args = parser.parse_args()

    result = analyze(args.input, args.stylesheet)

    # Write JSON
    data = {
        "document_title": result.document_title,
        "file_name": result.file_name,
        "stats": asdict(result.stats),
        "citations": asdict(result.citations),
        "top_trigrams": [asdict(it) for it in result.top_trigrams],
        "top_fivegrams": [asdict(it) for it in result.top_fivegrams],
        "flagged_sentences": [asdict(it) for it in result.flagged_sentences],
        "long_quotes": [asdict(it) for it in result.long_quotes],
        "internal_similarity_risk_0_100": result.internal_similarity_risk_0_100,
        "notes": result.notes,
    }

    os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Write HTML
    html = render_html(result, args.stylesheet)
    os.makedirs(os.path.dirname(args.html_output) or ".", exist_ok=True)
    with open(args.html_output, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
