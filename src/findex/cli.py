"""Installable CLI for findex built with Typer, Rich, and standard logging (Lab 4)."""

import heapq
import json
import logging
import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from findex.index import build_index
from findex.query import QuerySyntaxError
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import search as run_search
from findex.store import StorageFormat, open_index, save

log = logging.getLogger(__name__)

app = typer.Typer(
    name="findex",
    help="Full-text search engine CLI: index a corpus, run ranked queries, and inspect stats.",
    no_args_is_help=True,
)

stdout_console = Console()
stderr_console = Console(stderr=True)

_HL_START = "\x00"
_HL_END = "\x01"


class ScorerChoice(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


class FormatChoice(StrEnum):
    binary = "binary"
    pickle = "pickle"


def _configure_logging(verbosity: int) -> None:
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _render_highlighted_snippet(snippet: str) -> Text:
    """Convert sentinel-marked snippet text into a safe Rich Text object."""
    text = Text()
    pos = 0
    while True:
        start = snippet.find(_HL_START, pos)
        if start == -1:
            text.append(snippet[pos:])
            break
        text.append(snippet[pos:start])
        end = snippet.find(_HL_END, start + len(_HL_START))
        if end == -1:
            text.append(snippet[start:])
            break
        text.append(snippet[start + len(_HL_START) : end], style="bold yellow")
        pos = end + len(_HL_END)
    return text


def _term_freq_key(item: tuple[str, int]) -> int:
    return item[1]


@app.callback()
def root_callback(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Increase diagnostic logging on stderr (-v INFO, -vv DEBUG).",
        ),
    ] = 0,
) -> None:
    """Configure global CLI options before running a subcommand."""
    _configure_logging(verbose)


@app.command(name="index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(help="Path to corpus directory or .jsonl file.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output path for the saved index.")],
    positions: Annotated[
        bool,
        typer.Option("--positions", help="Store token positions to enable phrase queries."),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Index only the first N documents."),
    ] = None,
    fmt: Annotated[
        FormatChoice,
        typer.Option("--format", help="Index serialization format."),
    ] = FormatChoice.binary,
    no_text: Annotated[
        bool,
        typer.Option("--no-text", help="Omit raw document texts from the saved index."),
    ] = False,
) -> None:
    """Build and persist an inverted index from a corpus."""
    if not corpus.exists():
        stderr_console.print(f"[red]Error:[/red] Corpus path not found: {corpus}")
        raise typer.Exit(code=1)

    t0 = time.perf_counter()
    storage_fmt: StorageFormat = "binary" if fmt is FormatChoice.binary else "pickle"
    log.info("Starting index build from %s (positions=%s, limit=%s)", corpus, positions, limit)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.completed:,} docs"),
            TimeElapsedColumn(),
            console=stderr_console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("Indexing corpus...", total=limit)

            def _on_doc(count: int) -> None:
                progress.update(task_id, completed=count)

            built = build_index(
                corpus,
                include_positions=positions,
                store_text=not no_text,
                limit=limit,
                on_doc=_on_doc,
            )

        save(built, out, fmt=storage_fmt)
    except (OSError, ValueError) as exc:
        stderr_console.print(f"[red]Error:[/red] Failed to build or save index: {exc}")
        raise typer.Exit(code=1) from None

    elapsed = time.perf_counter() - t0
    size_mib = out.stat().st_size / (1024 * 1024)
    log.info("Saved %r to %s (%.2f MiB) in %.2f s", built, out, size_mib, elapsed)

    stdout_console.print(
        f"Indexed [bold]{built.num_docs:,}[/bold] docs "
        f"([bold]{len(built):,}[/bold] terms) -> {out} "
        f"({size_mib:.2f} MiB, {elapsed:.2f} s)"
    )


@app.command(name="search")
def search_cmd(
    index_path: Annotated[Path, typer.Argument(help="Path to a saved findex index file.")],
    query: Annotated[str, typer.Argument(help="Query string (supports AND, OR, NOT, (), quotes).")],
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="Maximum results to return.")] = 10,
    scorer: Annotated[
        ScorerChoice,
        typer.Option("--scorer", help="Ranking algorithm."),
    ] = ScorerChoice.bm25,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit results as JSON lines to stdout for piping."),
    ] = False,
) -> None:
    """Search a persisted index and print ranked results with snippets."""
    if not index_path.exists():
        stderr_console.print(f"[red]Error:[/red] Index file not found: {index_path}")
        raise typer.Exit(code=1)

    active_scorer: Scorer = BM25() if scorer is ScorerChoice.bm25 else TfIdf()
    hl_start = "**" if json_output else _HL_START
    hl_end = "**" if json_output else _HL_END

    try:
        t0 = time.perf_counter()
        with open_index(index_path) as ix:
            results = run_search(
                ix,
                query,
                scorer=active_scorer,
                k=k,
                highlight_start=hl_start,
                highlight_end=hl_end,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    except (QuerySyntaxError, ValueError, OSError, TypeError) as exc:
        stderr_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    log.info(
        "Query %r (%s) returned %d hits in %.2f ms", query, scorer.value, len(results), elapsed_ms
    )

    if json_output:
        for rank, res in enumerate(results, start=1):
            payload = {
                "rank": rank,
                "doc_id": res.doc_id,
                "score": round(res.score, 6),
                "title": res.title,
                "snippet": res.snippet,
            }
            typer.echo(json.dumps(payload, ensure_ascii=False))
        return

    if not results:
        stdout_console.print(f"No matches for [bold]{query!r}[/bold].")
        return

    table = Table(title=f"Results for {query!r} (scorer={scorer.value}, {elapsed_ms:.1f} ms)")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("ID", justify="right", style="dim", no_wrap=True)
    table.add_column("Score", justify="right", style="green", no_wrap=True)
    table.add_column("Title", style="bold")
    table.add_column("Snippet")

    for rank, res in enumerate(results, start=1):
        table.add_row(
            str(rank),
            str(res.doc_id),
            f"{res.score:.4f}",
            res.title,
            _render_highlighted_snippet(res.snippet),
        )

    stdout_console.print(table)


@app.command(name="stats")
def stats_cmd(
    index_path: Annotated[Path, typer.Argument(help="Path to a saved findex index file.")],
    top: Annotated[int, typer.Option("--top", min=1, help="Number of top terms to display.")] = 20,
) -> None:
    """Display summary statistics and top terms from a persisted index."""
    if not index_path.exists():
        stderr_console.print(f"[red]Error:[/red] Index file not found: {index_path}")
        raise typer.Exit(code=1)

    try:
        with open_index(index_path) as ix:
            num_docs = ix.num_docs
            vocab_size = len(ix)
            total_tokens = sum(ix.doc_lengths.values())
            avg_len = ix.avg_doc_length
            has_positions = ix.has_positions
            has_texts = bool(ix.doc_texts)
            total_postings = sum(len(plist) for plist in ix.values())
            term_totals = ((term, sum(p.tf for p in plist)) for term, plist in ix.items())
            top_terms = heapq.nlargest(top, term_totals, key=_term_freq_key)
    except (OSError, ValueError, TypeError) as exc:
        stderr_console.print(f"[red]Error:[/red] Could not read index: {exc}")
        raise typer.Exit(code=1) from None

    file_mib = index_path.stat().st_size / (1024 * 1024)
    log.info("Loaded stats for %s (%d docs, %d terms)", index_path, num_docs, vocab_size)

    summary = Table(title=f"Index Statistics — {index_path.name}")
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Documents", f"{num_docs:,}")
    summary.add_row("Total tokens", f"{total_tokens:,}")
    summary.add_row("Vocabulary size", f"{vocab_size:,}")
    summary.add_row("Total postings", f"{total_postings:,}")
    summary.add_row("Avg doc length", f"{avg_len:.1f} tokens")
    summary.add_row("Positions stored", "yes" if has_positions else "no")
    summary.add_row("Texts stored", "yes" if has_texts else "no")
    summary.add_row("File size", f"{file_mib:.2f} MiB")
    stdout_console.print(summary)

    terms_table = Table(title=f"Top-{len(top_terms)} Terms")
    terms_table.add_column("#", justify="right", style="cyan")
    terms_table.add_column("Term", style="bold")
    terms_table.add_column("Total TF", justify="right", style="green")
    for rank, (term, tf_sum) in enumerate(top_terms, start=1):
        terms_table.add_row(str(rank), term, f"{tf_sum:,}")
    stdout_console.print(terms_table)
