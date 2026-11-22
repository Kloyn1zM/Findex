"""CLI integration tests using Typer's CliRunner and slow marker tests (Lab 4, M2–M3)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import Index

runner = CliRunner()


def test_cli_index_and_stats(mini_corpus_dir: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "cli_index.bin"
    res_index = runner.invoke(
        app,
        ["-v", "index", str(mini_corpus_dir), "--out", str(out_file), "--positions"],
    )
    assert res_index.exit_code == 0, res_index.output
    assert out_file.exists()
    assert "Indexed 5 docs" in res_index.output

    res_stats = runner.invoke(app, ["stats", str(out_file), "--top", "5"])
    assert res_stats.exit_code == 0, res_stats.output
    assert "Index Statistics" in res_stats.output
    assert "Top-5 Terms" in res_stats.output


def test_cli_search_rich_table_and_tfidf(saved_index_path: Path) -> None:
    res = runner.invoke(
        app,
        ["search", str(saved_index_path), "python", "--k", "3", "--scorer", "tfidf"],
    )
    assert res.exit_code == 0, res.output
    assert "01_python" in res.output
    assert "scorer=tfidf" in res.output


def test_cli_search_json_lines_output(saved_index_path: Path) -> None:
    res = runner.invoke(
        app,
        ["search", str(saved_index_path), '"event loop"', "--json"],
    )
    assert res.exit_code == 0, res.output
    lines = [line for line in res.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["rank"] == 1
    assert row["title"] == "01_python"
    assert "**event**" in row["snippet"]


def test_cli_search_no_matches(saved_index_path: Path) -> None:
    res = runner.invoke(app, ["search", str(saved_index_path), "nonexistentterm"])
    assert res.exit_code == 0
    assert "No matches" in res.output


def test_cli_expected_errors_exit_nonzero_without_traceback(
    saved_index_path: Path,
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "does_not_exist.bin"

    res_missing_corpus = runner.invoke(
        app,
        ["index", str(missing_path), "--out", str(tmp_path / "x.bin")],
    )
    assert res_missing_corpus.exit_code == 1
    assert "Traceback" not in res_missing_corpus.output

    res_missing_index = runner.invoke(app, ["search", str(missing_path), "python"])
    assert res_missing_index.exit_code == 1
    assert "Traceback" not in res_missing_index.output

    res_bad_query = runner.invoke(app, ["search", str(saved_index_path), "(unclosed"])
    assert res_bad_query.exit_code == 1
    assert "Traceback" not in res_bad_query.output

    res_missing_stats = runner.invoke(app, ["stats", str(missing_path)])
    assert res_missing_stats.exit_code == 1
    assert "Traceback" not in res_missing_stats.output


def test_cli_phrase_query_fails_cleanly_without_positions(
    mini_corpus_dir: Path,
    tmp_path: Path,
) -> None:
    no_pos_index = tmp_path / "nopos.bin"
    res_build = runner.invoke(app, ["index", str(mini_corpus_dir), "--out", str(no_pos_index)])
    assert res_build.exit_code == 0

    res_phrase = runner.invoke(app, ["search", str(no_pos_index), '"event loop"'])
    assert res_phrase.exit_code == 1
    assert "Traceback" not in res_phrase.output


def test_cli_verbosity_debug_flag(saved_index_path: Path) -> None:
    res = runner.invoke(app, ["-vv", "search", str(saved_index_path), "python"])
    assert res.exit_code == 0


def test_session_fixture_index_contains_expected_unicode_terms(mini_index: Index) -> None:
    assert mini_index.num_docs == 5
    assert "café" in mini_index
    assert "київ" in mini_index
    assert "strasse" in mini_index


@pytest.mark.slow
def test_build_and_search_medium_synthetic_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "medium"
    corpus.mkdir()
    for i in range(250):
        (corpus / f"doc_{i:03d}.txt").write_text(
            f"Document {i} discusses python search ranking and bm25 algorithms number {i % 17}.",
            encoding="utf-8",
        )
    out = tmp_path / "medium.bin"
    res_idx = runner.invoke(app, ["index", str(corpus), "--out", str(out), "--positions"])
    assert res_idx.exit_code == 0
    res_q = runner.invoke(app, ["search", str(out), '"search ranking"', "--json"])
    assert res_q.exit_code == 0
    assert len(res_q.stdout.splitlines()) == 10
