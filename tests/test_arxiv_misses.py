import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "arxiv_misses", pathlib.Path(__file__).parent.parent / "scripts" / "arxiv_misses.py")
misses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(misses)


def _line(day, answered, tries, pid, name):
    return (f"{day} 03:26:07,018 INFO: [arxiv-outcome] answered={answered} tries={tries} "
            f"project_id={pid} name={name} [in /app/scripts/nightly_crawl.py:160]\n")


def test_parses_names_with_slashes_and_spaces_and_ignores_other_lines():
    out = misses.parse_outcomes([
        "2026-10-29 01:00:37,413 INFO: Related Work Agent startup [in /app/app/__init__.py:47]\n",
        _line("2026-10-29", "yes", 1, 3, "FUGU/FlexOlmoBar"),
        _line("2026-10-29", "no", 3, 12, "VLM self ident"),
    ])
    assert out[3] == {"name": "FUGU/FlexOlmoBar", "nights": {"2026-10-29": (True, 1)}}
    assert out[12]["nights"]["2026-10-29"] == (False, 3)


def test_three_misses_in_a_row_fire_the_trigger_but_scattered_ones_do_not():
    nights = ["2026-10-29", "2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02",
              "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-07"]
    scattered = [_line(d, "no" if i in (1, 4) else "yes", 1, 1, "A") for i, d in enumerate(nights)]
    report, fired = misses.summarize(misses.parse_outcomes(scattered))
    assert not fired and "2/10 project-nights missed (20%)" in report[-1]

    streak = [_line(d, "no" if i in (1, 2, 3) else "yes", 3, 1, "A") for i, d in enumerate(nights)]
    report, fired = misses.summarize(misses.parse_outcomes(streak))
    assert fired and "longest miss streak 3" in report[1]


def test_miss_rate_above_20_percent_fires():
    lines = [_line("2026-10-29", "no", 3, 1, "A")] + [_line("2026-10-29", "yes", 1, i, f"P{i}") for i in range(2, 5)]
    report, fired = misses.summarize(misses.parse_outcomes(lines))
    assert fired and "1/4 project-nights missed (25%)" in report[-1]


def test_main_reads_rotated_logs(tmp_path, capsys):
    (tmp_path / "crawl.log.1").write_text(_line("2026-10-29", "no", 3, 1, "A"))
    (tmp_path / "crawl.log").write_text(_line("2026-10-30", "yes", 2, 1, "A"))
    import sys
    from unittest.mock import patch
    with patch.object(sys, "argv", ["arxiv_misses.py", str(tmp_path)]):
        misses.main()
    out = capsys.readouterr().out
    assert "read 2 log file(s)" in out and "1/2 project-nights missed" in out and "1 needed a retry" in out
