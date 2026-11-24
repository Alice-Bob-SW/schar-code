import re
import subprocess
import sys
from pathlib import Path

import shutil

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_paper_outputs.py"

JOBS = (
    "compute_exponent",
    "reduce_exponent",
    "compute_residue",
    "compute_residue_adjoint",
    "accumulate_residue",
    "operation_count_recap",
    "logical_qubit_count",
    "version_comparison",
)
"""Table jobs; the last job, `tables_pdf`, writes no table."""
COLUMNS = {
    "compute_exponent": ["part", "calls", "operation", "size"],
    "reduce_exponent": ["part", "calls", "operation", "size"],
    "compute_residue": ["part", "calls", "operation", "size"],
    "compute_residue_adjoint": ["part", "calls", "operation", "size"],
    "accumulate_residue": ["part", "calls", "operation", "size"],
    "operation_count_recap": [
        "Subroutine",
        "Calls",
        "Register size",
        "Address size",
        "Q-Q Adder",
        "Q-C Adder",
        "Lookup",
        "Phaseup",
        "CX",
        "M_X",
    ],
    "logical_qubit_count": ["subroutine", "code", "out of scope (paper)", "code peak"],
    "version_comparison": ["subroutine", "operation", "delayed_mbu", "immediate_mbu"],
}
SYMBOLS = re.compile(r"num_primes|l_p|n_e|log_n_e|kappa|w_[134]")


def run(out: Path, *args: str) -> None:
    subprocess.run([sys.executable, SCRIPT, "--out", out, *args], check=True)


@pytest.fixture(scope="module")
def outputs(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("paper")
    run(out)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("job", JOBS)
def test_job_writes_its_four_files(outputs, job):
    tables = outputs / "tables"
    for suffix in ("symbolic", "rsa2048"):
        df = pd.read_csv(tables / f"{job}_{suffix}.csv", dtype=str, keep_default_na=False)
        assert list(df.columns) == COLUMNS[job]
        assert len(df) > 0
        tex = (tables / f"{job}_{suffix}.tex").read_text()
        assert tex.startswith(r"\begin{tabular}") and r"\end{tabular}" in tex
    symbolic = (tables / f"{job}_symbolic.csv").read_text()
    rsa = (tables / f"{job}_rsa2048.csv").read_text()
    assert SYMBOLS.search(symbolic), "the symbolic table has no parameter"
    assert not SYMBOLS.search(rsa), "the RSA-2048 table still has a parameter"


@pytest.mark.slow
def test_no_placeholder_outputs(outputs):
    written = {p.name for p in (outputs / "tables").iterdir()}
    expected = {
        f"{j}_{s}.{e}" for j in JOBS for s in ("symbolic", "rsa2048") for e in ("csv", "tex")
    }
    assert written == expected


@pytest.mark.slow
def test_tables_pdf(outputs):
    """figures/ holds only the document gathering the tables, and its PDF if tectonic exists."""
    figures = outputs / "figures"
    source = (figures / "tables.tex").read_text()
    for job in JOBS:
        for suffix in ("symbolic", "rsa2048"):
            table = (outputs / "tables" / f"{job}_{suffix}.tex").read_text().strip()
            assert table in source
    pdf = {"tables.pdf"} if shutil.which("tectonic") else set()
    assert {p.name for p in figures.iterdir()} == {"tables.tex"} | pdf


@pytest.mark.slow
def test_recap_at_rsa2048(outputs):
    """A few values of Table operation_count_recap at RSA-2048 (Table algo_parameter_value)."""
    df = pd.read_csv(outputs / "tables" / "operation_count_recap_rsa2048.csv", dtype=str)
    rows = df.set_index("Subroutine")
    assert rows.loc["ComputeExponent", "Calls"] == str((22658 + 1) * 214)
    assert rows.loc["ComputeResidue (outer loop)", "Calls"] == str(22658 * 5 * 7)
    assert rows.loc["AccumulateResidue", "Q-Q Adder"] == "2.5"
    assert rows.loc["AccumulateResidue", "Lookup"] == "1.5"
    assert rows.loc["AccumulateResidue", "Register size"] == "34"
    assert rows.loc["Last Phaseup (ComputeExponent)", "Calls"] == "214"


@pytest.mark.slow
def test_version_comparison_totals(outputs):
    """CX total is N_PCX = 262 |P| of 05-5_final_count.tex; only the Phaseups differ."""
    df = pd.read_csv(outputs / "tables" / "version_comparison_rsa2048.csv", dtype=str)
    totals = df[df["subroutine"] == "Total"].set_index("operation")
    assert totals.loc["CX", "delayed_mbu"] == str(262 * 22658)
    differ = totals[totals["delayed_mbu"] != totals["immediate_mbu"]]
    assert list(differ.index) == ["Phaseup"]


@pytest.mark.slow
def test_only_runs_the_chosen_jobs(tmp_path):
    run(tmp_path, "--only", "compute_residue")
    assert {p.name for p in (tmp_path / "tables").iterdir()} == {
        f"compute_residue_{s}.{e}" for s in ("symbolic", "rsa2048") for e in ("csv", "tex")
    }
    assert not (tmp_path / "figures").exists()  # tables_pdf did not run
