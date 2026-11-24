"""Regenerate the operation-count tables of Appendix B from the bloqs.

pixi run paper                          # every table, into outputs/tables/
pixi run paper --only compute_residue accumulate_residue

Each job writes its table four times: `<job>_symbolic.{csv,tex}` in the parameters of
Table `parameter_definition` (|P|, l, n_e, log n_e, kappa, w1, w3, w4), and
`<job>_rsa2048.{csv,tex}` at the RSA-2048 values of Table `algo_parameter_value`. The
LaTeX uses the macros of the paper (\\Lookup, \\Adder, ...), so it is meant to be \\input there.

The last job, `tables_pdf`, gathers all these LaTeX tables in `outputs/figures/tables.pdf`
(compiled with tectonic) for reading them outside the paper.

Counts come from the call graphs of `scharcode.analysis.costs.paper_op_counts`, whose
docstring lists the counting conventions. The tables describe the delayed-MBU version (the
one of Appendix B), except `version_comparison`, which sets it next to the immediate one.
"""

import argparse
import shutil
import subprocess
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import sympy as sp
from qualtran import Bloq
from qualtran.resource_counting import (
    QubitCount,
    SympySymbolAllocator,
    get_bloq_callee_counts,
    get_cost_value,
)

from scharcode.analysis import costs
from scharcode.analysis.costs import (
    CATEGORIES,
    COMPARISON,
    CX,
    LOOKUP,
    M_X,
    PHASEUP,
    QC_ADDER,
    QQ_ADDER,
    UNDERFLOW_PHASEFIX,
    PaperOp,
    Subroutine,
    add_counts,
    evaluate,
    kappa,
    l_p,
    log_n_e,
    n_e,
    num_primes,
    paper_op_counts,
    subroutines,
    w1,
    w3,
    w4,
)
from scharcode.bloqs.operations.lookup import LookupAbstract, LookupUncompute

PAPER_VERSION = "delayed_mbu"

# ---- formatting ------------------------------------------------------------------------

TEX_NAMES = {
    QQ_ADDER: r"Q-Q \Adder{}",
    QC_ADDER: r"Q-C \Adder{}",
    COMPARISON: r"Q-Q \Comparison{}",
    LOOKUP: r"\Lookup{}",
    PHASEUP: r"\Phaseup{}",
    UNDERFLOW_PHASEFIX: r"\UnderflowPhasefix{}",
    M_X: r"$M_X$",
    CX: r"$\CX$",
}
TEX_SUBROUTINES = {
    "ComputeExponent": r"\ComputeExponent{}",
    "ReduceExponent": r"\ReduceExponent{}",
    "ReduceExponent†": r"\ReduceExponent$^\dagger$",
    "ComputeResidue": r"\ComputeResidue{}",
    "ComputeResidue†": r"\ComputeResidue$^\dagger$",
    "AccumulateResidue": r"\AccumulateResidue{}",
    "UnderflowPhasefix": r"\UnderflowPhasefix{}",
    "Last Phaseup": r"Last \Phaseup{}",
}


def _tidy(expr) -> sp.Expr:
    return sp.factor(sp.simplify(sp.sympify(expr)))


def sym_str(expr) -> str:
    return str(_tidy(expr))


def sym_tex(expr) -> str:
    return f"${costs.latex(_tidy(expr))}$"


def num_str(expr) -> str:
    """A count at RSA-2048: an integer, or a decimal for expected counts."""
    value = evaluate(expr)
    if value.is_Integer:
        return str(int(value))
    if value.is_Rational:
        return f"{float(value):g}"
    return str(value)


def tex_name(text: str) -> str:
    """Paper macro for a subroutine label such as "ComputeResidue† (outer loop)"."""
    head, _, tail = text.partition(" (")
    head = TEX_SUBROUTINES.get(head, head)
    return f"{head} ({tail}" if tail else head


def op_tex(op: PaperOp) -> str:
    return TEX_NAMES[op.category] + (" (subtraction)" if op.subtraction else "")


def write_table(out: Path, name: str, symbolic: pd.DataFrame, rsa: pd.DataFrame, tex) -> None:
    """Write `<name>_symbolic` and `<name>_rsa2048` as CSV, and their LaTeX given by `tex`."""
    tables = out / "tables"
    for suffix, df in (("symbolic", symbolic), ("rsa2048", rsa)):
        df.to_csv(tables / f"{name}_{suffix}.csv", index=False)
        (tables / f"{name}_{suffix}.tex").write_text(
            tex(df, suffix).to_latex(index=False, escape=False)
        )


# ---- operation tables (one per subroutine table of Appendix B) ---------------------------


def op_rows(part: str, counts: dict[PaperOp, sp.Expr]) -> list[dict]:
    return [{"part": part, "op": op, "calls": n} for op, n in counts.items()]


def underflow_phasefix_part(sub: Subroutine) -> dict[PaperOp, sp.Expr]:
    """Operations inside the UnderflowPhasefix calls of `sub`, summed over the oracle."""
    expanded = sub.counts()
    outside = {
        op: n
        for op, n in sub.counts(expand_underflow_phasefix=False).items()
        if op.category != UNDERFLOW_PHASEFIX
    }
    return add_counts((1, expanded), (-1, outside))


def write_op_table(out: Path, name: str, rows: list[dict]) -> None:
    """Columns as in the paper: Calls, Operation, Operation size; plus the part it is in."""
    symbolic = pd.DataFrame(
        {
            "part": r["part"],
            "calls": sym_str(r["calls"]),
            "operation": r["op"].name,
            "size": r["op"].size_str(sym_str),
        }
        for r in rows
    )
    rsa = pd.DataFrame(
        {
            "part": r["part"],
            "calls": num_str(r["calls"]),
            "operation": r["op"].name,
            "size": r["op"].size_str(num_str),
        }
        for r in rows
    )

    def tex(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        fmt = sym_tex if suffix == "symbolic" else num_str
        return pd.DataFrame(
            {
                "Part": tex_name(r["part"]),
                "Calls": fmt(r["calls"]),
                "Operation": op_tex(r["op"]),
                "Operation size": r["op"].size_str(fmt),
            }
            for r in rows
        )

    write_table(out, name, symbolic, rsa, tex)


def compute_exponent(out: Path) -> None:
    """Table `compute_exponent_operation_count`: ComputeExponent and its deferred Phaseups."""
    subs = subroutines(PAPER_VERSION)
    rows = op_rows("ComputeExponent", subs["ComputeExponent"].counts())
    rows += op_rows("Last Phaseup", subs["Last Phaseup"].counts())
    write_op_table(out, "compute_exponent", rows)


def reduce_exponent(out: Path) -> None:
    """Table `reduce_exponent_operation_count`: ReduceExponent."""
    subs = subroutines(PAPER_VERSION)
    write_op_table(
        out, "reduce_exponent", op_rows("ReduceExponent", subs["ReduceExponent"].counts())
    )


def compute_residue(out: Path) -> None:
    """Table `compute_residue_operation_count`: ComputeResidue."""
    subs = subroutines(PAPER_VERSION)
    write_op_table(
        out, "compute_residue", op_rows("ComputeResidue", subs["ComputeResidue"].counts())
    )


def _with_underflow_phasefix(out: Path, name: str, sub: Subroutine) -> None:
    rows = op_rows(sub.name, sub.counts(expand_underflow_phasefix=False))
    rows += op_rows("UnderflowPhasefix", underflow_phasefix_part(sub))
    write_op_table(out, name, rows)


def compute_residue_adjoint(out: Path) -> None:
    """
    Table `compute_residue_adjoint_operation_count`: ComputeResidue† and its UnderflowPhasefix.
    """
    _with_underflow_phasefix(
        out, "compute_residue_adjoint", subroutines(PAPER_VERSION)["ComputeResidue†"]
    )


def accumulate_residue(out: Path) -> None:
    """Table `accumulate_residue_operation_count`: AccumulateResidue and its UnderflowPhasefix."""
    _with_underflow_phasefix(
        out, "accumulate_residue", subroutines(PAPER_VERSION)["AccumulateResidue"]
    )


# ---- Table operation_count_recap ----------------------------------------------------------

_c3 = sp.ceiling(l_p / w3)
_c4 = sp.ceiling(l_p / w4)
_c1 = sp.ceiling(n_e / w1)


def _is_lookup(b: Bloq) -> bool:
    return isinstance(b, LookupAbstract)


def _is_unload(b: Bloq) -> bool:
    return isinstance(b, LookupUncompute)


RECAP_ROWS: tuple[tuple[str, str, Callable[[Bloq], bool] | None, sp.Expr], ...] = (
    ("ComputeExponent", "ComputeExponent", None, (num_primes + 1) * _c1),
    ("ReduceExponent", "ReduceExponent", None, num_primes * log_n_e),
    ("ComputeResidue (initialization)", "ComputeResidue", _is_lookup, num_primes),
    (
        "ComputeResidue (outer loop)",
        "ComputeResidue",
        lambda b: not _is_lookup(b),
        num_primes * (_c3 - 2) * _c3,
    ),
    ("AccumulateResidue", "AccumulateResidue", None, num_primes * _c4),
    (
        "ComputeResidue† (outer loop)",
        "ComputeResidue†",
        lambda b: not _is_unload(b),
        2 * num_primes * (_c3 - 2) * _c3,
    ),
    ("ComputeResidue† (unloading)", "ComputeResidue†", _is_unload, num_primes),
    ("ReduceExponent†", "ReduceExponent†", None, num_primes * log_n_e),
    ("Last Phaseup (ComputeExponent)", "Last Phaseup", None, _c1),
)
"""Rows of Table `operation_count_recap`: label, subroutine, which of its direct callees the
row covers (None: all), and the Calls of the paper, by which the code totals are divided."""

RECAP_COLUMNS = (QQ_ADDER, QC_ADDER, LOOKUP, PHASEUP, CX, M_X)


def recap_counts(sub: Subroutine, select: Callable[[Bloq], bool] | None) -> dict:
    """Totals over the oracle, comparisons as Q-Q Adders, of the callees picked by `select`."""
    if select is None:
        return sub.counts(comparison_as_adder=True)
    terms = [
        (sub.calls * n, paper_op_counts(callee, comparison_as_adder=True))
        for callee, n in get_bloq_callee_counts(sub.bloq, ssa=SympySymbolAllocator())
        if select(callee)
    ]
    return add_counts(*terms)


def recap_rows(version_: str) -> list[dict]:
    """Per row: Calls, register and address sizes, and operations per call by category.

    Per-call counts are the code totals divided by the paper's Calls. A non-constant entry
    means that the code and the Calls of the paper disagree.
    """
    subs = subroutines(version_)
    rows = []
    for label, sub_name, select, calls in RECAP_ROWS:
        if sub_name not in subs:
            continue
        counts = recap_counts(subs[sub_name], select)
        registers = [op.sizes[-1] for op in counts if op.category in (QQ_ADDER, QC_ADDER, LOOKUP)]
        addresses = [op.sizes[0] for op in counts if op.category in (LOOKUP, PHASEUP)]
        row = {
            "subroutine": label,
            "calls": calls,
            "register size": sp.Max(*registers) if registers else None,
            "address size": sp.Max(*addresses) if addresses else None,
        }
        for category in RECAP_COLUMNS:
            total = sum((n for op, n in counts.items() if op.category == category), sp.Integer(0))
            row[category] = sp.simplify(total / calls)
        rows.append(row)
    return rows


def operation_count_recap(out: Path) -> None:
    """Table `operation_count_recap`, comparisons counted as Q-Q Adders."""
    rows = recap_rows(PAPER_VERSION)

    def frame(fmt, name_fmt=str, dash="-") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Subroutine": name_fmt(r["subroutine"]),
                "Calls": fmt(r["calls"]),
                "Register size": dash if r["register size"] is None else fmt(r["register size"]),
                "Address size": dash if r["address size"] is None else fmt(r["address size"]),
                **{c: fmt(r[c]) for c in RECAP_COLUMNS},
            }
            for r in rows
        )

    def tex(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        df = frame(sym_tex if suffix == "symbolic" else num_str, tex_name)
        return df.rename(columns={**TEX_NAMES, QQ_ADDER: r"Q-Q \Adder", QC_ADDER: r"Q-C \Adder"})

    write_table(out, "operation_count_recap", frame(sym_str), frame(num_str), tex)


# ---- Table logical_qubit_count ---------------------------------------------

PAPER_OUT_OF_SCOPE = {
    "ComputeExponent": kappa,
    "ReduceExponent": n_e + kappa,
    "ComputeResidue": n_e + log_n_e + kappa,
    "AccumulateResidue": n_e + log_n_e + l_p,
    "ComputeResidue†": n_e + log_n_e + kappa,
    "ReduceExponent†": n_e + kappa,
    "Last Phaseup": kappa,
}
"""Column "Out of scope" of Table `logical_qubit_count` (04_logical-implementation.tex)."""


def logical_qubit_count(out: Path) -> None:
    """Static qubit count of each subroutine (`my_static_costs`) next to the paper's
    out-of-scope qubits, for a row-by-row comparison with Table `logical_qubit_count`.

    "Code peak" is the code's count plus the out-of-scope registers: the total number of
    logical qubits while the subroutine runs, according to the code.
    """
    rows = []
    for name, sub in subroutines(PAPER_VERSION).items():
        code = get_cost_value(sub.bloq, QubitCount())
        rows.append(
            {
                "subroutine": name,
                "code": code,
                "out of scope (paper)": PAPER_OUT_OF_SCOPE[name],
                "code peak": code + PAPER_OUT_OF_SCOPE[name],
            }
        )

    def frame(fmt, name_fmt=str) -> pd.DataFrame:
        return pd.DataFrame(
            {"subroutine": name_fmt(r["subroutine"])}
            | {k: fmt(v) for k, v in r.items() if k != "subroutine"}
            for r in rows
        )

    def tex(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return frame(sym_tex if suffix == "symbolic" else num_str, tex_name)

    write_table(out, "logical_qubit_count", frame(sym_str), frame(num_str), tex)


# ---- immediate vs delayed -----------------------------------------------------------------


def version_comparison(out: Path) -> None:
    """Operations per subroutine and category, summed over sizes, for both versions."""
    totals = {v: {} for v in costs.VERSIONS}
    for v in costs.VERSIONS:
        for name, sub in subroutines(v).items():
            for op, n in sub.counts().items():
                key = (name, op.category)
                totals[v][key] = totals[v].get(key, 0) + n
    names = list(subroutines(PAPER_VERSION))
    keys = [(s, c) for s in names for c in CATEGORIES if any((s, c) in totals[v] for v in totals)]
    for c in CATEGORIES:
        if any(k[1] == c for k in keys):
            keys.append(("Total", c))
    for v in totals:
        for c in CATEGORIES:
            total = sum((n for (s, c_), n in totals[v].items() if c_ == c), sp.Integer(0))
            if total != 0:
                totals[v][("Total", c)] = total

    def frame(fmt, name_fmt=str, cat_fmt=str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "subroutine": name_fmt(s),
                "operation": cat_fmt(c),
                **{v: fmt(totals[v].get((s, c), 0)) for v in costs.VERSIONS},
            }
            for s, c in keys
        )

    def tex(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        fmt = sym_tex if suffix == "symbolic" else num_str
        df = frame(fmt, tex_name, TEX_NAMES.get)
        return df.rename(columns={"delayed_mbu": "delayed MBU", "immediate_mbu": "immediate MBU"})

    write_table(out, "version_comparison", frame(sym_str), frame(num_str), tex)


# ---- all tables in one PDF ----------------------------------------------------------------

TABLE_TITLES = {
    "compute_exponent": (r"\ComputeExponent{}", "computing_dlog_operation_count"),
    "reduce_exponent": (r"\ReduceExponent{}", "dlog_modular_reduction_operation_count"),
    "compute_residue": (r"\ComputeResidue{}", "modexp_operation_count"),
    "compute_residue_adjoint": (r"\ComputeResidue$^\dagger$", "unmodexp_operation_count"),
    "accumulate_residue": (r"\AccumulateResidue{}", "accumulation_operation_count"),
    "operation_count_recap": ("Operation count recap", "operation_count_recap"),
    "logical_qubit_count": ("Static qubit counts", "logical_qubit_count"),
    "version_comparison": ("Immediate vs delayed MBU", None),
}
"""Section title of each table job in the PDF, and the label of its table in the paper."""

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,landscape,margin=1.5cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,adjustbox}
\usepackage[T1]{fontenc}
\usepackage{newunicodechar}
\newunicodechar{†}{$^\dagger$}
% Macros of the paper (schar_paper/tex_files/ZZ_header.tex)
\newcommand{\C}{\mathord{\mathrm{C}}}
\newcommand{\CX}{\mathord{\C X}}
\newcommand{\Adder}{\textsc{Adder}}
\newcommand{\Comparison}{\textsc{Comparison}}
\newcommand{\Lookup}{\textsc{Lookup}}
\newcommand{\Phaseup}{\textsc{Phaseup}}
\newcommand{\ComputeExponent}{\textsc{ComputeExponent}}
\newcommand{\ReduceExponent}{\textsc{ReduceExponent}}
\newcommand{\ComputeResidue}{\textsc{ComputeResidue}}
\newcommand{\AccumulateResidue}{\textsc{AccumulateResidue}}
\newcommand{\UnderflowPhasefix}{\textsc{UnderflowPhasefix}}
\setlength{\parindent}{0pt}
"""


def _tt(text: str) -> str:
    """Escape underscores for \\texttt."""
    return text.replace("_", r"\_")


def tables_pdf(out: Path) -> None:
    """All the LaTeX tables in `outputs/tables/` in one document, `outputs/figures/tables.pdf`.

    The source `tables.tex` embeds each table, so it compiles on its own. Compiling needs
    `tectonic` (in the pixi environment); without it only `tables.tex` is written.
    """
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    body = [
        r"\begin{center}{\Large Appendix B operation counts, generated by"
        r" \texttt{scripts/make\_paper\_outputs.py}}\\[1ex]"
        rf"scharcode {version('scharcode')}, "
        r"delayed-MBU version unless stated otherwise\end{center}",
    ]
    for job, (title, label) in TABLE_TITLES.items():
        for suffix, what in (("symbolic", "symbolic"), ("rsa2048", "RSA-2048")):
            path = out / "tables" / f"{job}_{suffix}.tex"
            if not path.exists():
                continue
            paper = rf" --- paper: Table \texttt{{{_tt(label)}}}" if label else ""
            file = rf"\texttt{{{_tt(path.name)}}}"
            body += [
                r"\clearpage" if suffix == "symbolic" else r"\bigskip",
                rf"\section*{{{title} ({what})}}",
                rf"{{\small {file}{paper}}}\par\medskip",
                r"\begin{adjustbox}{max width=\textwidth,max totalheight=0.85\textheight}",
                path.read_text().strip(),
                r"\end{adjustbox}",
            ]
    source = figures / "tables.tex"
    source.write_text(PREAMBLE + "\\begin{document}\n" + "\n".join(body) + "\n\\end{document}\n")
    if shutil.which("tectonic") is None:
        print("[paper] tectonic not found: wrote figures/tables.tex but no PDF")
        return
    subprocess.run(
        ["tectonic", "--chatter", "minimal", source.name],
        cwd=figures,
        check=True,
    )


JOBS = {
    "compute_exponent": compute_exponent,
    "reduce_exponent": reduce_exponent,
    "compute_residue": compute_residue,
    "compute_residue_adjoint": compute_residue_adjoint,
    "accumulate_residue": accumulate_residue,
    "operation_count_recap": operation_count_recap,
    "logical_qubit_count": logical_qubit_count,
    "version_comparison": version_comparison,
    "tables_pdf": tables_pdf,  # last: gathers the .tex files written by the jobs above
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--only", nargs="+", choices=list(JOBS))
    args = parser.parse_args(argv)

    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    for name in args.only or list(JOBS):
        print(f"[paper] {name}")
        JOBS[name](args.out)


if __name__ == "__main__":
    main()
