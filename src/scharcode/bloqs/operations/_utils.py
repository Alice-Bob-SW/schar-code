from collections.abc import Iterable
import sympy as sp

from qualtran.symbolics import is_symbolic, SymbolicInt


T = sp.Symbol("T")


def check_equal(number: SymbolicInt, thresh: SymbolicInt = 0, ret_bool_symb: bool = True) -> bool:
    if is_symbolic(number) or is_symbolic(thresh):
        return ret_bool_symb
    else:
        return number == thresh


def check_bigger_than(number: SymbolicInt, thresh: int = 0) -> bool:
    """Check that `number` is strictly greater than `thresh`.

    Symbolic values always return True since they can't be compared directly.

    Args:
        number (SymbolicInt): The number to check, which may be a symbolic or
            concrete integer.
        thresh (int, optional): The threshold to compare against. Defaults to 0.

    Returns:
        bool: True if `number` is symbolic or greater than `thresh`, False otherwise.
    """
    if is_symbolic(number):
        return True
    else:
        return number > thresh


def to_big_endian_indices(length: int, indices: int | Iterable[int]) -> int | tuple[int, ...]:
    """Convert little-endian indices to their big-endian equivalents."""
    if isinstance(indices, int):
        return length - indices - 1
    return tuple(length - index - 1 for index in indices)


def tuple_or_symbolic(table):
    """Convert concrete nested table data to hashable tuples, preserving symbolic values."""
    if is_symbolic(table):
        return table
    if hasattr(table, "item") and not isinstance(table, Iterable):
        return table.item()
    if isinstance(table, str | bytes):
        return table
    if isinstance(table, Iterable):
        return tuple(tuple_or_symbolic(item) for item in table)
    return table
