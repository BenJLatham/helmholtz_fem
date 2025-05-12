"""
T-Conf Meshing Utilities

This module provides general-purpose helper functions for the T-Conf meshing
package, such as checking for missing indices in sets and ensuring that the
Gmsh Python API is available before attempting mesh operations.
"""

from typing import Sequence, Tuple, List


def find_missing_numbers(lst: Sequence[int], arr: Sequence[int]) -> Tuple[bool, List[int]]:
    """
    Identify which integers in `lst` are not present in `arr`.

    Parameters
    ----------
    lst : Sequence[int]
        A sequence of integer tags (e.g., corner point IDs) that were found.
    arr : Sequence[int]
        The expected sequence of integer tags.

    Returns
    -------
    Tuple[bool, List[int]]
        A tuple where the first element is True if any numbers are missing,
        and the second element is the sorted list of missing integers.

    Example
    -------
    >>> found, missing = find_missing_numbers([1, 3, 4], [1, 2, 3, 4])
    >>> found
    True
    >>> missing
    [2]
    """
    lst_set = set(lst)
    arr_set = set(arr)
    missing = sorted(list(arr_set - lst_set))
    return bool(missing), missing


def ensure_gmsh_available() -> None:
    """
    Verify that the Gmsh Python API is importable and that libGLU is present.
    Raises a RuntimeError if it is not available.

    This should be called at the start of any operation that requires Gmsh.

    Raises
    ------
    RuntimeError
        If the Gmsh Python API cannot be imported or initialized.
    """
    try:
        import gmsh  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "gmsh Python package is required but not installed. "
            "Please install gmsh (e.g., pip install gmsh) and ensure libGLU.so.1 is available."
        ) from e


__all__ = [
    "find_missing_numbers",
    "ensure_gmsh_available",
]
