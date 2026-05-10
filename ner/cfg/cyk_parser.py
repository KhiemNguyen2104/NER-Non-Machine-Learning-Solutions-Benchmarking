"""
cyk_parser.py
=============
Standard CYK (Cocke-Younger-Kasami) chart-parsing algorithm.

Complexity: O(N³ · |G|)  where N = number of tokens, |G| = grammar size.

The parser returns a full parse tree (as nested ``TreeNode`` objects) when
the start symbol S spans the entire sentence, or ``None`` on parse failure.

Tree traversal
--------------
After parsing, call ``extract_entities(tree, tagged_tokens)`` to extract
NER entities by walking the tree and collecting:
  - NP nodes whose leaves are all NNP  → Object (to be disambiguated)
  - TIME_PHRASE nodes                   → Time
  - MON_PHRASE nodes                    → Monetary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ner.cfg.grammar import BINARY_LOOKUP, TERMINAL_LOOKUP
from ner.cfg.lexicon import TIME_TERMINAL, MON_TERMINAL

# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    label: str
    children: list["TreeNode"] = field(default_factory=list)
    token: Optional[str] = None   # set for leaf nodes only

    def is_leaf(self) -> bool:
        return self.token is not None

    def leaves(self) -> list[str]:
        if self.is_leaf():
            return [self.token]  # type: ignore[list-item]
        result: list[str] = []
        for child in self.children:
            result.extend(child.leaves())
        return result

    def __repr__(self) -> str:
        if self.is_leaf():
            return f"[{self.label}: {self.token!r}]"
        return f"({self.label} {' '.join(repr(c) for c in self.children)})"


# ---------------------------------------------------------------------------
# CYK algorithm
# ---------------------------------------------------------------------------

# Cell type: dict[non_terminal_str → TreeNode]
Cell = dict[str, TreeNode]


def _unary_closure(tags: list[str], token: str) -> Cell:
    """
    Given the list of terminal tags for a token, apply unary productions
    exhaustively and return a Cell mapping non-terminal → TreeNode.
    """
    cell: Cell = {}
    for tag in tags:
        leaf = TreeNode(label=tag, token=token)
        cell[tag] = leaf
        for lhs in TERMINAL_LOOKUP.get(tag, []):
            if lhs not in cell:
                cell[lhs] = TreeNode(label=lhs, children=[leaf])
    return cell


def parse(tagged_tokens: list[tuple[str, str]]) -> Optional[TreeNode]:
    """
    Run CYK on *tagged_tokens* (list of (word, terminal_tag) pairs).

    Returns the root ``TreeNode`` (label == 'S') covering the full sentence,
    or ``None`` if no complete parse exists.

    Parameters
    ----------
    tagged_tokens:
        Output of ``Lexicon.tag_sentence()``, e.g.
        ``[("Edward", "NNP"), ("runs", "VB"), ...]``
    """
    n = len(tagged_tokens)
    if n == 0:
        return None

    # chart[i][j] = Cell of non-terminals spanning tokens[i:j+1]
    chart: list[list[Cell]] = [[{} for _ in range(n)] for _ in range(n)]

    # ------------------------------------------------------------------
    # Step 1: Fill diagonal (length-1 spans) with unary productions
    # ------------------------------------------------------------------
    for i, (token, tag) in enumerate(tagged_tokens):
        cell = _unary_closure([tag], token)
        chart[i][i] = cell

    # ------------------------------------------------------------------
    # Step 2: Fill spans of increasing length (standard CYK bottom-up)
    # ------------------------------------------------------------------
    for length in range(2, n + 1):          # span length
        for i in range(n - length + 1):    # start index
            j = i + length - 1             # end index
            cell: Cell = {}

            for k in range(i, j):          # split point
                left_cell  = chart[i][k]
                right_cell = chart[k + 1][j]

                for b in left_cell:
                    for c in right_cell:
                        if (b, c) in BINARY_LOOKUP:
                            for lhs in BINARY_LOOKUP[(b, c)]:
                                if lhs not in cell:
                                    node = TreeNode(
                                        label=lhs,
                                        children=[left_cell[b], right_cell[c]],
                                    )
                                    cell[lhs] = node

            chart[i][j] = cell

    # ------------------------------------------------------------------
    # Step 3: Check if S spans the whole sentence
    # ------------------------------------------------------------------
    root_cell = chart[0][n - 1]
    return root_cell.get("S")


# ---------------------------------------------------------------------------
# Entity extraction from parse tree
# ---------------------------------------------------------------------------

def _all_leaves_are_proper(node: TreeNode) -> bool:
    """True if every leaf under *node* has tag NNP or NNPS."""
    return all(
        child.label in ("NNP", "NNPS")
        for child in _iter_leaves(node)
    )


def _iter_leaves(node: TreeNode):
    """Generator over leaf TreeNodes."""
    if node.is_leaf():
        yield node
    else:
        for child in node.children:
            yield from _iter_leaves(child)


def _traverse(node: TreeNode, entities: list[tuple[str, str]]) -> None:
    """Recursively walk tree, collecting NER entities."""
    if node.label == "TIME_PHRASE":
        text = " ".join(node.leaves())
        entities.append((text, "Time"))
        return  # no need to recurse into terminal

    if node.label == "MON_PHRASE":
        text = " ".join(node.leaves())
        entities.append((text, "Monetary"))
        return

    if node.label == "NP" and _all_leaves_are_proper(node):
        text = " ".join(node.leaves())
        entities.append((text, "Object"))  # caller will disambiguate
        # Still recurse – nested NPs may exist

    for child in node.children:
        _traverse(child, entities)


def extract_entities(tree: TreeNode) -> list[tuple[str, str]]:
    """
    Traverse *tree* and return raw entities as (text, raw_type) tuples.

    raw_type is one of "Object", "Time", "Monetary".
    Object entities need further disambiguation (→ Person/Location/Organization).
    """
    entities: list[tuple[str, str]] = []
    _traverse(tree, entities)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for text, etype in entities:
        key = (text.lower(), etype)
        if key not in seen:
            seen.add(key)
            unique.append((text, etype))

    return unique
