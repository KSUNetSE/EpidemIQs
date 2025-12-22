#!/usr/bin/env python3
"""
auto_split_table.py
A linter that automatically breaks LaTeX tables with >5 data columns
into two side-by-side tables (≤5 columns each).

Usage:
    python auto_split_table.py your_file.tex   # rewrites in-place
    python auto_split_table.py your_file.tex > fixed.tex
"""

import re
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Regexes
# ----------------------------------------------------------------------
TABLE_ENV_RE = re.compile(
    r"(\\begin\{table\*?\}.*?)"          # start of environment + options
    r"(\\begin\{tabular\}[\{\[][^}]*\}[\{\[])"  # \begin{tabular}{...}
    r"(.*?)"
    r"(\\end\{tabular\})"
    r"(.*?)"
    r"(\\end\{table\*?\})",
    re.DOTALL,
)

COLUMN_SPEC_RE = re.compile(r"\{([^}]+)\}")   # captures the column spec string


def count_data_columns(col_spec: str) -> int:
    """Count columns after the first (row-header) column."""
    # strip possible @{} etc.
    clean = re.sub(r"@\{.*?\}", "", col_spec)
    return len([c for c in clean if c in "lcrpbd"])


def split_columns(col_spec: str, max_data_cols: int = 5):
    """Return (left_spec, right_spec, split_idx) where split_idx is the
    index of the first column that goes to the right table (0-based)."""
    clean = re.sub(r"@\{.*?\}", "", col_spec)
    letters = [c for c in clean if c in "lcrpbd"]
    data_cols = letters[1:]                     # exclude row-header
    if len(data_cols) <= max_data_cols:
        return col_spec, None, None

    # keep first column + max_data_cols data columns
    keep = 1 + max_data_cols
    left_letters = letters[:keep]
    right_letters = letters[keep:]

    # reconstruct specs (preserve @{} if any)
    left_spec = col_spec[: col_spec.index(letters[0])]
    left_spec += "".join(left_letters)
    right_spec = col_spec[: col_spec.index(letters[0])]
    right_spec += "".join(right_letters)
    return left_spec, right_spec, keep


def split_row(row: str, split_idx: int):
    """Split a single table line at the given & position."""
    parts = [p.strip() for p in row.split("&")]
    if len(parts) <= split_idx:
        return row, ""
    left = " & ".join(parts[:split_idx])
    right = " & ".join(parts[split_idx:])
    return left, right


def process_table(match):
    pre, tabular_start, body, tabular_end, post, env_end = match.groups()

    # ------------------------------------------------------------------
    # Parse column spec
    # ------------------------------------------------------------------
    col_spec_match = COLUMN_SPEC_RE.search(tabular_start)
    if not col_spec_match:
        return match.group(0)          # give up – malformed tabular
    col_spec = col_spec_match.group(1)

    data_col_count = count_data_columns(col_spec)
    if data_col_count <= 5:
        return match.group(0)          # nothing to do

    # ------------------------------------------------------------------
    # Split column specs
    # ------------------------------------------------------------------
    left_spec, right_spec, split_idx = split_columns(col_spec, max_data_cols=5)
    if right_spec is None:
        return match.group(0)

    # ------------------------------------------------------------------
    # Split every line of the body
    # ------------------------------------------------------------------
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    left_lines, right_lines = [], []

    for raw in lines:
        # preserve booktabs commands as whole lines
        if raw.strip().startswith(("\\toprule", "\\midrule", "\\bottomrule", "\\hline")):
            left_lines.append(raw)
            right_lines.append(raw)
            continue

        l, r = split_row(raw, split_idx)
        left_lines.append(l)
        right_lines.append(r)

    left_body = "\n".join(left_lines)
    right_body = "\n".join(right_lines)

    # ------------------------------------------------------------------
    # Re-build LaTeX
    # ------------------------------------------------------------------
    caption_match = re.search(r"\\caption\{(.*?)\}", pre + post, re.DOTALL)
    label_match = re.search(r"\\label\{(.*?)\}", pre + post)

    caption = caption_match.group(1) if caption_match else "AUTO-GENERATED"
    label   = label_match.group(1) if label_match else "auto:table"

    # ---- left table -------------------------------------------------
    left_tabular = (
        f"\\begin{{tabular}}{{{left_spec}}}\n"
        f"{left_body}\n"
        f"\\end{{tabular}}"
    )

    # ---- right table ------------------------------------------------
    right_tabular = (
        f"\\begin{{tabular}}{{{right_spec}}}\n"
        f"{right_body}\n"
        f"\\end{{tabular}}"
    )

    # ---- final two-column layout ------------------------------------
    result = f"""\\begin{{table*}}[ht]
    \\centering
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\begin{{minipage}}{{.48\\textwidth}}
    \\centering
    {left_tabular}
    \\end{{minipage}}%
    \\hfill
    \\begin{{minipage}}{{.48\\textwidth}}
    \\centering
    {right_tabular}
    \\end{{minipage}}
\\end{{table*}}"""
    return result


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python auto_split_table.py <file.tex>", file=sys.stderr)
        sys.exit(1)

    tex_path = Path(sys.argv[1])
    content = tex_path.read_text(encoding="utf-8")

    new_content = TABLE_ENV_RE.sub(process_table, content)

    # Overwrite or print
    if sys.stdout.isatty():
        tex_path.write_text(new_content, encoding="utf-8")
        print(f"Table in {tex_path} has been split (if needed).")
    else:
        print(new_content)


if __name__ == "__main__":
    main()