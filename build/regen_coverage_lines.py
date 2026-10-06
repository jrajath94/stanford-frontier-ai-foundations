#!/usr/bin/env python3
"""Recompute the 'File line' column of a lesson's coverage-map table
by matching each row's 'Covered in' text against live headings.
Usage: regen_coverage_lines.py <lesson.md> [--write]
Without --write, reports mismatches between table values and computed lines.
"""
import re, sys

path = sys.argv[1]
write = len(sys.argv) > 2 and sys.argv[2] == "--write"

with open(path) as f:
    lines = f.readlines()

# Collect headings: (line_no, text_without_hashes)
headings = []
for i, ln in enumerate(lines, start=1):
    m = re.match(r"^(#{2,4})\s+(.*)$", ln.rstrip("\n"))
    if m:
        headings.append((i, m.group(2).strip()))

def find_heading(target):
    t = target.strip()
    for no, text in headings:
        if t.lower() in text.lower():
            return no
    return None

# Locate the coverage table: header row containing 'Covered in' and 'File line'
out = []
mismatches = []
in_table = False
for i, ln in enumerate(lines, start=1):
    if re.search(r"\|\s*Covered in\s*\|", ln) and re.search(r"File line", ln):
        in_table = True
        out.append(ln)
        continue
    if in_table:
        if not ln.startswith("|"):
            in_table = False
            out.append(ln)
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[1] and cells[2].isdigit():
            target, old = cells[1], int(cells[2])
            new = find_heading(target)
            if new is None:
                mismatches.append((i, target, "NO HEADING MATCH", old))
            elif new != old:
                mismatches.append((i, target, new, old))
            if write and new is not None and new != old:
                cells[2] = str(new)
                ln = "| " + " | ".join(cells) + " |\n"
        out.append(ln)
    else:
        out.append(ln)

if write:
    with open(path, "w") as f:
        f.writelines(out)
    print(f"wrote {path}: {len(mismatches)} rows checked")
for m in mismatches:
    print(m)
print(f"total mismatches: {len(mismatches)}")
