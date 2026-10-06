#!/usr/bin/env python3
"""Two-phase coverage-line regen that only touches rows that were
consistent before editing (table value == live heading line).
Phase 1: python3 regen2.py snapshot <lessons...>
Phase 2: python3 regen2.py apply <lessons...>
"""
import re, sys, json, os

SNAP = "/tmp/cov_snap.json"

def headings_of(lines):
    return [(i, l.rstrip("\n")) for i, l in enumerate(lines, 1)
            if re.match(r"^#{2,4}\s+", l)]

def find_heading(heads, target):
    t = target.strip().lower()
    for no, text in heads:
        txt = re.sub(r"^#{2,4}\s+", "", text).strip().lower()
        if t in txt:
            return no
    return None

def table_rows(lines):
    rows = []  # (file_line_no, target, value)
    in_table = False
    for i, ln in enumerate(lines, 1):
        if re.search(r"\|\s*Covered in\s*\|", ln) and "File line" in ln:
            in_table = True
            continue
        if in_table:
            if not ln.startswith("|"):
                in_table = False
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) >= 3 and cells[1] and cells[2].isdigit():
                rows.append((i, cells[1], int(cells[2])))
    return rows

def snapshot(lessons):
    snap = {}
    for path in lessons:
        with open(path) as f:
            lines = f.readlines()
        heads = headings_of(lines)
        snap[path] = [(r[0], r[1], r[2]) for r in table_rows(lines)
                      if find_heading(heads, r[1]) == r[2]]
    with open(SNAP, "w") as f:
        json.dump(snap, f)
    for p, rows in snap.items():
        print(p, "consistent rows snapshotted:", len(rows))

def apply(lessons):
    with open(SNAP) as f:
        snap = json.load(f)
    for path in lessons:
        with open(path) as f:
            lines = f.readlines()
        heads = headings_of(lines)
        # index current table rows by their 'Covered in' target
        row_idx = {}
        in_table = False
        for i, ln in enumerate(lines):
            if re.search(r"\|\s*Covered in\s*\|", ln) and "File line" in ln:
                in_table = True
                continue
            if in_table:
                if not ln.startswith("|"):
                    in_table = False
                    continue
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                if len(cells) >= 3 and cells[1] and cells[2].isdigit():
                    row_idx.setdefault(cells[1], []).append(i)
        changed = 0
        missing = 0
        for (_rowno, target, old) in snap.get(path, []):
            new = find_heading(heads, target)
            if new is None or new == old:
                continue
            if target not in row_idx:
                missing += 1
                continue
            for idx in row_idx[target]:
                ln = lines[idx]
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                cells[2] = str(new)
                lines[idx] = "| " + " | ".join(cells) + " |\n"
                changed += 1
        with open(path, "w") as f:
            f.writelines(lines)
        print(path, "rows updated:", changed, "targets missing:", missing)

if __name__ == "__main__":
    cmd, lessons = sys.argv[1], sys.argv[2:]
    {"snapshot": snapshot, "apply": apply}[cmd](lessons)
