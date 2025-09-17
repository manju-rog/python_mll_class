from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any


EMPLOYEES = [
    "Manju","Rahul","Priya","Aisha","Vikram","Kiran","Anil","Deepa","Rohit","Neha","Arjun","Meera"
]


@dataclass
class MarkAbsentArgs:
    employee: Optional[str]
    date: Optional[str]


def _to_iso(date_str: str) -> Optional[str]:
    try:
        import dateparser
        dt = dateparser.parse(date_str)
        if dt:
            return dt.date().isoformat()
    except Exception:
        return None
    return None


def extract_absence_entities(text: str) -> MarkAbsentArgs:
    t = text.strip()
    # Employee: choose first matching known name (very naive demo)
    emp = None
    for name in EMPLOYEES:
        if re.search(rf"\b{name}\b", t, re.I):
            emp = name
            break

    # Date: today/tomorrow/yesterday or ISO date or weekday phrase
    date_patterns = [
        r"\b(today|tomorrow|yesterday)\b",
        r"\bon\s+(\d{4}-\d{2}-\d{2})\b",
        r"\bon\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(this week|last week|next week|this month|last month)\b",
    ]
    dt = None
    for pat in date_patterns:
        m = re.search(pat, t, re.I)
        if m:
            # pick the first capturing group if present else the whole match
            dt = m.group(1) if m.lastindex else m.group(0)
            break

    # Normalize simple date to ISO when possible
    iso = _to_iso(dt) if dt else None
    return MarkAbsentArgs(employee=emp, date=iso or dt)


def perform_mark_absent(args: MarkAbsentArgs) -> Dict[str, Any]:
    # Simulated side-effect; in real system, call HR/attendance API with authz & auditing.
    if not args.employee or not args.date:
        missing = []
        if not args.employee:
            missing.append("employee")
        if not args.date:
            missing.append("date")
        return {
            "status": "NEEDS_CONFIRMATION",
            "message": f"I can mark someone absent, but I still need: {', '.join(missing)}.",
            "missing": missing,
        }

    return {
        "status": "SIMULATED",
        "message": f"Okay — marking {args.employee} absent for {args.date} (demo only).",
        "employee": args.employee,
        "date": args.date,
    }
