"""소분류 라벨 검수용 엑셀 생성.

    uv run --with openpyxl python eval/review_subcategory.py            # 계층 위반 건만
    uv run --with openpyxl python eval/review_subcategory.py --all      # 100건 전부

  gold_draft.jsonl의 소분류는 검수 전 초안이다(생성라벨 44 + 기계초안 56).
  중분류는 사람 검수를 마쳤으므로 **중분류를 기준으로 삼고 소분류를 맞춘다.**

  기본은 계층 위반 건만 뽑는다. SUBCATEGORIES_BY_CATEGORY에 없는 조합이라
  대본을 보지 않아도 틀린 것이 확정된 건들이다.

  '수정할 소분류' 칸은 해당 중분류에 속한 소분류만 고를 수 있게 드롭다운을 건다.
  → 검수 결과가 다시 계층을 깨는 일이 구조적으로 없다.
"""

import argparse
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from callwrap_ai.schemas import SUBCATEGORIES_BY_CATEGORY

AI = Path(__file__).resolve().parents[1]
DATASETS = AI / "eval/datasets"
OUT = AI / "data/scenarios/subcategory_review.xlsx"

BY_CAT = {c.value: [s.value for s in subs] for c, subs in SUBCATEGORIES_BY_CATEGORY.items()}
OWNER = {s: c for c, subs in BY_CAT.items() for s in subs}

COLS = [
    ("ID", 10),
    ("중분류 (확정)", 16),
    ("현재 소분류", 18),
    ("현재 소분류의 소속", 18),
    ("판정", 10),
    ("수정할 소분류", 20),
    ("비고", 24),
    ("대본", 120),
]


def load() -> list[dict]:
    draft = {r["scenario_id"]: r for r in map(json.loads, (DATASETS / "gold_draft.jsonl").open(encoding="utf-8"))}
    rows = []
    for line in (DATASETS / "golden_category.jsonl").open(encoding="utf-8"):
        g = json.loads(line)
        d = draft.get(g["scenario_id"], {})
        sub = d.get("consultation_subcategory")
        rows.append({
            "id": g["scenario_id"],
            "category": g["category"],
            "sub": sub,
            "owner": OWNER.get(sub, "(계약에 없는 값)"),
            "status": d.get("label_status", ""),
            "text": g["text"],
            "valid": sub in BY_CAT[g["category"]],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="소분류 검수 엑셀 생성")
    ap.add_argument("--all", action="store_true", help="계층 위반뿐 아니라 100건 전부")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = load()
    target = rows if args.all else [r for r in rows if not r["valid"]]
    if not target:
        print("대상이 없습니다.")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "소분류 검수"

    head = Font(bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="4F6228")
    for i, (name, width) in enumerate(COLS, start=1):
        c = ws.cell(row=1, column=i, value=name)
        c.font, c.fill = head, fill
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.freeze_panes = "C2"

    dv_verdict = DataValidation(type="list", formula1='"소분류 수정,현재값 유지,중분류 재검토"')
    ws.add_data_validation(dv_verdict)
    bad = PatternFill("solid", fgColor="FFC7CE")

    for n, r in enumerate(target, start=2):
        ws.cell(row=n, column=1, value=r["id"])
        ws.cell(row=n, column=2, value=r["category"])
        c_sub = ws.cell(row=n, column=3, value=r["sub"])
        ws.cell(row=n, column=4, value=r["owner"])
        if not r["valid"]:
            c_sub.fill = bad
        dv_verdict.add(ws.cell(row=n, column=5))

        # 수정 후보는 이 행의 중분류에 속한 소분류로만 제한한다.
        dv = DataValidation(type="list", formula1='"' + ",".join(BY_CAT[r["category"]]) + '"')
        ws.add_data_validation(dv)
        dv.add(ws.cell(row=n, column=6))

        ws.cell(row=n, column=7, value=r["status"])
        t = ws.cell(row=n, column=8, value=r["text"])
        t.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[n].height = 90

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)

    n_bad = sum(1 for r in target if not r["valid"])
    print(f"{len(target)}건 -> {args.out}")
    print(f"  계층 위반 {n_bad}건 (현재 소분류 칸이 붉게 표시됨)")
    print("  '수정할 소분류'는 해당 중분류의 소분류만 고를 수 있다.")


if __name__ == "__main__":
    main()
