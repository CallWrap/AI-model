"""골든셋 검수 엑셀 생성 — 큐레이션 100건 + 라벨 초안을 검수용 시트로 조립.

라벨 출처:
  v2 : 생성 시 부여된 새 체계 라벨 (label_status=generated)
  v1 : relabel_v1.py가 단 새 체계 초안 (label_status=machine_draft)

사람은 이 시트에서 초안을 보고 맞으면 두고 틀리면 고친다. 확정 라벨이 골든셋이 된다.
시트 2장: '검수'(한 행=한 콜, 라벨 편집), '대본'(콜별 발화 전문).

실행: uv run python eval/build_review_sheet.py
출력: ai/data/scenarios/golden_review.xlsx
"""

import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from callwrap_ai.schemas import (
    SUBCATEGORIES_BY_CATEGORY,
    ConsultationCategory,
    ConsultationGrade,
    ConsultationType,
)

AI_DIR = Path(__file__).resolve().parents[1]
SCN_DIR = AI_DIR / "data/scenarios"
OUT = SCN_DIR / "golden_review.xlsx"

HEADER_FILL = PatternFill("solid", fgColor="2F5496")
EDIT_FILL = PatternFill("solid", fgColor="FFF2CC")  # 검수자가 고칠 칸(노랑)
DRAFT_FILL = PatternFill("solid", fgColor="E2EFDA")  # 초안 값(연녹)
HEADER_FONT = Font(color="FFFFFF", bold=True)
WRAP = Alignment(wrap_text=True, vertical="top")


def load_labels() -> dict[str, dict]:
    """콜 id → 새 체계 라벨. v2는 생성 라벨, v1은 초안."""
    labels = {}
    for line in (SCN_DIR / "scenarios_v2.jsonl").open(encoding="utf-8"):
        s = json.loads(line)
        lb, sm = s["labels"], s.get("summary_gold", {})
        labels[s["spec_id"]] = {
            "type": lb["consultation_type"],
            "category": lb["consultation_category"],
            "subcategory": lb["consultation_subcategory"],
            "grade": lb["grade"],
            "followup": lb["needs_followup"],
            "faq": lb["needs_faq"],
            "title": sm.get("title", ""),
            "inquiry": sm.get("inquiry", ""),
            "resolution": sm.get("resolution", ""),
            "note": "",
            "status": "생성라벨",
        }
    for line in (SCN_DIR / "v1_label_drafts.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        labels[d["id"]] = {
            "type": d["consultation_type"],
            "category": d["consultation_category"],
            "subcategory": d["consultation_subcategory"],
            "grade": d["grade"],
            "followup": d["needs_followup"],
            "faq": d["needs_faq"],
            "title": d["title"],
            "inquiry": d.get("inquiry", ""),
            "resolution": d.get("resolution", ""),
            "note": d.get("note", ""),
            "status": "기계초안",
        }
    return labels


def main() -> None:
    curated = [json.loads(line) for line in (SCN_DIR / "curated_100.jsonl").open(encoding="utf-8")]
    labels = load_labels()

    wb = Workbook()

    # ── 시트 1: 검수 ────────────────────────────────
    ws = wb.active
    ws.title = "검수"
    cols = [
        ("콜ID", 10), ("출처", 8), ("라벨상태", 10), ("시나리오형", 12),
        ("상담유형", 14), ("중분류", 16), ("소분류", 18), ("등급", 9),
        ("후속처리", 9), ("FAQ필요", 9), ("요청제목", 30),
        ("검수의견/note", 40), ("확정OK", 8),
    ]
    for c, (name, width) in enumerate(cols, 1):
        cell = ws.cell(1, c, name)
        cell.fill, cell.font, cell.alignment = HEADER_FILL, HEADER_FONT, WRAP
        ws.column_dimensions[get_column_letter(c)].width = width
    ws.freeze_panes = "A2"

    # 드롭다운 검증 (편집 칸)
    dv_type = DataValidation(type="list", formula1='"' + ",".join(t.value for t in ConsultationType) + '"')
    dv_cat = DataValidation(type="list", formula1='"' + ",".join(c.value for c in ConsultationCategory) + '"')
    dv_grade = DataValidation(type="list", formula1='"' + ",".join(g.value for g in ConsultationGrade) + '"')
    dv_bool = DataValidation(type="list", formula1='"TRUE,FALSE"')
    dv_ok = DataValidation(type="list", formula1='"O,수정"')
    for dv in (dv_type, dv_cat, dv_grade, dv_bool, dv_ok):
        ws.add_data_validation(dv)

    for r, scn in enumerate(curated, 2):
        cid = scn["_id"]
        lb = labels.get(cid, {})
        row = [
            cid, scn["_origin"], lb.get("status", "?"), scn["scenario_type"],
            lb.get("type", ""), lb.get("category", ""), lb.get("subcategory", ""),
            lb.get("grade", ""), lb.get("followup", ""), lb.get("faq", ""),
            lb.get("title", ""), lb.get("note", ""), "",
        ]
        for c, val in enumerate(row, 1):
            cell = ws.cell(r, c, val)
            cell.alignment = WRAP
            if 5 <= c <= 11:  # 라벨 칸 = 초안(연녹, 고칠 수 있음)
                cell.fill = DRAFT_FILL
            if c == 13:  # 확정OK 칸(노랑)
                cell.fill = EDIT_FILL
        dv_type.add(ws.cell(r, 5)); dv_cat.add(ws.cell(r, 6))
        dv_grade.add(ws.cell(r, 8))
        dv_bool.add(ws.cell(r, 9)); dv_bool.add(ws.cell(r, 10))
        dv_ok.add(ws.cell(r, 13))

    # ── 시트 2: 대본 ────────────────────────────────
    ws2 = wb.create_sheet("대본")
    for c, (name, width) in enumerate([("콜ID", 10), ("화자", 8), ("발화", 100)], 1):
        cell = ws2.cell(1, c, name)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        ws2.column_dimensions[get_column_letter(c)].width = width
    ws2.freeze_panes = "A2"
    r = 2
    for scn in curated:
        for u in scn["utterances"]:
            ws2.cell(r, 1, scn["_id"])
            ws2.cell(r, 2, u["speaker"])
            ws2.cell(r, 3, u["text"]).alignment = WRAP
            r += 1
        r += 1  # 콜 사이 빈 줄

    # ── 시트 3: 소분류 참고 ─────────────────────────
    ws3 = wb.create_sheet("소분류표")
    ws3.cell(1, 1, "중분류").fill = HEADER_FILL
    ws3.cell(1, 1).font = HEADER_FONT
    ws3.cell(1, 2, "소분류").fill = HEADER_FILL
    ws3.cell(1, 2).font = HEADER_FONT
    ws3.column_dimensions["A"].width = 18
    ws3.column_dimensions["B"].width = 60
    r = 2
    for cat in ConsultationCategory:
        subs = ", ".join(s.value for s in SUBCATEGORIES_BY_CATEGORY[cat])
        ws3.cell(r, 1, cat.value)
        ws3.cell(r, 2, subs).alignment = WRAP
        r += 1

    wb.save(OUT)

    # 검수 전에도 채점기를 돌려볼 수 있게 초안 정답 파일을 함께 출력한다.
    # 검수가 끝나면 확정본으로 대체한다 (label_status로 구분).
    draft = AI_DIR / "eval/datasets/gold_draft.jsonl"
    with draft.open("w", encoding="utf-8") as f:
        for scn in curated:
            lb = labels.get(scn["_id"], {})
            f.write(
                json.dumps(
                    {
                        "scenario_id": scn["_id"],
                        "label_status": lb.get("status", "?"),
                        "consultation_type": lb.get("type"),
                        "consultation_category": lb.get("category"),
                        "consultation_subcategory": lb.get("subcategory"),
                        "grade": lb.get("grade"),
                        "needs_followup": lb.get("followup"),
                        "needs_faq": lb.get("faq"),
                        "summary": {
                            "title": lb.get("title", ""),
                            "inquiry": lb.get("inquiry", ""),
                            "resolution": lb.get("resolution", ""),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"저장: {OUT.name} (검수 {len(curated)}행) · {draft.name} (초안 정답)")
    stat = {}
    for scn in curated:
        s = labels.get(scn["_id"], {}).get("status", "?")
        stat[s] = stat.get(s, 0) + 1
    print("  라벨 상태:", stat)


if __name__ == "__main__":
    main()
