"""P1-a: 도메인 코퍼스 수집.

세 개 공개 FAQ 사이트와 100문100답 PDF에서 Q/A를 모아 두 가지 형태로 낸다.

    data/faq/*.jsonl    {question, answer, category, source, url}
                        -> P2/P3 시나리오 생성의 사실 근거
    data/corpus/*.txt   한 줄 = 한 문단
                        -> P4-a DAPT(MLM) 코퍼스

robots.txt 확인 결과 (2026-07-28):
    socialservice.or.kr  /user/join/faq/* 허용 (Disallow: /ad /qn /my /user/cleanCenter/*)
    sjss.or.kr           Allow:/ (Disallow: /ecms/ 만)
    ableservice.or.kr    robots.txt 없음 (제약 명시 없음)

공공기관 공개 FAQ이며 산학협력 과제 목적이다. 요청 간 지연을 두고 순차 수집한다.

사용:
    python -m callwrap.collect --all
    python -m callwrap.collect --site socialservice --max-pages 5
    python -m callwrap.collect --pdf "장애인 활동지원제도 100문 100답_최종.pdf"
"""

from __future__ import annotations

import argparse
import hashlib
import html as H
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

# HTTP 헤더는 latin-1로 인코딩되므로 ASCII만 쓴다 (한글을 넣으면 UnicodeEncodeError).
USER_AGENT = ("Mozilla/5.0 (compatible; CallWrapResearchBot/0.1; "
              "academic research; contact: jaehyuk2.lee@gmail.com)")
DEFAULT_DELAY = 1.0   # 요청 간 지연(초). 공공 사이트에 부담을 주지 않는다.
DEFAULT_TIMEOUT = 30


# ---------------------------------------------------------------------------
# 자료 구조
# ---------------------------------------------------------------------------
@dataclass
class FAQItem:
    question: str
    answer: str
    source: str
    category: str = ""
    subcategory: str = ""
    url: str = ""
    item_id: str = ""

    def key(self) -> str:
        return hashlib.sha1(
            (self.question + self.answer[:200]).encode("utf-8")
        ).hexdigest()[:16]


@dataclass
class Stats:
    fetched: int = 0
    items: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Fetcher:
    """지연 + 재시도 + 디스크 캐시가 있는 단순 HTTP 클라이언트.

    캐시가 있으면 재실행 시 서버를 다시 때리지 않는다. 파서를 고쳐가며
    반복 실행할 때 중요하다.
    """

    def __init__(self, cache_dir: Path | None = None, delay: float = DEFAULT_DELAY,
                 timeout: int = DEFAULT_TIMEOUT, retries: int = 3) -> None:
        self.cache_dir = cache_dir
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self._last_call = 0.0
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url: str, data: bytes | None) -> Path | None:
        if not self.cache_dir:
            return None
        h = hashlib.sha1((url + (data.decode("utf-8", "replace") if data else ""))
                         .encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{h}.html"

    def get(self, url: str, params: dict | None = None,
            data: dict | None = None, *, use_cache: bool = True) -> str:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        body = urllib.parse.urlencode(data).encode("utf-8") if data else None

        cache = self._cache_path(url, body)
        if use_cache and cache and cache.exists():
            return cache.read_text(encoding="utf-8")

        last_err: Exception | None = None
        for attempt in range(self.retries):
            wait = self.delay - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                req = urllib.request.Request(
                    url, data=body,
                    headers={"User-Agent": USER_AGENT,
                             "Accept": "text/html,application/xhtml+xml",
                             "Accept-Language": "ko-KR,ko;q=0.9"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    charset = resp.headers.get_content_charset() or "utf-8"
                self._last_call = time.time()
                text = raw.decode(charset, errors="replace")
                if cache:
                    cache.write_text(text, encoding="utf-8")
                return text
            except Exception as exc:  # noqa: BLE001 — 네트워크 실패는 재시도로 흡수
                last_err = exc
                self._last_call = time.time()
                time.sleep(2 ** attempt)
        raise RuntimeError(f"요청 실패 ({self.retries}회): {url} — {last_err}")


# ---------------------------------------------------------------------------
# HTML 유틸
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(fragment: str) -> str:
    s = re.sub(r"<(script|style).*?</\1>", " ", fragment, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(p|div|tr|li|h[1-6]|td|dd|dt|table)>", "\n", s, flags=re.I)
    s = _TAG_RE.sub("", s)
    s = H.unescape(s)
    # NBSP, zero-width space, BOM 등 눈에 안 보이는 문자 제거
    s = s.replace("\xa0", " ").replace("​", "").replace("﻿", "")
    s = re.sub(r"[ \t]+", " ", s)
    # <br> 연속으로 생긴 공백뿐인 줄을 버린 뒤 문단을 다시 잇는다
    lines = [ln.strip() for ln in s.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(ln for ln in lines if ln)).strip()


def extract_block(html: str, cls: str) -> str | None:
    """class에 `cls`를 포함하는 첫 태그의 내용을 중첩을 고려해 잘라낸다."""
    m = re.search(r'<(\w+)[^>]*class="[^"]*\b' + re.escape(cls) + r'\b[^"]*"[^>]*>',
                  html)
    if not m:
        return None
    tag, start = m.group(1), m.end()
    pat = re.compile(rf"<{tag}\b[^>]*>|</{tag}\s*>", re.I)
    depth, i = 1, start
    while depth:
        mm = pat.search(html, i)
        if not mm:
            return html[start:]
        depth += -1 if mm.group(0).startswith("</") else 1
        i = mm.end()
        if depth == 0:
            return html[start:mm.start()]
    return None


def clean_qa(text: str) -> str:
    """선행 'Q'/'A' 마커와 군더더기를 제거한다."""
    text = text.strip()
    text = re.sub(r"^\s*[QA]\s*[\.\):]?\s*\n?", "", text)
    text = re.sub(r"^\s*(질문|답변)\s*[\.\):]?\s*\n?", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 어댑터 1 — 사회서비스 전자바우처 FAQ
#   목록: POST /user/join/faq/list.do  (pageIndex)  -> 10건/페이지
#         tr = [id, 대분류, 중분류, 제목, 등록일, 조회수], onclick=fnView(id)
#   상세: POST /user/join/faq/view.do  (p_bbs_sn)   -> .detail_tit / .detail_body
# ---------------------------------------------------------------------------
SS_BASE = "https://www.socialservice.or.kr:444"


def collect_socialservice(fetcher: Fetcher, max_pages: int = 50,
                          stats: Stats | None = None) -> list[FAQItem]:
    stats = stats or Stats()
    items: list[FAQItem] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        html = fetcher.get(f"{SS_BASE}/user/join/faq/list.do",
                           data={"pageIndex": str(page)})
        stats.fetched += 1

        tbody = re.search(r"<tbody[^>]*>(.*?)</tbody>", html, re.S)
        if not tbody:
            break
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tbody.group(1), re.S)
        page_ids: list[tuple[str, str, str, str]] = []
        for row in rows:
            cells = [html_to_text(c) for c in
                     re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
            m = re.search(r"fnView\('?(\d+)'?\)", row)
            if not m or len(cells) < 4:
                continue
            page_ids.append((m.group(1), cells[1], cells[2], cells[3]))

        if not page_ids or all(i[0] in seen for i in page_ids):
            break  # 마지막 페이지를 넘어가면 같은 내용이 반복된다

        for bbs_sn, major, minor, title in page_ids:
            if bbs_sn in seen:
                continue
            seen.add(bbs_sn)
            try:
                view = fetcher.get(f"{SS_BASE}/user/join/faq/view.do",
                                   data={"p_bbs_sn": bbs_sn, "pageIndex": str(page)})
                stats.fetched += 1
                body = extract_block(view, "detail_body")
                tit = extract_block(view, "detail_tit")
                answer = clean_qa(html_to_text(body)) if body else ""
                question = clean_qa(html_to_text(tit)) if tit else title
                if not answer:
                    continue
                items.append(FAQItem(
                    question=question, answer=answer,
                    source="사회서비스 전자바우처 FAQ",
                    category=major, subcategory=minor,
                    url=f"{SS_BASE}/user/join/faq/view.do?p_bbs_sn={bbs_sn}",
                    item_id=bbs_sn,
                ))
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"socialservice#{bbs_sn}: {exc}")

        print(f"  socialservice p{page}: 누적 {len(items)}건", flush=True)

    stats.items += len(items)
    return items


# ---------------------------------------------------------------------------
# 어댑터 2 — 세종시지역사회서비스지원단 FAQ
#   GET /board.es?mid=...&bid=0023&nPage=N   (한 페이지에 Q/A가 통째로 들어있음)
#   .question / .answer 쌍
# ---------------------------------------------------------------------------
SJ_BASE = "https://www.sjss.or.kr"


def collect_sjss(fetcher: Fetcher, max_pages: int = 20,
                 stats: Stats | None = None) -> list[FAQItem]:
    stats = stats or Stats()
    items: list[FAQItem] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        html = fetcher.get(f"{SJ_BASE}/board.es",
                           params={"mid": "a10607000000", "bid": "0023",
                                   "nPage": str(page)})
        stats.fetched += 1

        qs = re.findall(r'class="question"[^>]*>(.*?)</(?:a|div|dt|p|strong|h\d)>',
                        html, re.S)
        ans = re.findall(r'class="answer"[^>]*>(.*?)</div>', html, re.S)
        if not qs:
            break

        added = 0
        for q, a in zip(qs, ans):
            question = clean_qa(html_to_text(q))
            answer = clean_qa(html_to_text(a))
            if not question or not answer:
                continue
            k = hashlib.sha1(question.encode()).hexdigest()[:12]
            if k in seen:
                continue
            seen.add(k)
            items.append(FAQItem(
                question=question, answer=answer,
                source="세종시지역사회서비스지원단 FAQ",
                url=f"{SJ_BASE}/board.es?mid=a10607000000&bid=0023&nPage={page}",
                item_id=k,
            ))
            added += 1

        print(f"  sjss p{page}: 누적 {len(items)}건", flush=True)
        if added == 0:
            break

    stats.items += len(items)
    return items


# ---------------------------------------------------------------------------
# 어댑터 3 — 장애인활동지원 자주묻는 질문
#   목록: POST /jng/faq/getOHGF0001M0.do  (pageIndex, dasFaqSeCd)
#         table.data-list > tbody > tr = [문의분류, 제목(dtlBbs(id)), 조회수]
#   상세: POST /jng/faq/getOHGF0001M1.do  (faqMngNo) -> .view-title / .view-contents
# ---------------------------------------------------------------------------
AB_BASE = "https://www.ableservice.or.kr"
AB_CATEGORIES = ["", "01", "02", "03", "04", "05"]  # ""=전체


def collect_ableservice(fetcher: Fetcher, max_pages: int = 50,
                        stats: Stats | None = None) -> list[FAQItem]:
    stats = stats or Stats()
    items: list[FAQItem] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        html = fetcher.get(f"{AB_BASE}/jng/faq/getOHGF0001M0.do",
                           data={"pageIndex": str(page), "dasFaqSeCd": "",
                                 "comd": "123", "scwdSe": "all"})
        stats.fetched += 1

        table = extract_block(html, "data-list")
        if not table:
            break
        tbody = re.search(r"<tbody[^>]*>(.*?)</tbody>", table, re.S)
        if not tbody:
            break

        page_rows: list[tuple[str, str, str]] = []
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", tbody.group(1), re.S):
            m = re.search(r"dtlBbs\(\s*['\"]?(\d+)['\"]?\s*\)", row)
            if not m:
                continue
            cells = [html_to_text(c) for c in
                     re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
            page_rows.append((m.group(1),
                              cells[0] if cells else "",
                              cells[1] if len(cells) > 1 else ""))

        if not page_rows or all(r[0] in seen for r in page_rows):
            break

        for faq_no, cat, title in page_rows:
            if faq_no in seen:
                continue
            seen.add(faq_no)
            try:
                view = fetcher.get(f"{AB_BASE}/jng/faq/getOHGF0001M1.do",
                                   data={"faqMngNo": faq_no, "pageIndex": str(page),
                                         "comd": "123", "scwdSe": "all"})
                stats.fetched += 1
                body = (extract_block(view, "view-contents")
                        or extract_block(view, "answerBox"))
                tit = extract_block(view, "view-title")
                answer = clean_qa(html_to_text(body)) if body else ""
                question = clean_qa(html_to_text(tit)) if tit else title
                if not answer:
                    continue
                items.append(FAQItem(
                    question=question, answer=answer,
                    source="장애인활동지원 자주묻는 질문",
                    category=cat.strip("[]"),
                    url=f"{AB_BASE}/jng/faq/getOHGF0001M1.do?faqMngNo={faq_no}",
                    item_id=faq_no,
                ))
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"ableservice#{faq_no}: {exc}")

        print(f"  ableservice p{page}: 누적 {len(items)}건", flush=True)

    stats.items += len(items)
    return items


# ---------------------------------------------------------------------------
# 어댑터 4 — 100문100답 PDF
# ---------------------------------------------------------------------------
def collect_pdf(path: Path, stats: Stats | None = None) -> list[FAQItem]:
    """`장애인 활동지원제도 100문 100답` 류 PDF에서 Q/A를 뽑는다.

    레이아웃이 문서마다 다르므로, Q 패턴을 못 찾으면 페이지 텍스트를 문단으로
    떨어뜨려 DAPT 코퍼스로만 쓴다 (질문 없이 answer만 채운다).
    """
    stats = stats or Stats()
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit("pypdf가 필요합니다: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    stats.fetched += len(pages)
    full = "\n".join(pages)
    full = re.sub(r"\n\s*\d+\s*\n", "\n", full)          # 페이지 번호 제거
    full = re.sub(r"[ \t]+", " ", full)

    # "Q1." / "질문 1." / "1. ..." 형태를 순서대로 시도
    patterns = [
        r"(?m)^\s*Q\s*[\.\)]?\s*(\d+)\s*[\.\)]?\s*(.+?)(?=^\s*A\s*[\.\)]?\s|\Z)",
        r"(?m)^\s*(\d{1,3})\s*[\.\)]\s*(.+?)(?=^\s*\d{1,3}\s*[\.\)]|\Z)",
    ]
    items: list[FAQItem] = []
    for pat in patterns:
        blocks = re.findall(pat, full, re.S)
        if len(blocks) >= 10:
            for num, body in blocks:
                body = body.strip()
                parts = re.split(r"(?m)^\s*A\s*[\.\)]?\s*", body, maxsplit=1)
                question = parts[0].strip().split("\n")[0][:300]
                answer = (parts[1] if len(parts) > 1 else
                          "\n".join(body.split("\n")[1:])).strip()
                if len(answer) < 20:
                    continue
                items.append(FAQItem(
                    question=question, answer=answer,
                    source=path.stem, item_id=str(num), url=str(path),
                ))
            break

    if not items:
        print(f"  ⚠ {path.name}: Q/A 패턴을 찾지 못했습니다. "
              "문단만 추출해 DAPT 코퍼스로 사용합니다.")
        for i, para in enumerate(p for p in re.split(r"\n{2,}", full)
                                 if len(p.strip()) >= 40):
            items.append(FAQItem(question="", answer=para.strip(),
                                 source=path.stem, item_id=f"p{i}", url=str(path)))

    stats.items += len(items)
    print(f"  {path.name}: {len(items)}건")
    return items


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def dedupe(items: list[FAQItem]) -> list[FAQItem]:
    seen: set[str] = set()
    out = []
    for it in items:
        k = it.key()
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def write_faq_jsonl(items: list[FAQItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def write_corpus_txt(items: list[FAQItem], path: Path, min_chars: int = 30) -> int:
    """DAPT용 평문. 한 줄 = 한 문단.

    dapt.py 의 read_corpus() 가 그대로 읽는 형식이다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for it in items:
        if it.question:
            lines.append(it.question.replace("\n", " ").strip())
        for para in re.split(r"\n{1,}", it.answer):
            para = para.strip()
            if len(para) >= min_chars:
                lines.append(para)
    lines = [l for l in lines if len(l) >= min_chars]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


SITES = {
    "socialservice": collect_socialservice,
    "sjss": collect_sjss,
    "ableservice": collect_ableservice,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="P1-a 도메인 코퍼스 수집")
    ap.add_argument("--site", choices=list(SITES), action="append",
                    help="수집할 사이트 (반복 지정 가능). 미지정+--all 이면 전체")
    ap.add_argument("--all", action="store_true", help="세 사이트 모두")
    ap.add_argument("--pdf", type=Path, action="append", help="PDF 경로 (반복 가능)")
    ap.add_argument("--out-faq", type=Path, default=Path("data/faq"))
    ap.add_argument("--out-corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    ap.add_argument("--cache-dir", type=Path, default=Path(".cache/http"))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    targets = list(SITES) if args.all else (args.site or [])
    if not targets and not args.pdf:
        ap.error("--all, --site, --pdf 중 하나는 지정해야 합니다.")

    fetcher = Fetcher(None if args.no_cache else args.cache_dir, delay=args.delay)
    stats = Stats()
    all_items: list[FAQItem] = []

    for name in targets:
        print(f"\n[{name}] 수집 시작")
        try:
            items = dedupe(SITES[name](fetcher, args.max_pages, stats))
        except Exception as exc:  # noqa: BLE001 — 한 사이트 실패가 전체를 막지 않는다
            print(f"  ✘ 실패: {exc}")
            stats.errors.append(f"{name}: {exc}")
            continue
        write_faq_jsonl(items, args.out_faq / f"{name}.jsonl")
        n = write_corpus_txt(items, args.out_corpus / f"{name}.txt")
        print(f"  -> {len(items)}건 / 코퍼스 {n}문단")
        all_items.extend(items)

    for pdf in args.pdf or []:
        if not pdf.exists():
            print(f"\n[pdf] ✘ 파일 없음: {pdf}")
            stats.errors.append(f"pdf: 파일 없음 {pdf}")
            continue
        print(f"\n[pdf] {pdf.name}")
        items = dedupe(collect_pdf(pdf, stats))
        slug = re.sub(r"[^0-9A-Za-z가-힣]+", "_", pdf.stem)[:40]
        write_faq_jsonl(items, args.out_faq / f"{slug}.jsonl")
        n = write_corpus_txt(items, args.out_corpus / f"{slug}.txt")
        print(f"  -> {len(items)}건 / 코퍼스 {n}문단")
        all_items.extend(items)

    all_items = dedupe(all_items)
    if all_items:
        write_faq_jsonl(all_items, args.out_faq / "_all.jsonl")

    print("\n" + "=" * 60)
    print(f"요청 {stats.fetched}회 / 수집 {len(all_items)}건 (중복 제거 후)")
    print(f"  FAQ    : {args.out_faq}/*.jsonl   -> P2/P3 시나리오 근거")
    print(f"  코퍼스 : {args.out_corpus}/*.txt  -> P4-a DAPT")
    if stats.errors:
        print(f"\n오류 {len(stats.errors)}건:")
        for e in stats.errors[:10]:
            print("  -", e)
    if all_items:
        print(f"\n다음 단계: python -m callwrap.dapt --corpus {args.out_corpus} "
              "--out runs/dapt-roberta-large")


if __name__ == "__main__":
    main()
