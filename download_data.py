import argparse
import concurrent.futures as futures
import hashlib
import html
import io
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")

WIKI_USER_AGENT = (
    "ClyxDatasetBuilder/3.0 "
    "(https://www.mediawiki.org/wiki/API:Etiquette; local educational dataset build) "
    f"Python/{sys.version_info.major}.{sys.version_info.minor}"
)
DEFAULT_BATCH_SIZE = 50
DEFAULT_WORKERS = 8

AD_PATTERNS = [
    re.compile(r"(?i)\b(click here|subscribe now|buy now|limited offer|sponsored by)\b"),
    re.compile(r"(?i)\b(подпишитесь|реклама|купить сейчас|перейти по ссылке)\b"),
    re.compile(r"(?i)https?://\S+\.(jpg|jpeg|png|gif|webp|svg)\b"),
    re.compile(r"(?i)\[?(ad|ads|advertisement)\]?"),
]

BOILERPLATE_HEADINGS = {
    "см. также",
    "примечания",
    "литература",
    "ссылки",
    "источники",
    "see also",
    "references",
    "external links",
    "further reading",
    "bibliography",
    "notes",
}

NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*\[?\d+\]?\s*$"),
    re.compile(r"^\s*(isbn|issn|doi|pmid|pmc)\s*[:\d]", re.IGNORECASE),
    re.compile(r"^\s*(archived from|retrieved|accessed|дата обращения)\b", re.IGNORECASE),
    re.compile(r"^\s*(edit|править|source|источник)\s*$", re.IGNORECASE),
    re.compile(r"^[\W_]{3,}$", re.UNICODE),
]

CATEGORIES_RU = [
    "Категория:Программирование",
    "Категория:Языки_программирования",
    "Категория:Алгоритмы",
    "Категория:Компьютерные_сети",
    "Категория:Информатика",
    "Категория:Программное_обеспечение",
    "Категория:Операционные_системы",
    "Категория:Базы_данных",
    "Категория:Искусственный_интеллект",
]
CATEGORIES_EN = [
    "Category:Programming_languages",
    "Category:Algorithms",
    "Category:Computer_networking",
    "Category:Computer_science",
    "Category:Software",
    "Category:Operating_systems",
    "Category:Databases",
    "Category:Artificial_intelligence",
    "Category:Machine_learning",
    "Category:Software_engineering",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Builds a clean IT pretraining text dataset in Clyx/data/raw/dataset.txt.")
    p.add_argument("--out", type=str, default=os.path.join(RAW_DIR, "dataset.txt"))
    p.add_argument("--max_mb", type=int, default=500)
    p.add_argument("--min_chars", type=int, default=700)
    p.add_argument("--timeout", type=float, default=45.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--state", type=str, default=os.path.join(RAW_DIR, "download_state.json"))
    p.add_argument("--seed_ru_docs", type=int, default=5000)
    p.add_argument("--seed_en_docs", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--quality", choices=["loose", "balanced", "strict"], default="balanced")
    p.add_argument("--flush_docs", type=int, default=32)
    p.add_argument("--wiki_sleep", type=float, default=0.0)
    p.add_argument("--no_rfc", action="store_true")
    p.add_argument("--no_shuffle", action="store_true")
    p.add_argument("--wiki_first", action="store_true", help="Download Wikipedia before RFC.")
    p.add_argument("--no_ssl_verify", action="store_true")
    return p.parse_args()


def setup_stdout() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(msg.encode(enc, errors="replace").decode(enc, errors="replace"), flush=True)


def make_session(verify_ssl: bool, workers: int = DEFAULT_WORKERS) -> requests.Session:
    session = requests.Session()
    session.verify = verify_ssl
    session.headers.update(
        {
            "User-Agent": WIKI_USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate",
        }
    )
    retry_kwargs = dict(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
    )
    try:
        retry = Retry(allowed_methods=frozenset(["GET", "POST"]), **retry_kwargs)
    except TypeError:
        retry = Retry(method_whitelist=frozenset(["GET", "POST"]), **retry_kwargs)
    adapter = HTTPAdapter(pool_connections=max(4, workers * 2), pool_maxsize=max(4, workers * 2), max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session


def effective_title_cap(max_mb: int, seed: int) -> int:
    estimated = max(50, int(max_mb * 45))
    return min(seed, estimated)


def _is_ssl_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "certificate_verify_failed" in msg or "ssl" in msg


def request_json(
    session: requests.Session,
    url: str,
    timeout: float,
    *,
    params: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, str]] = None,
    tries: int = 4,
) -> Dict:
    last_exc: Optional[BaseException] = None
    for attempt in range(tries):
        try:
            if data is not None:
                response = session.post(url, data=data, timeout=(10.0, timeout))
            else:
                response = session.get(url, params=params, timeout=(10.0, timeout))
            if response.status_code == 429:
                wait = 2.0 + attempt * 2.0
                log(f"  rate limit, pause {wait:.0f}s")
                time.sleep(wait)
                continue
            if response.status_code == 403 and "user-agent" in response.text.lower():
                raise RuntimeError("Wikipedia 403: User-Agent was rejected.")
            response.raise_for_status()
            return response.json()
        except BaseException as exc:
            last_exc = exc
            if _is_ssl_error(exc):
                break
            if attempt + 1 < tries:
                time.sleep(0.4 + attempt * 0.8)
    raise RuntimeError(str(last_exc) if last_exc else "request_json failed")


def request_text(session: requests.Session, url: str, timeout: float, tries: int = 4) -> str:
    last_exc: Optional[BaseException] = None
    for attempt in range(tries):
        try:
            response = session.get(url, timeout=(10.0, timeout))
            if response.status_code == 429:
                time.sleep(2.0 + attempt * 2.0)
                continue
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except BaseException as exc:
            last_exc = exc
            if _is_ssl_error(exc):
                break
            if attempt + 1 < tries:
                time.sleep(0.4 + attempt * 0.8)
    raise RuntimeError(str(last_exc) if last_exc else "request_text failed")


def mw_api(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def normalize_unicode(text: str) -> str:
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\u00ad": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def repair_fragmented_lines(lines: Sequence[str]) -> List[str]:
    repaired: List[str] = []
    buffer: List[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        if len(buffer) >= 4:
            joined = "".join(buffer)
            if len(joined) >= 4:
                repaired.append(joined)
            else:
                repaired.extend(buffer)
        else:
            repaired.extend(buffer)
        buffer = []

    for raw in lines:
        line = raw.strip()
        if line and len(line) <= 2 and re.search(r"[A-Za-zА-Яа-яЁё0-9]", line):
            buffer.append(line)
            continue
        flush_buffer()
        repaired.append(raw)
    flush_buffer()
    return repaired


def is_noise_line(line: str) -> bool:
    normalized = line.strip().lower().strip("=:-. ")
    if not normalized:
        return False
    if normalized in BOILERPLATE_HEADINGS:
        return True
    if any(pattern.search(line) for pattern in NOISE_LINE_PATTERNS):
        return True
    if len(line) <= 2:
        return True
    visible = len(line.replace(" ", ""))
    letters = len(re.findall(r"[A-Za-zА-Яа-яЁё]", line))
    if visible >= 12 and letters == 0:
        return True
    if visible >= 20:
        punct = len(re.findall(r"[^A-Za-zА-Яа-яЁё0-9\s]", line))
        if punct / max(1, visible) > 0.55:
            return True
    return False


def dedupe_paragraphs(text: str) -> str:
    paragraphs = re.split(r"\n{2,}", text)
    output: List[str] = []
    seen: Set[str] = set()
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        key = re.sub(r"\s+", " ", para).lower()
        if len(key) > 80:
            h = stable_hash(key)
            if h in seen:
                continue
            seen.add(h)
        output.append(para)
    return "\n\n".join(output)


def strip_boilerplate_sections(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    skipping = False
    heading_re = re.compile(r"^(={2,})\s*(.*?)\s*\1$")
    for line in lines:
        match = heading_re.match(line.strip())
        if match:
            heading_title = match.group(2).lower().strip("=:-. ")
            if heading_title in BOILERPLATE_HEADINGS:
                skipping = True
            else:
                skipping = False
        if not skipping:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = strip_boilerplate_sections(text)
    text = normalize_unicode(text)
    text = re.sub(r"\{\\displaystyle.*?\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\displaystyle\s*[^ \n]+", " ", text)
    text = re.sub(r"(?<=[A-Za-zА-Яа-яЁё])-\n(?=[A-Za-zА-Яа-яЁё])", "", text)

    for pattern in AD_PATTERNS:
        text = pattern.sub(" ", text)

    allowed = r"[^\x09\x0A\x0D\x20-\x7E\u0400-\u04FF]"
    text = re.sub(allowed, " ", text)
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    lines = repair_fragmented_lines(text.split("\n"))
    cleaned_lines: List[str] = []
    previous_blank = False
    for raw in lines:
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line:
            if cleaned_lines and not previous_blank:
                cleaned_lines.append("")
                previous_blank = True
            continue
        if is_noise_line(line):
            continue
        cleaned_lines.append(line)
        previous_blank = False

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = dedupe_paragraphs(text)
    return text.strip()


def quality_thresholds(level: str) -> Tuple[float, int, int]:
    if level == "strict":
        return 0.42, 35, 220
    if level == "loose":
        return 0.25, 12, 80
    return 0.34, 22, 140


def is_good_document(doc: str, min_chars: int, quality: str) -> bool:
    if len(doc) < min_chars:
        return False

    # Filter out index/TOC pages with dotted/dashed lines (e.g. ........ 12)
    toc_lines = sum(1 for line in doc.splitlines() if re.search(r"\.{4,}\s*\d+", line))
    if toc_lines > 2:
        return False

    compact = re.sub(r"\s+", "", doc)
    if len(compact) < min_chars * 0.65:
        return False

    alpha_ratio_min, min_avg_line, min_unique_words = quality_thresholds(quality)
    letters = len(re.findall(r"[A-Za-zА-Яа-яЁё]", doc))
    visible = len(compact)
    if letters / max(1, visible) < alpha_ratio_min:
        return False

    lines = [line for line in doc.splitlines() if line.strip()]
    if lines:
        avg_line = sum(len(line) for line in lines) / len(lines)
        very_short = sum(1 for line in lines if len(line.strip()) <= 3)
        if avg_line < min_avg_line:
            return False
        if very_short / max(1, len(lines)) > 0.12:
            return False

    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9_+-]{2,}", doc.lower())
    if len(set(words)) < min_unique_words and len(doc) > 5000:
        return False
    return True


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def atomic_replace(src: str, dst: str) -> None:
    if os.path.exists(dst):
        backup = dst + ".bak"
        try:
            if os.path.exists(backup):
                os.remove(backup)
            os.replace(dst, backup)
        except OSError:
            pass
    os.replace(src, dst)


def iter_category_titles(session: requests.Session, lang: str, category: str, limit: int, timeout: float) -> Iterator[str]:
    if limit <= 0:
        return
    got = 0
    cont: Optional[str] = None
    while got < limit:
        params: Dict[str, str] = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "page",
            "cmlimit": str(min(500, limit - got)),
        }
        if cont:
            params["cmcontinue"] = cont
        data = request_json(session, mw_api(lang), timeout, params=params)
        members = data.get("query", {}).get("categorymembers", [])
        for item in members:
            title = item.get("title")
            if title:
                got += 1
                yield title
                if got >= limit:
                    return
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont or not members:
            return


def mw_extract_batch(session: requests.Session, lang: str, titles: Sequence[str], timeout: float) -> List[str]:
    if not titles:
        return []
    payload = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "wiki",
        "redirects": "1",
        "titles": "|".join(titles),
    }
    data = request_json(session, mw_api(lang), timeout, data=payload)
    pages = data.get("query", {}).get("pages") or []
    output: List[str] = []
    if isinstance(pages, list):
        for page in pages:
            if isinstance(page, dict):
                extract = page.get("extract")
                if isinstance(extract, str) and extract.strip():
                    output.append(extract)
    elif isinstance(pages, dict):
        for page in pages.values():
            extract = page.get("extract")
            if isinstance(extract, str) and extract.strip():
                output.append(extract)
    return output


def mw_extract_adaptive(session: requests.Session, lang: str, titles: Sequence[str], timeout: float) -> List[str]:
    if not titles:
        return []
    try:
        return mw_extract_batch(session, lang, titles, timeout)
    except Exception as exc:
        if len(titles) <= 1:
            if titles:
                log(f"  skip wiki article '{titles[0][:60]}': {exc}")
            return []
        mid = len(titles) // 2
        return mw_extract_adaptive(session, lang, titles[:mid], timeout) + mw_extract_adaptive(
            session, lang, titles[mid:], timeout
        )


def load_state(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj
    except OSError:
        pass
    return {}


def save_state(path: str, state: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, path)


def write_doc(out_f: io.TextIOWrapper, doc: str) -> int:
    out_f.write(doc)
    out_f.write("\n\n")
    return len((doc + "\n\n").encode("utf-8", errors="ignore"))


def parse_rfc_numbers(index_text: str) -> List[int]:
    nums = {int(m.group(1)) for m in re.finditer(r"\bRFC\s*(\d{1,5})\b", index_text, flags=re.IGNORECASE)}
    if not nums:
        nums = {int(m.group(1)) for m in re.finditer(r"\brfc(\d{1,5})\b", index_text, flags=re.IGNORECASE)}
    return sorted(n for n in nums if n >= 1)


def normalize_rfc_text(text: str) -> str:
    text = normalize_unicode(text)
    raw_pages = text.split("\x0c")
    cleaned_pages = []
    months_re = re.compile(
        r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}$",
        re.IGNORECASE,
    )
    for page in raw_pages:
        lines = [line.rstrip() for line in page.splitlines()]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            continue

        # Strip footer (e.g. [Page 1] or [Page iv])
        if re.search(r"\[Page\s+[a-z0-9]+\]\s*$", lines[-1].strip(), re.IGNORECASE):
            lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            continue

        # Strip headers (up to 2 lines)
        for _ in range(2):
            if lines:
                fs = lines[0].strip()
                if (
                    months_re.match(fs)
                    or re.match(r"^rfc\s+\d+$", fs, re.IGNORECASE)
                    or fs.lower() == "internet protocol"
                    or (
                        re.match(r"^[A-Za-z\s]+(?:,\s*[A-Za-z\s]+)*$", fs)
                        and len(fs) < 40
                        and any(w in fs.lower() for w in ["postel", "editor", "darpa", "specification"])
                    )
                ):
                    lines.pop(0)

        while lines and not lines[0].strip():
            lines.pop(0)
        if lines:
            cleaned_pages.append("\n".join(lines))

    cleaned_text = "\n\n".join(cleaned_pages)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def chunked(seq: Sequence[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def build_dataset(
    out_path: str,
    state_path: str,
    max_bytes: int,
    min_chars: int,
    timeout: float,
    resume: bool,
    seed_ru_docs: int,
    seed_en_docs: int,
    batch_size: int,
    workers: int,
    quality: str,
    flush_docs: int,
    wiki_sleep: float,
    rfc_fill: bool,
    do_shuffle: bool,
    wiki_first: bool,
    verify_ssl: bool,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    main_session = make_session(verify_ssl, workers)
    thread_local = threading.local()
    workers = max(1, workers)
    batch_size = max(1, min(50, batch_size))
    flush_docs = max(1, flush_docs)

    def worker_session() -> requests.Session:
        session = getattr(thread_local, "session", None)
        if session is None:
            session = make_session(verify_ssl, workers)
            thread_local.session = session
        return session

    if not verify_ssl:
        log("[!] SSL verification is OFF (--no_ssl_verify)")

    state = load_state(state_path) if resume else {}
    seen_hashes: Set[str] = set(state.get("seen_hashes", [])) if isinstance(state.get("seen_hashes"), list) else set()
    wiki_ru_done = set(state.get("wiki_ru_done", [])) if isinstance(state.get("wiki_ru_done"), list) else set()
    wiki_en_done = set(state.get("wiki_en_done", [])) if isinstance(state.get("wiki_en_done"), list) else set()
    rfc_done = set(state.get("rfc_done", [])) if isinstance(state.get("rfc_done"), list) else set()

    out_tmp = out_path if resume else out_path + ".new"
    if resume:
        written = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        log(f"[resume] already written: {written / (1024 * 1024):.2f} MB")
    else:
        written = 0
        for path in (out_tmp, out_path + ".new"):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    docs_written = int(state.get("docs_written", 0)) if resume else 0
    docs_skipped = int(state.get("docs_skipped", 0)) if resume else 0
    last_log_mb = -1
    started_at = time.time()

    max_mb = max_bytes // (1024 * 1024)
    ru_cap = effective_title_cap(max_mb, seed_ru_docs)
    en_cap = effective_title_cap(max_mb, seed_en_docs)

    log(f"Target: {max_bytes / (1024 * 1024):.2f} MB -> {out_path}")
    log(f"Quality: {quality}, min_chars={min_chars}, workers={workers}, wiki_batch={batch_size}")
    log(f"Article title cap: RU<={ru_cap}, EN<={en_cap}")

    def checkpoint() -> None:
        save_state(
            state_path,
            {
                "written_bytes": written,
                "docs_written": docs_written,
                "docs_skipped": docs_skipped,
                "seen_hashes": list(seen_hashes)[-250000:],
                "wiki_ru_done": list(wiki_ru_done),
                "wiki_en_done": list(wiki_en_done),
                "rfc_done": list(rfc_done),
                "updated_at": time.time(),
            },
        )

    def progress_log(force: bool = False) -> None:
        nonlocal last_log_mb
        mb = written / (1024 * 1024)
        whole = int(mb)
        if force or whole > last_log_mb:
            last_log_mb = whole
            elapsed = max(0.001, time.time() - started_at)
            speed = mb / elapsed
            log(f"  -> {mb:.2f}/{max_mb:.0f} MB, docs={docs_written}, skipped={docs_skipped}, avg={speed:.2f} MB/s")

    def maybe_write(out_f: io.TextIOWrapper, doc: str) -> bool:
        nonlocal written, docs_written, docs_skipped
        if written >= max_bytes:
            return False
        doc = clean_text(doc)
        if not is_good_document(doc, min_chars, quality):
            docs_skipped += 1
            return True
        doc_key = re.sub(r"\s+", " ", doc).strip().lower()
        h = stable_hash(doc_key)
        if h in seen_hashes:
            docs_skipped += 1
            return True

        doc_with_markers = doc
        if not doc_with_markers.endswith("<|endtext|><STOP>"):
            doc_with_markers = doc_with_markers.rstrip() + "<|endtext|><STOP>"

        payload = (doc_with_markers + "\n\n").encode("utf-8", errors="ignore")
        if written + len(payload) > max_bytes:
            # Prevent truncating the document; discard it and stop download
            return False

        written += write_doc(out_f, doc_with_markers)
        seen_hashes.add(h)
        docs_written += 1
        if docs_written % flush_docs == 0:
            out_f.flush()
            checkpoint()
        progress_log()
        return written < max_bytes

    def fetch_rfc(num: int) -> Tuple[int, str]:
        text = request_text(worker_session(), f"https://www.rfc-editor.org/rfc/rfc{num}.txt", timeout=timeout)
        return num, normalize_rfc_text(text)

    def run_rfc(out_f: io.TextIOWrapper) -> None:
        if not rfc_fill or written >= max_bytes:
            return
        log("[RFC] downloading protocol/network documents...")
        try:
            index_text = request_text(main_session, "https://www.rfc-editor.org/rfc/rfc-index.txt", timeout=timeout)
            rfc_nums = parse_rfc_numbers(index_text)
            log(f"  RFC index: {len(rfc_nums)} documents")
        except Exception as exc:
            log(f"  RFC index unavailable: {exc}")
            return

        popular = [
            793, 791, 1034, 1035, 1122, 1812, 1918, 2049, 2131, 2616, 2828, 3261,
            3330, 3986, 4251, 5321, 7230, 7231, 7540, 8259, 8446, 9110, 9112, 9592,
        ]
        if do_shuffle:
            random.shuffle(rfc_nums)
        ordered: List[int] = []
        seen_num: Set[int] = set()
        for n in popular + rfc_nums:
            if n not in seen_num:
                seen_num.add(n)
                ordered.append(n)

        pending = [n for n in ordered if str(n) not in rfc_done]
        window = max(4, workers * 3)
        submitted = 0
        completed = 0
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            active: Dict[futures.Future, int] = {}

            def submit_more() -> None:
                nonlocal submitted
                while len(active) < window and submitted < len(pending) and written < max_bytes:
                    n = pending[submitted]
                    submitted += 1
                    active[pool.submit(fetch_rfc, n)] = n

            submit_more()
            while active and written < max_bytes:
                done, _ = futures.wait(active, return_when=futures.FIRST_COMPLETED)
                for future in done:
                    n = active.pop(future)
                    completed += 1
                    try:
                        rfc_num, text = future.result()
                    except Exception:
                        rfc_done.add(str(n))
                        continue
                    rfc_done.add(str(rfc_num))
                    if text and not maybe_write(out_f, text):
                        return
                    if completed % 40 == 0:
                        checkpoint()
                        log(f"  RFC fetched: {completed}/{len(pending)}")
                submit_more()

    def collect_wiki_titles(lang: str, categories: Sequence[str], cap: int, done_set: Set[str]) -> List[str]:
        collected: List[str] = []
        per_cat = max(20, (cap + len(categories) - 1) // len(categories))
        for category in categories:
            if len(collected) >= cap or written >= max_bytes:
                break
            need = min(per_cat, cap - len(collected))
            log(f"  category: {category}")
            try:
                for title in iter_category_titles(main_session, lang, category, need, timeout):
                    if title not in done_set and title not in collected:
                        collected.append(title)
                    if len(collected) >= cap:
                        break
            except Exception as exc:
                log(f"  title list error: {exc}")
            log(f"  collected titles: {len(collected)}")
        if do_shuffle:
            random.shuffle(collected)
        return [title for title in collected if title not in done_set]

    def run_wiki_lang(
        out_f: io.TextIOWrapper,
        lang: str,
        categories: Sequence[str],
        cap: int,
        done_set: Set[str],
        label: str,
    ) -> None:
        if written >= max_bytes or cap <= 0:
            return
        log(label)
        pending = collect_wiki_titles(lang, categories, cap, done_set)
        batches = list(chunked(pending, batch_size))
        if not batches:
            return
        wiki_workers = max(1, min(workers, 4))
        log(f"  wiki batches: {len(batches)}, workers={wiki_workers}")

        def fetch_batch(batch: Sequence[str]) -> Tuple[List[str], List[str]]:
            return list(batch), mw_extract_adaptive(worker_session(), lang, batch, timeout)

        if wiki_workers == 1:
            iterator = ((batch, mw_extract_adaptive(main_session, lang, batch, timeout)) for batch in batches)
            for idx, (batch, texts) in enumerate(iterator, 1):
                if written >= max_bytes:
                    break
                done_set.update(batch)
                for text in texts:
                    if not maybe_write(out_f, text):
                        return
                if idx % 5 == 0 or idx == len(batches):
                    log(f"  batch {idx}/{len(batches)}, accepted_docs={docs_written}")
                    checkpoint()
                if wiki_sleep > 0:
                    time.sleep(wiki_sleep)
            return

        submitted = 0
        completed = 0
        window = max(wiki_workers, wiki_workers * 3)
        with futures.ThreadPoolExecutor(max_workers=wiki_workers) as pool:
            active: Dict[futures.Future, List[str]] = {}

            def submit_more() -> None:
                nonlocal submitted
                while len(active) < window and submitted < len(batches) and written < max_bytes:
                    batch = batches[submitted]
                    submitted += 1
                    active[pool.submit(fetch_batch, batch)] = batch

            submit_more()
            while active and written < max_bytes:
                done, _ = futures.wait(active, return_when=futures.FIRST_COMPLETED)
                for future in done:
                    completed += 1
                    batch = active.pop(future)
                    if written >= max_bytes:
                        break
                    try:
                        done_titles, texts = future.result()
                    except Exception as exc:
                        done_set.update(batch)
                        log(f"  wiki batch failed: {exc}")
                        continue
                    done_set.update(done_titles)
                    for text in texts:
                        if not maybe_write(out_f, text):
                            return
                    if completed % 5 == 0 or completed == len(batches):
                        log(f"  batch {completed}/{len(batches)}, accepted_docs={docs_written}")
                        checkpoint()
                    if wiki_sleep > 0:
                        time.sleep(wiki_sleep)
                submit_more()

    with open(out_tmp, "a" if resume else "w", encoding="utf-8", newline="\n") as out_f:
        if rfc_fill and not wiki_first:
            run_rfc(out_f)
        if written < max_bytes:
            run_wiki_lang(out_f, "ru", CATEGORIES_RU, ru_cap, wiki_ru_done, "[Wikipedia RU]")
        if written < max_bytes:
            run_wiki_lang(out_f, "en", CATEGORIES_EN, en_cap, wiki_en_done, "[Wikipedia EN]")
        if rfc_fill and written < max_bytes and wiki_first:
            run_rfc(out_f)
        out_f.flush()
        checkpoint()

    if not resume:
        atomic_replace(out_tmp, out_path)

    final_mb = os.path.getsize(out_path) / (1024 * 1024) if os.path.exists(out_path) else 0
    progress_log(force=True)
    log(f"Done: {out_path} - {final_mb:.2f} MB, docs={docs_written}, skipped={docs_skipped}")
    if docs_written == 0:
        log("Nothing was downloaded. Check network, SSL, or use --no_ssl_verify if your system certificates are broken.")


def main() -> None:
    setup_stdout()
    args = parse_args()
    verify_ssl = not bool(args.no_ssl_verify)
    if os.environ.get("DATASET_INSECURE", "").strip().lower() in {"1", "true", "yes"}:
        verify_ssl = False
    build_dataset(
        out_path=args.out,
        state_path=args.state,
        max_bytes=int(args.max_mb) * 1024 * 1024,
        min_chars=int(args.min_chars),
        timeout=float(args.timeout),
        resume=bool(args.resume),
        seed_ru_docs=int(args.seed_ru_docs),
        seed_en_docs=int(args.seed_en_docs),
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        quality=str(args.quality),
        flush_docs=int(args.flush_docs),
        wiki_sleep=float(args.wiki_sleep),
        rfc_fill=not bool(args.no_rfc),
        do_shuffle=not bool(args.no_shuffle),
        wiki_first=bool(args.wiki_first),
        verify_ssl=verify_ssl,
    )


if __name__ == "__main__":
    main()
