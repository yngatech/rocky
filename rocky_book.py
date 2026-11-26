"""
The book Rocky comes from, for his think_back tool (rocky_harness): its text cut into passages, and the few that best
match what he wants to remember, ranked by BM25 over their words.

The book is copyrighted, like rocky_lines.txt: its text and the passages cut from it are gitignored, and never go
anywhere but Rocky's own requests. Without them he has no think_back.

    uv run --with-requirements requirements.txt python rocky_book.py   # rocky_book.txt to rocky_book.json
"""

import json
import math
import re
import sys
import threading
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
BOOK_PATH = HERE / "rocky_book.txt"      # the novel as text, a paragraph a line
INDEX_PATH = HERE / "rocky_book.json"    # its passages, built from it
PASSAGE_WORDS = 200  # a passage's length: whole paragraphs, to about this many words
OVERLAP = 1 / 3      # how much of a passage the next one starts with, so a scene cut between two is whole in one
PASSAGES = 3         # how many a look-up returns

# The extraction's headings: a chapter starts at one, and the book ends at the dedication
HEADING = re.compile(r"^(Chapter \S+|Dedication|Acknowledgments), ")
WORD = re.compile(r"[a-z0-9]+")
STOP = set("""a an and are as at be been but by can could did do does for from had has have he her him his how i if in
into is it its me my no not of on or our she so than that the their them then there these they this those to up us
was we were what when where which who whom why will with would you your""".split())


def available():
    return INDEX_PATH.exists()


# The story's passages: whole paragraphs, PASSAGE_WORDS long, each chapter on its own, overlapping by OVERLAP
def passages(text):
    chapters, chapter = [], []  # what comes before the first heading is kept only if there are none
    before = chapter
    for line in text.splitlines():
        line = line.strip()
        if (heading := HEADING.match(line)) and heading[1] in ("Dedication", "Acknowledgments"):
            break
        if heading:
            chapters.append(chapter := [])
        elif line and not line.startswith("#") and line != "—":
            chapter.append(line)
    chapters = chapters or [before]
    found = []
    for paras in chapters:
        start = 0
        while start < len(paras):
            end, words = start, 0
            while end < len(paras) and words < PASSAGE_WORDS:
                words += len(paras[end].split())
                end += 1
            found.append("\n\n".join(paras[start:end]))
            if end == len(paras):
                break
            back, words = end, 0
            while back > start + 1 and words < PASSAGE_WORDS * OVERLAP:
                back -= 1
                words += len(paras[back].split())
            start = back
    return found


def words(text):
    return [w for w in WORD.findall(text.lower().replace("’", "'")) if w not in STOP]


class Index:
    def __init__(self, found, k1=1.2, b=0.75):
        self.passages = found
        self.counts = [Counter(words(p)) for p in found]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.mean = sum(self.lengths) / max(len(found), 1)
        df = Counter(w for c in self.counts for w in c)
        self.idf = {w: math.log(1 + (len(found) - n + .5) / (n + .5)) for w, n in df.items()}
        self.k1, self.b = k1, b

    def search(self, what, k=PASSAGES):
        scores = [0.0] * len(self.passages)
        for w in set(words(what)) & self.idf.keys():
            for i, c in enumerate(self.counts):
                if tf := c.get(w):
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.mean)
                    scores[i] += self.idf[w] * tf * (self.k1 + 1) / (tf + norm)
        best = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])[:k]
        return [self.passages[i] for i in best]


index = None  # (which INDEX_PATH it was read from: its inode, size and modification time; its Index)
loading = threading.Lock()


# The passages that best match `what`, best first. The index is read from INDEX_PATH the first time, and again once
# it's rebuilt.
def search(what, k=PASSAGES):
    global index
    with loading:
        stamp = ((s := INDEX_PATH.stat()).st_ino, s.st_size, s.st_mtime_ns)
        if index is None or index[0] != stamp:
            found = json.loads(INDEX_PATH.read_text())["passages"]
            if not isinstance(found, list) or not all(isinstance(p, str) for p in found):
                raise ValueError("not a list of passages")
            index = stamp, Index(found)
    return index[1].search(what, k)


def build(book=BOOK_PATH, out=INDEX_PATH):
    found = passages(Path(book).read_text())
    tmp = Path(out).with_suffix(".tmp")
    tmp.write_text(json.dumps({"passages": found}, ensure_ascii=False))
    tmp.replace(out)
    return found


if __name__ == "__main__":
    found = build(*sys.argv[1:2])
    print(f"{len(found)} passages from {sys.argv[1] if len(sys.argv) > 1 else BOOK_PATH.name} in {INDEX_PATH.name}")
