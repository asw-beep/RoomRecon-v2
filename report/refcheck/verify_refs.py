"""Look up candidate references on arXiv (title search) and print what arXiv records:
authors, first-version date, id, comment and journal_ref (venue when stated). Report aid only."""
import json, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
def look(title):
    q = urllib.parse.quote('ti:"%s"' % re.sub(r'[^\w\s-]', ' ', title).strip())
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=3"
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    out = []
    for e in root.findall("a:entry", NS):
        out.append({"title": " ".join(e.find("a:title", NS).text.split()),
                    "id": e.find("a:id", NS).text.rsplit("/", 1)[-1],
                    "published": e.find("a:published", NS).text[:10],
                    "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
                    "comment": (e.find("x:comment", NS).text if e.find("x:comment", NS) is not None else ""),
                    "journal_ref": (e.find("x:journal_ref", NS).text if e.find("x:journal_ref", NS) is not None else "")})
    return out
if __name__ == "__main__":
    titles = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    res = {}
    for t in titles:
        try:
            res[t] = look(t)
        except Exception as ex:
            res[t] = [{"error": str(ex)}]
        time.sleep(3)                                  # arXiv API etiquette
    json.dump(res, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for t, hits in res.items():
        h = hits[0] if hits else {}
        print(("OK  " if h.get("id") else "??  ") + t[:70], "|", h.get("id", ""), h.get("published", ""), "|", ", ".join(h.get("authors", [])[:3]), "|", (h.get("journal_ref") or h.get("comment") or "")[:60])
