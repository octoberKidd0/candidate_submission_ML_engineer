#!/usr/bin/env python3
"""Постобработка OCR паспорта РФ: принять (и выдать поля) или отклонить.

Запуск:  python postprocess.py data/test_predictions.jsonl answers.jsonl

Идея: каждое поле принимаем только если его подтверждает независимый источник.
  * ФИО, пол, дата рождения, дата выдачи, код подразделения, серия/номер —
    сверяются с MRZ (транслитерация + контрольные цифры). Если MRZ валидна по всем
    контрольным цифрам, а визуальная зона ошиблась в цифровом поле — берём значение из MRZ.
  * Место рождения и «кем выдан» подтвердить нечем: проверяем число строк по высоте
    рамки (детектор часто теряет строки), словарь токенов и уверенность по символам.
  * Паспорт должен быть действителен на 01.10.2026 (строго, без 90-дневного льготного срока).
Любое сомнение -> reject.
"""
import csv
import datetime as dt
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DATE = dt.date(2026, 10, 1)

FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]

# Пороги (подбираются на train, см. tune в отчёте)
CFG = {
    "place_min_conf": 0.97,    # минимальная уверенность по любому символу места рождения
    "org_min_conf": 0.97,      # то же для «кем выдан»
    "viz_min_conf": 0.0,       # для полей, подтверждённых MRZ, порог не нужен
    "require_vocab_org": True,
    "require_vocab_place": False,
    "place_strict_end": True,   # место рождения: сокращения ОБЛ/КР/РЕСП/ОКР в конце — только с точкой
}

# Транслитерация MRZ внутреннего паспорта РФ (выведена по train, Ё в данных нет)
TR = {"А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ж": "J", "З": "Z",
      "И": "I", "Й": "Q", "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P",
      "Р": "R", "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "3",
      "Ш": "4", "Щ": "W", "Ъ": "X", "Ы": "Y", "Ь": "9", "Э": "6", "Ю": "7", "Я": "8"}
TR["-"] = "<"
TR[" "] = "<"


def norm(s):
    return " ".join(str(s or "").split()).upper()


def translit(s):
    out = []
    for c in s:
        if c not in TR:
            return None
        out.append(TR[c])
    return "".join(out)


# ---------------------------------------------------------------- MRZ

_W = (7, 3, 1)


def _val(c):
    if c.isdigit():
        return int(c)
    if c == "<":
        return 0
    if "A" <= c <= "Z":
        return ord(c) - 55
    return None


def check_digit(s):
    t = 0
    for i, c in enumerate(s):
        v = _val(c)
        if v is None:
            return None
        t += v * _W[i % 3]
    return str(t % 10)


def parse_mrz(mrz):
    """-> dict или None. Требуем ровно две строки по 44 символа."""
    if not mrz or not mrz.get("lines") or len(mrz["lines"]) != 2:
        return None
    a, b = mrz["lines"][0]["text"], mrz["lines"][1]["text"]
    if len(a) != 44 or len(b) != 44:
        return None
    ok_num = check_digit(b[0:9]) == b[9]
    ok_birth = check_digit(b[13:19]) == b[19]
    ok_issue = check_digit(b[28:42]) == b[42]
    ok_final = check_digit(b[0:10] + b[13:20] + b[21:43]) == b[43]
    return {
        "l1": a, "l2": b,
        "ok_num": ok_num, "ok_birth": ok_birth, "ok_issue": ok_issue, "ok_final": ok_final,
        "all_ok": ok_num and ok_birth and ok_issue and ok_final and b[10:13] == "RUS",
    }


def _date(dd, mm, yyyy):
    try:
        return dt.date(int(yyyy), int(mm), int(dd))
    except (ValueError, TypeError):
        return None


def parse_date(s):
    m = re.fullmatch(r"(\d\d)\.(\d\d)\.(\d{4})", norm(s))
    return _date(*m.groups()) if m else None


def fmt_date(d):
    return d.strftime("%d.%m.%Y")


def add_years(d, y):
    try:
        return d.replace(year=d.year + y)
    except ValueError:
        return d.replace(year=d.year + y, day=28)


def passport_valid(birth, issue):
    """Строго: до 20-летия (если выдан 14-19), до 45-летия (если выдан 20-44), далее бессрочно."""
    if issue > REF_DATE or issue < add_years(birth, 14):
        return False
    if issue >= add_years(birth, 45):
        return True
    expiry = add_years(birth, 45) if issue >= add_years(birth, 20) else add_years(birth, 20)
    return REF_DATE < expiry


# ---------------------------------------------------------------- VIZ helpers

def viz_text(viz, f):
    v = viz.get(f)
    if not v or not v.get("lines"):
        return None
    return " ".join(l["text"] for l in v["lines"]).strip()


def viz_min_conf(viz, f):
    v = viz.get(f)
    if not v or not v.get("lines"):
        return 0.0
    cs = [c for l in v["lines"] for c in l["char_conf"]]
    return min(cs) if cs else 0.0


def expected_lines(h):
    """По высоте рамки поля: 1 строка < 42px, 2 строки < 73px, иначе 3 (подобрано по train)."""
    return 1 if h < 42 else 2 if h < 73 else 3


def lines_complete(viz, f):
    v = viz.get(f)
    return bool(v and v.get("lines") and len(v["lines"]) == expected_lines(v["box"][3]))


TOKEN = re.compile(r"[А-ЯЁ]+|\d+")


def tokens(s):
    return TOKEN.findall(norm(s))


def load_vocab():
    voc = set()
    path = os.path.join(HERE, "reference", "subdivisions.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                voc.update(tokens(r["name"]))
    vp = os.path.join(HERE, "reference", "vocab.json")  # строится build_vocab.py по train-разметке
    org, place = set(), set()
    if os.path.exists(vp):
        with open(vp, encoding="utf-8") as fh:
            d = json.load(fh)
        org, place = set(d["org"]), set(d["place"])
    return voc | org, voc | place


# ---------------------------------------------------------------- решение

def decide(doc, vocab_org, vocab_place, cfg=CFG, why=None):
    """-> (decision, fields). why — список для причин отказа (для отладки)."""
    def rej(reason):
        if why is not None:
            why.append(reason)
        return "reject", {}

    viz = doc["viz"]
    txt = {f: viz_text(viz, f) for f in FIELDS}
    if any("Ё" in (t or "") for t in txt.values()):
        return rej("yo")

    m = parse_mrz(doc.get("mrz"))
    if m is None:
        return rej("mrz_missing")
    l1, l2 = m["l1"], m["l2"]
    out = {}

    # --- ФИО: целиком сверяем первую строку MRZ
    names = [txt["Last_name_ru"], txt["First_name_ru"], txt["Middle_name_ru"]]
    if any(n is None for n in names):
        return rej("name_null")
    parts = [translit(n) for n in names]
    if any(p is None for p in parts):
        return rej("name_charset")
    raw = "PNRUS" + parts[0] + "<<" + parts[1] + "<" + parts[2]
    if len(raw) > 44:
        return rej("name_too_long")
    if (raw + "<" * 44)[:44] != l1:
        return rej("name_mrz_mismatch")
    for f, n in zip(FIELDS[:3], names):
        out[f] = norm(n)

    # --- Пол
    sex_mrz = {"M": "МУЖ.", "F": "ЖЕН."}.get(l2[20])
    if sex_mrz is None or not m["ok_final"]:
        return rej("sex")
    if txt["Sex_ru"] is not None and norm(txt["Sex_ru"]) != sex_mrz:
        return rej("sex_mismatch")
    out["Sex_ru"] = sex_mrz

    # --- Даты, код, номер: из MRZ, если она валидна по всем контрольным цифрам
    if not m["all_ok"]:
        return rej("mrz_checks")

    # дата выдачи (век: 97..99 -> 19xx, иначе 20xx)
    yy = int(l2[29:31]) if l2[29:35].isdigit() else None
    if yy is None:
        return rej("issue_digits")
    issue = _date(l2[33:35], l2[31:33], (1900 if yy >= 97 else 2000) + yy)
    if issue is None:
        return rej("issue_date")
    # дата рождения: век определяем по выдаче (>=14 лет) и по визуальной зоне
    if not l2[13:19].isdigit():
        return rej("birth_digits")
    by = int(l2[13:15])
    cands = []
    for cen in (1900, 2000):
        d = _date(l2[17:19], l2[15:17], cen + by)
        if d and add_years(d, 14) <= issue and add_years(d, 110) > REF_DATE:
            cands.append(d)
    vb = parse_date(txt["Birth_date"]) if txt["Birth_date"] else None
    if len(cands) == 1:
        birth = cands[0]
    elif len(cands) == 2 and vb in cands:
        birth = vb
    else:
        return rej("birth_century")
    if not passport_valid(birth, issue):
        return rej("invalid_passport")

    code = l2[35:38] + "-" + l2[38:41]
    if not (l2[35:41].isdigit() and l2[0:9].isdigit() and l2[28].isdigit()):
        return rej("digits")
    licence = f"{l2[0:2]} {l2[2]}{l2[28]} {l2[3:9]}"
    # серия: год бланка на 0..3 меньше года выдачи (закономерность по train)
    if (issue.year % 100 - int(l2[2] + l2[28])) % 100 > 3:
        return rej("series_year")

    out.update({"Birth_date": fmt_date(birth), "Issue_date": fmt_date(issue),
                "Issue_organisation_code": code, "Licence_number": licence})

    # --- Место рождения и орган выдачи: сверять не с чем
    for f, voc, key, need in (("Birth_place_ru", vocab_place, "place", cfg["require_vocab_place"]),
                              ("Issue_organization_ru", vocab_org, "org", cfg["require_vocab_org"])):
        t = txt[f]
        if not t:
            return rej(f + "_null")
        if not lines_complete(viz, f):
            return rej(f + "_lines")
        if viz_min_conf(viz, f) < cfg[key + "_min_conf"]:
            return rej(f + "_conf")
        if need and any(w not in voc for w in tokens(t)):
            return rej(f + "_vocab")
        if f == "Birth_place_ru" and cfg["place_strict_end"]:
            if re.search(r"(ОБЛ|КР|РЕСП|ОКР)$", norm(t)):
                return rej(f + "_end_dot")
        out[f] = norm(t)

    return "accept", out


def main(inp, outp):
    vocab_org, vocab_place = load_vocab()
    n_acc = 0
    with open(inp, encoding="utf-8") as fi, open(outp, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            doc = json.loads(line)
            try:
                dec, fields = decide(doc, vocab_org, vocab_place)
            except Exception:  # любая неожиданность -> отказ
                dec, fields = "reject", {}
            n_acc += dec == "accept"
            fo.write(json.dumps({"doc_id": doc["doc_id"], "decision": dec, "fields": fields},
                                ensure_ascii=False) + "\n")
    print(f"accepted {n_acc}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])