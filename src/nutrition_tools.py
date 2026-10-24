import difflib
import numpy as np, pandas as pd

# ───────── 설정 (기준값은 임시: 대한당뇨병학회 등 근거 확인 후 교체) ─────────
LEVEL_CFG = {"net_carb_low": 15, "net_carb_high": 40, "sugar_bump": 15}   # g, 1회 기준
LEVELS = ["낮음", "보통", "높음"]
NOTICE = "혈당 부담 등급은 순탄수화물 기준의 임시 기준이에요. 혈당 수치를 예측하지 않고 부담 정도만 추정한 참고용이에요."

def load_master(path):
    m = pd.read_parquet(path)
    m["_key"] = m["ai_hub_name"].str.replace(r"\s+", "", regex=True)
    m["_key2"] = m["식품명"].astype(str).str.replace(r"\s+", "", regex=True)
    return m.set_index("food_id", drop=False)

def find_food(master, query, n=5):
    """이름으로 음식 찾기: 정확일치 → 포함 → 비슷한 이름 순. food_id 리스트 반환"""
    q = "".join(str(query).split())
    if not q:
        return []
    out = []
    for col in ("_key", "_key2"):
        out += master.index[master[col] == q].tolist()
    for col in ("_key", "_key2"):
        hit = master[master[col].str.contains(q, regex=False)]
        out += hit.loc[hit[col].str.len().sort_values().index].index.tolist()
    if len(out) < n:
        names = master["_key"].tolist()
        for nm in difflib.get_close_matches(q, names, n=n, cutoff=0.5):
            out += master.index[master["_key"] == nm].tolist()
    return list(dict.fromkeys(out))[:n]

def _num(v):
    return 0.0 if pd.isna(v) else float(v)

def nutrients(row, mult=1.0):
    """한 음식의 영양(기준량 × mult). 기준이 100g이면 100g당, 1인분이면 1인분당"""
    carb, fib, sug = _num(row["탄수화물(g)"]), _num(row["총 식이섬유(g)"]), _num(row["총당류(g)"])
    return {
        "에너지(㎉)": _num(row["에너지(㎉)"]) * mult,
        "탄수화물(g)": carb * mult,
        "식이섬유(g)": fib * mult,
        "당류(g)": sug * mult,
        "순탄수(g)": max(carb - fib, 0) * mult,
        "단백질(g)": _num(row["단백질(g)"]) * mult,
        "지방(g)": _num(row["지방(g)"]) * mult,
        "나트륨(㎎)": _num(row["나트륨(㎎)"]) * mult,
    }

def level_of(net_carb, sugar, cfg=LEVEL_CFG):
    """순탄수로 1차 등급, 당류가 많으면 한 단계 올림. (등급, 이유)"""
    lo, hi = cfg["net_carb_low"], cfg["net_carb_high"]
    i = 0 if net_carb < lo else (1 if net_carb < hi else 2)
    why = [f"순탄수 {net_carb:.0f}g"]
    if sugar >= cfg["sugar_bump"] and i < 2:
        i += 1
        why.append(f"당류 {sugar:.0f}g로 많아 한 단계 올림")
    return LEVELS[i], ", ".join(why)

# 보조 라벨: 식약처 1일 영양성분 기준치를 3끼로 나눈 자체 참고 기준 (공식 분류 아님)
SIDE_CFG = {"열량": ("에너지(㎉)", 500, 900, "kcal"), "당류": ("당류(g)", 10, 35, "g"),
            "나트륨": ("나트륨(㎎)", 400, 1000, "mg")}
SIDE_NOTICE = "열량·당류·나트륨 구간은 식약처 1일 기준치를 3끼로 나눈 자체 참고 기준이에요."

def side_labels(n):
    out = {}
    for name, (key, lo, hi, unit) in SIDE_CFG.items():
        v = n.get(key, 0)
        out[name] = (LEVELS[0 if v < lo else (1 if v < hi else 2)], v, unit)
    return out

def lookup(master, food_id, mult=1.0):
    r = master.loc[food_id]
    n = nutrients(r, mult)
    lv, why = level_of(n["순탄수(g)"], n["당류(g)"])
    basis = f'{r["기준"]} 기준' + (f' ({r["1회제공량"]:g}g)' if r["기준"] == "1인분" else "")
    return {
        "food_id": food_id, "음식": r["ai_hub_name"], "식품명": r["식품명"], "기준": basis, "배수": mult,
        "영양": n, "혈당부담": lv, "이유": why,
        "보조": side_labels(n), "근사": bool(r["근사표시"]),
        "안내": f'가장 가까운 음식({r["식품명"]}) 기준이에요.' if r["근사표시"] else "",
    }

def lookup_text(master, query, mult=1.0):
    ids = find_food(master, query)
    if not ids:
        return f"'{query}'은(는) 데이터에 없는 음식이에요."
    d = lookup(master, ids[0], mult)
    n = d["영양"]
    s = (f'{d["음식"]} ({d["기준"]}{"" if mult == 1 else f", ×{mult:g}"})\n'
         f'열량 {n["에너지(㎉)"]:.0f}kcal · 탄수화물 {n["탄수화물(g)"]:.1f}g · 식이섬유 {n["식이섬유(g)"]:.1f}g · 당류 {n["당류(g)"]:.1f}g\n'
         f'혈당 부담: {d["혈당부담"]} ({d["이유"]})')
    s += "\n참고: " + " · ".join(f"{k} {v[0]}({v[1]:.0f}{v[2]})" for k, v in d["보조"].items())
    if d["안내"]: s += "\nⓘ " + d["안내"]
    if len(ids) > 1: s += "\n비슷한 이름: " + ", ".join(master.loc[i, "ai_hub_name"] for i in ids[1:4])
    return s + "\n(" + NOTICE + " " + SIDE_NOTICE + ")"

def meal_sum(master, items):
    """items: [(food_id, 배수), ...] → 합계와 등급. 일부가 100g 기준이면 안내"""
    tot, rows, notes = {}, [], []
    for fid, mult in items:
        d = lookup(master, fid, mult)
        rows.append(d)
        for k, v in d["영양"].items():
            tot[k] = tot.get(k, 0.0) + v
        if d["근사"]: notes.append(f'{d["음식"]}: {d["안내"]}')
        if d["기준"].startswith("100g"): notes.append(f'{d["음식"]}: 100g 기준 값이라 먹은 양에 맞게 배수를 조정하세요.')
    lv, why = level_of(tot.get("순탄수(g)", 0), tot.get("당류(g)", 0))
    return {"항목": rows, "합계": tot, "혈당부담": lv, "이유": why, "보조": side_labels(tot), "참고": notes}

class DailyLog:
    """하루 식사 기록: add(끼니, food_id, 배수) → summary()"""
    MEALS = ["아침", "점심", "저녁", "간식"]
    def __init__(self, master): self.m, self.items = master, []
    def add(self, meal, food_id, mult=1.0):
        assert meal in self.MEALS, f"끼니는 {self.MEALS} 중 하나예요"
        self.items.append((meal, food_id, mult))
    def remove(self, idx): self.items.pop(idx)
    def summary(self):
        out, allitems = {}, []
        for meal in self.MEALS:
            its = [(f, x) for ml, f, x in self.items if ml == meal]
            if its:
                out[meal] = meal_sum(self.m, its); allitems += its
        day = meal_sum(self.m, allitems) if allitems else None
        return {"끼니별": out, "하루합계": day}


# ═════════ 3번 기능: 조건에 맞는 메뉴 / 대체 메뉴 찾기 ═════════
import json, re

def _prep(M):
    d = M[~M["영양누락"]].copy()
    d["순탄수"] = (d["탄수화물(g)"] - d["총 식이섬유(g)"].fillna(0)).clip(lower=0)
    return d.drop_duplicates("식품코드")

def _row(r):
    n = nutrients(r)
    lv, _ = level_of(n["순탄수(g)"], n["당류(g)"])
    return {"food_id": r["food_id"], "음식": r["ai_hub_name"], "식품명": r["식품명"],
            "분량": f'{r["1회제공량"]:g}g' if r["기준"] == "1인분" else "100g",
            "열량": n["에너지(㎉)"], "탄수화물": n["탄수화물(g)"], "순탄수": n["순탄수(g)"],
            "혈당부담": lv, "근사": bool(r["근사표시"])}

def search_foods(M, kinds=None, max_carb=None, max_kcal=None, min_protein=None,
                 sort="탄수화물", n=5, include_100g=False, exclude_codes=(), cuisine=None):
    """조건 검색. kinds: 이름/분류에 들어 있어야 할 단어들(하나라도 맞으면 통과)
    cuisine: 요리 종류(한식/중식/일식/양식/분식/기타). master에 '요리종류' 열이 있을 때만 적용"""
    d = _prep(M)
    if not include_100g:
        d = d[d["기준"] == "1인분"]
    if kinds:
        text = (d["식품명"].fillna("") + " " + d["식품대분류"].fillna("") + " "
                + d["식품상세분류"].fillna("") + " " + d["ai_hub_name"].fillna(""))
        mask = pd.Series(False, index=d.index)
        for k in kinds:
            k = str(k)
            hit = text.str.contains(k, regex=False)
            if k == "국":                                   # '국'은 '국수'에 걸리지 않게
                hit = text.str.contains("국(?!수)", regex=True)
            mask |= hit
        d = d[mask]
    if cuisine and "요리종류" in d.columns:
        d = d[d["요리종류"] == cuisine]
    if max_carb is not None:   d = d[d["탄수화물(g)"] <= max_carb]
    if max_kcal is not None:   d = d[d["에너지(㎉)"] <= max_kcal]
    if min_protein is not None: d = d[d["단백질(g)"] >= min_protein]
    d = d[~d["식품코드"].isin(exclude_codes)]
    key = {"탄수화물": "탄수화물(g)", "열량": "에너지(㎉)", "순탄수": "순탄수"}.get(sort, "탄수화물(g)")
    return [_row(r) for _, r in d.sort_values(key).head(n).iterrows()]

def find_alternatives(M, query, n=5):
    """query 음식과 같은 분류에서 탄수화물이 더 적은 메뉴. (원래 음식, 대체 리스트) 반환"""
    ids = find_food(M, query, n=1)
    if not ids:
        return None, []
    base = M.loc[ids[0]]
    d = _prep(M)
    d = d[(d["기준"] == base["기준"]) & (d["식품코드"] != base["식품코드"])]
    bn = nutrients(base)
    pools = [d[d["식품상세분류"] == base["식품상세분류"]], d[d["식품대분류"] == base["식품대분류"]]]
    out = []
    for p in pools:
        c = p[(p["순탄수"] < bn["순탄수(g)"])]
        if base["기준"] == "1인분":                       # 너무 적은 분량(과자 등)은 대체로 안 봄
            c = c[(c["1회제공량"] >= 0.5 * base["1회제공량"]) & (c["에너지(㎉)"] >= 0.4 * base["에너지(㎉)"])]
        for _, r in c.sort_values("순탄수").iterrows():
            if r["food_id"] not in [o["food_id"] for o in out]:
                out.append(_row(r))
        if len(out) >= n: break
    return _row(base), out[:n]

def _fmt(x):
    s = f'- {x["음식"]} ({x["분량"]}): 탄수화물 {x["탄수화물"]:.0f}g · {x["열량"]:.0f}kcal · 혈당 부담 {x["혈당부담"]}'
    return s + (" ⓘ가장 가까운 음식 기준" if x["근사"] else "")

PARSE_PROMPT = '''사용자 질문에서 음식 검색 조건을 JSON으로만 뽑아 주세요.
- "intent": "recommend"(조건에 맞는 메뉴 추천) / "alternative"(어떤 음식 대신 먹을 것) / "other"(그 외)
- "target": alternative일 때 대체하려는 음식 이름, 아니면 ""
- "kinds": 음식 종류를 나타내는 단어 목록. 한국 음식 이름에 실제로 들어갈 법한 짧은 단어로 (예: 국물 요리 → ["국","탕","찌개","전골"], 면 → ["면","국수"], 밥 → ["밥"])
- "max_carb": 탄수화물 상한(g) 숫자, 없으면 null ("적은"만 있으면 null)
- "max_kcal": 열량 상한(kcal) 숫자, 없으면 null
- "min_protein": 단백질 하한(g) 숫자, 없으면 null
- "sort": 정렬 기준 "탄수화물" 또는 "열량"
JSON 예: {{"intent":"recommend","target":"","kinds":["국","탕"],"max_carb":null,"max_kcal":null,"min_protein":null,"sort":"탄수화물"}}

질문: {q}'''

def parse_query(text, call_llm=None):
    """call_llm(prompt)->JSON 문자열. 실패하면 간단한 규칙으로 대신 분석"""
    if call_llm is not None:
        try:
            raw = re.sub(r"^```(json)?|```$", "", call_llm(PARSE_PROMPT.format(q=text)).strip()).strip()
            p = json.loads(raw)
            p.setdefault("kinds", []); p["kinds"] = [str(k) for k in p["kinds"] if k]
            return p
        except Exception as e:
            print("질문 분석 LLM 실패 → 규칙으로 처리:", e)
    t = text.replace(" ", "")
    m = re.search(r"(.+?)(대신|말고|대체)", text)
    if m:
        target = re.sub(r"(을|를|이|가|은|는)$", "", m.group(1).strip().split()[-1])
        return {"intent": "alternative", "target": target, "kinds": [], "max_carb": None,
                "max_kcal": None, "min_protein": None, "sort": "탄수화물"}
    kinds = []
    for k, ws in {"국물": ["국", "탕", "찌개", "전골"], "면": ["면", "국수"], "밥": ["밥"], "샐러드": ["샐러드"]}.items():
        if k in t: kinds += ws
    if not kinds and not any(w in t for w in ["추천", "적은", "낮은", "메뉴", "먹을", "뭐먹"]):
        return {"intent": "other", "target": "", "kinds": [], "max_carb": None,
                "max_kcal": None, "min_protein": None, "sort": "탄수화물"}
    nums = lambda pat: (lambda mm: float(mm.group(1)) if mm else None)(re.search(pat, t))
    return {"intent": "recommend", "target": "", "kinds": kinds, "max_carb": nums(r"탄수화물(\d+)g?이하"),
            "max_kcal": nums(r"(\d+)(?:kcal|칼로리)이하"), "min_protein": None,
            "sort": "열량" if "칼로리" in t or "열량" in t else "탄수화물"}

def ask(M, text, call_llm=None, n=5):
    """질문 → 조건 분석 → DB에서 걸러 답변. 숫자는 모두 DB 값이에요."""
    p = parse_query(text, call_llm)
    if p.get("intent") == "alternative" and p.get("target"):
        base, alts = find_alternatives(M, p["target"], n)
        if base is None:
            return f"'{p['target']}'은(는) 데이터에 없는 음식이에요."
        head = f'{base["음식"]}({base["분량"]})은 탄수화물 {base["탄수화물"]:.0f}g이에요.'
        if not alts:
            return head + "\n같은 분류에서 탄수화물이 더 적은 메뉴를 찾지 못했어요.\n(" + NOTICE + ")"
        return head + "\n같은 분류에서 탄수화물이 더 적은 메뉴예요:\n" + "\n".join(_fmt(x) for x in alts) + "\n(" + NOTICE + ")"
    if p.get("intent") == "recommend":
        res = search_foods(M, p.get("kinds"), p.get("max_carb"), p.get("max_kcal"), p.get("min_protein"),
                           p.get("sort", "탄수화물"), n)
        relaxed = False
        if not res and (p.get("max_carb") or p.get("max_kcal") or p.get("min_protein")):
            # 숫자 조건에 딱 맞는 게 없으면 숫자 조건을 빼고 가장 가까운(수치가 낮은) 메뉴를 보여줌
            res = search_foods(M, p.get("kinds"), None, None, None, p.get("sort", "탄수화물"), n)
            relaxed = bool(res)
        if not res:
            return "조건에 맞는 메뉴를 찾지 못했어요. 음식 종류나 조건을 바꿔서 다시 물어봐 주세요."
        cond = ", ".join(x for x in [
            "/".join(p["kinds"]) if p.get("kinds") else "", f'탄수화물 {p["max_carb"]:g}g 이하' if p.get("max_carb") else "",
            f'열량 {p["max_kcal"]:g}kcal 이하' if p.get("max_kcal") else ""] if x)
        if relaxed:
            return (f'조건({cond})에 딱 맞는 메뉴는 없었어요. 가장 가까운 메뉴를 {"열량" if p.get("sort")=="열량" else "탄수화물"}이 적은 순으로 보여드릴게요:\n'
                    + "\n".join(_fmt(x) for x in res) + "\n(" + NOTICE + ")")
        return f'조건({cond or "전체"})에서 {"열량" if p.get("sort")=="열량" else "탄수화물"}이 적은 순이에요:\n' + "\n".join(_fmt(x) for x in res) + "\n(" + NOTICE + ")"
    return "조건에 맞는 메뉴 추천이나 대체 메뉴 질문만 도와드릴 수 있어요. 예: '탄수화물 적은 국물 요리 추천해줘', '짜장면 대신 먹을 거 있어?'"
