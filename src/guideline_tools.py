"""
기능 2: 식사 방법 물어보기 (탱)
공식 식사요법 문서(질병관리청 국가건강정보포털, 학회 자료 등)를 청크로 나눠 임베딩하고,
질문과 비슷한 문서 내용을 찾아 돌려준다.

사용법 (main.ipynb)
    import guideline_tools as gt
    gidx = gt.GuidelineIndex(DOCS_DIR, client, cache_path=f'{DOCS_DIR}/guideline_index.pkl').build()
    gidx.search('외식할 때 주의할 점', situations=['외식'])

문서 넣는 법
    - 질병관리청 문서: 브라우저에서 Ctrl+S → "웹페이지, HTML만" → 파일명을 cntnts_sn 번호로 (예: 3388.html)
      → DOCS_DIR 바로 아래(또는 DOCS_DIR/kdca/)에 업로드. (Colab은 해외 서버라 사이트 직접 접속이 막혀요)
    - 그 외 자료(txt/pdf): DOCS_DIR/ 에 업로드. 첫 줄에 `출처: 문서명 | 기관 | URL` 을 적으면 출처로 쓰여요.
"""
import os, re, glob, pickle
import numpy as np
from bs4 import BeautifulSoup

SOURCE_KIND = 'guideline'

# ───────── 문서 목록 ─────────
KDCA_URL = ('https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/'
            'gnrlzHealthInfoView.do?cntnts_sn={}')
KDCA_ORG = '질병관리청 국가건강정보포털'
KDCA_DOCS = [
    {'sn': 3388, 'title': '당뇨환자의 식이요법',               'target': '일반'},
    {'sn': 5305, 'title': '당뇨병',                           'target': '일반'},
    {'sn': 6694, 'title': '비만',                             'target': '체중조절'},
    {'sn': 6547, 'title': '건강한 체중조절을 위한 식사',        'target': '체중조절'},
    {'sn': 6693, 'title': '식이영양',                         'target': '일반'},
    {'sn': 5214, 'title': '식이영양(임산부)',                  'target': '임산부'},
    {'sn': 6763, 'title': '영양표시 읽는 법! 알려드리겠습니다!', 'target': '일반'},
    {'sn': 5271, 'title': '임신당뇨병',                       'target': '임산부'},
    {'sn': 5863, 'title': '건강체중을 위한 식이 조절',          'target': '체중조절'},
    {'sn': 5211, 'title': '이상지질혈증 식사요법',             'target': '일반'},
]
TARGETS = ['일반', '임산부', '체중조절']

SITUATION_KEYWORDS = {
    '외식':     ['외식', '식당', '중식', '한식', '양식', '일식', '분식', '패스트푸드', '뷔페', '회식'],
    '간식':     ['간식', '야식', '과일', '떡', '과자', '빵', '고구마'],
    '음주':     ['술', '음주', '알코올', '맥주', '소주', '와인', '안주'],
    '음료':     ['음료', '커피', '주스', '탄산', '우유'],
    '먹는방법': ['순서', '천천히', '규칙', '거르', '아침', '식사량', '골고루', '채소'],
    '영양표시': ['영양표시', '영양성분', '표시', '라벨', '당류'],
    '조리법':   ['튀김', '구이', '찜', '볶음', '조림', '국물', '기름'],
    '명절':     ['명절', '잔치', '부침', '송편', '차례'],
}


# ───────── HTML / 텍스트 파싱 ─────────
def html_to_sections(container):
    """소제목(h2~h5) 단위로 [(소제목, 본문)] 리스트"""
    sections, head, buf = [], '본문', []
    for el in container.find_all(['h2', 'h3', 'h4', 'h5', 'p', 'li', 'td', 'dt', 'dd']):
        text = el.get_text(' ', strip=True)
        if not text:
            continue
        if el.name in ('h2', 'h3', 'h4', 'h5'):
            if buf:
                sections.append((head, '\n'.join(buf)))
            head, buf = text, []
        elif el.find_parent(['li', 'td', 'dd']) is None or el.name in ('li', 'td', 'dd'):
            if not buf or buf[-1] != text:
                buf.append(text)
    if buf:
        sections.append((head, '\n'.join(buf)))
    return sections


def find_main(soup):
    """링크 비율이 낮고 글이 가장 많은 영역을 본문으로 고른다 (메뉴·탭·사이드바 제외)"""
    cands = []
    for el in soup.find_all(['div', 'section', 'article', 'main']):
        total = len(el.get_text(' ', strip=True))
        if total < 300:
            continue
        link = sum(len(a.get_text(' ', strip=True)) for a in el.find_all('a'))
        if link / total > 0.3:
            continue
        cands.append((total, el))
    if not cands:
        return soup.body or soup
    best = max(t for t, _ in cands)
    return min([c for c in cands if c[0] >= 0.8 * best], key=lambda c: c[0])[1]


def _decode(raw_bytes):
    for enc in ('utf-8', 'cp949', 'euc-kr'):
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw_bytes.decode('utf-8', errors='ignore')


def load_kdca(docs_dir, docs=KDCA_DOCS, verbose=True):
    """<번호>.html 을 읽어 섹션 리스트로. docs_dir 바로 아래와 docs_dir/kdca/ 둘 다 찾는다"""
    out = []
    for d in docs:
        files = (glob.glob(f"{docs_dir}/{d['sn']}*.htm*") +
                 glob.glob(f"{docs_dir}/kdca/{d['sn']}*.htm*"))
        if not files:
            if verbose:
                print(f"  ❌ {d['title']}: {d['sn']}.html 없음")
            continue
        soup = BeautifulSoup(_decode(open(files[0], 'rb').read()), 'html.parser')
        for t in soup(['script', 'style', 'nav', 'header', 'footer']):
            t.decompose()
        secs = html_to_sections(find_main(soup))
        out += [{'doc_title': d['title'], 'org': KDCA_ORG, 'url': KDCA_URL.format(d['sn']),
                 'target': d['target'], 'section': h, 'text': t} for h, t in secs]
        if verbose:
            print(f"  ✅ {d['title']}: 섹션 {len(secs)}개")
    return out


def _read_local(path):
    if path.lower().endswith('.pdf'):
        from pypdf import PdfReader
        return '\n'.join((p.extract_text() or '') for p in PdfReader(path).pages)
    return _decode(open(path, 'rb').read())


def load_local(docs_dir, verbose=True):
    """docs_dir 의 txt/pdf (첫 줄 `출처: 문서명 | 기관 | URL` 선택)"""
    out = []
    for path in sorted(glob.glob(f'{docs_dir}/*.txt') + glob.glob(f'{docs_dir}/*.pdf')):
        text = _read_local(path)
        first, _, body = text.partition('\n')
        name = os.path.splitext(os.path.basename(path))[0]
        title, org, url = name, '기타 자료', ''
        if first.strip().startswith('출처:'):
            parts = [p.strip() for p in first.split(':', 1)[1].split('|')]
            title = parts[0] or name
            org = parts[1] if len(parts) > 1 else org
            url = parts[2] if len(parts) > 2 else ''
        else:
            body = text
        target = ('임산부' if ('임산부' in name or '임신' in name)
                  else '체중조절' if ('체중' in name or '비만' in name) else '일반')
        blocks = re.split(r'\n\s*\n|\n(?=[■●○◆▶]|\d+\.\s)', body)
        secs = [{'doc_title': title, 'org': org, 'url': url, 'target': target,
                 'section': b.strip().split('\n')[0][:40], 'text': b.strip()}
                for b in blocks if len(b.strip()) > 30]
        out += secs
        if verbose:
            print(f'  ✅ {os.path.basename(path)}: 섹션 {len(secs)}개')
    return out


# ───────── 청크 ─────────
def tag_situations(text):
    return [k for k, kws in SITUATION_KEYWORDS.items() if any(w in text for w in kws)]


def split_text(text, max_len=500, overlap=1):
    sents = [s.strip() for s in re.split(r'(?<=[.!?다요])\s+|\n', text) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        if cur and len(' '.join(cur + [s])) > max_len:
            chunks.append(' '.join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.append(s)
    if cur:
        chunks.append(' '.join(cur))
    return chunks


def make_chunks(sections, min_len=40):
    chunks = []
    for sec in sections:
        for piece in split_text(sec['text']):
            if len(piece) < min_len:
                continue
            chunks.append({**sec, 'text': piece,
                           'embed_text': f"[{sec['doc_title']} > {sec['section']}] {piece}",
                           'situations': tag_situations(sec['section'] + ' ' + piece)})
    return chunks


# ───────── 검색 인덱스 ─────────
class GuidelineIndex:
    def __init__(self, docs_dir, client, cache_path=None,
                 embed_model='text-embedding-3-small', min_score=0.30):
        self.docs_dir, self.client = docs_dir, client
        self.cache_path = cache_path or f'{docs_dir}/guideline_index.pkl'
        self.embed_model, self.min_score = embed_model, min_score
        self.chunks, self.vecs = [], None

    def _embed(self, texts, batch=100):
        out = []
        for i in range(0, len(texts), batch):
            res = self.client.embeddings.create(model=self.embed_model, input=texts[i:i + batch])
            out += [d.embedding for d in res.data]
        v = np.array(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def build(self, force=False, verbose=True):
        if not os.path.isdir(self.docs_dir):
            raise FileNotFoundError(f'문서 폴더가 없어요: {self.docs_dir}')
        if verbose:
            print('[문서 읽기]')
        sections = load_kdca(self.docs_dir, verbose=verbose) + load_local(self.docs_dir, verbose=verbose)
        self.chunks = make_chunks(sections)
        if not self.chunks:
            raise RuntimeError(f'읽은 문서가 없어요. {self.docs_dir} 에 3388.html 같은 파일이 있는지 확인하세요.')
        sig = [c['embed_text'] for c in self.chunks]
        if not force and os.path.exists(self.cache_path):
            cache = pickle.load(open(self.cache_path, 'rb'))
            if cache.get('sig') == sig and cache.get('model') == self.embed_model:
                self.vecs = cache['vecs']
                if verbose:
                    print(f'[임베딩] 캐시 사용 · 청크 {len(self.chunks)}개')
                return self
        self.vecs = self._embed(sig)
        pickle.dump({'sig': sig, 'model': self.embed_model, 'vecs': self.vecs}, open(self.cache_path, 'wb'))
        if verbose:
            print(f'[임베딩] 새로 생성 · 청크 {len(self.chunks)}개')
        return self

    def search(self, query, situations=None, target='일반', k=4):
        """질문과 비슷한 문서 내용 k개. 점수가 낮으면 빈 결과 + 안내"""
        if self.vecs is None:
            raise RuntimeError('build()를 먼저 실행하세요.')
        qv = self._embed([query])[0]
        base = self.vecs @ qv
        situations = set(situations or [])
        scored = []
        for i, s in enumerate(base):
            c = self.chunks[i]
            s = float(s) + 0.03 * len(situations & set(c['situations']))
            if target != '일반' and c['target'] == target:
                s += 0.05
            if target == '일반' and c['target'] == '임산부':
                s -= 0.05
            scored.append((s, i))
        scored.sort(reverse=True)
        hits = [{'score': round(s, 3), '문서': self.chunks[i]['doc_title'], '기관': self.chunks[i]['org'],
                 '섹션': self.chunks[i]['section'], '내용': self.chunks[i]['text'], 'url': self.chunks[i]['url']}
                for s, i in scored[:k] if s >= self.min_score]
        if not hits:
            return {'결과': [], '안내': '근거 문서에서 관련 내용을 찾지 못했어요.'}
        return {'결과': hits}


def sources_of(result):
    """search() 결과에서 출처 라벨 목록: ['질병관리청「당뇨환자의 식이요법」', ...]"""
    out = []
    for h in result.get('결과', []):
        label = f"{h['기관'].split()[0]}「{h['문서']}」"
        if label not in out:
            out.append(label)
    return out


# ───────── LLM 도구 설명 ─────────
SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'search_guideline',
        'description': ('[기능 2] 당뇨 식사요법 공식 문서(질병관리청 등)에서 먹는 방법, 상황별 요령(외식·간식·음주 등), '
                        '원칙과 이유를 검색한다. "어떻게 먹어?", "뭘 조심해?" 같은 식사 방법 질문에는 항상 호출한다.'),
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': '검색할 내용 (예: 중식 외식 시 주의할 점)'},
                'situations': {'type': 'array', 'items': {'type': 'string', 'enum': list(SITUATION_KEYWORDS)}},
                'target': {'type': 'string', 'enum': TARGETS, 'description': '임신 중이면 임산부, 체중 감량 목적이면 체중조절'},
            },
            'required': ['query'],
        },
    },
}
