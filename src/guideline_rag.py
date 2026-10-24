"""
기능 1: 식사 방법 물어보기 (탱) — LangChain 버전

guideline_tools.py 에서 바뀐 것
    - 문서 읽기·청킹(load_kdca, load_local, make_chunks)은 guideline_tools 것을 그대로 불러 써요.
    - 임베딩: OpenAI 직접 호출 → LangChain OpenAIEmbeddings (밖에서 만들어 넘겨요)
    - 벡터 저장: numpy 배열 + pickle → FAISS (save_local / load_local)
    - 검색 결과 형식, 상황·대상 가감점, min_score 거절은 예전과 같아요.
      그래서 노트북의 make_footer, gt.sources_of 를 그대로 쓸 수 있어요.

사용법 (노트북)
    from langchain_openai import OpenAIEmbeddings
    import guideline_rag as grag
    rag = grag.GuidelineRAG(DOCS_DIR, OpenAIEmbeddings(model='text-embedding-3-small')).build()
    rag.search('외식할 때 주의할 점', situations=['외식'])
"""
import os, json, hashlib
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

import guideline_tools as gt

META_KEYS = ('doc_title', 'org', 'url', 'target', 'section', 'text', 'situations')


def load_documents(docs_dir, verbose=True):
    """guideline_tools의 청크를 LangChain Document로 바꿔요.

    page_content: 임베딩할 글 = "[문서명 > 소제목] 본문" (제목을 붙여 맥락을 같이 임베딩)
    metadata    : 출처·대상·상황 태그, 그리고 답변에 쓸 본문(text)
    """
    sections = gt.load_kdca(docs_dir, verbose=verbose) + gt.load_local(docs_dir, verbose=verbose)
    chunks = gt.make_chunks(sections)
    return [Document(page_content=c['embed_text'], metadata={k: c[k] for k in META_KEYS})
            for c in chunks]


def _signature(docs, model_name):
    """문서·청킹·임베딩 모델 중 하나라도 바뀌면 값이 달라져서 인덱스를 다시 만들어요."""
    raw = json.dumps([model_name] + [[d.page_content, d.metadata] for d in docs], ensure_ascii=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


class GuidelineRAG:
    def __init__(self, docs_dir, embeddings, index_dir=None, min_score=0.30, fetch_k=40):
        """
        embeddings: LangChain 임베딩 객체 (예: OpenAIEmbeddings)
        index_dir : FAISS 인덱스를 저장할 폴더 (기본: DOCS_DIR/faiss_index)
        min_score : 가감점 반영 후 이 점수 미만이면 결과에서 빼요 (예전과 같은 0.30)
        fetch_k   : FAISS에서 먼저 넉넉히 가져올 개수. 여기에 가감점을 매겨 다시 정렬해요.
        """
        self.docs_dir, self.embeddings = docs_dir, embeddings
        self.index_dir = index_dir or f'{docs_dir}/faiss_index'
        self.min_score, self.fetch_k = min_score, fetch_k
        self.model_name = getattr(embeddings, 'model', type(embeddings).__name__)
        self.vs, self.docs = None, []

    def build(self, force=False, verbose=True):
        os.makedirs(f'{self.docs_dir}/kdca', exist_ok=True)
        if verbose:
            print('[문서 읽기]')
        self.docs = load_documents(self.docs_dir, verbose=verbose)
        if not self.docs:
            raise RuntimeError(f'읽은 문서가 없어요. {self.docs_dir}/kdca/ 에 HTML을 올렸는지 확인하세요.')

        sig = _signature(self.docs, self.model_name)
        sig_path = f'{self.index_dir}/signature.txt'
        if not force and os.path.exists(sig_path) and open(sig_path).read().strip() == sig:
            # 직접 저장한 인덱스라서 pickle 로딩을 허용해요 (남이 준 인덱스 폴더는 열지 마세요)
            self.vs = FAISS.load_local(self.index_dir, self.embeddings,
                                       allow_dangerous_deserialization=True,
                                       distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
            if verbose:
                print(f'[FAISS] 저장된 인덱스 사용 · 청크 {len(self.docs)}개')
            return self

        # OpenAI 임베딩은 길이가 1로 정규화돼 있어서, 내적(MAX_INNER_PRODUCT) = 코사인 유사도예요.
        self.vs = FAISS.from_documents(self.docs, self.embeddings,
                                       distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        self.vs.save_local(self.index_dir)
        with open(sig_path, 'w') as f:
            f.write(sig)
        if verbose:
            print(f'[FAISS] 새로 생성 · 청크 {len(self.docs)}개 → {self.index_dir}')
        return self

    def retriever(self, k=4):
        """LangChain 표준 retriever. 나중에 BM25 앙상블·리랭커를 붙일 때 이걸 감싸요."""
        return self.vs.as_retriever(search_kwargs={'k': k})

    def search(self, query, situations=None, target='일반', k=4):
        """질문과 비슷한 문서 내용 k개. 반환 형식은 guideline_tools.GuidelineIndex.search 와 같아요."""
        if self.vs is None:
            raise RuntimeError('build()를 먼저 실행하세요.')
        situations = set(situations or [])
        pairs = self.vs.similarity_search_with_score(query, k=min(self.fetch_k, len(self.docs)))

        scored = []
        for doc, base in pairs:
            m = doc.metadata
            s = float(base) + 0.03 * len(situations & set(m['situations']))
            if target != '일반' and m['target'] == target:
                s += 0.05
            if target == '일반' and m['target'] == '임산부':
                s -= 0.05
            scored.append((s, m))
        scored.sort(key=lambda x: -x[0])

        hits = [{'score': round(s, 3), '문서': m['doc_title'], '기관': m['org'], '섹션': m['section'],
                 '내용': m['text'], 'url': m['url']}
                for s, m in scored[:k] if s >= self.min_score]
        if not hits:
            return {'결과': [], '안내': '근거 문서에서 관련 내용을 찾지 못했어요.'}
        return {'결과': hits}


# 출처 라벨은 예전 함수를 그대로 써요
sources_of = gt.sources_of
SITUATIONS = list(gt.SITUATION_KEYWORDS)
TARGETS = gt.TARGETS
