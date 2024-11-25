from kobert_tokenizer import KoBERTTokenizer
import torch
from transformers import BertModel
from langchain_community.embeddings import HuggingFaceEmbeddings
import Levenshtein
from jamo import h2j, j2h, j2hcj
from kiwipiepy import Kiwi

# KoBERT 토크나이저와 모델 초기화
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertModel.from_pretrained('skt/kobert-base-v1')

class KoBERTEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, documents):
        embeddings = []
        for doc in documents:
            inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
        return embeddings

def query_embedding(query, k, embeddings_model, vector_store):
    query_embedded = embeddings_model.embed_documents([query])[0]
    results1 = vector_store.similarity_search_by_vector(query_embedded, k=k)
    results2 = vector_store.max_marginal_relevance_search_by_vector(query_embedded, k=k)
    #results3 = vector_store.search_by_vector(query_embedded, k=k)
    return results1, results2#, results3

def format_docs(docs):
    return '\n'.join([f'"{d.page_content}"' for d in docs])

def format_dict(dict):
    return '\n\n'.join([f"{key}: {', '.join(value) if isinstance(value, list) else value}" for key, value in dict.items()])

# 한글을 자모로 변환하는 함수
def hangul_to_jamo(text):
    return ''.join(j2hcj(h2j(text)))

# 자모를 한글로 변환하는 함수
def jamo_to_hangul(jamo_text):
    return j2h(jamo_text)

def documents_to_jamodict(documents):
    kiwi = Kiwi()
    jamodict = {}
    for doc in documents:
        docnouns = [""]  # 첫 번째 항목은 모든 명사를 합친 문자열
        for sentence in kiwi.analyze(doc):
            for word, tag, _, _ in sentence[0]:
                if tag in ['NNG', 'NNP']:  # 일반 명사(NNG) 및 고유 명사(NNP)
                    jamo_noun = hangul_to_jamo(word)
                    docnouns[0] += jamo_noun
                    docnouns.append(jamo_noun)
        jamodict[doc] = docnouns
    return jamodict

# 자모 사전을 검색하여 레벤슈타인 거리가 가장 가까운 결과를 반환하는 함수
def jamodict_search(query, jamodict):
    jamo_query = hangul_to_jamo(query)
    distance_list = []

    for k, v in jamodict.items():
        k_distance = Levenshtein.distance(hangul_to_jamo(k), jamo_query)
        v_distances = [Levenshtein.distance(jamo, jamo_query) for jamo in v]
        distance_list.append([k, k_distance, v_distances])

    sorted_distance_list = sorted(distance_list, key=lambda x: min(x[2]))
    return sorted_distance_list

def parse_output_string(response):
    # "output: " 부분을 제거하고, 대괄호와 따옴표를 제거한 후 공백을 기준으로 나누어 리스트로 변환
    cleaned_string = response.replace('output: ', '').strip().strip('[]').replace('"', '')
    result = cleaned_string.split(", ")
    return result
