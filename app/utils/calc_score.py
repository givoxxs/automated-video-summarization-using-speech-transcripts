# app/services/scoring.py
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set

from app.config import get_config
from app.models.base import Segment, TimedWord

config = get_config()

def preprocess_text(text: str) -> str:
    text_cleaned = ''.join(filter(str.isalnum, text)).lower()
    return text_cleaned if text_cleaned else None

def calc_term_frrequencies(segments: List[Segment]) ->Tuple[Dict[int, Counter], Counter, float, int, Dict[int, int]]:
    """Tính tần suất từ, độ dài segment và các thông số cần thiết."""
    n_i_w = defaultdict(Counter)  # Tần suất từ trong từng segment
    n_w = Counter()  # Tần suất từ w trên toàn bộ video
    L_i = {}  # Lưu độ dài (số từ đã lọc) của từng segment
    total_words_in_video = 0
    num_segments = len(segments)
    
    for segment in segments:
        filtered_words = []
        for word in segment.words:
            filtered_word = preprocess_text(word.word)
            if filtered_word:
                filtered_words.append(filtered_word)
        
        n_i_w[segment.id].update(filtered_words)
        n_w.update(filtered_words)
        L_i[segment.id] = len(filtered_words)
        total_words_in_video += len(filtered_words)
    
    A_L = total_words_in_video / num_segments if num_segments > 0 else 0
    return n_i_w, n_w, A_L, num_segments, L_i
  
# tính điểm quan trọng (s_i,w) cho mỗi từ (w) trong mỗi segment (i)
def calc_word_scores(
    n_i_w: Dict[int, Counter],
    n_w: Counter,
    A_L: float, 
    num_segments: int,
    L_i: Dict[int, int],
) -> Dict[str, float]:
    """Tính toán điểm cho từng từ."""
    words_scores = defaultdict(dict)
    k = config.SCORING_K
    b = config.SCORING_B
    N = num_segments
    
    if N == 0: return words_scores
    
    inv_doc_freq = {} # Tần suất từ trong toàn bộ video
    for word, freq in n_w.items():
        if freq > 0: # xuất hiện ít nhất 1 lần
            idf_val = math.log(N / freq) if freq < N else 0 # if freq == N thì log (1)=0
            inv_doc_freq[word] = idf_val
    
    for segment_id, words in n_i_w.items():
        Li = L_i.get(segment_id, 0)
        if Li == 0 or A_L == 0: continue # Bỏ qua segment rỗng
        
        part_denominator = k * (1 - b) + b * (Li / A_L)
        
        for w, niw in words.items():
            idf = inv_doc_freq.get(w, 0.0)
            if niw <= 0: continue
            
            numerator = (k + 1) * niw
            denominator = part_denominator + niw
            score_siw = (numerator / denominator) * idf
            words_scores[segment_id][w] = score_siw
            
    return words_scores

def calc_log_likelihood(a, b, c, d, N):
    def log(x):
        return math.log(x) if x > 0 else 0.0

    term1 = a * log(a) + b * log(b) + c * log(c) + d * log(d)
    term2 = (a + b) * log(a + b)
    term3 = (a + c) * log(a + c)
    term4 = (b + d) * log(b + d)
    term5 = (c + d) * log(c + d)
    term6 = N * log(N)
    
    lambda_val = term1 - term2 - term3 - term4 - term5 + term6
    return max(0.0, lambda_val)

def detect_dominant_pairs(
    segments: List[Segment],
    n_i_w: Dict[int, Counter],
    num_segments: int,
    top_n: int
) -> List[Tuple[int, int, float]]:
    if num_segments == 0: return []
    
    # 1. Lấy tập hợp từ unique cho mỗi segment (đã lọc)
    segment_pairs: Dict[int, Set[str]] = defaultdict(set)
    for seg_idx, counts in n_i_w.items():
        segment_pairs[seg_idx] = set(counts.keys())
    
    # 2. Lấy toàn bộ từ vựng (unique, đã lọc)
    vocab = set()
    for w_set in segment_pairs.values():
        vocab.update(w_set)
    vocab_list = sorted(list(vocab))

    # 3. Tính log-likelihood cho từng cặp
    pair_likelihoods = []
    processed_pairs = 0
    total_pairs = len(vocab_list) * (len(vocab_list) - 1) // 2
    
    for i in range(len(vocab_list)):
        w1 = vocab_list[i]
        for j in range(i + 1, len(vocab_list)):
            w2 = vocab_list[j]
            processed_pairs += 1
            
            if processed_pairs % 50000 == 0: # Log tiến trình
                print(f"Processing pairs: {processed_pairs}/{total_pairs}")
            
            a, b, c, d = 0, 0, 0, 0
            for seg_idx, w_set in segment_pairs.items():
                has_w1 = w1 in w_set
                has_w2 = w2 in w_set
                if has_w1 and has_w2:
                    a += 1
                elif has_w1 and not has_w2:
                    b += 1
                elif not has_w1 and has_w2:
                    c += 1
                else:
                    d += 1
            d = num_segments - a - b - c - d
            
            if a >= 0 and b >= 0 and c >= 0 and d >= 0:
                lambda_val = calc_log_likelihood(a, b, c, d, num_segments)
                if lambda_val > 1e-6:
                   pair_likelihoods.append((lambda_val, (w1, w2)))
    
    # 4. Sắp xếp và lấy top_n cặp có độ khả thi cao nhất
    pair_likelihoods.sort(reverse=True, key=lambda x: x[0])
    top_pairs = [pair for _, pair in pair_likelihoods[:top_n]]
    
    print(f"Top {top_n} pairs: {top_pairs}")
    
    return top_pairs                                   

async def calc_score_segments(segments: List[Segment]) -> List[Segment]:
    if not segments:
        return []
    
    N = len(segments)
    # --- Bước 6: Tính Trọng số Đoạn dựa trên Từ đơn ---
    # 1. Tính tần suất
    n_i_w, n_w, A_L, _, L_i = calc_term_frrequencies(segments)
    
    # 2. Tính điểm cho từng từ s(i, w)
    words_scores = calc_word_scores(n_i_w, n_w, A_L, N, L_i)
    
    # 3. Tính điểm cho từng đoạn s(i) = avg(s(i, w))
    for segment in segments:
        scores_dict = words_scores.get(segment.id, {})
        Li = L_i.get(segment.id, 0)
        if Li > 0 and scores_dict:
            segment.score = sum(scores_dict.values()) / Li
        else:
            segment.score = 0.0
    
    # --- Bước 7: Phát hiện Cặp từ Nổi bật & Tăng cường Điểm ---
    # 1. Tìm cặp từ nổi bật
    top_pairs = detect_dominant_pairs(segments, n_i_w, N, config.DOMINANT_PAIR_COUNT)
    top_pairs_set = set()
    
    for p1, p2 in top_pairs:
        top_pairs_set.add(tuple(sorted([p1, p2])))
    
    # 2. Tăng cường điểm cho các đoạn chứa cặp từ nổi bật
    boost_factor = config.DOMINANT_PAIR_BOOST
    boosted_count = 0
    
    if top_pairs_set:
        for segment in segments:
            # lay lai tap hop tu unique trong segment
            words_in_segment = set(n_i_w.get(segment.id, Counter()).keys())
            word_list = sorted(list(words_in_segment))
            found_dominant_pair = False
            
            for i in range(len(word_list)):
                for j in range(i + 1, len(word_list)):
                    pair = tuple(sorted([word_list[i], word_list[j]]))
                    if pair in top_pairs_set:
                        segment.score *= boost_factor
                        found_dominant_pair = True
                        boosted_count += 1
                        break
                if found_dominant_pair:
                    break
    return segments