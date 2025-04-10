# app/services/scoring.py
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import logging
import itertools # Để tạo cặp từ

from app.models.summarization import Segment, TimedWord
# Giả sử config chứa các hằng số: SCORING_K, SCORING_B, DOMINANT_PAIR_COUNT, DOMINANT_PAIR_BOOST
try:
    from app.core.config import SCORING_K, SCORING_B, DOMINANT_PAIR_COUNT, DOMINANT_PAIR_BOOST
except ImportError:
    logging.warning("Config not found, using default scoring parameters K=2.0, B=0.75, Pairs=30, Boost=1.2")
    SCORING_K = 2.0
    SCORING_B = 0.75
    DOMINANT_PAIR_COUNT = 30
    DOMINANT_PAIR_BOOST = 1.2

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _preprocess_word(word: str) -> Optional[str]:
    """Chuẩn hóa từ: lowercase và chỉ giữ alphanumeric."""
    cleaned = ''.join(filter(str.isalnum, word)).lower()
    return cleaned if cleaned else None

def calculate_term_frequencies(segments: List[Segment]) -> Tuple[Dict[int, Counter], Counter, float, int, Dict[int, int]]:
    """Tính tần suất từ, độ dài segment và các thông số cần thiết."""
    segment_tf = defaultdict(Counter)
    global_tf = Counter()
    segment_lengths = {} # Lưu độ dài (số từ đã lọc) của từng segment
    total_words_in_video = 0
    num_segments = len(segments)

    for segment in segments:
        filtered_words = []
        for word_obj in segment.words:
            processed_word = _preprocess_word(word_obj.word)
            if processed_word:
                filtered_words.append(processed_word)

        segment_tf[segment.id].update(filtered_words)
        global_tf.update(filtered_words)
        segment_length = len(filtered_words)
        segment_lengths[segment.id] = segment_length
        total_words_in_video += segment_length

    average_segment_length = (total_words_in_video / num_segments) if num_segments > 0 else 0

    logger.info(f"TF Calculation: Segments={num_segments}, TotalWords={total_words_in_video}, AvgLen={average_segment_length:.2f}")
    return segment_tf, global_tf, average_segment_length, num_segments, segment_lengths

def calculate_word_scores(
    segment_tf: Dict[int, Counter],
    global_tf: Counter,
    average_segment_length: float,
    num_segments: int,
    segment_lengths: Dict[int, int]
) -> Dict[int, Dict[str, float]]:
    """Tính điểm s(i,w) cho mỗi từ trong mỗi segment (Eq 1)."""
    word_scores = defaultdict(dict) # segment_id -> {word -> score}
    k = SCORING_K
    b = SCORING_B
    N = num_segments
    if N == 0: return word_scores

    inv_doc_freq = {} # log(N / nw) - Inverse Document Frequency
    for word, nw in global_tf.items():
        if nw > 0: # Chỉ xét từ xuất hiện ít nhất 1 lần
            # Nếu nw = N (từ xuất hiện trong mọi segment), idf = log(1) = 0
            idf_val = math.log(N / nw) if nw < N else 0 # Tránh log(0), nếu nw=N thì idf=0
            inv_doc_freq[word] = idf_val
        # Các từ có idf=0 sẽ có score=0

    for segment_id, term_counts in segment_tf.items():
        Li = segment_lengths.get(segment_id, 0)
        if Li == 0 or average_segment_length == 0: continue # Bỏ qua segment rỗng

        # Tính phần chuẩn hóa độ dài (Document length normalization part)
        normalization_factor_base = k * ((1 - b) + b * (Li / average_segment_length))

        for word, niw in term_counts.items(): # niw: số lần từ w xuất hiện trong segment i
            idf = inv_doc_freq.get(word, 0.0) # Lấy idf, mặc định là 0 nếu từ không có trong idf dict
            if idf <= 0: continue # Bỏ qua từ có idf=0 (xuất hiện trong mọi segment)

            # Công thức (1)
            numerator = (k + 1) * niw
            denominator = normalization_factor_base + niw
            score_siw = (numerator / denominator) * idf
            word_scores[segment_id][word] = score_siw

    return word_scores

def _calculate_log_likelihood(a, b, c, d, N):
    """Tính toán giá trị log-likelihood (Eq 4), xử lý log(0)."""
    # Sử dụng hàm log riêng để xử lý log(0) -> trả về 0
    def log(x):
        return math.log(x) if x > 0 else 0.0

    term1 = a * log(a) + b * log(b) + c * log(c) + d * log(d)
    term2 = (a + b) * log(a + b)
    term3 = (a + c) * log(a + c)
    term4 = (b + d) * log(b + d)
    term5 = (c + d) * log(c + d)
    term6 = N * log(N)

    # Lambda = 2 * (term1 - term2 - term3 - term4 - term5 + term6) # Nhân 2 thường thấy trong G-test
    # Tuy nhiên, paper không có số 2, chỉ cần tính phần log terms
    lambda_val = term1 - term2 - term3 - term4 - term5 + term6

    # Đôi khi lambda có thể âm nhẹ do sai số, kẹp về 0
    return max(0.0, lambda_val)


def detect_dominant_pairs(
    segments: List[Segment],
    segment_tf: Dict[int, Counter], # Dùng lại tf đã tính
    num_segments: int,
    top_n: int
) -> List[Tuple[str, str]]:
    """Phát hiện cặp từ nổi bật sử dụng log-likelihood (Eq 4)."""
    if num_segments == 0: return []
    logger.info(f"Detecting top {top_n} dominant pairs using log-likelihood...")

    # 1. Lấy tập hợp từ unique cho mỗi segment (đã lọc)
    segment_word_sets: Dict[int, Set[str]] = defaultdict(set)
    for seg_id, counts in segment_tf.items():
        segment_word_sets[seg_id] = set(counts.keys())

    # 2. Lấy toàn bộ từ vựng (unique, đã lọc)
    vocab = set()
    for word_set in segment_word_sets.values():
        vocab.update(word_set)
    vocab_list = sorted(list(vocab))
    logger.info(f"Vocabulary size for pair detection: {len(vocab_list)}")

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
                 logger.info(f"Processing pairs: {processed_pairs}/{total_pairs}")

            # Đếm a, b, c, d
            a = 0 # count(w1, w2)
            b = 0 # count(w1, not w2)
            c = 0 # count(not w1, w2)
            for seg_id, word_set in segment_word_sets.items():
                has_w1 = w1 in word_set
                has_w2 = w2 in word_set
                if has_w1 and has_w2:
                    a += 1
                elif has_w1 and not has_w2:
                    b += 1
                elif not has_w1 and has_w2:
                    c += 1
            # d = N - a - b - c (N = num_segments)
            d = num_segments - a - b - c

            # Tính log-likelihood
            # Kiểm tra các giá trị count hợp lệ (>=0)
            if a >= 0 and b >= 0 and c >= 0 and d >= 0:
                 lambda_val = _calculate_log_likelihood(a, b, c, d, num_segments)
                 if lambda_val > 1e-6 : # Chỉ lưu các cặp có giá trị đáng kể
                     pair_likelihoods.append((lambda_val, (w1, w2)))
            else:
                 logger.warning(f"Invalid counts for pair ({w1}, {w2}): a={a},b={b},c={c},d={d},N={num_segments}")


    # 4. Sắp xếp và lấy top N
    pair_likelihoods.sort(key=lambda x: x[0], reverse=True)
    dominant_pairs = [pair for _, pair in pair_likelihoods[:top_n]]

    logger.info(f"Detected {len(dominant_pairs)} dominant pairs: {dominant_pairs}")
    return dominant_pairs


# --- Hàm chính ---
async def score_segments(segments: List[Segment]) -> List[Segment]:
    """Tính điểm cuối cùng cho các segment, bao gồm cả boost từ dominant pairs."""
    if not segments:
        return []
    logger.info(f"Starting scoring for {len(segments)} segments...")
    N = len(segments)

    # --- Bước 6: Tính Trọng số Đoạn dựa trên Từ đơn ---
    # 1. Tính tần suất
    segment_tf, global_tf, avg_len, _, segment_lengths = calculate_term_frequencies(segments)

    # 2. Tính điểm từng từ s(i,w)
    word_scores = calculate_word_scores(segment_tf, global_tf, avg_len, N, segment_lengths)

    # 3. Tính điểm cơ bản cho segment S(i) = avg(s(i,w)) (Eq 3)
    for segment in segments:
        scores_dict = word_scores.get(segment.id, {})
        Li = segment_lengths.get(segment.id, 0)
        if Li > 0 and scores_dict:
            total_score = sum(scores_dict.values())
            segment.score = total_score / Li # Điểm trung bình
        else:
            segment.score = 0.0 # Segment rỗng hoặc không có từ nào được tính điểm
        # logger.debug(f"Segment {segment.id} base score: {segment.score:.4f}")


    # --- Bước 7: Phát hiện Cặp từ Nổi bật & Tăng cường Điểm ---
    # 4. Phát hiện cặp từ nổi bật
    dominant_pairs = detect_dominant_pairs(segments, segment_tf, N, DOMINANT_PAIR_COUNT)
    dominant_pair_set = set()
    # Sắp xếp cặp từ trong tuple để đảm bảo thứ tự khi kiểm tra
    for p1, p2 in dominant_pairs:
        dominant_pair_set.add(tuple(sorted((p1, p2))))

    # 5. Tăng điểm cho segment chứa cặp từ nổi bật
    boost_factor = DOMINANT_PAIR_BOOST
    boosted_count = 0
    if dominant_pair_set: # Chỉ thực hiện nếu có cặp từ nổi bật
        for segment in segments:
            # Lấy lại tập hợp từ unique, đã lọc của segment
            words_in_segment = set(segment_tf.get(segment.id, Counter()).keys())
            word_list = sorted(list(words_in_segment))
            found_dominant = False
            # Kiểm tra tất cả các cặp từ có thể có trong segment
            for i in range(len(word_list)):
                for j in range(i + 1, len(word_list)):
                    # Tạo cặp đã sắp xếp để kiểm tra
                    current_pair = tuple(sorted((word_list[i], word_list[j])))
                    if current_pair in dominant_pair_set:
                        original_score = segment.score
                        segment.score *= boost_factor
                        # logger.debug(f"Boosting score for segment {segment.id} (orig: {original_score:.4f}) due to pair {current_pair}. New score: {segment.score:.4f}")
                        found_dominant = True
                        boosted_count += 1
                        break # Chỉ boost 1 lần cho mỗi segment
                if found_dominant:
                    break # Thoát vòng lặp ngoài nếu đã boost

    logger.info(f"Segment scoring complete. Boosted {boosted_count} segments based on {len(dominant_pair_set)} dominant pairs.")
    # Trả về list các segment đã được cập nhật điểm
    return segments