import json
import random
import collections
import codecs
from pathlib import Path
 
 
def load_jsonl(file_path):
    data = []
    error_count = 0
 
    decoder = json.JSONDecoder()
    text = ""
    idx = 0
    line_num = 1
 
    # 'replace' でデコードエラーを潰し、末尾の不完全なUTF-8でも落とさない
    inc = codecs.getincrementaldecoder("utf-8")(errors="replace")
 
    def _parse_available_text(is_final: bool = False) -> None:
        nonlocal text, idx, line_num, error_count
 
        while idx < len(text):
            # 空白をスキップ
            while idx < len(text) and text[idx].isspace():
                if text[idx] == "\n":
                    line_num += 1
                idx += 1
            if idx >= len(text):
                break
 
            try:
                obj, end_idx = decoder.raw_decode(text, idx) # end_idx is the end character index
                data.append(obj)
                idx = end_idx
                continue
            except json.JSONDecodeError as e:
                # チャンク境界でJSONが未完成の可能性があるので、改行が無ければ次チャンクを待つ
                next_newline = text.find("\n", idx)
                if next_newline == -1:
                    if is_final:
                        error_count += 1
                        print(
                            f"Warning: Trailing/incomplete JSON around line {line_num}, "
                            f"position {idx}: {e}"
                        )
                    break
 
                # 壊れた/不完全な行をスキップして続行
                error_count += 1
                print(
                    f"Warning: Error parsing around line {line_num}, "
                    f"position {idx}: {e}"
                )
                idx = next_newline + 1
                line_num += 1
                if error_count > 100:
                    print("  Too many errors, stopping...")
                    idx = len(text)
                    break
 
        # バッファが肥大化しないように定期的に詰める
        if idx > 1_000_000:
            text = text[idx:]
            idx = 0
 
    chunk_size = 8 * 1024 * 1024
    with open(file_path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            text += inc.decode(b)
            _parse_available_text(is_final=False)
 
    # finalize decoder + parse remainder
    text += inc.decode(b"", final=True)
    _parse_available_text(is_final=True)
 
    if error_count > 0:
        print(f"  Total errors: {error_count} objects/lines skipped")
    return data # len(data) = 100, i.e., 100 samples for debug, each sample is a dict with keys=dict_keys(['uuid', 'professional_persona', 'sports_persona', 'arts_persona', 'travel_persona', 'culinary_persona', 'persona', 'cultural_background', 'skills_and_expertise', 'skills_and_expertise_list', 'hobbies_and_interests', 'hobbies_and_interests_list', 'career_goals_and_ambitions', 'sex', 'age', 'marital_status', 'education_level', 'occupation', 'region', 'area', 'prefecture', 'country', 'age_band', '_all_text', '_core_text', '_core_len', '_attr_key', 'score_finance', 'score_safety', 'score_vocab', 'score_public', 'score_tools', 'score_life', 'score_geo', 'score_culture', '_geo_text', '_tools_text', '_kw_hits', 'jc_category', 'max_score_any', '_public_bonus', '_religion_pen', 'jc_theme', 'topic_category', 'jcqa_data', 'jcqa_data__reasoning_trace', 'quality_metrics', 'quality_metrics__reasoning_trace', 'clarity_score', 'difficulty'])
 
 
def save_jsonl(data, file_path):
    """データをJSONLファイルに保存する"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
 
 
def split_train_valid(data, train_ratio=0.9, seed=42):
    """
    データをtrainとvalidに分割する
    """
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    split_idx = int(len(shuffled_data) * train_ratio)
    train_data = shuffled_data[:split_idx]
    valid_data = shuffled_data[split_idx:]
    return train_data, valid_data
 
def _count_jcqa_labels_raw(data):
    """元の data（load_jsonl の直後）から jcqa の answer_index 分布を数える"""
    counts = collections.Counter()
    total = 0
    for item in data:
        jcqa = item.get("jcqa_data")
        if jcqa is None:
            continue
        idx = jcqa.get("answer_index")
        if idx is None:
            continue
        counts[str(idx)] += 1 # NOTE idx = answer choice index
        total += 1
    return counts, total
    # counts=Counter({'0': 73, '1': 15, '2': 6, '3': 5, '4': 1}) , total=100
 
def _rebalance_one_jcqa(jcqa, target_label):
    choices = []
    for i in range(5):
        choices.append(jcqa.get(f"choice{i}", ""))
 
    orig_idx = jcqa.get("answer_index")
    if orig_idx is None:
        return
 
    correct_text = choices[orig_idx]
    other_texts = [c for i, c in enumerate(choices) if i != orig_idx]
 
    random.shuffle(other_texts)
 
    new_choices = [None] * 5
    new_choices[target_label] = correct_text
 
    it = iter(other_texts)
    for i in range(5):
        if new_choices[i] is None:
            new_choices[i] = next(it)
 
    # jcqa_data を上書き
    for i in range(5):
        jcqa[f"choice{i}"] = new_choices[i]
    jcqa["answer_index"] = target_label
 
 
def rebalance_jcqa_labels_in_place(data, seed=42):
    random.seed(seed)
 
    # before: 元の分布を出力
    before_counts, n = _count_jcqa_labels_raw(data)
    print(f"[jcqa] total jcqa_data examples: {n}")
    print(f"[jcqa] label counts BEFORE rebalance: {dict(before_counts)}")
 
    jcqa_indices = [i for i, item in enumerate(data) if "jcqa_data" in item]
    n_jcqa = len(jcqa_indices)
    if n_jcqa == 0:
        print("[jcqa] No jcqa_data found, skip rebalance.")
        return
 
    # 目標件数：ほぼ N/5 ずつ
    base = n_jcqa // 5
    rem = n_jcqa % 5  # 余りは先頭から +1 して配る
    target_counts = {str(l): base for l in range(5)}
    for l in range(rem):
        target_counts[str(l)] += 1
 
    print(f"[jcqa] target label counts: {target_counts}")
 
    # シャッフルした順番で、各サンプルにターゲットラベルを割り当てる
    random.shuffle(jcqa_indices)
    current_counts = {str(l): 0 for l in range(5)}
 
    for idx in jcqa_indices:
        item = data[idx]
        jcqa = item["jcqa_data"]
 
        # まだ目標に達していないラベルの中から選ぶ
        available_labels = [
            l for l in range(5)
            if current_counts[str(l)] < target_counts[str(l)]
        ]
        if not available_labels:
            # すべて埋まっていたら、何か適当に
            available_labels = list(range(5))
 
        target_label = random.choice(available_labels)
        current_counts[str(target_label)] += 1
 
        # 1サンプルの choice 並びと answer_index を target_label に揃える
        _rebalance_one_jcqa(jcqa, target_label)
 
    print(f"[jcqa] label counts AFTER  rebalance: {current_counts}")
 
def convert_jcommonsenseqa(item):
    """jcommonsenseqaデータをmessagesフォーマットに変換"""
    jcqa = item.get('jcqa_data', {})
 
    question = jcqa.get('question', '')
    choices_lines = []
    for i in range(5):
        choice_key = f'choice{i}'
        if choice_key in jcqa:
            choices_lines.append(f"    {i}. {jcqa[choice_key]}")
    choices_text = '\n'.join(choices_lines)
 
    user_content = (
        "以下の質問に答えてください。\n\n"
        f"質問:\n{question}\n\n"
        "選択肢:\n"
        f"{choices_text}\n\n"
        "最も適切な選択肢の番号だけを、半角数字で1つ出力してください。"
    ) # '以下の質問に答えてください。\n\n質問:\nあなたは大阪の公共の駅で、現金で名古屋行きの片道切符を購入したいです。どの施設で切符を買うべきですか？\n\n選択肢:\n    0. 旅行代理店\n    1. 自動券売機\n    2. 乗車カードチャージ機\n    3. 自動改札機\n    4. 窓口販売\n\n最も適切な選択肢の番号だけを、半角数字で1つ出力してください。'
 
    ground_truth = jcqa.get('answer_index')
    ground_truth_str = str(ground_truth) if ground_truth is not None else ""
    assistant_content = f"<think></think>\n\n{ground_truth_str}"
 
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content} # '<think></think>\n\n1'
        ],
        "extra_env_info": {
            "dataset_type": "jcommonsenseqa",
            "ground_truth": ground_truth_str # '1'
        }
    }
 
def count_labels_in_converted(data, name):
    counts = collections.Counter()
    total = 0
    for ex in data:
        gt = ex.get("extra_env_info", {}).get("ground_truth")
        if gt is None:
            continue
        counts[gt] += 1
        total += 1
    print(f"[{name}] total examples: {total}")
    print(f"[{name}] label counts: {dict(counts)}")
 
from collections import defaultdict
 
def convert_data(input_file_path, output_dir):
    print(f"Loading {input_file_path}...")
    data = load_jsonl(input_file_path)
    print(f"  Loaded {len(data)} examples")
 
    # ★ jcommonsenseqa のラベル分布をリバランス（choice 並び＋answer_index を書き換える）
    print("\nRebalancing jcommonsenseqa label indices (0-4)...")
    rebalance_jcqa_labels_in_place(data, seed=42)
 
    # 変換
    jcommonsenseqa_data = []
 
    for idx, item in enumerate(data):
        if 'jcqa_data' in item:
            converted = convert_jcommonsenseqa(item)
            jcommonsenseqa_data.append(converted) # {'messages': [{'role': 'user', 'content': '以下の質問に答えてください。\n\n質問:\nあなたは大阪の公共の駅で、現金で名古屋行きの片道切符を購入したいです。どの施設で切符を買うべきですか？\n\n選択肢:\n    0. 旅行代理店\n    1. 自動券売機\n    2. 乗車カードチャージ機\n    3. 自動改札機\n    4. 窓口販売\n\n最も適切な選択肢の番号だけを、半角数字で1つ出力してください。'}, {'role': 'assistant', 'content': '<think></think>\n\n1'}], 'extra_env_info': {'dataset_type': 'jcommonsenseqa', 'ground_truth': '1'}}

 
    print(f"\nJCommonsenseQA (converted): {len(jcommonsenseqa_data)} examples")
 
    # 変換後のラベル分布も表示
    count_labels_in_converted(jcommonsenseqa_data, "jcommonsenseqa_converted")
 
    # 出力ディレクトリを作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
 
    # train/valid に分割
    print("\nSplitting data into train/valid...")
    jcommonsenseqa_train, jcommonsenseqa_valid = split_train_valid(jcommonsenseqa_data)
 
    print(f"  JCommonsenseQA:     train={len(jcommonsenseqa_train)},     valid={len(jcommonsenseqa_valid)}")
 
    # train/valid ごとのラベル分布も確認
    count_labels_in_converted(jcommonsenseqa_train, "jcommonsenseqa_train")
    count_labels_in_converted(jcommonsenseqa_valid, "jcommonsenseqa_valid")
 
    # ファイルを保存
    # JCommonsenseQA
    jcommonsenseqa_train_file = output_path / "jcommonsenseqa_train.jsonl"
    jcommonsenseqa_valid_file = output_path / "jcommonsenseqa_valid.jsonl"
    print(f"\nSaving JCommonsenseQA train data to {jcommonsenseqa_train_file}...")
    save_jsonl(jcommonsenseqa_train, jcommonsenseqa_train_file)
    print(f"Saving JCommonsenseQA valid data to {jcommonsenseqa_valid_file}...")
    save_jsonl(jcommonsenseqa_valid, jcommonsenseqa_valid_file)
 
    print("\n✓ Done!")
    return str(jcommonsenseqa_train_file)
 
 
if __name__ == "__main__":
    base_dir = Path(__file__).parent
    print(base_dir)
    import ipdb; ipdb.set_trace()
 
    # 元のファイル名は環境に合わせて変更してください
    input_file = base_dir / "with_seed_data_100.jsonl"
    output_dir = base_dir / "simple_format_arrange"
 
    print("=" * 60)
    print("Converting to simple format with balanced jcommonsenseqa labels...")
    print("=" * 60)
 
    jcommonsenseqa_file = convert_data(
        str(input_file),
        str(output_dir)
    )
 
    print(f"\nOutput files:")
    print(f"  JCommonsenseQA:     {jcommonsenseqa_file}")



