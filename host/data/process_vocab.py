import random

def process_vocab_file(input_file, output_file, mode='sort'):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 去除空白與換行
    lines = [line.strip() for line in lines if line.strip()]

    # 剔除英文重複的單字
    seen = set()
    unique_lines = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) < 2:
            continue  # 忽略格式錯誤的行
        eng = parts[0].strip().lower()
        if eng not in seen:
            seen.add(eng)
            unique_lines.append(line)

    # 處理排序或隨機打亂
    if mode == 'sort':
        unique_lines.sort(key=lambda x: x.split('\t')[0].lower())
    elif mode == 'sort_len':
        unique_lines.sort(key=lambda x: len(x.split('\t')[0].strip()))
    elif mode == 'shuffle':
        random.shuffle(unique_lines)

    with open(output_file, "w", encoding="utf-8") as f:
        for line in unique_lines:
            f.write(line + '\n')

    print(f"✅ 已完成：{mode}，共 {len(unique_lines)} 個不重複單字 → {output_file}")

# 範例用法（自行修改路徑或模式）
process_vocab_file("vocab.txt", "vocab_sort.txt", mode='sort')    # 字母排序
process_vocab_file("vocab.txt", "vocab_sort_len.txt", mode='sort_len')  # 隨機打亂
process_vocab_file("vocab.txt", "vocab_shuffled.txt", mode='shuffle')  # 隨機打亂

