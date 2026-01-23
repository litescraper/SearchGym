import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.name_prompts import country_names, city_names, first_names, last_names, university_names, company_names, language_names
from utils.llm_utils_ds import call_model
import json
import argparse

def ensure_unique_names(all_generated_names):
    """Ensure all names across all categories are unique"""
    seen_names = set()
    unique_results = {}
    statistics = {}

    for category, names_list in all_generated_names.items():
        unique_names = []
        duplicate_count = 0

        for name in names_list:
            if name not in seen_names:
                seen_names.add(name)
                unique_names.append(name)
            else:
                duplicate_count += 1

        unique_results[category] = unique_names
        statistics[category] = {
            'original_count': len(names_list),
            'duplicate_count': duplicate_count,
            'unique_count': len(unique_names)
        }

    return unique_results, statistics

def generate_first_and_last_names(count, existing_names=None):
    """生成first names和last names，然后组合成full names"""
    import random

    if existing_names is None:
        existing_names = set()

    # Add existing names to the prompt to avoid duplicates
    existing_names_str = ", ".join(list(existing_names)[:20]) if existing_names else "none"

    # 生成first names，多生成一些以便有更多选择
    first_names_count = int(count * 1.5)  # 多生成50%的first names
    first_names_prompt = first_names.format(count=first_names_count, existing_names=existing_names_str)
    first_names_data = call_model(first_names_prompt, response_format={"type": "json_object"}, model="gemini-2.5-pro")
    first_names_list = first_names_data.get('names', [])

    # 生成last names，多生成一些以便有更多组合选择
    last_names_count = int(count * 1.5)  # 多生成50%的last names
    last_names_prompt = last_names.format(count=last_names_count, existing_names=existing_names_str)
    last_names_data = call_model(last_names_prompt, response_format={"type": "json_object"}, model="gemini-2.5-pro")
    last_names_list = last_names_data.get('names', [])

    # 打乱顺序确保随机性
    random.shuffle(first_names_list)
    random.shuffle(last_names_list)

    return first_names_list, last_names_list

def combine_names(first_names_list, last_names_list, target_count):
    """将first names和last names随机组合成full names，避免同性或同名过多"""
    import random

    full_names = []
    used_combinations = set()

    # 随机打乱两个列表的顺序
    first_names_shuffled = first_names_list.copy()
    last_names_shuffled = last_names_list.copy()
    random.shuffle(first_names_shuffled)
    random.shuffle(last_names_shuffled)

    # 使用多种策略组合名字
    attempts = 0
    max_attempts = target_count * 10  # 避免无限循环

    while len(full_names) < target_count and attempts < max_attempts:
        attempts += 1

        # 策略1: 完全随机选择 (70%概率)
        if random.random() < 0.7:
            first_name = random.choice(first_names_list)
            last_name = random.choice(last_names_list)
        # 策略2: 从打乱的列表中顺序选择 (20%概率)
        elif random.random() < 0.9:
            first_idx = len(full_names) % len(first_names_shuffled)
            last_idx = (len(full_names) + len(first_names_shuffled) // 2) % len(last_names_shuffled)
            first_name = first_names_shuffled[first_idx]
            last_name = last_names_shuffled[last_idx]
        # 策略3: 交叉选择 (10%概率)
        else:
            first_idx = len(full_names) % len(first_names_list)
            last_idx = (len(first_names_list) - 1 - first_idx) % len(last_names_list)
            first_name = first_names_list[first_idx]
            last_name = last_names_list[last_idx]

        full_name = f"{first_name} {last_name}"

        # 确保组合唯一
        if full_name not in used_combinations:
            used_combinations.add(full_name)
            full_names.append(full_name)

    # 如果还没达到目标数量，使用更激进的随机策略
    while len(full_names) < target_count:
        # 重新打乱列表
        random.shuffle(first_names_list)
        random.shuffle(last_names_list)

        # 创建新的组合
        for i in range(min(len(first_names_list), len(last_names_list), target_count - len(full_names))):
            first_name = first_names_list[i]
            last_name = last_names_list[(i + len(first_names_list) // 3) % len(last_names_list)]
            full_name = f"{first_name} {last_name}"

            if full_name not in used_combinations:
                used_combinations.add(full_name)
                full_names.append(full_name)

                if len(full_names) >= target_count:
                    break

    return full_names[:target_count]

def generate_names(type: str, count: dict, existing_names=None):
    if existing_names is None:
        existing_names = set()

    # Add existing names to the prompt to avoid duplicates
    existing_names_str = ", ".join(list(existing_names)[:20]) if existing_names else "none"

    if type == "country":
        country_names_prompt = country_names.format(count=count.get("country"), existing_names=existing_names_str)
        return call_model(country_names_prompt, response_format={"type": "json_object"}, model="gemini-2.5-pro")
    elif type == "city":
        city_names_prompt = city_names.format(count=count.get("city"), existing_names=existing_names_str)
        return call_model(city_names_prompt, response_format={"type": "json_object"}, model="gemini-2.5-pro")
    elif type == "person":
        # 固定生成100个first names和100个last names
        first_count = 100
        last_count = 100

        first_names_list, last_names_list = generate_first_and_last_names(
            max(first_count, last_count), existing_names
        )

        # 组合生成full names，严格按照配置的数量来生成
        target_count = count.get("person")
        full_names = combine_names(first_names_list, last_names_list, target_count)

        return {"names": full_names}
    elif type == "university":
        university_names_prompt = university_names.format(count=count.get("university"), existing_names=existing_names_str)
        return call_model(university_names_prompt, response_format={"type": "json_object"}, model="gemini-2.5-pro")
    elif type == "company":
        company_names_prompt = company_names.format(count=count.get("company"), existing_names=existing_names_str)
        return call_model(company_names_prompt, response_format={"type": "json_object"}, model="gemini-2.5-pro")
    elif type == "language":
        language_names_prompt = language_names.format(count=count.get("language"), existing_names=existing_names_str)
        return call_model(language_names_prompt, response_format={"type": "json_object"}, model="gemini-2.5-pro")
    else:
        raise ValueError(f"Invalid type: {type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, nargs='+', default="all", choices=["country", "city", "person", "university", "company", "language", "all"])
    parser.add_argument("--count", type=dict, default={"country": 200, "city": 400, "person": 2000, "university": 200, "company": 200, "language": 50})
    args = parser.parse_args()

    if "all" in args.type:
        types = ["country", "city", "person", "university", "company", "language"]
    else:
        types = args.type

    os.makedirs("outputs/names", exist_ok=True)

    # Track all generated names to ensure uniqueness across categories
    all_generated_names = {}
    existing_names = set()

    # Generate names for each type
    for t in types:
        names_data = generate_names(t, args.count, existing_names)

        # Handle case where LLM returns None (failed API call)
        if names_data is None:
            print(f"ERROR: Failed to generate names for {t}. Skipping this category.")
            all_generated_names[t] = []
            continue

        all_generated_names[t] = names_data["names"]

        # Update existing names with newly generated names
        existing_names.update(names_data["names"])

        print(f"Generated {len(names_data['names'])} names for {t}")

        # Save individual category to json file
        save_path = f"outputs/names/names_{t}.json"
        with open(save_path, "w") as f:
            json.dump(names_data, f)

    # Ensure uniqueness across all categories and get statistics
    unique_results, statistics = ensure_unique_names(all_generated_names)

    # Save individual category files with unique names (overwrite original files)
    for category, unique_names in unique_results.items():
        save_path = f"outputs/names/names_{category}.json"
        unique_data = {"names": unique_names}
        with open(save_path, "w") as f:
            json.dump(unique_data, f, indent=2)
        print(f"Overwrote {save_path} with {len(unique_names)} unique names")

    # Print detailed statistics
    print("\n" + "="*50)
    print("DUPLICATE REMOVAL STATISTICS")
    print("="*50)

    total_original = 0
    total_duplicates = 0
    total_unique = 0

    for category, stats in statistics.items():
        original = stats['original_count']
        duplicates = stats['duplicate_count']
        unique = stats['unique_count']

        total_original += original
        total_duplicates += duplicates
        total_unique += unique

        print(f"{category.upper()}:")
        print(f"  Original:  {original:4d} names")
        print(f"  Duplicates: {duplicates:4d} removed")
        print(f"  Unique:    {unique:4d} retained")
        print()

    print(f"TOTAL:")
    print(f"  Original:  {total_original:4d} names")
    print(f"  Duplicates: {total_duplicates:4d} removed")
    print(f"  Unique:    {total_unique:4d} retained")

    if total_duplicates > 0:
        print(f"\nSuccessfully removed {total_duplicates} duplicate names across all categories!")
    else:
        print(f"\nNo duplicates found. All {total_original} names are unique.")