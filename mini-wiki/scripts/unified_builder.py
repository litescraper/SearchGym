#!/usr/bin/env python3
"""
统一的知识图谱构建脚本
包含所有步骤：名称生成、节点构建、知识图谱构建、路径生成、复杂路径生成
"""
import sys
import os
import json
import random
import argparse
from datetime import datetime
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.name_prompts import country_names, city_names, first_names, last_names, university_names, company_names, language_names
from utils.llm_utils_ds import call_model
from utils.build_utils import load_schema

# =============================================================================
# 统一配置参数
# =============================================================================

UNIFIED_CONFIG = {
    # 名称生成配置
    "name_generation": {
        "country": 200,      # 国家名称数量 100
        "city": 400,         # 城市名称数量 200
        "person": 2000,      # 人物名称数量 800
        "university": 200,   # 大学名称数量 100
        "company": 200,      # 公司名称数量 100
        "language": 50       # 语言名称数量 50
    },
    
    "path_generation": {
        "forward_paths_need_access": {
            "1-hop": 6000,        # 1-hop路径数量 (need_access=true)
            "2-hop": 5000,        # 2-hop路径数量 (need_access=true)
            "3-hop": 5000,        # 3-hop路径数量 (need_access=true)
            "4-hop": 5000,        # 4-hop路径数量 (need_access=true)
            "5-hop": 5000,        # 5-hop路径数量 (need_access=true)
            "6-hop": 5000         # 6-hop路径数量 (need_access=true)
        },
        "forward_paths_no_need_access": {
            "1-hop": 6000,        # 1-hop路径数量 (need_access=false)
            "2-hop": 5000,        # 2-hop路径数量 (need_access=false)
            "3-hop": 4000,        # 3-hop路径数量 (need_access=false)
            "4-hop": 3000,        # 4-hop路径数量 (need_access=false)
            "5-hop": 3000,        # 5-hop路径数量 (need_access=false)
            "6-hop": 2000         # 6-hop路径数量 (need_access=false)
        }
    },

    # 复杂路径生成配置 - 并行路径
    "complex_paths": {
        # 相同数值类型的并行路径
        "parallel_same_type_numerical": {
            "single_hop": 1000,        # 两个单跳路径
            "single_multi_hop": 1000,  # 一个单跳 + 一个多跳
            "multi_hop": 1000          # 两个多跳路径
        },

        # 相同实体类型的并行路径
        "parallel_same_type_entity": {
            "single_hop": 1000,        # 两个单跳路径
            "single_multi_hop": 1000,  # 一个单跳 + 一个多跳
            "multi_hop": 1000          # 两个多跳路径
        },

        # 不同类型的并行路径
        "parallel_different_type": {
            "single_hop": 1000,        # 两个单跳路径
            "single_multi_hop": 1000,  # 一个单跳 + 一个多跳
            "multi_hop": 1000          # 两个多跳路径
        },

        # 组合路径
        "combo": {
            "7_hop": 1000,       # 7-hop组合路径
            "8_hop": 1000,       # 8-hop组合路径
            "9_hop": 1000,       # 9-hop组合路径
            "10_hop": 1000,      # 10-hop组合路径
            "11_hop": 1000,      # 11-hop组合路径
            "12_hop": 1000       # 12-hop组合路径
        }
    },

    # 文件路径配置
    "file_paths": {
        "names_dir": "outputs/names/",
        "nodes_dir": "outputs/nodes/",
        "edges_dir": "outputs/edges/",
        "paths_dir": "outputs/paths/",
        "complex_paths_dir": "outputs/complex_paths/",
        "qa_dir": "outputs/qa/",
        "summary_dir": "outputs/summary/"
    },

    # QA生成配置
    "qa_generation": {
        "generate_path": True,      # 是否生成路径（包括简单路径和复杂路径）
        "generate_qa": False        # 是否生成QA（需要先生成路径）
    },

    # Simple QA模板配置
    "simple_qa_templates_config": {
        "1-hop": {
            "simple_qa_prompt_all_template": 0.3,
            "simple_qa_prompt_scenario_1_hop_template": 0.3,
            "simple_qa_prompt_123_hop_template": 0.4
        },
        "2-hop": {
            "simple_qa_prompt_all_template": 0.3,
            "simple_qa_prompt_scenario_2_hop_template": 0.3,
            "simple_qa_prompt_123_hop_template": 0.4
        },
        "3-hop": {
            "simple_qa_prompt_all_template": 0.4,
            "simple_qa_prompt_scenario_3_hop_template": 0.2,
            "simple_qa_prompt_123_hop_template": 0.4
        },
        "4-hop": {
            "simple_qa_prompt_all_template": 0.5,
            "simple_qa_prompt_456_hop_template": 0.5
        },
        "5-hop": {
            "simple_qa_prompt_all_template": 0.5,
            "simple_qa_prompt_456_hop_template": 0.5
        },
        "6-hop": {
            "simple_qa_prompt_all_template": 0.5,
            "simple_qa_prompt_456_hop_template": 0.5
        }
    },

    # 其他配置参数
    "misc": {
        "random_seed": 42,          # 随机种子
        "batch_size": 50,           # 批处理大小
        "max_attempts": 20000,       # 最大尝试次数
        "abstracts_selection_ratio": 0.5  # 选择到abstracts的属性比例
    }
}

# 数值类型定义
NUMERICAL_TYPES = {
    'Year', 'GDP', 'Population Count', 'Area', 'University Rank', 'Student Count',
    'Professor Count', 'Research Project Count', 'Research Publication Count', 'Research Award Count',
    'Research Patent Count', 'Employee Count', 'Department Count', 'Product Count',
    'Franchise Count', 'Franchise Revenue', 'University Count', 'Company Count', 'Ethic Group Count'
}

# 节点类型定义
NODE_TYPES = ["Person", "Company", "Country", "City", "University"]

# =============================================================================
# 配置访问函数
# =============================================================================

def get_name_generation_config():
    """获取名称生成配置"""
    return UNIFIED_CONFIG["name_generation"]

def get_path_generation_config():
    """获取路径生成配置"""
    return UNIFIED_CONFIG["path_generation"]

def get_complex_paths_config():
    """获取复杂路径生成配置"""
    return UNIFIED_CONFIG["complex_paths"]

def get_file_paths_config():
    """获取文件路径配置"""
    return UNIFIED_CONFIG["file_paths"]

def get_misc_config():
    """获取其他配置"""
    return UNIFIED_CONFIG["misc"]

def get_qa_generation_config():
    """获取QA生成配置"""
    return UNIFIED_CONFIG["qa_generation"]

def get_simple_qa_templates_config():
    """获取Simple QA模板配置"""
    return UNIFIED_CONFIG["simple_qa_templates_config"]

def update_config(config_path, value):
    """更新配置值，支持点分隔的路径，如 'name_generation.country' """
    keys = config_path.split('.')
    config = UNIFIED_CONFIG

    # 导航到目标位置
    for key in keys[:-1]:
        if key not in config:
            raise KeyError(f"配置路径 '{config_path}' 不存在")
        config = config[key]

    # 设置最终值
    final_key = keys[-1]
    if final_key not in config:
        raise KeyError(f"配置键 '{final_key}' 不存在于路径 '{config_path}' 中")

    config[final_key] = value
    print(f"配置已更新: {config_path} = {value}")

def names_folder_exists():
    """检查outputs/names文件夹是否存在（保留向后兼容）"""
    names_dir = get_file_paths_config()["names_dir"]
    return os.path.exists(names_dir)

def check_names_files_exist():
    """检查各个类别的名称文件是否都存在"""
    names_dir = get_file_paths_config()["names_dir"]
    name_types = ["country", "city", "person", "university", "company", "language"]

    missing_files = []
    existing_files = []

    for name_type in name_types:
        file_path = os.path.join(names_dir, f"names_{name_type}.json")
        if os.path.exists(file_path):
            existing_files.append(name_type)
        else:
            missing_files.append(name_type)

    return len(missing_files) == 0, missing_files, existing_files

def print_config():
    """打印所有配置"""
    print("当前配置:")
    print(json.dumps(UNIFIED_CONFIG, indent=2, ensure_ascii=False))

# =============================================================================
# 名称生成相关函数
# =============================================================================

def ensure_unique_names(all_generated_names):
    """确保所有名称跨类别都是唯一的"""
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
    if existing_names is None:
        existing_names = set()

    existing_names_str = ", ".join(list(existing_names)[:20]) if existing_names else "none"

    # 生成first names
    first_names_count = int(count * 1.5)
    first_names_prompt = first_names.format(count=first_names_count, existing_names=existing_names_str)
    first_names_data = call_model(first_names_prompt, response_format={"type": "json_object"})
    first_names_list = first_names_data.get('names', [])

    # 生成last names
    last_names_count = int(count * 1.5)
    last_names_prompt = last_names.format(count=last_names_count, existing_names=existing_names_str)
    last_names_data = call_model(last_names_prompt, response_format={"type": "json_object"})
    last_names_list = last_names_data.get('names', [])

    random.shuffle(first_names_list)
    random.shuffle(last_names_list)

    return first_names_list, last_names_list

def combine_names(first_names_list, last_names_list, target_count):
    """将first names和last names随机组合成full names"""
    full_names = []
    used_combinations = set()

    first_names_shuffled = first_names_list.copy()
    last_names_shuffled = last_names_list.copy()
    random.shuffle(first_names_shuffled)
    random.shuffle(last_names_shuffled)

    attempts = 0
    max_attempts = target_count * 10

    while len(full_names) < target_count and attempts < max_attempts:
        attempts += 1

        if random.random() < 0.7:
            first_name = random.choice(first_names_list)
            last_name = random.choice(last_names_list)
        elif random.random() < 0.9:
            first_idx = len(full_names) % len(first_names_shuffled)
            last_idx = (len(full_names) + len(first_names_shuffled) // 2) % len(last_names_shuffled)
            first_name = first_names_shuffled[first_idx]
            last_name = last_names_shuffled[last_idx]
        else:
            first_idx = len(full_names) % len(first_names_list)
            last_idx = (len(first_names_list) - 1 - first_idx) % len(last_names_list)
            first_name = first_names_list[first_idx]
            last_name = last_names_list[last_idx]

        full_name = f"{first_name} {last_name}"

        if full_name not in used_combinations:
            used_combinations.add(full_name)
            full_names.append(full_name)

    # 如果还没达到目标数量，使用更激进的随机策略
    while len(full_names) < target_count:
        random.shuffle(first_names_list)
        random.shuffle(last_names_list)

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

    existing_names_str = ", ".join(list(existing_names)[:20]) if existing_names else "none"

    if type == "country":
        country_names_prompt = country_names.format(count=count.get("country"), existing_names=existing_names_str)
        return call_model(country_names_prompt, response_format={"type": "json_object"})
    elif type == "city":
        city_names_prompt = city_names.format(count=count.get("city"), existing_names=existing_names_str)
        return call_model(city_names_prompt, response_format={"type": "json_object"})
    elif type == "person":
        first_count = 100
        last_count = 100

        first_names_list, last_names_list = generate_first_and_last_names(
            max(first_count, last_count), existing_names
        )

        target_count = count.get("person", 800)
        full_names = combine_names(first_names_list, last_names_list, target_count)

        return {"names": full_names}
    elif type == "university":
        university_names_prompt = university_names.format(count=count.get("university"), existing_names=existing_names_str)
        return call_model(university_names_prompt, response_format={"type": "json_object"})
    elif type == "company":
        company_names_prompt = company_names.format(count=count.get("company"), existing_names=existing_names_str)
        return call_model(company_names_prompt, response_format={"type": "json_object"})
    elif type == "language":
        language_names_prompt = language_names.format(count=count.get("language"), existing_names=existing_names_str)
        return call_model(language_names_prompt, response_format={"type": "json_object"})
    else:
        raise ValueError(f"Invalid type: {type}")

def step1_generate_names():
    """步骤1：生成名称"""
    print("=== 步骤1：生成名称 ===")

    config = UNIFIED_CONFIG["name_generation"]
    types = ["country", "city", "person", "university", "company", "language"]

    os.makedirs(UNIFIED_CONFIG["file_paths"]["names_dir"], exist_ok=True)

    all_generated_names = {}
    existing_names = set()

    # 生成各类名称
    for t in types:
        print(f"生成 {t} 名称...")
        names_data = generate_names(t, config, existing_names)
        all_generated_names[t] = names_data["names"]
        existing_names.update(names_data["names"])

        print(f"生成了 {len(names_data['names'])} 个 {t} 名称")

        # 保存单独的类别文件
        save_path = os.path.join(UNIFIED_CONFIG["file_paths"]["names_dir"], f"names_{t}.json")
        with open(save_path, "w") as f:
            json.dump(names_data, f)

    # 确保唯一性
    unique_results, statistics = ensure_unique_names(all_generated_names)

    # 保存唯一名称文件（覆盖原文件）
    for category, unique_names in unique_results.items():
        save_path = os.path.join(UNIFIED_CONFIG["file_paths"]["names_dir"], f"names_{category}.json")
        unique_data = {"names": unique_names}
        with open(save_path, "w") as f:
            json.dump(unique_data, f, indent=2)
        print(f"更新 {save_path} 包含 {len(unique_names)} 个唯一名称")

    print("名称生成完成！\n")

# =============================================================================
# 节点构建相关函数
# =============================================================================

def load_names_from_json(filepath):
    """从JSON文件中加载名称列表"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('names', [])
    except FileNotFoundError:
        print(f"错误: 文件未找到 {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"错误: 文件格式不正确 {filepath}")
        return []

def generate_national_day():
    """生成唯一的国庆日"""
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    days = list(range(1, 29))
    return f"{random.choice(months)} {random.choice(days)}th"

def calculate_university_rank(students, professors, projects, publications, awards, patents):
    """根据多个因素计算大学排名"""
    student_score = min(students / 50000, 1.0) * 0.2
    professor_score = min(professors / 1000, 1.0) * 0.2
    project_score = min(projects / 1000, 1.0) * 0.15
    publication_score = min(publications / 10000, 1.0) * 0.2
    award_score = min(awards / 100, 1.0) * 0.15
    patent_score = min(patents / 5000, 1.0) * 0.1

    total_score = student_score + professor_score + project_score + publication_score + award_score + patent_score
    return max(1, int(1000 * (1 - total_score)))

def select_random_attributes(attributes_dict):
    """根据配置比例随机选择属性"""
    if not attributes_dict:
        return {}

    total_attrs = len(attributes_dict)
    selection_ratio = UNIFIED_CONFIG["misc"]["abstracts_selection_ratio"]
    num_to_select = max(1, int(total_attrs * selection_ratio))

    selected_keys = random.sample(list(attributes_dict.keys()), num_to_select)
    return {key: attributes_dict[key] for key in selected_keys}

class CompleteNodeGenerator:
    def __init__(self, schema):
        self.schema = schema
        self.current_year = 2025

        self.used_values = {
            'country_pop': set(), 'country_gdp': set(), 'country_area': set(),
            'city_pop': set(), 'city_area': set(), 'city_gdp': set(),
            'university_students': set(), 'university_rank': set(),
            'national_day': set(), 'company_employees': set()
        }

        self.countries = []
        self.cities = []
        self.universities = []
        self.companies = []
        self.persons = []

        self.sister_country_pairs = set()
        self.sister_city_pairs = set()
        self.spouse_pairs = set()
        self.best_friend_pairs = set()

        self.country_leaders = {}
        self.city_mayors = {}

    def generate_nodes(self):
        """生成所有节点"""
        print("开始生成节点...")

        self.countries = self.create_country_nodes()
        self.cities = self.create_city_nodes()
        self.universities = self.create_university_nodes()
        self.companies = self.create_company_nodes()
        self.persons = self.create_person_nodes()

        print(f"生成完成: {len(self.countries)}个国家, {len(self.cities)}个城市, "
              f"{len(self.universities)}所大学, {len(self.companies)}家公司, {len(self.persons)}个人")

    def create_country_nodes(self):
        """创建国家节点"""
        country_names = load_names_from_json(os.path.join(UNIFIED_CONFIG["file_paths"]["names_dir"], 'names_country.json'))
        language_names = load_names_from_json(os.path.join(UNIFIED_CONFIG["file_paths"]["names_dir"], 'names_language.json'))
        num_countries = len(country_names)
        selected_names = random.sample(country_names, num_countries)

        nodes = []

        for i, country_name in enumerate(selected_names):
            node = {
                "node_id": f"country_{i}",
                "node_type": "Country",
                "name": country_name,
                "attributes": {}
            }

            # 必有属性
            population = random.randint(100000000, 1000000000)
            while population in self.used_values['country_pop']:
                population += 1
            self.used_values['country_pop'].add(population)
            node['attributes']['population'] = {"name": population, "type": "Population Count"}

            gdp = int(population * random.uniform(10000, 50000))
            while gdp in self.used_values['country_gdp']:
                gdp += 1
            self.used_values['country_gdp'].add(gdp)
            node['attributes']['gdp'] = {"name": gdp, "type": "GDP"}

            area = int(population * random.uniform(0.01, 0.1))
            while area in self.used_values['country_area']:
                area += 1
            self.used_values['country_area'].add(area)
            node['attributes']['total_area'] = {"name": area, "type": "Area"}

            node['attributes']['official_language'] = {"name": random.choice(language_names), "type": "Language"}

            if random.random() < 0.4:
                node['attributes']['number_of_ethnic_groups'] = {"name": random.randint(1, 10), "type": "Ethic Group Count"}

            national_day = generate_national_day()
            while national_day in self.used_values['national_day']:
                national_day = generate_national_day()
            self.used_values['national_day'].add(national_day)
            node['attributes']['national_day'] = {"name": national_day, "type": "Date"}

            node['attributes_in_abstracts'] = select_random_attributes(node['attributes'])
            nodes.append(node)

        self._setup_sister_country_relationships(nodes)
        return nodes

    def create_city_nodes(self):
        """创建城市节点"""
        city_names = load_names_from_json(os.path.join(UNIFIED_CONFIG["file_paths"]["names_dir"], 'names_city.json'))
        num_cities = len(city_names)
        selected_names = random.sample(city_names, num_cities)

        nodes = []
        capital_cities = set()

        # 为每个国家分配一个首都
        for country in self.countries:
            capital_city_name = random.choice(selected_names)
            selected_names.remove(capital_city_name)

            node = {
                "node_id": f"city_{len(nodes)}",
                "node_type": "City",
                "name": capital_city_name,
                "attributes": {}
            }

            population = random.randint(5000000, 30000000)
            while population in self.used_values['city_pop']:
                population += 1
            self.used_values['city_pop'].add(population)
            node['attributes']['population'] = {"name": population, "type": "Population Count"}

            area = int(population * random.uniform(0.001, 0.01))
            while area in self.used_values['city_area']:
                area += 1
            self.used_values['city_area'].add(area)
            node['attributes']['area'] = {"name": area, "type": "Area"}

            node['attributes']['located_country'] = {
                "name": country['name'],
                "type": "Country",
                "node_id": country['node_id']
            }

            gdp = int(population * random.uniform(5000, 20000))
            while gdp in self.used_values['city_gdp']:
                gdp += 1
            self.used_values['city_gdp'].add(gdp)
            node['attributes']['gdp'] = {"name": gdp, "type": "GDP"}

            node['attributes']['capital_of'] = {
                "name": country['name'],
                "type": "Country",
                "node_id": country['node_id']
            }
            capital_cities.add(node['node_id'])

            num_universities = max(1, int(population / 2000000))
            node['attributes']['number_of_universities'] = {"name": num_universities, "type": "University Count"}

            num_companies = max(1, int(gdp / 100000000))
            node['attributes']['number_of_companies'] = {"name": num_companies, "type": "Company Count"}

            node['attributes_in_abstracts'] = select_random_attributes(node['attributes'])
            nodes.append(node)

        # 生成其他城市
        for city_name in selected_names[:num_cities - len(self.countries)]:
            node = {
                "node_id": f"city_{len(nodes)}",
                "node_type": "City",
                "name": city_name,
                "attributes": {}
            }

            population = random.randint(100000, 10000000)
            while population in self.used_values['city_pop']:
                population += 1
            self.used_values['city_pop'].add(population)
            node['attributes']['population'] = {"name": population, "type": "Population Count"}

            area = int(population * random.uniform(0.001, 0.01))
            while area in self.used_values['city_area']:
                area += 1
            self.used_values['city_area'].add(area)
            node['attributes']['area'] = {"name": area, "type": "Area"}

            country = random.choice(self.countries)
            node['attributes']['located_country'] = {
                "name": country['name'],
                "type": "Country",
                "node_id": country['node_id']
            }

            gdp = int(population * random.uniform(5000, 20000))
            while gdp in self.used_values['city_gdp']:
                gdp += 1
            self.used_values['city_gdp'].add(gdp)
            node['attributes']['gdp'] = {"name": gdp, "type": "GDP"}

            num_universities = max(1, int(population / 2000000))
            num_companies = max(1, int(gdp / 100000000))
            node['attributes']['number_of_universities'] = {"name": num_universities, "type": "University Count"}
            node['attributes']['number_of_companies'] = {"name": num_companies, "type": "Company Count"}

            node['attributes_in_abstracts'] = select_random_attributes(node['attributes'])
            nodes.append(node)

        self._setup_sister_city_relationships(nodes)
        return nodes

    def create_university_nodes(self):
        """创建大学节点"""
        university_names = load_names_from_json(os.path.join(UNIFIED_CONFIG["file_paths"]["names_dir"], 'names_university.json'))
        num_universities = len(university_names)
        selected_names = random.sample(university_names, num_universities)

        nodes = []

        for i, university_name in enumerate(selected_names):
            node = {
                "node_id": f"university_{i}",
                "node_type": "University",
                "name": university_name,
                "attributes": {}
            }

            city = random.choice(self.cities)
            node['attributes']['located_city'] = {
                "name": city['name'],
                "type": "City",
                "node_id": city['node_id']
            }

            students = random.randint(1000, 50000)
            while students in self.used_values['university_students']:
                students += 1
            self.used_values['university_students'].add(students)
            node['attributes']['registered_students'] = {"name": students, "type": "Student Count"}

            faculties = max(3, int(students / 5000))
            node['attributes']['number_of_faculties'] = {"name": faculties, "type": "Faculty Count"}

            professors = max(50, int(students / 100))
            node['attributes']['number_of_professors'] = {"name": professors, "type": "Professor Count"}

            projects = random.randint(professors, professors * 3)
            node['attributes']['number_of_research_projects'] = {"name": projects, "type": "Research Project Count"}

            publications = random.randint(professors * 10, professors * 50)
            node['attributes']['number_of_research_publications'] = {"name": publications, "type": "Research Publication Count"}

            awards = random.randint(max(1, professors // 10), professors // 2)
            node['attributes']['number_of_research_awards'] = {"name": awards, "type": "Research Award Count"}

            patents = random.randint(max(1, professors // 5), professors * 2)
            node['attributes']['number_of_research_patents'] = {"name": patents, "type": "Research Patent Count"}

            established_year = random.randint(1600, 2000)
            node['attributes']['established_year'] = {"name": established_year, "type": "Year"}

            rank = calculate_university_rank(students, professors, projects, publications, awards, patents)
            while rank in self.used_values['university_rank']:
                rank += 1
            self.used_values['university_rank'].add(rank)
            node['attributes']['university_rank'] = {"name": rank, "type": "University Rank"}

            node['attributes_in_abstracts'] = select_random_attributes(node['attributes'])
            nodes.append(node)

        return nodes

    def create_company_nodes(self):
        """创建公司节点"""
        company_names = load_names_from_json(os.path.join(UNIFIED_CONFIG["file_paths"]["names_dir"], 'names_company.json'))
        num_companies = len(company_names)
        selected_names = random.sample(company_names, num_companies)

        nodes = []

        for i, company_name in enumerate(selected_names):
            node = {
                "node_id": f"company_{i}",
                "node_type": "Company",
                "name": company_name,
                "attributes": {}
            }

            city = random.choice(self.cities)
            node['attributes']['headquarter_city'] = {
                "name": city['name'],
                "type": "City",
                "node_id": city['node_id']
            }

            employees = random.randint(50, 50000)
            while employees in self.used_values['company_employees']:
                employees += 1
            self.used_values['company_employees'].add(employees)
            node['attributes']['number_of_employees'] = {"name": employees, "type": "Employee Count"}

            departments = max(1, int(employees / 1000))
            node['attributes']['number_of_departments'] = {"name": departments, "type": "Department Count"}

            products = random.randint(1, 100)
            node['attributes']['number_of_products'] = {"name": products, "type": "Product Count"}

            franchise_count = random.randint(products // 2, products * 5)
            node['attributes']['franchise_count'] = {"name": franchise_count, "type": "Franchise Count"}

            franchise_revenue = random.randint(products * 1000000, products * 10000000)
            node['attributes']['franchise_revenue'] = {"name": franchise_revenue, "type": "Franchise Revenue"}

            founded_year = random.randint(1800, 2020)
            node['attributes']['founded_year'] = {"name": founded_year, "type": "Year"}

            node['attributes_in_abstracts'] = select_random_attributes(node['attributes'])
            nodes.append(node)

        return nodes

    def create_person_nodes(self):
        """创建人物节点"""
        person_names = load_names_from_json(os.path.join(UNIFIED_CONFIG["file_paths"]["names_dir"], 'names_person.json'))
        num_persons = len(person_names)
        selected_names = random.sample(person_names, num_persons)

        nodes = []

        for i, person_name in enumerate(selected_names):
            node = {
                "node_id": f"person_{i}",
                "node_type": "Person",
                "name": person_name,
                "attributes": {}
            }

            birth_year = random.randint(1950, 2010)
            node['attributes']['birth_year'] = {"name": birth_year, "type": "Year"}
            age = self.current_year - birth_year

            gender = random.choice(['Male', 'Female'])
            node['attributes']['gender'] = {"name": gender, "type": "Gender"}

            birth_city = random.choice(self.cities)
            node['attributes']['birth_city'] = {
                "name": birth_city['name'],
                "type": "City",
                "node_id": birth_city['node_id']
            }

            current_cities = [c for c in self.cities if c['node_id'] != birth_city['node_id']]
            if current_cities:
                current_city = random.choice(current_cities)
                node['attributes']['current_living_city'] = {
                    "name": current_city['name'],
                    "type": "City",
                    "node_id": current_city['node_id']
                }

            if 16 <= age <= 70 and random.random() < 0.6:
                university = random.choice(self.universities)
                node['attributes']['university_attended'] = {
                    "name": university['name'],
                    "type": "University",
                    "node_id": university['node_id']
                }

            if 22 <= age <= 65 and random.random() < 0.7:
                company = random.choice(self.companies)
                node['attributes']['current_working_company'] = {
                    "name": company['name'],
                    "type": "Company",
                    "node_id": company['node_id']
                }

            node['attributes_in_abstracts'] = select_random_attributes(node['attributes'])
            nodes.append(node)

        self._setup_family_relationships(nodes)
        self._setup_spouse_relationships(nodes)
        self._setup_leadership_positions(nodes)
        self._setup_best_friend_relationships(nodes)

        return nodes

    def _setup_sister_country_relationships(self, countries):
        """设置姐妹国家关系"""
        for i in range(0, len(countries) - 1, 2):
            if i + 1 < len(countries):
                country1, country2 = countries[i], countries[i + 1]

                pair1 = (country1['node_id'], country2['node_id'])
                pair2 = (country2['node_id'], country1['node_id'])

                if pair1 not in self.sister_country_pairs and pair2 not in self.sister_country_pairs:
                    country1['attributes']['sister_country'] = {
                        "name": country2['name'],
                        "type": "Country",
                        "node_id": country2['node_id']
                    }
                    country2['attributes']['sister_country'] = {
                        "name": country1['name'],
                        "type": "Country",
                        "node_id": country1['node_id']
                    }
                    self.sister_country_pairs.add(pair1)
                    self.sister_country_pairs.add(pair2)

    def _setup_sister_city_relationships(self, cities):
        """设置姐妹城市关系"""
        for i in range(0, len(cities) - 1, 2):
            if i + 1 < len(cities):
                city1, city2 = cities[i], cities[i + 1]

                pair1 = (city1['node_id'], city2['node_id'])
                pair2 = (city2['node_id'], city1['node_id'])

                if pair1 not in self.sister_city_pairs and pair2 not in self.sister_city_pairs:
                    city1['attributes']['sister_city'] = {
                        "name": city2['name'],
                        "type": "City",
                        "node_id": city2['node_id']
                    }
                    city2['attributes']['sister_city'] = {
                        "name": city1['name'],
                        "type": "City",
                        "node_id": city1['node_id']
                    }
                    self.sister_city_pairs.add(pair1)
                    self.sister_city_pairs.add(pair2)

    def _setup_family_relationships(self, persons):
        """设置家庭关系"""
        persons_by_age = sorted(persons, key=lambda p: p['attributes']['birth_year']['name'])

        for i, person in enumerate(persons_by_age):
            age = self.current_year - person['attributes']['birth_year']['name']

            if age <= 40 and random.random() < 0.8:
                if 'father' not in person['attributes'] and 'mother' not in person['attributes']:
                    potential_fathers = [p for p in persons_by_age
                                    if p['attributes']['gender']['name'] == 'Male' and
                                    25 <= (p['attributes']['birth_year']['name'] - person['attributes']['birth_year']['name']) <= 50]

                    if potential_fathers:
                        father = random.choice(potential_fathers)
                        person['attributes']['father'] = {
                            "name": father['name'],
                            "type": "Person",
                            "node_id": father['node_id']
                        }

        for i, person in enumerate(persons_by_age):
            age = self.current_year - person['attributes']['birth_year']['name']
            if age <= 40 and random.random() < 0.8:
                if 'father' not in person['attributes'] and 'mother' not in person['attributes']:
                    potential_mothers = [p for p in persons_by_age
                                        if p['attributes']['gender']['name'] == 'Female' and
                                        25 <= (p['attributes']['birth_year']['name'] - person['attributes']['birth_year']['name']) <= 50]

                    if potential_mothers:
                        mother = random.choice(potential_mothers)
                        person['attributes']['mother'] = {
                            "name": mother['name'],
                            "type": "Person",
                            "node_id": mother['node_id']
                        }

    def _setup_spouse_relationships(self, persons):
        """设置配偶关系"""
        males = [p for p in persons if p['attributes']['gender']['name'] == 'Male' and self.current_year - p['attributes']['birth_year']['name'] >= 18]
        females = [p for p in persons if p['attributes']['gender']['name'] == 'Female' and self.current_year - p['attributes']['birth_year']['name'] >= 18]

        for male in males:
            if random.random() < 0.4:
                male_age = self.current_year - male['attributes']['birth_year']['name']
                potential_spouses = [f for f in females if
                                   abs((self.current_year - f['attributes']['birth_year']['name']) - male_age) <= 10 and
                                   'spouse' not in f['attributes']]

                if potential_spouses:
                    spouse = random.choice(potential_spouses)

                    pair1 = (male['node_id'], spouse['node_id'])
                    pair2 = (spouse['node_id'], male['node_id'])

                    if pair1 not in self.spouse_pairs and pair2 not in self.spouse_pairs:
                        male['attributes']['spouse'] = {
                            "name": spouse['name'],
                            "type": "Person",
                            "node_id": spouse['node_id']
                        }
                        spouse['attributes']['spouse'] = {
                            "name": male['name'],
                            "type": "Person",
                            "node_id": male['node_id']
                        }
                        self.spouse_pairs.add(pair1)
                        self.spouse_pairs.add(pair2)

    def _setup_leadership_positions(self, persons):
        """设置领导职位"""
        for country in self.countries:
            potential_leaders = [p for p in persons if
                               50 <= (self.current_year - p['attributes']['birth_year']['name']) <= 100]

            if potential_leaders:
                leader = random.choice(potential_leaders)
                leader['attributes']['leader_of_country'] = {
                    "name": country['name'],
                    "type": "Country",
                    "node_id": country['node_id']
                }
                self.country_leaders[country['node_id']] = leader['node_id']

        for city in self.cities:
            potential_mayors = [p for p in persons if
                             50 <= (self.current_year - p['attributes']['birth_year']['name']) <= 100 and
                             'leader_of_country' not in p['attributes']]

            if potential_mayors:
                mayor = random.choice(potential_mayors)
                mayor['attributes']['mayor_of_city'] = {
                    "name": city['name'],
                    "type": "City",
                    "node_id": city['node_id']
                }
                self.city_mayors[city['node_id']] = mayor['node_id']

    def _setup_best_friend_relationships(self, persons):
        """设置好朋友关系"""
        for i, person in enumerate(persons):
            if random.random() < 0.3:
                age = self.current_year - person['attributes']['birth_year']['name']

                potential_friends = [p for p in persons[i+1:] if
                                   abs((self.current_year - p['attributes']['birth_year']['name']) - age) <= 15]

                if potential_friends:
                    friend = random.choice(potential_friends)

                    if 'best_friend' not in person['attributes'] and 'best_friend' not in friend['attributes']:
                        pair1 = (person['node_id'], friend['node_id'])
                        pair2 = (friend['node_id'], person['node_id'])

                        if pair1 not in self.best_friend_pairs and pair2 not in self.best_friend_pairs:
                            person['attributes']['best_friend'] = {
                                "name": friend['name'],
                                "type": "Person",
                                "node_id": friend['node_id']
                            }
                            friend['attributes']['best_friend'] = {
                                "name": person['name'],
                                "type": "Person",
                                "node_id": person['node_id']
                            }
                            self.best_friend_pairs.add(pair1)
                            self.best_friend_pairs.add(pair2)

    def save_nodes(self):
        """保存节点到文件"""
        output_dir = UNIFIED_CONFIG["file_paths"]["nodes_dir"]
        os.makedirs(output_dir, exist_ok=True)

        node_groups = {
            'countries': self.countries,
            'cities': self.cities,
            'universities': self.universities,
            'companies': self.companies,
            'persons': self.persons
        }

        for group_name, nodes in node_groups.items():
            filename = os.path.join(output_dir, f'nodes_{group_name}.jsonl')
            with open(filename, 'w', encoding='utf-8') as f:
                for node in nodes:
                    f.write(json.dumps(node, ensure_ascii=False) + '\n')

        print(f"节点已保存到 {output_dir}")

def step2_build_nodes():
    """步骤2：构建节点"""
    print("=== 步骤2：构建节点 ===")

    schema = load_schema('schema_config.json')
    generator = CompleteNodeGenerator(schema)
    generator.generate_nodes()
    generator.save_nodes()

    print("节点构建完成！\n")

# =============================================================================
# 知识图谱构建相关函数
# =============================================================================

def load_nodes_batch(filepath, batch_size=None):
    """分批加载节点以减少内存使用"""
    if batch_size is None:
        batch_size = UNIFIED_CONFIG["misc"]["batch_size"]

    nodes = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                node = json.loads(line.strip())
                nodes.append(node)
                if len(nodes) >= batch_size:
                    yield nodes
                    nodes = []
            if nodes:
                yield nodes
    except FileNotFoundError:
        print(f"错误: 文件未找到 {filepath}")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件格式不正确 {filepath}")
        return

def load_node_attributes():
    """加载所有节点的attributes_in_abstracts映射"""
    print("加载节点attributes_in_abstracts映射...")

    node_attributes = {}
    file_paths = [
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_countries.jsonl'),
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_cities.jsonl'),
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_universities.jsonl'),
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_companies.jsonl'),
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_persons.jsonl')
    ]

    for filepath in file_paths:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        node = json.loads(line.strip())
                        node_id = node['node_id']

                        abstracts_attrs = set()
                        if 'attributes_in_abstracts' in node:
                            for attr_name in node['attributes_in_abstracts'].keys():
                                abstracts_attrs.add(attr_name)

                        node_attributes[node_id] = abstracts_attrs
            except Exception as e:
                print(f"警告: 加载 {filepath} 时出错: {e}")

    print(f"加载了 {len(node_attributes)} 个节点的attributes信息")
    return node_attributes

def check_need_access_for_edge(edge, node_attributes):
    """检查edge的need_access状态"""
    source_id = edge['source_id']
    target_id = edge['target_id']
    edge_type = edge['edge_type']
    source_type = edge['source_type']

    if source_type in ['City', 'Country', 'Person', 'Company', 'University']:
        if source_id in node_attributes:
            return edge_type not in node_attributes[source_id]

    if target_id in node_attributes:
        original_edge_type = edge_type
        if edge_type.endswith('_of'):
            original_edge_type = edge_type[:-3]

        reverse_to_original = {
            'birth_year_of': 'birth_year',
            'birth_city_of': 'birth_city',
            'current_living_city_of': 'current_living_city',
            'university_attended_by': 'university_attended',
            'current_working_company_of': 'current_working_company',
            'father_of': 'father',
            'mother_of': 'mother',
            'spouse_of': 'spouse',
            'leader_of_country_for': 'leader_of_country',
            'mayor_of_city_for': 'mayor_of_city',
            'best_friend_of': 'best_friend',
            'sister_country_of': 'sister_country',
            'sister_city_of': 'sister_city',
            'located_country_of': 'located_country',
            'capital_city': 'capital_of',
            'located_city_of': 'located_city',
            'headquarter_city_of': 'headquarter_city'
        }

        if edge_type in reverse_to_original:
            original_edge_type = reverse_to_original[edge_type]

        return original_edge_type not in node_attributes[target_id]

    return True

def get_reverse_relation_type(relation_type):
    """获取反向关系类型"""
    reverse_mapping = {
        'located_country': 'located_country_of',
        'capital_of': 'capital_city',
        'located_city': 'located_city_of',
        'headquarter_city': 'headquarter_city_of',
        'birth_city': 'birth_city_of',
        'current_living_city': 'current_living_city_of',
        'university_attended': 'university_attended_by',
        'current_working_company': 'current_working_company_of',
        'father': 'father_of',
        'mother': 'mother_of',
        'spouse': 'spouse_of',
        'leader_of_country': 'leader_of_country_for',
        'mayor_of_city': 'mayor_of_city_for',
        'best_friend': 'best_friend_of',
        'sister_country': 'sister_country_of',
        'sister_city': 'sister_city_of'
    }

    if relation_type in reverse_mapping:
        return reverse_mapping[relation_type]
    elif not relation_type.endswith('_of'):
        return relation_type + '_of'
    else:
        return relation_type[:-3]

def get_reverse_mapping_type(mapping_type):
    """反转映射类型"""
    if mapping_type == '1-1':
        return '1-1'
    elif mapping_type == '1-n':
        return 'n-1'
    elif mapping_type == 'n-1':
        return '1-n'
    else:
        return mapping_type

def build_forward_edges():
    """构建正向关系"""
    print("开始构建正向关系...")

    schema = load_schema('schema_config.json')

    attr_mapping = {}
    for entity_type in ['Country', 'City', 'University', 'Company', 'Person']:
        attrs = schema.get(f'{entity_type}_Attributes', [])
        for attr in attrs:
            attr_name = attr['name']
            mapping_type = attr['type'][2] if len(attr['type']) > 2 else '1-1'
            attr_mapping[attr_name] = mapping_type

    node_attributes = load_node_attributes()

    forward_edges = []
    edge_id_counter = 0

    file_paths = [
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_countries.jsonl'),
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_cities.jsonl'),
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_universities.jsonl'),
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_companies.jsonl'),
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_persons.jsonl')
    ]

    entity_map = {}

    print("构建实体映射...")
    for filepath in file_paths:
        if os.path.exists(filepath):
            for batch in load_nodes_batch(filepath, 100):
                for node in batch:
                    entity_map[node['node_id']] = {
                        'name': node['name'],
                        'type': node['node_type']
                    }

    print(f"加载了 {len(entity_map)} 个实体")

    print("构建边...")
    for filepath in file_paths:
        if os.path.exists(filepath):
            for batch_idx, batch in enumerate(load_nodes_batch(filepath, 50)):
                print(f"  处理 {filepath} 批次 {batch_idx + 1}...")

                for node in batch:
                    node_type = node['node_type']
                    attributes = node['attributes']

                    for attr_name, attr_value in attributes.items():
                        if isinstance(attr_value, dict) and 'node_id' in attr_value:
                            target_node_id = attr_value['node_id']
                            if target_node_id in entity_map:
                                target_entity = entity_map[target_node_id]

                                mapping_type = attr_mapping.get(attr_name, '1-1')

                                edge = {
                                    'edge_id': f'e_{edge_id_counter}',
                                    'edge_type': attr_name,
                                    'source_id': node['node_id'],
                                    'source_type': node_type,
                                    'source_name': node['name'],
                                    'target_id': target_node_id,
                                    'target_type': target_entity['type'],
                                    'target_name': target_entity['name'],
                                    'mapping_type': mapping_type
                                }

                                edge['need_access'] = check_need_access_for_edge(edge, node_attributes)
                                forward_edges.append(edge)
                                edge_id_counter += 1

                        elif isinstance(attr_value, dict) and 'name' in attr_value:
                            attr_value_name = attr_value['name']
                            target_type = attr_value.get('type', 'Value')

                            mapping_type = attr_mapping.get(attr_name, '1-1')

                            edge = {
                                'edge_id': f'e_{edge_id_counter}',
                                'edge_type': attr_name,
                                'source_id': node['node_id'],
                                'source_type': node_type,
                                'source_name': node['name'],
                                'target_id': f"attr_{attr_name}_{attr_value_name}",
                                'target_type': target_type,
                                'target_name': str(attr_value_name),
                                'mapping_type': mapping_type
                            }

                            edge['need_access'] = check_need_access_for_edge(edge, node_attributes)
                            forward_edges.append(edge)
                            edge_id_counter += 1

    print(f"正向关系构建完成，共 {len(forward_edges)} 条关系")
    return forward_edges

def build_reverse_edges(forward_edges, node_attributes):
    """构建反向关系"""
    print("开始构建反向关系...")

    reverse_edges = []
    edge_id_counter = len(forward_edges)

    reverse_mapping_stats = defaultdict(set)

    for edge in forward_edges:
        reverse_edge_type = get_reverse_relation_type(edge['edge_type'])
        reverse_mapping_stats[(edge['target_id'], reverse_edge_type)].add(edge['source_id'])

    print("生成反向关系...")
    for i, edge in enumerate(forward_edges):
        if i % 500 == 0:
            print(f"  处理反向关系 {i+1}/{len(forward_edges)}...")

        reverse_edge_type = get_reverse_relation_type(edge['edge_type'])

        key = (edge['target_id'], reverse_edge_type)
        actual_targets = reverse_mapping_stats.get(key, set())

        if len(actual_targets) == 1:
            actual_reverse_mapping = '1-1'
        else:
            actual_reverse_mapping = '1-n'

        reverse_edge = {
            'edge_id': f'e_{edge_id_counter}',
            'edge_type': reverse_edge_type,
            'source_id': edge['target_id'],
            'source_type': edge['target_type'],
            'source_name': edge['target_name'],
            'target_id': edge['source_id'],
            'target_type': edge['source_type'],
            'target_name': edge['source_name'],
            'mapping_type': actual_reverse_mapping
        }

        reverse_edge['need_access'] = check_need_access_for_edge(reverse_edge, node_attributes)
        reverse_edges.append(reverse_edge)
        edge_id_counter += 1

    print(f"反向关系构建完成，共 {len(reverse_edges)} 条关系")
    return reverse_edges

def save_edges(edges, filename):
    """保存边到文件"""
    print(f"保存边到 {filename}...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        for edge in edges:
            f.write(json.dumps(edge, ensure_ascii=False) + '\n')

def generate_summary(forward_edges, reverse_edges):
    """生成摘要"""
    all_edges = forward_edges + reverse_edges

    mapping_stats = defaultdict(int)
    edge_types = defaultdict(int)
    source_types = defaultdict(int)
    need_access_stats = defaultdict(int)

    for edge in all_edges:
        mapping_stats[edge['mapping_type']] += 1
        edge_types[edge['edge_type']] += 1
        source_types[edge['source_type']] += 1

        need_access_status = edge.get('need_access', True)
        need_access_stats['need_access_true' if need_access_status else 'need_access_false'] += 1

    summary = {
        'total_edges': len(all_edges),
        'forward_edges': len(forward_edges),
        'reverse_edges': len(reverse_edges),
        'mapping_types': dict(mapping_stats),
        'edge_types': dict(edge_types),
        'source_types': dict(source_types),
        'need_access_stats': dict(need_access_stats)
    }

    return summary

def step3_build_kg():
    """步骤3：构建知识图谱"""
    print("=== 步骤3：构建知识图谱 ===")

    forward_edges = build_forward_edges()
    save_edges(forward_edges, os.path.join(UNIFIED_CONFIG["file_paths"]["edges_dir"], 'forward_edges.jsonl'))

    node_attributes = load_node_attributes()
    reverse_edges = build_reverse_edges(forward_edges, node_attributes)
    save_edges(reverse_edges, os.path.join(UNIFIED_CONFIG["file_paths"]["edges_dir"], 'reverse_edges.jsonl'))

    all_edges = forward_edges + reverse_edges
    save_edges(all_edges, os.path.join(UNIFIED_CONFIG["file_paths"]["edges_dir"], 'all_edges.jsonl'))

    summary = generate_summary(forward_edges, reverse_edges)
    save_edges([summary], os.path.join(UNIFIED_CONFIG["file_paths"]["edges_dir"], 'graph_summary.json'))

    print(f"知识图谱构建完成!")
    print(f"  总关系数: {summary['total_edges']}")
    print(f"  正向关系: {summary['forward_edges']}")
    print(f"  反向关系: {summary['reverse_edges']}")

    need_access_stats = summary.get('need_access_stats', {})
    need_access_true = need_access_stats.get('need_access_true', 0)
    need_access_false = need_access_stats.get('need_access_false', 0)
    print(f"  Need_Access=True:  {need_access_true} ({need_access_true/summary['total_edges']*100:.1f}%)")
    print(f"  Need_Access=False: {need_access_false} ({need_access_false/summary['total_edges']*100:.1f}%)")

    print("知识图谱构建完成！\n")

# =============================================================================
# 路径生成相关函数
# =============================================================================

def load_edges(filepath):
    """加载边数据"""
    edges = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            edge = json.loads(line.strip())
            edges.append(edge)
    return edges

def load_nodes(filepath):
    """加载节点数据"""
    nodes = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            node = json.loads(line.strip())
            nodes[node['node_id']] = node
    return nodes

def build_graph(edges):
    """构建图结构"""
    graph = defaultdict(list)
    edge_info = {}
    target_types = {}

    for edge in edges:
        source_id = edge['source_id']
        target_id = edge['target_id']
        edge_type = edge['edge_type']
        mapping_type = edge['mapping_type']
        target_types[target_id] = edge['target_type']

        graph[source_id].append((target_id, edge_type, mapping_type))
        edge_info[(source_id, target_id, edge_type)] = edge

    return graph, edge_info, target_types

def get_node_name(node_id, nodes):
    """获取节点名称"""
    if node_id in nodes:
        return nodes[node_id]['name']
    elif node_id.startswith('attr_'):
        parts = node_id.split('_')
        return parts[-1] if len(parts) > 1 else node_id
    else:
        return node_id

def find_forward_paths(graph, edge_info, nodes, samples_per_hop_config):
    """查找第一类问题路径"""
    print("查找第一类问题路径...")

    valid_edges = [edge for edge in edge_info.values() if edge['mapping_type'] in ['1-1', 'n-1']]
    print(f"有效边数量 (1-1 + n-1): {len(valid_edges)}")

    forward_paths = []
    path_signatures = set()

    min_hop = min(int(hop.split('-')[0]) for hop in samples_per_hop_config.keys())
    max_hop = max(int(hop.split('-')[0]) for hop in samples_per_hop_config.keys())

    for hop_length in range(min_hop, max_hop + 1):
        print(f"生成 {hop_length}-hop 路径...")
        hop_paths = []

        if samples_per_hop_config and f"{hop_length}-hop" in samples_per_hop_config:
            target_count = samples_per_hop_config[f"{hop_length}-hop"]
        else:
            continue

        start_nodes = list(graph.keys())
        random.shuffle(start_nodes)

        max_attempts = min(len(start_nodes), UNIFIED_CONFIG["misc"]["max_attempts"])
        attempts = 0

        for start_node in start_nodes:
            if attempts >= max_attempts or len(hop_paths) >= target_count:
                break

            path = find_single_forward_path(
                graph, start_node, hop_length, edge_info, nodes, path_signatures
            )

            if path:
                hop_paths.append(path)

            attempts += 1

        forward_paths.extend(hop_paths)
        print(f"  {hop_length}-hop 路径: 生成 {len(hop_paths)} 条 (目标: {target_count})")

    return forward_paths

def find_forward_paths_by_access(graph, edge_info, nodes, samples_per_hop_config, need_access=True):
    """查找第一类问题路径，按need_access分类"""
    access_type = "need_access" if need_access else "no_need_access"
    print(f"查找第一类问题路径 ({access_type})...")

    # 根据need_access要求过滤有效边
    if need_access:
        # 至少有一个edge的need_access为true
        valid_edges = [edge for edge in edge_info.values()
                      if edge['mapping_type'] in ['1-1', 'n-1'] and edge.get('need_access', False)]
    else:
        # 所有edge的need_access都为false
        valid_edges = [edge for edge in edge_info.values()
                      if edge['mapping_type'] in ['1-1', 'n-1'] and not edge.get('need_access', False)]

    print(f"有效边数量 (1-1 + n-1, {access_type}): {len(valid_edges)}")

    forward_paths = []
    path_signatures = set()

    min_hop = min(int(hop.split('-')[0]) for hop in samples_per_hop_config.keys())
    max_hop = max(int(hop.split('-')[0]) for hop in samples_per_hop_config.keys())

    for hop_length in range(min_hop, max_hop + 1):
        print(f"生成 {hop_length}-hop 路径 ({access_type})...")
        hop_paths = []

        if samples_per_hop_config and f"{hop_length}-hop" in samples_per_hop_config:
            target_count = samples_per_hop_config[f"{hop_length}-hop"]
        else:
            continue

        start_nodes = list(graph.keys())
        random.shuffle(start_nodes)

        max_attempts = min(len(start_nodes), UNIFIED_CONFIG["misc"]["max_attempts"])
        attempts = 0

        for start_node in start_nodes:
            if attempts >= max_attempts or len(hop_paths) >= target_count:
                break

            path = find_single_forward_path_by_access(
                graph, start_node, hop_length, edge_info, nodes, path_signatures, need_access
            )

            if path:
                hop_paths.append(path)

            attempts += 1

        forward_paths.extend(hop_paths)
        print(f"  {hop_length}-hop 路径 ({access_type}): 生成 {len(hop_paths)} 条 (目标: {target_count})")

    return forward_paths

def find_single_forward_path(graph, start_node, hop_length, edge_info, nodes, path_signatures):
    """查找单个正向路径"""
    stack = [(start_node, [], set())]

    while stack:
        current_node, path, visited = stack.pop()

        if len(path) >= hop_length:
            continue

        neighbors = graph[current_node]
        random.shuffle(neighbors)

        for neighbor, edge_type, mapping_type in neighbors:
            if mapping_type not in ['1-1', 'n-1']:
                continue

            if neighbor in visited:
                continue

            edge_id = edge_info[(current_node, neighbor, edge_type)]['edge_id']
            if edge_id in [p['edge_id'] for p in path]:
                continue

            new_path = path + [{
                'edge_id': edge_id,
                'source_id': current_node,
                'target_id': neighbor,
                'edge_type': edge_type,
                'mapping_type': mapping_type
            }]
            new_visited = visited | {current_node, neighbor}

            all_nodes_in_new_path = set()
            for edge in new_path:
                all_nodes_in_new_path.add(edge['source_id'])
                all_nodes_in_new_path.add(edge['target_id'])

            if len(all_nodes_in_new_path) != len(new_path) + 1:
                continue

            node_sequence = []
            if new_path:
                node_sequence.append(new_path[0]['source_id'])
                for edge in new_path:
                    node_sequence.append(edge['target_id'])

            seen_nodes = set()
            for node_id in node_sequence:
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)

            if len(new_path) == hop_length:
                path_sig = tuple(sorted([p['edge_id'] for p in new_path]))
                if path_sig not in path_signatures:
                    path_signatures.add(path_sig)
                    return new_path
            else:
                stack.append((neighbor, new_path, new_visited))

    return None

def find_single_forward_path_by_access(graph, start_node, hop_length, edge_info, nodes, path_signatures, need_access=True):
    """查找单个正向路径，按need_access分类"""
    stack = [(start_node, [], set())]

    while stack:
        current_node, path, visited = stack.pop()

        if len(path) >= hop_length:
            continue

        neighbors = graph[current_node]
        random.shuffle(neighbors)

        for neighbor, edge_type, mapping_type in neighbors:
            if mapping_type not in ['1-1', 'n-1']:
                continue

            if neighbor in visited:
                continue

            edge_id = edge_info[(current_node, neighbor, edge_type)]['edge_id']
            if edge_id in [p['edge_id'] for p in path]:
                continue

            edge_data = edge_info[(current_node, neighbor, edge_type)]
            edge_need_access = edge_data.get('need_access', False)

            new_path = path + [{
                'edge_id': edge_id,
                'source_id': current_node,
                'target_id': neighbor,
                'edge_type': edge_type,
                'mapping_type': mapping_type
            }]
            new_visited = visited | {current_node, neighbor}

            all_nodes_in_new_path = set()
            for edge in new_path:
                all_nodes_in_new_path.add(edge['source_id'])
                all_nodes_in_new_path.add(edge['target_id'])

            if len(all_nodes_in_new_path) != len(new_path) + 1:
                continue

            node_sequence = []
            if new_path:
                node_sequence.append(new_path[0]['source_id'])
                for edge in new_path:
                    node_sequence.append(edge['target_id'])

            seen_nodes = set()
            for node_id in node_sequence:
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)

            if len(new_path) == hop_length:
                # 检查路径是否符合need_access要求
                if need_access:
                    # 至少有一个edge的need_access为true
                    has_need_access = any(edge_info[(edge['source_id'], edge['target_id'], edge['edge_type'])].get('need_access', False) for edge in new_path)
                    if not has_need_access:
                        continue
                else:
                    # 所有edge的need_access都为false
                    has_any_need_access = any(edge_info[(edge['source_id'], edge['target_id'], edge['edge_type'])].get('need_access', False) for edge in new_path)
                    if has_any_need_access:
                        continue

                path_sig = tuple(sorted([p['edge_id'] for p in new_path]))
                if path_sig not in path_signatures:
                    path_signatures.add(path_sig)
                    return new_path
            else:
                # 对于路径中间步骤，也进行need_access检查以提高效率
                if need_access:
                    # 如果当前路径还没有need_access=true的边，优先选择need_access=true的边
                    has_need_access_so_far = any(edge_info[(edge['source_id'], edge['target_id'], edge['edge_type'])].get('need_access', False) for edge in new_path)
                    if not has_need_access_so_far and not edge_need_access:
                        # 优先跳过没有need_access的边，除非已经找到有need_access的边
                        continue
                else:
                    # 如果需要所有边都没有need_access，跳过有need_access的边
                    if edge_need_access:
                        continue

                stack.append((neighbor, new_path, new_visited))

    return None

def generate_path_description(path, nodes, all_edges_by_edge_id=None):
    """生成具体的路径描述"""
    if not path:
        return ""

    descriptions = []
    for i, edge in enumerate(path):
        source_name = get_node_name(edge['source_id'], nodes)
        target_name = get_node_name(edge['target_id'], nodes)

        edge_type_display = edge['edge_type']
        if all_edges_by_edge_id and edge['edge_id'] in all_edges_by_edge_id:
            edge_data = all_edges_by_edge_id[edge['edge_id']]
            if edge_data.get('need_access', False):
                edge_type_display = f"{edge['edge_type']}(need_access)"

        if i == 0:
            descriptions.append(f"{source_name} -> {edge_type_display} -> {target_name}")
        else:
            descriptions.append(f"{edge_type_display} -> {target_name}")

    return " -> ".join(descriptions)

def format_path_data(path, nodes, graph, edge_info, target_types, all_edges_by_edge_id, path_type="forward", path_number=1):
    """格式化路径数据"""
    if not path:
        return None

    hop_length = len(path)
    path_id = f"{path_type}_{hop_length}_hop_{path_number}"

    path_edges = [edge['edge_id'] for edge in path]
    path_description = generate_path_description(path, nodes, all_edges_by_edge_id)
    answer = get_node_name(path[-1]['target_id'], nodes)
    answer_type = target_types[path[-1]['target_id']]

    access_count = 0
    has_need_access = False
    for edge in path:
        if edge['edge_id'] in all_edges_by_edge_id:
            if all_edges_by_edge_id[edge['edge_id']].get('need_access', False):
                access_count += 1
                has_need_access = True

    formatted_path = {
        'path_id': path_id,
        'path_type': f"{hop_length}-hop",
        'path_edges': path_edges,
        'path_description': path_description,
        'answer': answer,
        'answer_type': answer_type,
        'path_length': hop_length,
        'access_count': access_count,
        'need_access': has_need_access,  # 新增字段：路径是否包含need_access=true的边
        'detailed_path': [
            {
                'edge_id': edge['edge_id'],
                'source_id': edge['source_id'],
                'source_name': get_node_name(edge['source_id'], nodes),
                'source_type': all_edges_by_edge_id[edge['edge_id']]['source_type'],
                'target_id': edge['target_id'],
                'target_name': get_node_name(edge['target_id'], nodes),
                'target_type': all_edges_by_edge_id[edge['edge_id']]['target_type'],
                'edge_type': edge['edge_type'],
                'mapping_type': edge['mapping_type'],
                'need_access': all_edges_by_edge_id[edge['edge_id']].get('need_access', False)
            }
            for edge in path
        ]
    }

    return formatted_path

def step4_build_paths():
    """步骤4：构建路径"""
    print("=== 步骤4：构建路径 ===")

    random.seed(UNIFIED_CONFIG["misc"]["random_seed"])
    print(f"使用随机种子: {UNIFIED_CONFIG['misc']['random_seed']}")

    all_edges = load_edges(os.path.join(UNIFIED_CONFIG["file_paths"]["edges_dir"], 'all_edges.jsonl'))
    all_edges_by_edge_id = {edge['edge_id']: edge for edge in all_edges}
    all_nodes = {}

    node_files = [
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_countries.jsonl'),
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_cities.jsonl'),
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_universities.jsonl'),
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_companies.jsonl'),
        os.path.join(UNIFIED_CONFIG["file_paths"]["nodes_dir"], 'nodes_persons.jsonl')
    ]

    for file_path in node_files:
        if os.path.exists(file_path):
            nodes = load_nodes(file_path)
            all_nodes.update(nodes)

    print(f"加载了 {len(all_edges)} 条边和 {len(all_nodes)} 个节点")

    graph, edge_info, target_types = build_graph(all_edges)

    need_access_config = UNIFIED_CONFIG["path_generation"]["forward_paths_need_access"]
    no_need_access_config = UNIFIED_CONFIG["path_generation"]["forward_paths_no_need_access"]
    print(f"need_access路径配置: {need_access_config}")
    print(f"no_need_access路径配置: {no_need_access_config}")

    forward_paths_data_need_access = find_forward_paths_by_access(graph, edge_info, all_nodes, samples_per_hop_config=need_access_config, need_access=True)
    forward_paths_data_no_need_access = find_forward_paths_by_access(graph, edge_info, all_nodes, samples_per_hop_config=no_need_access_config, need_access=False)

    # 合并两种类型的路径，但保持随机化
    forward_paths_data = []
    all_hop_lengths = set(list(need_access_config.keys()) + list(no_need_access_config.keys()))

    for hop_length in sorted([int(hop.split('-')[0]) for hop in all_hop_lengths]):
        hop_key = f"{hop_length}-hop"

        # 获取当前hop长度的两种类型路径
        need_access_paths = [path for path in forward_paths_data_need_access if len(path) == hop_length]
        no_need_access_paths = [path for path in forward_paths_data_no_need_access if len(path) == hop_length]

        # 合并并随机打乱
        combined_paths = need_access_paths + no_need_access_paths
        random.shuffle(combined_paths)

        forward_paths_data.extend(combined_paths)

        print(f"  {hop_length}-hop 路径: {len(need_access_paths)} need_access, {len(no_need_access_paths)} no_need_access (总计: {len(combined_paths)})")

    forward_paths = []
    path_counters = defaultdict(int)

    for path in forward_paths_data:
        hop_length = len(path)
        path_counters[hop_length] += 1

        formatted = format_path_data(
            path, all_nodes, graph, edge_info, target_types,
            all_edges_by_edge_id, "forward", path_counters[hop_length]
        )
        if formatted:
            forward_paths.append(formatted)

    os.makedirs(UNIFIED_CONFIG["file_paths"]["paths_dir"], exist_ok=True)

    with open(os.path.join(UNIFIED_CONFIG["file_paths"]["paths_dir"], 'forward_paths.jsonl'), 'w', encoding='utf-8') as f:
        for path in forward_paths:
            f.write(json.dumps(path, ensure_ascii=False) + '\n')

    summary = {
        'total_analysis': {
            'total_forward_paths': len(forward_paths),
            'total_edges': len(all_edges),
            'total_nodes': len(all_nodes)
        },
        'forward_paths_by_hop': defaultdict(int),
        'path_uniqueness': {
            'forward_unique_edges': len(set(edge for path in forward_paths for edge in path['path_edges']))
        }
    }

    for path in forward_paths:
        summary['forward_paths_by_hop'][path['path_length']] += 1

    summary['forward_paths_by_hop'] = dict(summary['forward_paths_by_hop'])

    with open(os.path.join(UNIFIED_CONFIG["file_paths"]["paths_dir"], 'improved_paths_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("路径生成完成!")
    print(f"第一类问题路径: {len(forward_paths)} 条")
    print(f"正向路径分布: {summary['forward_paths_by_hop']}")

    print("路径构建完成！\n")

# =============================================================================
# 复杂路径生成相关函数
# =============================================================================

def is_numerical_type(answer_type):
    """检查answer_type是否为数值类型"""
    return answer_type in NUMERICAL_TYPES

def load_forward_qa_paths(filepath: str) -> List[Dict]:
    """从forward_qa.jsonl加载第一类问题路径"""
    paths = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            qa = json.loads(line.strip())
            if 'path_edges' in qa and 'detailed_path' in qa and 'answer_type' in qa:
                paths.append(qa)
    return paths

def extract_nodes_and_edges(path: Dict) -> Tuple[Set[str], Set[str]]:
    """提取路径中的节点和边"""
    nodes = set()
    edges = set(path['path_edges'])

    for edge_info in path['detailed_path']:
        nodes.add(edge_info['source_id'])
        nodes.add(edge_info['target_id'])

    return nodes, edges

def are_paths_disjoint(path1_nodes: Set[str], path1_edges: Set[str],
                      path2_nodes: Set[str], path2_edges: Set[str]) -> bool:
    """检查两条路径是否不相交"""
    return len(path1_nodes & path2_nodes) == 0 and len(path1_edges & path2_edges) == 0

def are_paths_globally_unique(path1_nodes: Set[str], path1_edges: Set[str],
                             path2_nodes: Set[str], path2_edges: Set[str],
                             global_used_path_sets: Set[frozenset]) -> bool:
    """检查两条路径的组合是否全局唯一"""
    path1_edges_frozen = frozenset(path1_edges)
    path2_edges_frozen = frozenset(path2_edges)

    # 使用frozenset的组合作为签名，这样 (pathA, pathB) 和 (pathB, pathA) 会被视为相同
    combination_signature = frozenset([path1_edges_frozen, path2_edges_frozen])
    return combination_signature not in global_used_path_sets

def format_parallel_path(pair: Tuple[Dict, Dict], index: int, path_type: str, is_diff_type: bool = False) -> Dict:
    """格式化并行路径"""
    path1, path2 = pair

    hop1 = path1['path_length']
    hop2 = path2['path_length']

    if hop1 == 1 and hop2 == 1:
        actual_path_type = "parallel single-hop"
    elif hop1 == 1 and hop2 > 1:
        actual_path_type = f"parallel single-hop + {hop2}-hop"
    elif hop1 > 1 and hop2 == 1:
        actual_path_type = f"parallel single-hop + {hop1}-hop"
    else:
        actual_path_type = f"parallel {hop1}-hop + {hop2}-hop"

    path_id = f"parallel_{path_type}_{index + 1:04d}"

    if is_diff_type:
        answer_type = [path1['answer_type'], path2['answer_type']]
    else:
        answer_type = path1['answer_type']

    formatted_path = {
        'path_id': path_id,
        'path_type': actual_path_type,
        'original_path_ids': [path1['path_id'], path2['path_id']],
        'path_edges': [path1['path_edges'], path2['path_edges']],
        'path_description': [path1['path_description'], path2['path_description']],
        'answers': [path1['answer'], path2['answer']],
        'answer_type': answer_type
    }

    return formatted_path

def generate_combo_paths(forward_paths: List[Dict], global_used_path_sets: Set[frozenset]) -> List[Dict]:
    """生成combo路径，避免重复组合"""
    print("生成combo路径...")

    # 构建连接点映射：target_name -> 以该节点为结束的路径列表
    target_to_paths = defaultdict(list)
    for path in forward_paths:
        if path['path_length'] <= 6:  # combo路径的子路径不超过6-hop
            detailed_path = path.get('detailed_path', [])
            if detailed_path:
                last_edge = detailed_path[-1]
                target_name = last_edge.get('target_name')
                if target_name:
                    target_to_paths[target_name].append(path)

    # 构建起始点映射：source_name -> 以该节点为开始的路径列表
    source_to_paths = defaultdict(list)
    for path in forward_paths:
        if path['path_length'] <= 6:
            detailed_path = path.get('detailed_path', [])
            if detailed_path:
                first_edge = detailed_path[0]
                source_name = first_edge.get('source_name')
                if source_name:
                    source_to_paths[source_name].append(path)

    combo_paths = []
    combo_config = UNIFIED_CONFIG["complex_paths"]["combo"]

    # 目标生成的combo路径数量
    target_counts = {
        7: combo_config.get("7_hop", 100),
        8: combo_config.get("8_hop", 100),
        9: combo_config.get("9_hop", 100),
        10: combo_config.get("10_hop", 100),
        11: combo_config.get("11_hop", 100),
        12: combo_config.get("12_hop", 100)
    }

    print(f"目标combo路径数量: {target_counts}")

    # 生成combo路径
    for target_name, paths_ending_here in target_to_paths.items():
        paths_starting_here = source_to_paths.get(target_name, [])

        if not paths_ending_here or not paths_starting_here:
            continue

        # 尝试所有可能的组合
        for path1 in paths_ending_here:
            for path2 in paths_starting_here:
                # 避免自连接
                if path1['path_id'] == path2['path_id']:
                    continue

                total_length = path1['path_length'] + path2['path_length']

                # 检查是否需要这个长度的combo路径
                if total_length in target_counts and target_counts[total_length] > 0:
                    # 检查节点和边是否不相交
                    nodes1, edges1 = extract_nodes_and_edges(path1)
                    nodes2, edges2 = extract_nodes_and_edges(path2)

                    if are_paths_disjoint(nodes1, edges1, nodes2, edges2):
                        # 检查组合是否全局唯一
                        if are_paths_globally_unique(nodes1, edges1, nodes2, edges2, global_used_path_sets):
                            # 创建combo路径
                            combo_path = {
                                'path_id': f"combo_combined_{total_length}_hop_{len(combo_paths) + 1}",
                                'path_type': f"combo {total_length}-hop",
                                'original_path_ids': [path1['path_id'], path2['path_id']],
                                'path_edges': [path1['path_edges'], path2['path_edges']],
                                'path_description': f"{path1.get('path_description', '')} + {path2.get('path_description', '')}",
                                'answers': [path1.get('answer', ''), path2.get('answer', '')],
                                'answer_types': [path1.get('answer_type', ''), path2.get('answer_type', '')],
                                'combination_method': 'path_combination',
                                'first_path_id': path1['path_id'],
                                'second_path_id': path2['path_id'],
                                'connection_point': target_name
                            }

                            combo_paths.append(combo_path)
                            target_counts[total_length] -= 1

                            # 记录已使用的组合签名
                            path1_edges_frozen = frozenset(edges1)
                            path2_edges_frozen = frozenset(edges2)
                            combination_signature = frozenset([path1_edges_frozen, path2_edges_frozen])
                            global_used_path_sets.add(combination_signature)

                            print(f"  生成combo {total_length}-hop: {path1['path_id']} -> {path2['path_id']} (连接点: {target_name})")

                            # 如果达到目标数量，停止生成该长度的路径
                            if target_counts[total_length] == 0:
                                print(f"  完成 {total_length}-hop combo路径生成")
                                break

            # 检查是否所有长度的路径都已生成完成
            if all(count <= 0 for count in target_counts.values()):
                break

    print(f"combo路径生成完成，实际生成 {len(combo_paths)} 条")
    return combo_paths

def step5_build_complex_paths():
    """步骤5：构建复杂路径"""
    print("=== 步骤5：构建复杂路径 ===")

    forward_paths = load_forward_qa_paths(os.path.join(UNIFIED_CONFIG["file_paths"]["paths_dir"], 'forward_paths.jsonl'))
    print(f"加载了 {len(forward_paths)} 条forward QA路径")

    parallel_paths = [p for p in forward_paths if p['path_length'] <= 6]

    numerical_comparison_paths = []
    different_type_paths = []

    global_used_path_sets = set()

    # 生成 parallel numerical comparison paths
    print("=== 生成 parallel numerical comparison paths ===")

    paths_by_type = defaultdict(list)
    for path in parallel_paths:
        if is_numerical_type(path['answer_type']):
            paths_by_type[path['answer_type']].append(path)

    print(f"数值类型分布: {dict((k, len(v)) for k, v in paths_by_type.items())}")

    # 简化版本：生成少量示例路径
    for numerical_type in list(paths_by_type.keys())[:3]:  # 只处理前3个数值类型
        single_hop_type_paths = [p for p in paths_by_type[numerical_type] if p['path_length'] == 1]
        if len(single_hop_type_paths) >= 2:
            for _ in range(min(10, len(single_hop_type_paths) // 2)):  # 最多生成10条
                path1, path2 = random.sample(single_hop_type_paths, 2)
                nodes1, edges1 = extract_nodes_and_edges(path1)
                nodes2, edges2 = extract_nodes_and_edges(path2)
                if are_paths_disjoint(nodes1, edges1, nodes2, edges2):
                    if are_paths_globally_unique(nodes1, edges1, nodes2, edges2, global_used_path_sets):
                        formatted = format_parallel_path((path1, path2), len(numerical_comparison_paths), "ss", False)
                        numerical_comparison_paths.append(formatted)

                        path1_edges_frozen = frozenset(edges1)
                        path2_edges_frozen = frozenset(edges2)
                        # 使用标准化的组合签名来避免顺序相关的重复
                        combination_signature = frozenset([path1_edges_frozen, path2_edges_frozen])
                        global_used_path_sets.add(combination_signature)

    print(f"生成了 {len(numerical_comparison_paths)} 条numerical comparison paths")

    # 生成 parallel different answer type paths（简化版本）
    print("=== 生成 parallel different answer type paths ===")

    single_hop_all_paths = [p for p in forward_paths if p['path_length'] == 1]
    for _ in range(min(20, len(single_hop_all_paths) // 2)):
        path1, path2 = random.sample(single_hop_all_paths, 2)
        if path1['answer_type'] != path2['answer_type']:
            nodes1, edges1 = extract_nodes_and_edges(path1)
            nodes2, edges2 = extract_nodes_and_edges(path2)
            if are_paths_disjoint(nodes1, edges1, nodes2, edges2):
                if are_paths_globally_unique(nodes1, edges1, nodes2, edges2, global_used_path_sets):
                    formatted = format_parallel_path((path1, path2), len(different_type_paths), "ss", True)
                    different_type_paths.append(formatted)

                    path1_edges_frozen = frozenset(edges1)
                    path2_edges_frozen = frozenset(edges2)
                    # 使用标准化的组合签名来避免顺序相关的重复
                    combination_signature = frozenset([path1_edges_frozen, path2_edges_frozen])
                    global_used_path_sets.add(combination_signature)

    print(f"生成了 {len(different_type_paths)} 条different type paths")

    # 生成 combo paths
    print("=== 生成 combo paths ===")
    combo_paths = generate_combo_paths(forward_paths, global_used_path_sets)
    print(f"生成了 {len(combo_paths)} 条combo paths")

    # 保存复杂路径
    os.makedirs(UNIFIED_CONFIG["file_paths"]["complex_paths_dir"], exist_ok=True)

    with open(os.path.join(UNIFIED_CONFIG["file_paths"]["complex_paths_dir"], 'parallel_same_type_numerical_qa_paths.jsonl'), 'w', encoding='utf-8') as f:
        for path in numerical_comparison_paths:
            f.write(json.dumps(path, ensure_ascii=False) + '\n')

    with open(os.path.join(UNIFIED_CONFIG["file_paths"]["complex_paths_dir"], 'parallel_diff_type_qa_paths.jsonl'), 'w', encoding='utf-8') as f:
        for path in different_type_paths:
            f.write(json.dumps(path, ensure_ascii=False) + '\n')

    with open(os.path.join(UNIFIED_CONFIG["file_paths"]["complex_paths_dir"], 'combo_qa_paths.jsonl'), 'w', encoding='utf-8') as f:
        for path in combo_paths:
            f.write(json.dumps(path, ensure_ascii=False) + '\n')

    summary = {
        "parallel_same_type_numerical": len(numerical_comparison_paths),
        "parallel_diff_type": len(different_type_paths),
        "combo_qa": len(combo_paths),
        "total": len(numerical_comparison_paths) + len(different_type_paths) + len(combo_paths)
    }

    with open(os.path.join(UNIFIED_CONFIG["file_paths"]["complex_paths_dir"], 'classified_paths_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("复杂路径生成完成!")
    print(f"总计: {summary['total']} 条复杂路径")

    print("复杂路径构建完成！\n")

# =============================================================================
# QA生成相关函数
# =============================================================================

def check_path_files_exist():
    """检查路径生成文件是否存在"""
    paths_dir = UNIFIED_CONFIG["file_paths"]["paths_dir"]
    complex_paths_dir = UNIFIED_CONFIG["file_paths"]["complex_paths_dir"]

    required_files = [
        os.path.join(paths_dir, "forward_paths.jsonl"),
        os.path.join(complex_paths_dir, "parallel_diff_type_qa_paths.jsonl"),
        os.path.join(complex_paths_dir, "combo_qa_paths.jsonl")
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    return len(missing_files) == 0, missing_files

def step6_generate_simple_qa():
    """步骤6：生成简单QA"""
    print("=== 步骤6：生成简单QA ===")

    # Import generate_simple_qa module
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_simple_qa",
                                                   os.path.join(os.path.dirname(__file__), "generate_simple_qa.py"))
    generate_simple_qa_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate_simple_qa_module)

    # Call the main function with default parameters
    generate_simple_qa_module.main()

    print("简单QA生成完成！\n")

def step7_generate_complex_qa():
    """步骤7：生成复杂QA"""
    print("=== 步骤7：生成复杂QA ===")

    # Import generate_complex_qa module
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_complex_qa",
                                                   os.path.join(os.path.dirname(__file__), "generate_complex_qa.py"))
    generate_complex_qa_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate_complex_qa_module)

    # Call the main function
    generate_complex_qa_module.main()

    print("复杂QA生成完成！\n")

# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数 - 执行所有步骤"""
    print("="*60)
    print("统一知识图谱构建流程")
    print("="*60)

    parser = argparse.ArgumentParser(description='统一知识图谱构建脚本')
    parser.add_argument('--steps', type=str, nargs='+',
                       choices=['names', 'nodes', 'kg', 'paths', 'complex_paths', 'simple_qa', 'complex_qa', 'all'],
                       default=['all'],
                       help='指定要执行的步骤')

    args = parser.parse_args()

    # 获取QA生成配置
    qa_config = get_qa_generation_config()
    generate_path = qa_config["generate_path"]
    generate_qa = qa_config["generate_qa"]

    print(f"配置: generate_path={generate_path}, generate_qa={generate_qa}")

    # 创建所有必要的目录
    for dir_path in UNIFIED_CONFIG["file_paths"].values():
        if isinstance(dir_path, str) and dir_path.endswith('/'):
            os.makedirs(dir_path, exist_ok=True)

    try:
        # 场景1: generate_path=True, generate_qa=False - 只生成路径
        if generate_path and not generate_qa:
            print("执行模式: 只生成路径（不生成QA）")
            if 'all' in args.steps or 'names' in args.steps:
                step1_generate_names()
            if 'all' in args.steps or 'nodes' in args.steps:
                step2_build_nodes()
            if 'all' in args.steps or 'kg' in args.steps:
                step3_build_kg()
            if 'all' in args.steps or 'paths' in args.steps:
                step4_build_paths()
            if 'all' in args.steps or 'complex_paths' in args.steps:
                step5_build_complex_paths()

        # 场景2: generate_path=False, generate_qa=True - 只生成QA（检查路径文件是否存在）
        elif not generate_path and generate_qa:
            print("执行模式: 只生成QA（基于已有路径文件）")

            # 检查必需的路径文件是否存在
            path_files_exist, missing_files = check_path_files_exist()
            if not path_files_exist:
                print("错误: 缺少必需的路径文件，请先运行 generate_path=True 来生成路径:")
                for file_path in missing_files:
                    print(f"  - {file_path}")
                return

            print("✓ 所有必需的路径文件都存在，开始生成QA...")

            if 'all' in args.steps or 'simple_qa' in args.steps:
                step6_generate_simple_qa()
            if 'all' in args.steps or 'complex_qa' in args.steps:
                step7_generate_complex_qa()

        # 场景3: generate_path=True, generate_qa=True - 生成所有内容
        elif generate_path and generate_qa:
            print("执行模式: 生成路径和QA")

            # 生成路径
            if 'all' in args.steps or 'names' in args.steps:
                step1_generate_names()
            if 'all' in args.steps or 'nodes' in args.steps:
                step2_build_nodes()
            if 'all' in args.steps or 'kg' in args.steps:
                step3_build_kg()
            if 'all' in args.steps or 'paths' in args.steps:
                step4_build_paths()
            if 'all' in args.steps or 'complex_paths' in args.steps:
                step5_build_complex_paths()

            # 生成QA
            if 'all' in args.steps or 'simple_qa' in args.steps:
                step6_generate_simple_qa()
            if 'all' in args.steps or 'complex_qa' in args.steps:
                step7_generate_complex_qa()

        else:
            print("错误: 无效的配置组合")
            return

        print("="*60)
        print("所有步骤执行完成！")
        print("="*60)

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()