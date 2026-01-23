#!/usr/bin/env python3
"""
完整版本：完全符合attributes_requirements.txt的所有要求
"""
import sys
import os
import json
import random
from datetime import datetime
from collections import defaultdict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.build_utils import load_schema

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
    days = list(range(1, 29))  # 避免无效日期
    return f"{random.choice(months)} {random.choice(days)}th"

def calculate_university_rank(students, professors, projects, publications, awards, patents):
    """根据多个因素计算大学排名"""
    # 归一化各项指标
    student_score = min(students / 50000, 1.0) * 0.2
    professor_score = min(professors / 1000, 1.0) * 0.2
    project_score = min(projects / 1000, 1.0) * 0.15
    publication_score = min(publications / 10000, 1.0) * 0.2
    award_score = min(awards / 100, 1.0) * 0.15
    patent_score = min(patents / 5000, 1.0) * 0.1

    total_score = student_score + professor_score + project_score + publication_score + award_score + patent_score
    return max(1, int(1000 * (1 - total_score)))  # 分数越高，排名越靠前

def select_random_attributes(attributes_dict):
    """随机选择50%的属性（向上取整）"""
    if not attributes_dict:
        return {}

    # 计算要选择的属性数量（50%向上取整）
    total_attrs = len(attributes_dict)
    num_to_select = (total_attrs + 1) // 2  # 等同于 math.ceil(total_attrs * 0.5)

    # 随机选择属性
    selected_keys = random.sample(list(attributes_dict.keys()), num_to_select)

    # 返回选中的属性字典
    return {key: attributes_dict[key] for key in selected_keys}

class CompleteNodeGenerator:
    def __init__(self, schema):
        self.schema = schema
        self.current_year = 2025

        # 用于追踪已使用的值，确保唯一性
        self.used_values = {
            'country_pop': set(), 'country_gdp': set(), 'country_area': set(),
            'city_pop': set(), 'city_area': set(), 'city_gdp': set(),
            'university_students': set(), 'university_rank': set(),
            'national_day': set(), 'company_employees': set()
        }

        # 存储生成的实体
        self.countries = []
        self.cities = []
        self.universities = []
        self.companies = []
        self.persons = []

        # 用于双向关系
        self.sister_country_pairs = set()
        self.sister_city_pairs = set()
        self.spouse_pairs = set()
        self.best_friend_pairs = set()

        # 领导者映射
        self.country_leaders = {}
        self.city_mayors = {}

    def generate_nodes(self):
        """生成所有节点"""
        print("开始生成节点...")

        # 1. 生成国家
        self.countries = self.create_country_nodes()

        # 2. 生成城市
        self.cities = self.create_city_nodes()

        # 3. 生成大学
        self.universities = self.create_university_nodes()

        # 4. 生成公司
        self.companies = self.create_company_nodes()

        # 5. 生成人物
        self.persons = self.create_person_nodes()

        print(f"生成完成: {len(self.countries)}个国家, {len(self.cities)}个城市, "
              f"{len(self.universities)}所大学, {len(self.companies)}家公司, {len(self.persons)}个人")

    def create_country_nodes(self):
        """创建国家节点"""
        country_names = load_names_from_json('outputs/names/names_country.json')
        language_names = load_names_from_json('outputs/names/names_language.json')
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

            # GDP与人口相关
            gdp = int(population * random.uniform(10000, 50000))
            while gdp in self.used_values['country_gdp']:
                gdp += 1
            self.used_values['country_gdp'].add(gdp)
            node['attributes']['gdp'] = {"name": gdp, "type": "GDP"}

            # 面积与人口相关
            area = int(population * random.uniform(0.01, 0.1))
            while area in self.used_values['country_area']:
                area += 1
            self.used_values['country_area'].add(area)
            node['attributes']['total_area'] = {"name": area, "type": "Area"}

            # 官方语言
            node['attributes']['official_language'] = {"name": random.choice(language_names), "type": "Language"}

            # 可选属性
            if random.random() < 0.4:
                node['attributes']['number_of_ethnic_groups'] = {"name": random.randint(1, 10), "type": "Ethic Group Count"}

            # 国庆日
            national_day = generate_national_day()
            while national_day in self.used_values['national_day']:
                national_day = generate_national_day()
            self.used_values['national_day'].add(national_day)
            node['attributes']['national_day'] = {"name": national_day, "type": "Date"}

            # 添加随机选择的属性到 abstracts
            node['attributes_in_abstracts'] = select_random_attributes(node['attributes'])

            nodes.append(node)

        # 设置姐妹国家关系（双向）
        self._setup_sister_country_relationships(nodes)

        return nodes

    def create_city_nodes(self):
        """创建城市节点"""
        city_names = load_names_from_json('outputs/names/names_city.json')
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

            # 首都人口应该较大
            population = random.randint(5000000, 30000000)
            while population in self.used_values['city_pop']:
                population += 1
            self.used_values['city_pop'].add(population)
            node['attributes']['population'] = {"name": population, "type": "Population Count"}

            # 面积与人口相关
            area = int(population * random.uniform(0.001, 0.01))
            while area in self.used_values['city_area']:
                area += 1
            self.used_values['city_area'].add(area)
            node['attributes']['area'] = {"name": area, "type": "Area"}

            # 位于国家
            node['attributes']['located_country'] = {
                "name": country['name'],
                "type": "Country",
                "node_id": country['node_id']
            }

            # GDP与人口相关
            gdp = int(population * random.uniform(5000, 20000))
            while gdp in self.used_values['city_gdp']:
                gdp += 1
            self.used_values['city_gdp'].add(gdp)
            node['attributes']['gdp'] = {"name": gdp, "type": "GDP"}

            # 首都属性
            node['attributes']['capital_of'] = {
                "name": country['name'],
                "type": "Country",
                "node_id": country['node_id']
            }
            capital_cities.add(node['node_id'])

            # 大学数量基于人口
            num_universities = max(1, int(population / 2000000))
            node['attributes']['number_of_universities'] = {"name": num_universities, "type": "University Count"}

            # 公司数量基于GDP
            num_companies = max(1, int(gdp / 100000000))
            node['attributes']['number_of_companies'] = {"name": num_companies, "type": "Company Count"}

            # 添加随机选择的属性到 abstracts
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

            # 人口
            population = random.randint(100000, 10000000)
            while population in self.used_values['city_pop']:
                population += 1
            self.used_values['city_pop'].add(population)
            node['attributes']['population'] = {"name": population, "type": "Population Count"}

            # 面积
            area = int(population * random.uniform(0.001, 0.01))
            while area in self.used_values['city_area']:
                area += 1
            self.used_values['city_area'].add(area)
            node['attributes']['area'] = {"name": area, "type": "Area"}

            # 位于国家
            country = random.choice(self.countries)
            node['attributes']['located_country'] = {
                "name": country['name'],
                "type": "Country",
                "node_id": country['node_id']
            }

            # GDP
            gdp = int(population * random.uniform(5000, 20000))
            while gdp in self.used_values['city_gdp']:
                gdp += 1
            self.used_values['city_gdp'].add(gdp)
            node['attributes']['gdp'] = {"name": gdp, "type": "GDP"}

            # 大学和公司数量
            num_universities = max(1, int(population / 2000000))
            num_companies = max(1, int(gdp / 100000000))
            node['attributes']['number_of_universities'] = {"name": num_universities, "type": "University Count"}
            node['attributes']['number_of_companies'] = {"name": num_companies, "type": "Company Count"}

            # 添加随机选择的属性到 abstracts
            node['attributes_in_abstracts'] = select_random_attributes(node['attributes'])

            nodes.append(node)

        # 设置姐妹城市关系（双向）
        self._setup_sister_city_relationships(nodes)

        return nodes

    def create_university_nodes(self):
        """创建大学节点"""
        university_names = load_names_from_json('outputs/names/names_university.json')
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

            # 位于城市
            city = random.choice(self.cities)
            node['attributes']['located_city'] = {
                "name": city['name'],
                "type": "City",
                "node_id": city['node_id']
            }

            # 注册学生数
            students = random.randint(1000, 50000)
            while students in self.used_values['university_students']:
                students += 1
            self.used_values['university_students'].add(students)
            node['attributes']['registered_students'] = {"name": students, "type": "Student Count"}

            # 学院数基于学生数
            faculties = max(3, int(students / 5000))
            node['attributes']['number_of_faculties'] = {"name": faculties, "type": "Faculty Count"}

            # 教授数基于学生数
            professors = max(50, int(students / 100))
            node['attributes']['number_of_professors'] = {"name": professors, "type": "Professor Count"}

            # 研究项目基于教授数
            projects = random.randint(professors, professors * 3)
            node['attributes']['number_of_research_projects'] = {"name": projects, "type": "Research Project Count"}

            # 研究成果基于教授数
            publications = random.randint(professors * 10, professors * 50)
            node['attributes']['number_of_research_publications'] = {"name": publications, "type": "Research Publication Count"}

            # 研究奖励基于教授数
            awards = random.randint(max(1, professors // 10), professors // 2)
            node['attributes']['number_of_research_awards'] = {"name": awards, "type": "Research Award Count"}

            # 研究专利基于教授数
            patents = random.randint(max(1, professors // 5), professors * 2)
            node['attributes']['number_of_research_patents'] = {"name": patents, "type": "Research Patent Count"}

            # 建校年份
            established_year = random.randint(1600, 2000)
            node['attributes']['established_year'] = {"name": established_year, "type": "Year"}

            # 计算排名
            rank = calculate_university_rank(students, professors, projects, publications, awards, patents)
            while rank in self.used_values['university_rank']:
                rank += 1
            self.used_values['university_rank'].add(rank)
            node['attributes']['university_rank'] = {"name": rank, "type": "University Rank"}

            # 添加随机选择的属性到 abstracts
            node['attributes_in_abstracts'] = select_random_attributes(node['attributes'])

            nodes.append(node)

        return nodes

    def create_company_nodes(self):
        """创建公司节点"""
        company_names = load_names_from_json('outputs/names/names_company.json')
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

            # 总部城市
            city = random.choice(self.cities)
            node['attributes']['headquarter_city'] = {
                "name": city['name'],
                "type": "City",
                "node_id": city['node_id']
            }

            # 员工数
            employees = random.randint(50, 50000)
            while employees in self.used_values['company_employees']:
                employees += 1
            self.used_values['company_employees'].add(employees)
            node['attributes']['number_of_employees'] = {"name": employees, "type": "Employee Count"}

            # 部门数
            departments = max(1, int(employees / 1000))
            node['attributes']['number_of_departments'] = {"name": departments, "type": "Department Count"}

            # 产品数
            products = random.randint(1, 100)
            node['attributes']['number_of_products'] = {"name": products, "type": "Product Count"}

            # 分店数基于产品数
            franchise_count = random.randint(products // 2, products * 5)
            node['attributes']['franchise_count'] = {"name": franchise_count, "type": "Franchise Count"}

            # 分店收入基于产品数
            franchise_revenue = random.randint(products * 1000000, products * 10000000)
            node['attributes']['franchise_revenue'] = {"name": franchise_revenue, "type": "Franchise Revenue"}

            # 成立年份
            founded_year = random.randint(1800, 2020)
            node['attributes']['founded_year'] = {"name": founded_year, "type": "Year"}

            # 添加随机选择的属性到 abstracts
            node['attributes_in_abstracts'] = select_random_attributes(node['attributes'])

            nodes.append(node)

        return nodes

    def create_person_nodes(self):
        """创建人物节点"""
        person_names = load_names_from_json('outputs/names/names_person.json')
        num_persons = len(person_names)
        selected_names = random.sample(person_names, num_persons)

        nodes = []

        # 生成基础人物信息
        for i, person_name in enumerate(selected_names):
            node = {
                "node_id": f"person_{i}",
                "node_type": "Person",
                "name": person_name,
                "attributes": {}
            }

            # 出生年份
            birth_year = random.randint(1950, 2010)
            node['attributes']['birth_year'] = {"name": birth_year, "type": "Year"}
            age = self.current_year - birth_year

            # 性别
            gender = random.choice(['Male', 'Female'])
            node['attributes']['gender'] = {"name": gender, "type": "Gender"}

            # 出生地
            birth_city = random.choice(self.cities)
            node['attributes']['birth_city'] = {
                "name": birth_city['name'],
                "type": "City",
                "node_id": birth_city['node_id']
            }

            # 当前居住地（与出生地不同）
            current_cities = [c for c in self.cities if c['node_id'] != birth_city['node_id']]
            if current_cities:
                current_city = random.choice(current_cities)
                node['attributes']['current_living_city'] = {
                    "name": current_city['name'],
                    "type": "City",
                    "node_id": current_city['node_id']
                }

            # 大学（年龄约束）
            if 16 <= age <= 70 and random.random() < 0.6:
                university = random.choice(self.universities)
                node['attributes']['university_attended'] = {
                    "name": university['name'],
                    "type": "University",
                    "node_id": university['node_id']
                }

            # 工作（年龄约束）
            if 22 <= age <= 65 and random.random() < 0.7:
                company = random.choice(self.companies)
                node['attributes']['current_working_company'] = {
                    "name": company['name'],
                    "type": "Company",
                    "node_id": company['node_id']
                }

            # 添加随机选择的属性到 abstracts
            node['attributes_in_abstracts'] = select_random_attributes(node['attributes'])

            nodes.append(node)

        # 设置复杂关系
        self._setup_family_relationships(nodes)
        self._setup_spouse_relationships(nodes)
        self._setup_leadership_positions(nodes)
        self._setup_best_friend_relationships(nodes)

        return nodes

    def _setup_sister_country_relationships(self, countries):
        """设置姐妹国家关系（双向）"""
        for i in range(0, len(countries) - 1, 2):
            if i + 1 < len(countries):
                country1, country2 = countries[i], countries[i + 1]

                # 检查是否已经设置了关系
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
        """设置姐妹城市关系（双向）"""
        for i in range(0, len(cities) - 1, 2):
            if i + 1 < len(cities):
                city1, city2 = cities[i], cities[i + 1]

                # 检查是否已经设置了关系
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
        # 按年龄排序，确保父母比子女大
        persons_by_age = sorted(persons, key=lambda p: p['attributes']['birth_year']['name'])

        # 为较年轻的人设置父母
        for i, person in enumerate(persons_by_age):
            age = self.current_year - person['attributes']['birth_year']['name']

            # 只有年轻人（40岁以下）才有父母在这个数据集中
            if age <= 40 and random.random() < 0.8:
                if 'father' not in person['attributes'] and 'mother' not in person['attributes']: # 如果已经设置了母亲，则不设置父亲，这样会让spouse的分配多一层限制
                    # 寻找合适的父亲（男性，年龄差25-50岁）
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
            if age <= 40 and random.random() < 0.8: # 如果已经设置了父亲，则不设置母亲，这样会让spouse的分配多一层限制
                # 寻找合适的母亲（女性，年龄差25-50岁）
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
        """设置配偶关系（双向）"""
        # 按年龄和性别分组
        males = [p for p in persons if p['attributes']['gender']['name'] == 'Male' and self.current_year - p['attributes']['birth_year']['name'] >= 18]
        females = [p for p in persons if p['attributes']['gender']['name'] == 'Female' and self.current_year - p['attributes']['birth_year']['name'] >= 18]

        # 配对
        for male in males:
            if random.random() < 0.4:  # 40%概率有配偶
                # 寻找年龄相仿且还没有配偶的女性
                male_age = self.current_year - male['attributes']['birth_year']['name']
                potential_spouses = [f for f in females if
                                   abs((self.current_year - f['attributes']['birth_year']['name']) - male_age) <= 10 and
                                   'spouse' not in f['attributes']]

                if potential_spouses:
                    spouse = random.choice(potential_spouses)

                    # 检查是否已经设置了关系
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
        # 为每个国家设置一个领导人
        for country in self.countries:
            # 寻找合适的领导人（50-100岁）
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

        # 为每个城市设置市长
        for city in self.cities:
            # 寻找合适的市长（50-100岁，不是国家领导人）
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
        """设置好朋友关系（双向）"""
        for i, person in enumerate(persons):
            if random.random() < 0.3:  # 30%概率有好朋友
                age = self.current_year - person['attributes']['birth_year']['name']

                # 寻找年龄相仿的人
                potential_friends = [p for p in persons[i+1:] if
                                   abs((self.current_year - p['attributes']['birth_year']['name']) - age) <= 15]

                if potential_friends:
                    friend = random.choice(potential_friends)

                    # 检查两个人是否都还没有best_friend关系
                    if 'best_friend' not in person['attributes'] and 'best_friend' not in friend['attributes']:
                        # 检查是否已经设置了关系
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
        output_dir = 'outputs/nodes'
        os.makedirs(output_dir, exist_ok=True)

        # 保存各个实体类型的节点
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

def main():
    """主函数"""
    # 加载schema
    schema = load_schema()

    # 创建节点生成器
    generator = CompleteNodeGenerator(schema)

    # 生成节点
    generator.generate_nodes()

    # 保存节点
    generator.save_nodes()

    print("完整版本节点生成完成!")

if __name__ == "__main__":
    main()