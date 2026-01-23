"""
复杂问题生成的LLM提示词模板
支持多种类型的复杂问题生成
"""

# 简单问题生成模板
simple_qa_prompt_all_template = """
You are a master question designer specializing in creating complex, multi-hop questions to benchmark advanced AI agents. Your mission is to transform a structured 'fact path' into a single, high-quality, natural language question.

## CONTEXT & INPUTS
You will be provided with the following information:
Path Type: The number of "hops" or relationships in the path (e.g., 2-hop, 3-hop).
Fact Path(s): A structured list of connected facts, like (Entity A) --[relationship]--> (Entity B).
Answer Name: The final entity in the fact path, which is the correct answer to the question you will create.

Your Inputs:
Path Type: {path_type}
Fact Path(s): 
{facts_string}
Intermediate Nodes: {intermediate_nodes}
Start Node: {start_node}
Answer Name:  {answer_name}

## CRITICAL RULES
Strictly Grounded: Base the question exclusively on the provided Fact Path(s). Do not infer or add any external information.
Conceal Intermediate Nodes & Answer: NEVER reveal the answer name ({answer_name}) or any intermediate nodes ({intermediate_nodes}) from the path in your question.
Follow the Path's Logic: The question MUST begin by naming the start node ({start_node}) of the fact path. The logical flow of the question must then follow the exact sequence of relationships provided.
Ask the right question: The question should be a question whose answer is {answer_name}({answer_type}).
Specific Phrasing: Do not use generic question words like "what" or "where." Be specific about the type of answer expected. For example, instead of "Where is...", use "In which city is..." or "At which company does...".
Final Output Format: Your final output must be ONLY the raw text of the question. Do not include any preamble, titles, labels (like "Question:"), or explanations.

## STEP-BY-STEP QUESTION DESIGN PROCESS
Deconstruct the Path: Identify the start node, all intermediate nodes, the final answer node, and every relationship connecting them.
Anchor the Question: Begin formulating the question by explicitly stating the name of the start node.
Weave the Narrative: Sequentially convert each relationship in the path into a descriptive clause. For (Entity A) --[relationship]--> (Entity B), this might become "...the [description of B] that [Entity A] [relationship]...".
Formulate the Query: Conclude by asking for the specific category of the final answer node.
Final Review: Read your generated question aloud. Does it sound natural? Is it unambiguous? Does it strictly adhere to all the CRITICAL RULES above?

## EXAMPLES (For Your Reference, please study them carefully before generating the question)

Example 1: 2-hop
Fact Path(s):
(Elara Vance(Person)) --[graduated from]--> (Astral University(University))
(Astral University(University)) --[is located in]--> (Silverwind City(City))
Intermediate Nodes: [Astral University]
Start Node: Elara Vance
Answer Name: Silverwind City
Excellent Question: In which city is the university that Elara Vance graduated from located?
Bad Question: In which city is the university, Astral University, that Elara Vance graduated from located? (Reason: Violates Rule #2 by revealing the intermediate node "Astral University").

Example 2: 3-hop
Fact Path(s):
(Elara Vance(Person)) --[graduated from]--> (Astral University(University))
(Astral University(University)) --[is located in]--> (Silverwind City(City))
(Silverwind City(City)) --[is the capital of]--> (Silverwind Country(Country))
Intermediate Nodes: [Astral University, Silverwind City]
Start Node: Elara Vance 
Answer Name: Silverwind Country
Excellent Question: Of which country is the capital city that contains the university Elara Vance graduated from?
Bad Question: In which country is Astral University located? (Reason: Violates Rule #3 by not including all relationships from the path).

Example 3: 3-hop
Fact Path(s):
(126(Research Award Count)) -> number_of_research_awards_of -> (Seeker University(University))
(Seeker University(University)) -> located_city -> (Zorvan(City))
(Zorvan(City)) -> sister_city -> (Xandor(City))
Intermediate Nodes: [Seeker University, Zorvan]
Start Node: 126
Answer Name: Xandor
Excellent Question: In which city is the university that has 126 research awards located?
Bad Question: Through how many research awards is the university located in the sister city of Xandor funded? (Reason: Violates Critical Rules by not asking the right question, not starting with the start node, and directly revealing the answer name.)

Example 4: 1-hop
Fact Path(s):
7(University Count) -> number_of_universities_of -> Zarnok(City)
Intermediate Nodes: []
Start Node: 7
Answer Name: Zarnok
Excellent Question: In which city are there 7 universities?
Bad Question: How many universities are located in the city where the number 7 is the number of universities? (Reason: Violates Critical Rule by not asking the right question.)

You will now receive the inputs. Generate the question. Make sure to follow all the CRITICAL RULES above.
"""

simple_qa_prompt_123_hop_template = """
You are a master instruction designer specializing in crafting complex, multi-hop reasoning instructions to benchmark advanced AI systems. Your goal is to transform a structured "fact path" into a single, natural, high-quality instruction whose answer is the final node in that path.

## CONTEXT & INPUTS

You will receive:
- **Path Type:** The number of "hops" (1-hop, 2-hop, 3-hop, etc.).
- **Fact Path(s):** A structured list of linked facts in the form (Entity A) --[relationship]--> (Entity B).
- **Answer Name:** The final entity in the path — the correct answer to your instruction.

Inputs provided:
Path Type: {path_type}
Fact Path(s):
{facts_string}
Intermediate Nodes: {intermediate_nodes}
Start Node: {start_node}
Answer Name: {answer_name}

---

## CRITICAL RULES

1. **Strictly Grounded:** Use *only* the provided Fact Path(s). Do not invent new entities, names, events, or relationships.  
2. **Conceal Hidden Nodes:** Never reveal the `{answer_name}` or any `{intermediate_nodes}` directly in the instruction.  
3. **Logical Sequence:** The instruction must begin with the **Start Node** and follow the exact order of relationships step by step.  
4. **Natural Scenario:** Wrap the relationships into a short, natural scenario or story. It should feel realistic and engaging, but not overly elaborate.  
5. **Explicit Ask:** The instruction must clearly request the final answer entity (the `{answer_name}`), without ambiguity.  
6. **Final Output:** Output *only* the instruction text — no preambles, labels, or explanations.

---

## INSTRUCTION DESIGN PROCESS

1. **Deconstruct the Path:** Identify the start, intermediates, and final nodes, and the relationships connecting them.  
2. **Anchor the Start Node:** Begin the instruction with the `{start_node}`.  
3. **Weave the Narrative:** Convert each relationship into a natural descriptive clause (e.g., “the university that she graduated from,” “the city where that university is located”).  
4. **Conclude Clearly:** End with a specific instruction that asks for the `{answer_name}` type.  
5. **Final Review:** Ensure the instruction sounds natural, logical, and conceals all hidden entities.

---

## EXAMPLES (For Your Reference, please study them carefully before generating the instruction)


Example 1: 2-hop
Fact Path(s):
(Elara Vance(Person)) --[graduated from]--> (Astral University(University))
(Astral University(University)) --[is located in]--> (Silverwind City(City))
Intermediate Nodes: [Astral University]
Start Node: Elara Vance
Answer Name: Silverwind City
Excellent Example: Please find the city where the university that Elara Vance graduated from is located.
Bad Example: Please find the city where the university, Astral University, that Elara Vance graduated from is located (Reason: Violates Rule #2 by revealing the intermediate node "Astral University").

Example 2: 3-hop
Fact Path(s):
(Elara Vance(Person)) --[graduated from]--> (Astral University(University))
(Astral University(University)) --[is located in]--> (Silverwind City(City))
(Silverwind City(City)) --[is the capital of]--> (Silverwind Country(Country))
Intermediate Nodes: [Astral University, Silverwind City]
Start Node: Elara Vance 
Answer Name: Silverwind Country
Excellent Example: Identify the country where the capital city contains the university Elara Vance graduated from.
Bad Example: Identify the country that Astral University is located in. (Reason: Violates Rule #3 by not including all relationships from the path).

Example 3: 3-hop
Fact Path(s):
(126(Research Award Count)) -> number_of_research_awards_of -> (Seeker University(University))
(Seeker University(University)) -> located_city -> (Zorvan(City))
(Zorvan(City)) -> sister_city -> (Xandor(City))
Intermediate Nodes: [Seeker University, Zorvan]
Start Node: 126
Answer Name: Xandor
Excellent Example: There is a university that has 126 research awards, and the city this university is located in has a sister city, please find the name of this sister city.
Bad Example: The university named Seeker University is located in Zorvan, which has a sister city called Xandor — what is the name of that sister city? (Reason: Violates Critical Rules by not starting with the start node, and directly revealing the answer name.)

You will now receive the inputs. Generate the instruction. Make sure to follow all the CRITICAL RULES above.
"""

simple_qa_prompt_456_hop_template = """
You are a master instruction designer specializing in crafting complex, multi-hop reasoning instructions to benchmark advanced AI systems. Your goal is to transform a structured "fact path" into a single, natural, high-quality instruction whose answer is the final node in that path.

## CONTEXT & INPUTS

You will receive:
- **Path Type:** The number of "hops" (4-hop, 5-hop, 6-hop, etc.).
- **Fact Path(s):** A structured list of linked facts in the form (Entity A) --[relationship]--> (Entity B).
- **Answer Name:** The final entity in the path — the correct answer to your instruction.

Inputs provided:
Path Type: {path_type}
Fact Path(s):
{facts_string}
Intermediate Nodes: {intermediate_nodes}
Start Node: {start_node}
Answer Name: {answer_name}

---

## CRITICAL RULES

1. **Strictly Grounded:** Use *only* the provided Fact Path(s). Do not invent new entities, names, events, or relationships.  
2. **Conceal Hidden Nodes:** Never reveal the `{answer_name}` or any `{intermediate_nodes}` directly in the instruction.  
3. **Logical Sequence:** The instruction must begin with the **Start Node** and follow the exact order of relationships step by step.  
4. **Natural Scenario:** Wrap the relationships into a short, natural scenario or story. It should feel realistic and engaging, but not overly elaborate.  
5. **Explicit Ask:** The instruction must clearly request the final answer entity (the `{answer_name}`), without ambiguity.  
6. **Final Output:** Output *only* the instruction text — no preambles, labels, or explanations.

---

## INSTRUCTION DESIGN PROCESS

1. **Deconstruct the Path:** Identify the start, intermediates, and final nodes, and the relationships connecting them.  
2. **Anchor the Start Node:** Begin the instruction with the `{start_node}`.  
3. **Weave the Narrative:** Convert each relationship into a natural descriptive clause (e.g., “the university that she graduated from,” “the city where that university is located”).  
4. **Conclude Clearly:** End with a specific instruction that asks for the `{answer_name}` type.  
5. **Final Review:** Ensure the instruction sounds natural, logical, and conceals all hidden entities.

---

## EXAMPLES (For Your Reference, please study them carefully before generating the instruction)

Example 1: 6-hop
Fact Path(s):
(313(University Rank)) --[university_rank_of]--> (Novellus Institute(University))
(Novellus Institute(University)) --[located_city]--> (Acantha(City))
(Acantha(City)) --[sister_city]--> (Qyrin(City))
(Qyrin(City)) --[located_country]--> (Volgrim(Country))
(Volgrim(Country)) --[leader_of_country_for]--> (Yasmir Falkenrath(Person))
(Yasmir Falkenrath(Person)) --[birth_year]--> (1972(Year))

Intermediate Nodes: [Novellus Institute, Acantha, Qyrin, Volgrim, Yasmir Falkenrath]
Start Node: 313
Answer Name: 1972

Excellent Instruction: Identify the exact year that satisfies the following conditions: It is the birth year of the leader of a country, and that country contains the sister city of the city where the university ranked 313 is located.

Example 2:5-hop
Fact Path(s):
(Ysoria Zolaris(Person)) --[current_living_city]--> (Rhovanor(City))
(Rhovanor(City)) --[capital_of]--> (Malakor(Country))
(Malakor(Country)) --[sister_country_of]--> (Orpheos(Country))
(Orpheos(Country)) --[capital_city]--> (Ophion(City))
(Ophion(City)) --[area]--> (133549(Area))

Intermediate Nodes: [Rhovanor, Malakor, Orpheos, Ophion]
Start Node: Ysoria Zolaris
Answer Name: 133549

Excellent Instruction: Find the area of the capital city of the sister country to the country whose capital is the city where Ysoria Zolaris currently lives.
"""

simple_qa_prompt_scenario_1_hop_template = """
You are a master question designer specializing in crafting complex, {path_type} reasoning/searching questions to benchmark advanced AI systems. Your goal is to transform a structured "fact path" into a single, natural, high-quality question whose answer is the final node in that path.

## CONTEXT & INPUTS

You will receive:
- **Fact Path(s):** A structured list of linked facts in the form (Entity A) --[relationship]--> (Entity B).
- **Answer Name:** The final entity in the path — the correct answer to your question.

Inputs provided:
Fact Path(s):
{facts_string}
Start Node: {start_node}
Answer Name: {answer_name}

---

## CRITICAL RULES

1. **Strictly Grounded:** Use *only* the provided Fact Path(s). Do not invent new entities, events, or relationships.  
2. **Conceal Hidden Nodes:** Never reveal the `{answer_name}` directly in the question.  
3. **Logical Sequence:** The question must begin with the **Start Node** and follow the exact order of relationships step by step.  
4. **Natural Scenario:** Wrap the relationships into a short, natural scenario or story. It should feel realistic and engaging, but not overly elaborate.  
5. **Specific Questioning:** Avoid vague question words like “what” or “where.” Instead, specify the category of the answer (e.g., “In which city…”, “At which company…”).  
6. **Explicit Ask:** The question must clearly request the final answer entity (the `{answer_name}`), without ambiguity.  
7. **Final Output:** Output *only* the question text — no preambles, labels, or explanations.

---

## QUESTION DESIGN PROCESS

1. **Deconstruct the Path:** Identify the start, intermediates, and final nodes, and the relationships connecting them.  
2. **Anchor the Start Node:** Begin the question with the `{start_node}`.  
3. **Weave the Narrative:** Convert each relationship into a natural descriptive clause (e.g., “the university that she graduated from,” “the city where that university is located”).  
4. **Conclude Clearly:** End with a specific question that asks for the `{answer_name}` type.  
5. **Final Review:** Ensure the question sounds natural, logical, and conceals all hidden entities.

---

## EXAMPLES (For Your Reference, please study them carefully before generating the question)

Example 1:
Fact Path(s):
Kyloq Espinay(Person) -> current_living_city -> Belltower(City)
Start Node: Kyloq Espinay
Answer Name: Belltower

Excellent Question:
Kyloq Espinay is updating a travel profile and needs the current residence listed; in which city is Kyloq Espinay currently living?

Bad Questions:
Where does Kyloq Espinay live—Belltower, right? ❌ (Reveals the answer)
What is Kyloq Espinay's location? ❌ (Too generic; not explicitly a city)
Which city is Belltower? ❌ (Asks about the answer itself rather than eliciting it)

Example 2:
Fact Path(s):
Kestrel (City) -> number_of_universities -> 1 (University Count)
Start Node: Kestrel
Answer Name: 1

Excellent Question:
Kestrel is preparing an education overview; state the number of universities located in Kestrel.

Bad Questions:
Which city has 1 university? ❌ (Does not begin with the start node and could match multiple cities)
How many universities are in Kestrel—1, correct? ❌ (Leads with or implies the answer)
What is the figure associated with Kestrel? ❌ (Too vague; not clearly about universities)

Example 3:
Fact Path(s):
114801502 (Franchise Revenue) -> franchise_revenue_of -> Riptide Propulsion (Company)
Start Node: 114801502
Answer Name: Riptide Propulsion

Excellent Question:
A business analyst was reviewing revenue reports and found a figure of 114,801,502 dollars attributed to a particular franchise. Which company generated that amount in franchise revenue?

Bad Question Examples:

“What is the franchise revenue of Riptide Propulsion?” ❌ (Reveals the answer; wrong direction.)
“Which franchise made 114801502?” ❌ (Unnatural phrasing, lacks context.)
“Riptide Propulsion earned how much?” ❌ (Inverts the path; asks the wrong thing.)

You will now receive the inputs. Generate the question. Make sure to follow all the CRITICAL RULES above.
"""

simple_qa_prompt_scenario_2_hop_template = """
You are a master question designer specializing in crafting complex, {path_type} reasoning questions to benchmark advanced AI systems. Your goal is to transform a structured "fact path" into a single, natural, high-quality question whose answer is the final node in that path.

## CONTEXT & INPUTS

You will receive:
- **Path Type:** The number of "hops" (1-hop, 2-hop, 3-hop, etc.).
- **Fact Path(s):** A structured list of linked facts in the form (Entity A) --[relationship]--> (Entity B).
- **Answer Name:** The final entity in the path — the correct answer to your question.

Inputs provided:
Fact Path(s):
{facts_string}
Intermediate Nodes: {intermediate_nodes}
Start Node: {start_node}
Answer Name: {answer_name}

---

## CRITICAL RULES

1. **Strictly Grounded:** Use *only* the provided Fact Path(s). Do not invent new entities, names, events, or relationships.  
2. **Conceal Hidden Nodes:** Never reveal the `{answer_name}` or any `{intermediate_nodes}` directly in the question.  
3. **Logical Sequence:** The question must begin with the **Start Node** and follow the exact order of relationships step by step.  
4. **Natural Scenario:** Wrap the relationships into a short, natural scenario or story. It should feel realistic and engaging, but not overly elaborate.  
5. **Specific Questioning:** Avoid vague question words like “what” or “where.” Instead, specify the category of the answer (e.g., “In which city…”, “At which company…”).  
6. **Explicit Ask:** The question must clearly request the final answer entity (the `{answer_name}`), without ambiguity.  
7. **Final Output:** Output *only* the question text — no preambles, labels, or explanations.

---

## QUESTION DESIGN PROCESS

1. **Deconstruct the Path:** Identify the start, intermediates, and final nodes, and the relationships connecting them.  
2. **Anchor the Start Node:** Begin the question with the `{start_node}`.  
3. **Weave the Narrative:** Convert each relationship into a natural descriptive clause (e.g., “the university that she graduated from,” “the city where that university is located”).  
4. **Conclude Clearly:** End with a specific question that asks for the `{answer_name}` type.  
5. **Final Review:** Ensure the question sounds natural, logical, and conceals all hidden entities.

---

## EXAMPLES (For Your Reference, please study them carefully before generating the question)

Example 1: 2-hop
Fact Path(s):
(37031562782510(GDP)) --[gdp_of]--> (Tulvir(Country))
(Tulvir(Country)) --[number_of_ethnic_groups]--> (10(Ethnic Group Count))
Intermediate Nodes:
[Tulvir]
Start Node:
37031562782510
Answer Name:
10

Excellent Question:
The GDP figure of 37031562782510 is being reviewed by a development economist compiling a diversity brief; after determining the country associated with this GDP figure, what is the number of ethnic groups recorded for that country?

Bad Question:
1. Tulvir has a GDP of 37031562782510—how many ethnic groups are there? (Reveals the intermediate node)
2. 37031562782510 is a GDP figure; name the country. (Stops at the first hop; omits the second relationship)
3. In which country are there 10 ethnic groups? (Leads with the answer value and ignores the start node and first hop)
4. Provide the number of ethnic groups in Tulvir. (Reveals intermediate node and skips the first hop logic from the start node)

Example 2: 2-hop
Fact Path(s):
(Phyla Krasnic(Person)) --[current_working_company]--> (Bioscape Terraforming(Company))
(Bioscape Terraforming(Company)) --[number_of_departments]--> (43(Department Count))
Intermediate Nodes:
[Bioscape Terraforming]
Start Node:
Phyla Krasnic
Answer Name:
43

Excellent Question:
Phyla Krasnic is preparing an internal orientation packet; for the company where Phyla Krasnic is currently employed, please find the total number of departments.

Bad Question:
1. What is the number of departments in the company where Phyla Krasnic is currently employed? (Not a natural question with scenario)
2. How many departments does Bioscape Terraforming have? (Reveals the intermediate node)
3. Phyla Krasnic works somewhere; how many departments are there? (Missing the second-hop anchor that the departments belong to the employer identified via the first hop; too vague)

Example 3: 2-hop
Fact Path(s):
(Shattrath(City)) --[sister_city_of]--> (Volantis(City))
(Volantis(City)) --[located_country]--> (Ikthos(Country))
Intermediate Nodes:
[Volantis]
Start Node:
Shattrath
Answer Name:
Ikthos

Excellent Question:
Shattrath is updating its cultural partnerships page; for the city that holds a sister-city relationship with Shattrath, name the country in which that partner city is located.

Bad Question:
1. Shattrath has a sister city; which city is it? (Stops after the first hop and does not reach the required answer type)
2. Name the sister city of Shattrath that is in Ikthos. (Reveals the answer and asks for the wrong entity)
3. Shattrath has cultural ties; name the country of its main partner. (Vague; skips the explicit sister-city relationship step)

You will now receive the inputs. Generate the question. Make sure to follow all the CRITICAL RULES above.
"""

simple_qa_prompt_scenario_3_hop_template = """
You are a master question designer specializing in crafting complex, {path_type} reasoning questions to benchmark advanced AI systems. Your goal is to transform a structured "fact path" into a single, natural, high-quality question whose answer is the final node in that path.

## CONTEXT & INPUTS

You will receive:
- **Path Type:** The number of "hops" (1-hop, 2-hop, 3-hop, etc.).
- **Fact Path(s):** A structured list of linked facts in the form (Entity A) --[relationship]--> (Entity B).
- **Answer Name:** The final entity in the path — the correct answer to your question.

Inputs provided:
Fact Path(s):
{facts_string}
Intermediate Nodes: {intermediate_nodes}
Start Node: {start_node}
Answer Name: {answer_name}

---

## CRITICAL RULES

1. **Strictly Grounded:** Use *only* the provided Fact Path(s). Do not invent new entities, names, events, or relationships.  
2. **Conceal Hidden Nodes:** Never reveal the `{answer_name}` or any `{intermediate_nodes}` directly in the question.  
3. **Logical Sequence:** The question must begin with the **Start Node** and follow the exact order of relationships step by step.  
4. **Natural Scenario:** Wrap the relationships into a short, natural scenario or story. It should feel realistic and engaging, but not overly elaborate.  
5. **Specific Questioning:** Avoid vague question words like “what” or “where.” Instead, specify the category of the answer (e.g., “In which city…”, “At which company…”).  
6. **Explicit Ask:** The question must clearly request the final answer entity (the `{answer_name}`), without ambiguity.  
7. **Final Output:** Output *only* the question text — no preambles, labels, or explanations.

---

## QUESTION DESIGN PROCESS

1. **Deconstruct the Path:** Identify the start, intermediates, and final nodes, and the relationships connecting them.  
2. **Anchor the Start Node:** Begin the question with the `{start_node}`.  
3. **Weave the Narrative:** Convert each relationship into a natural descriptive clause (e.g., “the university that she graduated from,” “the city where that university is located”).  
4. **Conclude Clearly:** End with a specific question that asks for the `{answer_name}` type.  
5. **Final Review:** Ensure the question sounds natural, logical, and conceals all hidden entities.

---

## EXAMPLES (For Your Reference, please study them carefully before generating the question)

Example 1: 3-hop
Fact Path(s):
(Brondar Drakaar(Person)) --[spouse]--> (Kryll Dornwald(Person))
(Kryll Dornwald(Person)) --[current_living_city]--> (Thunder(City))
(Thunder(City)) --[number_of_universities]--> (1(University Count))
Intermediate Nodes:
[Kryll Dornwald, Thunder]
Start Node:
Brondar Drakaar
Answer Name:
1

Excellent Question:
Brondar Drakaar’s biographer is tracing how family ties relate to local education access; beginning with Brondar Drakaar’s spouse and the city where that spouse currently lives, please find the total number of universities in that city.

Bad Question:
1. How many universities are in Thunder? (Reveals an intermediate node; doesn’t begin with the start node)
2. Starting from Brondar’s city, count universities. (Skips the spouse step; incorrect path order)
3. Name the spouse of Brondar Drakaar and their city. (Asks for intermediates, not the final answer)
4. Brondar Drakaar’s spouse lives in Thunder; confirm there is 1 university. (Reveals the intermediate city and presupposes the answer)

Example 2: 3-hop
Fact Path(s):
(14730(Employee Count)) --[number_of_employees_of]--> (Starfall Exoplanetary(Company))
(Starfall Exoplanetary(Company)) --[headquarter_city]--> (Evermore(City))
(Evermore(City)) --[located_city_of]--> (Oakhart Seminary(University))
Intermediate Nodes:
[Starfall Exoplanetary, Evermore]
Start Node:
14730
Answer Name:
Oakhart Seminary

Excellent Question:
A corporate analyst starts from a headcount figure of 14730 to identify the firm it belongs to; after determining that company's headquarters city, what is the university located in that city.

Bad Question:
1. Which university is in Evermore? (Reveals an intermediate node; ignores the required chain from the start node)
2. From the HQ city, find the company with 14730 staff. (Reverses the path and asks for extra information)
3. For the university in that city, how many employees are there? (Asks for the wrong target and muddles roles)

Example 3: 3-hop
Fact Path(s):
(Karthos(Country)) --[capital_city]--> (Cinderhollow(City))
(Cinderhollow(City)) --[sister_city]--> (Cerulea(City))
(Cerulea(City)) --[mayor_of_city_for]--> (Zennith Folara(Person))
Intermediate Nodes:
[Cinderhollow, Cerulea]
Start Node:
Karthos
Answer Name:
Zennith Folara

Excellent Question:
A person is preparing an international relations briefing; starting from Karthos's capital, then identifying that capital's sister city, what is the name of the person who serves as mayor of that sister city.

Bad Question:
1. Who is the mayor of Cerulea? (Reveals an intermediate node; skips the ordered chain from the start node)
2. Karthos has a capital; name it. (Stops after the first hop; does not reach the final entity)
3. Name the capital and its sister city. (Asks for intermediates; not the final answer)

You will now receive the inputs. Generate the question. Make sure to follow all the CRITICAL RULES above.
"""



combo_qa_prompt_template = """
You are a master question designer. Your task is to rewrite a second question (Question 2) to make it entirely dependent on the answer of a first question (Question 1).

## CONTEXT

You will be given two sequential questions. The answer to Question 1 is a piece of information required to answer Question 2. Your goal is to rephrase Question 2 by replacing the starting information with a dependent reference to the answer of Question 1.

## INPUT FORMAT

You will receive the following inputs:
*   **Question 1:** [The first question] | **Answer:** [The answer to Q1]
*   **Question 2:** [The second question, which contains the answer to Q1] | **Answer:** [The answer to Q2]

Your Inputs:
Sub Questions: 
Question 1: {sub_question1} Answer: {sub_answer1} ({sub_answer_type1})
Question 2: {sub_question2} Answer: {sub_answer2} ({sub_answer_type2})

## YOUR TASK & CRITICAL RULES

1.  **Analyze the Questions:** You will receive two questions (Question 1, Question 2) and their answers.
2.  **Identify the Link:** Find the specific piece of information (the "linking entity") in Question 2 that is identical to the answer of Question 1.
3.  **Rephrase Question 2:** Rewrite Question 2 by replacing the "linking entity" with a generic, dependent reference. This reference MUST point back to the answer of Question 1.
    *   **Examples of dependent references:** "the person from the answer of the first question," "the city you obtained from the answer of the first question"
4.  **Preserve Intent:** The rephrased question must follow the exact same logic as the original Question 2. Do not alter its core meaning and logic, your job is only to replace the starting information with a dependent reference to the answer of Question 1.


## EXAMPLE

### Example 1:

### Your Input:

*   **Question 1:** How many faculties does the university located in the birth city of Jackson Barton's spouse have? | **Answer:** 3 (Faculty Count)
*   **Question 2:** Who is the mother of the leader of the country where the city that headquarters a company with 3 products is located? | **Answer:** Devereux Bryan (Person)

### Your Required Output:
Example 1: Who is the mother of the leader of the country where the city that headquarters a company with n products is located, where n is the **numerical answer** to the first question.
Example 2: Who is the mother of the leader of the country where the city that headquarters a company with a number of products **equal to the answer of the first question** is located?

### Example 2:

### Your Input:

*   **Question 1:** What is the birth year of the mother of the person who is the mayor of the city where the university that has 179 research patents is located? | **Answer:** 1980 (Year)
*   **Question 2:** In which city is the company founded in the same year you obtained in Question 1 headquartered? | **Answer:** Zarnok (City)

### Your Required Output:
In which city is the company founded in the same year that you obtained from Question 1 headquartered?

## FINAL OUTPUT INSTRUCTIONS

Your final output must be the raw text of the slightly rephrased Question 2.**
*   Do NOT include any preamble, titles, or labels (like "Rephrased Question:").
*   Do NOT include explanations or markdown.

"""

parallel_same_numerical_sum_prompt_template = """
You are a master question designer specializing in creating a parallel question based on two sub-questions whose answers are the same type of numerical information.

## CONTEXT

You will be given two sub-questions and their answers. Your goal is to combine the two sub-questions into a single parallel question, where the answers of the sub-questions are of the same type and are numerical, e.g. Year, GDP, Population Count, and you should make up a new question that asks the summation of the answers of the two sub-questions in a natural way.

## INPUT FORMAT

You will receive the following inputs:
*   **Question 1:** [The first question] | **Answer:** [The answer to Q1]
*   **Question 2:** [The second question] | **Answer:** [The answer to Q2]

Your Inputs:
Sub Questions:
Question 1: {sub_question1} Answer: {sub_answer1} ({sub_answer_type1})
Question 2: {sub_question2} Answer: {sub_answer2} ({sub_answer_type2})

## CRITICAL RULES

1.  **Analyze the Questions:** You will receive two questions (Question 1, Question 2) and their answers.
2.  **Remove Scenario Descriptions:** IMPORTANT - When combining the sub-questions, you MUST remove any scenario descriptions, context explanations, or situational details from the original questions. Only keep the core question structure and entity information.
3.  **Identify the Link:** Based on the answers and questions, combine the two sub-questions and make up a new question that asks the summation of the answers of the two sub-questions in a natural way.
4.  **Keep the orginal logic:** While connecting the two sub-questions, you should strictly keep the original logic of the each sub-questions unchanged, but without scenario descriptions.
5.  **Final Output Format:** Your final output must be a JSON object with the following format:
    {
        "Question": "The combined question",
        "Answer": "The answer to the combined question"
    }

## Example

### Your Input:

*   **Question 1:** Khepran Quintoris is filling out a biographical form and needs to specify his birth year; in which year was Khepran Quintoris born? | **Answer:** 1968 (Year)
*   **Question 2:** Xantheus Varden is updating his biography and needs to include his birth year; in which year was Xantheus Varden born? | **Answer:** 1978 (Year)

### Your Required Output:
```json
{{
    "Question": "What is the summation of the birth years of Khepran Quintoris and Xantheus Varden?",
    "Answer": "3946"
}}
```

## Additional Example

### Your Input:

*   **Question 1:** What is the GDP of the country whose capital is City A? | **Answer:** 10000 (GDP)
*   **Question 2:** How much is the GDP of the country whose leader is the person whose spouse is person X? | **Answer:** 20000 (GDP)

### Your Required Output:
```json
{{
    "Question": "What is the summation of the GDP of the country whose leader is the person whose spouse is person X and the GDP of the country whose capital is City A?",
    "Answer": "30000"
}}
```
"""

parallel_same_numerical_difference_prompt_template = """
You are a master question designer specializing in creating a parallel question based on two sub-questions whose answers are the same type of numerical information.

## CONTEXT

You will be given two sub-questions and their answers. Your goal is to combine the two sub-questions into a single parallel question, where the answers of the sub-questions are of the same type and are numerical, e.g. Year, GDP, Population Count, and you should make up a new question that asks the absolute difference of the answers of the two sub-questions in a natural way.

## INPUT FORMAT

You will receive the following inputs:
*   **Question 1:** [The first question] | **Answer:** [The answer to Q1]
*   **Question 2:** [The second question] | **Answer:** [The answer to Q2]

Your Inputs:
Sub Questions:
Question 1: {sub_question1} Answer: {sub_answer1} ({sub_answer_type1})
Question 2: {sub_question2} Answer: {sub_answer2} ({sub_answer_type2})

## CRITICAL RULES

1.  **Analyze the Questions:** You will receive two questions (Question 1, Question 2) and their answers.
2.  **Remove Scenario Descriptions:** IMPORTANT - When combining the sub-questions, you MUST remove any scenario descriptions, context explanations, or situational details from the original questions. Only keep the core question structure and entity information.
3.  **Identify the Link:** Based on the answers and questions, combine the two sub-questions and make up a new question that asks the absolute difference of the answers of the two sub-questions in a natural way.
4.  **Keep the orginal logic:** While connecting the two sub-questions, you should strictly keep the original logic of the each sub-questions unchanged, but without scenario descriptions.
5.  **Final Output Format:** Your final output must be a JSON object with the following format:
    {
        "Question": "The combined question",
        "Answer": "The answer to the combined question"
    }

## Example

### Your Input:

*   **Question 1:** Khepran Quintoris is filling out a biographical form and needs to specify his birth year; in which year was Khepran Quintoris born? | **Answer:** 1968 (Year)
*   **Question 2:** Xantheus Varden is updating his biography and needs to include his birth year; in which year was Xantheus Varden born? | **Answer:** 1978 (Year)

### Your Required Output:
```json
{{
    "Question": "What is the absolute difference in birth years between Khepran Quintoris and Xantheus Varden?",
    "Answer": "10"
}}
```

## Additional Example

### Your Input:

*   **Question 1:** What is the GDP of the country whose capital is City A? | **Answer:** 10000 (GDP)
*   **Question 2:** How much is the GDP of the country whose leader is the person whose spouse is person X? | **Answer:** 20000 (GDP)

### Your Required Output:
```json
{{
    "Question": "What is the absolute difference in GDP between the country whose leader is the person whose spouse is person X and the country whose capital is City A?",
    "Answer": "10000"
}}
```
"""

parallel_same_entity_compare_prompt_template = """
You are a master question designer specializing in creating a parallel question based on two sub-questions whose answers are the same type of entity.

## CONTEXT

You will be given two sub-questions and their answers that are of the same type of entity, together with the numerical attribute of the answer entities. Your goal is to combine the two sub-questions into a single parallel question, where the answers of the sub-questions are of the same type of entity, e.g. Person, Company, City, University, and you should make up a new question that compares the attributes of the answer entities in a natural way.

## INPUT FORMAT

You will receive the following inputs:
*   **Question 1:** [The first question] | **Answer:** [The answer to Q1] | **Answer Entity Type:** [The type of the answer entity] | **Answer Entity Attribute:** [The attribute of the answer entity]
*   **Question 2:** [The second question] | **Answer:** [The answer to Q2] | **Answer Entity Type:** [The type of the answer entity] | **Answer Entity Attribute:** [The attribute of the answer entity]

Your Inputs:
Sub Questions:
Question 1: {sub_question1} | Answer: {sub_answer1} | Answer Entity Type: {sub_answer_entity_type1} | Answer Entity Attribute Relation: {sub_answer_entity_attribute_relation1} | Attribute Value: {sub_answer_entity_attribute_value1}
Question 2: {sub_question2} | Answer: {sub_answer2} | Answer Entity Type: {sub_answer_entity_type2} | Answer Entity Attribute Relation: {sub_answer_entity_attribute_relation2} | Attribute Value: {sub_answer_entity_attribute_value2}

## CRITICAL RULES

1.  **Analyze the Questions:** You will receive two questions (Question 1, Question 2), their answers and attributes of the answer entities.
2.  **Identify the Link:** Based on the answers, attributes of answer entities and questions, combine the two sub-questions and make up a new question that asks the comparison of the attributes of the answer entities in a natural way.
3.  **Keep the orginal logic:** While connecting the two sub-questions, you should strictly keep the original logic of the each sub-questions unchanged.
4.  **Final Output Format:** Your final output must be a JSON object with the following format:
    {
        "Question": "The combined question",
        "Answer": "The answer to the combined question"
    }

## Example

### Your Input:

*   Question 1: "What is the mayor of the city where the Person A was born?" | Answer: Person B | Answer Entity Type: Person | Answer Entity Attribute Relation: Person B -> birth_year -> 1968 | Attribute Value: 1968
*   Question 1: "Who is the leader of the sister country to the country where the headquarter city of Company X is located?" | Answer: Person C | Answer Entity Type: Person | Answer Entity Attribute Relation: Person C -> birth_year -> 1966 | Attribute Value: 1966

### Your Required Output:
```json
{{
    "Question": "Who is older, the mayor of the city where the Person A was born, or the leader of the sister country to the country where the headquarter city of Company X is located?",
    "Answer": "Person C"
}}
```
"""

parallel_same_entity_sum_prompt_template = """
You are a master question designer specializing in creating a parallel question based on two sub-questions whose answers are the same type of entity.

## CONTEXT

You will be given two sub-questions and their answers that are of the same type of entity, together with the numerical attribute of the answer entities. Your goal is to combine the two sub-questions into a single parallel question, where the answers of the sub-questions are of the same type of entity, e.g. Person, Company, City, University, and you should make up a new question that asks the SUM of the numerical attributes of the answer entities in a natural way.

## INPUT FORMAT

You will receive the following inputs:
*   **Question 1:** [The first question] | **Answer:** [The answer to Q1] | **Answer Entity Type:** [The type of the answer entity] | **Answer Entity Attribute Relation:** [The attribute relation of the answer entity] | **Attribute Value:** [The numerical value of the attribute]
*   **Question 2:** [The second question] | **Answer:** [The answer to Q2] | **Answer Entity Type:** [The type of the answer entity] | **Answer Entity Attribute Relation:** [The attribute relation of the answer entity] | **Attribute Value:** [The numerical value of the attribute]

Your Inputs:
Sub Questions:
Question 1: {sub_question1} | Answer: {sub_answer1} | Answer Entity Type: {sub_answer_entity_type1} | Answer Entity Attribute Relation: {sub_answer_entity_attribute_relation1} | Attribute Value: {sub_answer_entity_attribute_value1}
Question 2: {sub_question2} | Answer: {sub_answer2} | Answer Entity Type: {sub_answer_entity_type2} | Answer Entity Attribute Relation: {sub_answer_entity_attribute_relation2} | Attribute Value: {sub_answer_entity_attribute_value2}

## CRITICAL RULES

1.  **Analyze the Questions:** You will receive two questions (Question 1, Question 2), their answers and numerical attributes of the answer entities.
2.  **Remove Scenario Descriptions:** IMPORTANT - When combining the sub-questions, you MUST remove any scenario descriptions, context explanations, or situational details from the original questions. Only keep the core question structure and entity information.
3.  **Identify the Link:** Based on the answers, numerical attributes of answer entities and questions, combine the two sub-questions and make up a new question that asks the SUM of the numerical attributes of the answer entities in a natural way.
4.  **Keep the original logic:** While connecting the two sub-questions, you should strictly keep the original logic of each sub-question unchanged, but without scenario descriptions.
5.  **Final Output Format:** Your final output must be a JSON object with the following format:
    {
        "Question": "The combined question",
        "Answer": "The answer to the combined question"
    }

## Examples

### Example 1: Sum Operation

### Your Input:

*   Question 1: "Sisyphon Kimaro is updating a biographical form and needs to specify his best friend's information; who is the best friend of Sisyphon Kimaro?" | Answer: Morphax Serpentine | Answer Entity Type: Person | Answer Entity Attribute Relation: Morphax Serpentine -> birth_year -> 1987 | Attribute Value: 1987
*   Question 2: "Ruolan Ravenscar is preparing a social network map and needs to identify connections; who is the best friend of Ruolan Ravenscar?" | Answer: Lycomed Septimus | Answer Entity Type: Person | Answer Entity Attribute Relation: Lycomed Septimus -> birth_year -> 1964 | Attribute Value: 1964

### Your Required Output:
```json
{{
    "Question": "What is the summation of the birth years of the best friend of Sisyphon Kimaro and the best friend of Ruolan Ravenscar?",
    "Answer": "3951"
}}
```

### Example 2: Difference Operation with Complex Questions

### Your Input:

*   Question 1: "Starting from the country Sorgan, identify the best friend of the mayor of its sister country's capital city's sister city." | Answer: Person A | Answer Entity Type: Person | Answer Entity Attribute Relation: Person A -> birth_year -> 1980 | Attribute Value: 1980
*   Question 2: "Which person serves as the mayor of the sister city of the city in which the university Kronus Takamura attended is located?" | Answer: Person B | Answer Entity Type: Person | Answer Entity Attribute Relation: Person B -> birth_year -> 1964 | Attribute Value: 1964

### Your Required Output:
```json
{{
    "Question": "What is the absolute difference in birth years between the best friend of the mayor of country Sorgan's sister country's capital city's sister city and the mayor of the sister city of the city where Kronus Takamura attended university?",
    "Answer": "16"
}}
```
"""

parallel_same_entity_difference_prompt_template = """
You are a master question designer specializing in creating a parallel question based on two sub-questions whose answers are the same type of entity.

## CONTEXT

You will be given two sub-questions and their answers that are of the same type of entity, together with the numerical attribute of the answer entities. Your goal is to combine the two sub-questions into a single parallel question, where the answers of the sub-questions are of the same type of entity, e.g. Person, Company, City, University, and you should make up a new question that asks the ABSOLUTE DIFFERENCE of the numerical attributes of the answer entities in a natural way.

## INPUT FORMAT

You will receive the following inputs:
*   **Question 1:** [The first question] | **Answer:** [The answer to Q1] | **Answer Entity Type:** [The type of the answer entity] | **Answer Entity Attribute Relation:** [The attribute relation of the answer entity] | **Attribute Value:** [The numerical value of the attribute]
*   **Question 2:** [The second question] | **Answer:** [The answer to Q2] | **Answer Entity Type:** [The type of the answer entity] | **Answer Entity Attribute Relation:** [The attribute relation of the answer entity] | **Attribute Value:** [The numerical value of the attribute]

Your Inputs:
Sub Questions:
Question 1: {sub_question1} | Answer: {sub_answer1} | Answer Entity Type: {sub_answer_entity_type1} | Answer Entity Attribute Relation: {sub_answer_entity_attribute_relation1} | Attribute Value: {sub_answer_entity_attribute_value1}
Question 2: {sub_question2} | Answer: {sub_answer2} | Answer Entity Type: {sub_answer_entity_type2} | Answer Entity Attribute Relation: {sub_answer_entity_attribute_relation2} | Attribute Value: {sub_answer_entity_attribute_value2}

## CRITICAL RULES

1.  **Analyze the Questions:** You will receive two questions (Question 1, Question 2), their answers and numerical attributes of the answer entities.
2.  **Remove Scenario Descriptions:** IMPORTANT - When combining the sub-questions, you MUST remove any scenario descriptions, context explanations, or situational details from the original questions. Only keep the core question structure and entity information.
3.  **Identify the Link:** Based on the answers, numerical attributes of answer entities and questions, combine the two sub-questions and make up a new question that asks the ABSOLUTE DIFFERENCE of the numerical attributes of the answer entities in a natural way.
4.  **Keep the original logic:** While connecting the two sub-questions, you should strictly keep the original logic of each sub-question unchanged, but without scenario descriptions.
5.  **Final Output Format:** Your final output must be a JSON object with the following format:
    {
        "Question": "The combined question",
        "Answer": "The answer to the combined question"
    }

## Examples

### Example 1: Difference Operation

### Your Input:

*   Question 1: "Sisyphon Kimaro is updating a biographical form and needs to specify his best friend's information; who is the best friend of Sisyphon Kimaro?" | Answer: Morphax Serpentine | Answer Entity Type: Person | Answer Entity Attribute Relation: Morphax Serpentine -> birth_year -> 1987 | Attribute Value: 1987
*   Question 2: "Ruolan Ravenscar is preparing a social network map and needs to identify connections; who is the best friend of Ruolan Ravenscar?" | Answer: Lycomed Septimus | Answer Entity Type: Person | Answer Entity Attribute Relation: Lycomed Septimus -> birth_year -> 1964 | Attribute Value: 1964

### Your Required Output:
```json
{{
    "Question": "What is the absolute difference in birth years between the best friend of Sisyphon Kimaro and the best friend of Ruolan Ravenscar?",
    "Answer": "23"
}}
```

### Example 2: Difference Operation with Complex Questions

### Your Input:

*   Question 1: "Starting from the country Sorgan, identify the best friend of the mayor of its sister country's capital city's sister city." | Answer: Person A | Answer Entity Type: Person | Answer Entity Attribute Relation: Person A -> birth_year -> 1980 | Attribute Value: 1980
*   Question 2: "Which person serves as the mayor of the sister city of the city in which the university Kronus Takamura attended is located?" | Answer: Person B | Answer Entity Type: Person | Answer Entity Attribute Relation: Person B -> birth_year -> 1964 | Attribute Value: 1964

### Your Required Output:
```json
{{
    "Question": "What is the absolute difference in birth years between the best friend of the mayor of country Sorgan's sister country's capital city's sister city and the mayor of the sister city of the city where Kronus Takamura attended university?",
    "Answer": "16"
}}
```
"""