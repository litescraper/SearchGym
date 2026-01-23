HTML_WIKI_PROMPT_TEMPLATE = """
You are a specialized content generation system. Your primary task is to create a JSON object containing two key pieces of information: a brief summary (abstract) and a complete, realistic, Wikipedia-style HTML page for an encyclopedia of a fictional world. But don't mention anything related to "Fictional" in the content.

Your goal is to generate this structured JSON output based on the core facts provided.

---
**Core Fact Sheet for "{node_name}" ({node_type})**
This information is the absolute truth and MUST be accurately and naturally integrated into both the abstract and the main HTML content.
{core_facts_string}
---

**Selected Facts for Abstract**
The following facts have been pre-selected for the abstract. You MUST use ONLY these facts to generate the abstract:
{abstract_facts}
---

**Generation Tasks & Output Structure:**

You must perform two tasks and then combine their results into a final JSON object.

**Task 1: Generate the Abstract**
First, write a concise and engaging summary of the entity "{node_name}". This summary will be the value for the "abstract" key in the final JSON.
-   **Strict Information Restriction**: The abstract MUST ONLY use the facts provided in "Selected Facts for Abstract" above. Do NOT use any other facts from the Core Fact Sheet, invent new information, add external knowledge, or include any details not explicitly listed in the pre-selected facts.
-   **Content**: Base the abstract ENTIRELY on the pre-selected facts. Do not elaborate or add additional context.

**Task 2: Generate the HTML Content**
Second, generate the full HTML document for the encyclopedia page. This will be the value for the "content" key in the final JSON. Use the provided HTML template structure:

{html_template}

**Content Elaboration Rules for HTML:**
-   **Adherence to Facts**: Strictly adhere to the complete "Core Fact Sheet" (not just the abstract facts)
-   **Plausible Details**: Invent plausible, non-contradictory details to enrich the text
-   **No New Specifics**: Do not invent new, specific, named entities, dates, or numbers not in the Core Fact Sheet
-   **No Fiction Relationships**: Do not invent relationships not present in the Core Fact Sheet

---
**Final Output Format: A Single JSON Object**

Your final output MUST be a single, valid JSON object and nothing else. It should follow this exact structure:

```json
{{
  "abstract": "The brief summary using ONLY the pre-selected facts goes here.",
  "content": "The generated HTML content based on the template goes here"
}}

Crucial: Do NOT include any explanations, comments, or markdown formatting like json ... around the output. The entire response must start with {{ and end with }}.
"""


PLAIN_TEXT_WIKI_PROMPT_TEMPLATE = """
You are a senior editor for the official encyclopedia of a fictional world. Your task is to write a vivid, detailed, and internally consistent encyclopedia article for the specified entity, based on the structured core facts provided below.
Entity Name: {node_name}
Entity Type: {node_type}

Core Fact Sheet (This information MUST be naturally integrated into the article):

{core_facts_string}
Writing Instructions:

**Truthfulness to Facts**: Strictly adhere to the "Core Fact Sheet" provided above. These facts are the absolute truth within this world, and apart from the numbers, date, and names and so on that are provided in the core facts, you should not add anything specific to the article.

**Enrich with Detail (Inject Noise)**: Elaborate on the core facts. You are encouraged to invent plausible, non-contradictory details about the entity itself/himself/herself to make the article feel more authentic. Yet, never add anything specific like names, dates, or numbers to the article, and never ellaborate on the entities (like companies, universities, cities, countries, etc.) core facts, also never add relationships of any kind that are not in the core facts. For example, assuming one's spouse is from the same country, you should not add that the spouse is from the same country.

**Encyclopedic Style**: Use an objective, neutral, third-person narrative style. The tone should be informative and authoritative.

**Length**: The article should be between 300 and 400 words.

Output Format: Return only the body of the article. Do not include any extra titles, headings, comments, or explanations.
"""