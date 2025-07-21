import json
from vLLM_Engine import vLLM_Engine
import os
import re
import pyarrow.parquet as pq
import random

CONFLICT_CHECK_PROMPT = '''You are an expert in constraint validation and logical consistency checking.

Your task is to analyze the given set of constraints (evaluation functions and verification questions) and determine whether:
- They contain **conflicting requirements**
- They impose **impossible or mutually exclusive conditions**
- They have **hidden risks** that would make it extremely difficult or impossible for a language model to satisfy all constraints simultaneously

Functions:
{functions}
Questions:
{questions}

Output:
"yes" if there are conflicts or risks
"no" if the constraints are consistent and feasible

Do not output any explanation, only "yes" or "no"
'''

Prompt = '''You are a helpful assistant that generates instructions and evaluation functions for response validation.

Given:
1. A persona description  
2. Multiple Python evaluation functions (with placeholders for parameters)
3. A list of natural language verification questions

Your task is to:  
- Generate an **instruction** that reflects the **persona** and includes two types of constraints:
  - **Objective constraints**, which are clearly verifiable via the provided evaluation functions
  - **Subjective qualities**, which are evaluable through the provided natural language verification questions
- Output only the **instruction**, followed by the **evaluation functions with concrete parameters**, and the **verification questions**, each wrapped in ```python``` code blocks  
- Only specify the parameters of the evaluate function; do not modify the function name and function body (e.g., avoid adding extra code like nltk.download)
- **Must contain** the "import" information such as "import re" or "import nltk" in the function body

Example Input:

Persona: You are a middle school science teacher helping students write clear experiment reports.  
Functions:
```python
def evaluate_max_words(response, max_words):
    """Validate total word count <= max_words."""
    import nltk
    from nltk.tokenize import word_tokenize
    words = word_tokenize(response)
    return len(words) <= max_words
```
```python
def evaluate_max_words_per_sentence(response, max_per_sentence):
    """Ensure no sentence exceeds max_per_sentence words."""
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    sentences = sent_tokenize(response)
    for sent in sentences:
        if len(word_tokenize(sent)) > max_per_sentence:
            return False
    return True
```
Questions:
```python
Is the originality in the response meaningful and contextually appropriate, rather than artificially imposed or distracting?
```
```python
Does the response provide a clear conclusion supported by the experimental observations?
```

Example Output:

Instruction: Write a science experiment report suitable for middle school, with no more than 300 words total and no sentence exceeding 25 words. Ensure the originality of the response is meaningful and fits the context naturally. The conclusion should clearly reflect the observations made in the experiment.
```python
def evaluate_max_words(response, max_words=300):
    """Validate total word count <= max_words."""
    import nltk
    from nltk.tokenize import word_tokenize
    words = word_tokenize(response)
    return len(words) <= max_words
```
```python
def evaluate_max_words_per_sentence(response, max_per_sentence=25):
    """Ensure no sentence exceeds max_per_sentence words."""
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    sentences = sent_tokenize(response)
    for sent in sentences:
        if len(word_tokenize(sent)) > max_per_sentence:
            return False
    return True
```
```python
Is the originality in the response meaningful and contextually appropriate, rather than artificially imposed or distracting?
```
```python
Does the response provide a clear conclusion supported by the experimental observations?
```

Now generate the output based on the following persona, evaluation functions, and verification questions:
Persona: {persona}
Functions:
{functions}
Questions:
{questions}
'''



functions = {
    "response_language_checker": '''
def evaluate(response, language):
    """Verify if the entire response is in the specified language"""
    import langdetect
    try:
        return langdetect.detect(response) == language
    except langdetect.LangDetectException:
        return False  # Changed from True to False as undetectable language should fail
''',

    "min_sentences_checker": '''
def evaluate(response, min_sentences):
    """Verify if response contains at least min_sentences sentences"""
    import nltk
    from nltk import data
    tokenizer = data.load("tokenizers/punkt/english.pickle")
    tokenized_sentences = tokenizer.tokenize(response)  # Fixed variable name from 'text' to 'response'
    return len(tokenized_sentences) >= min_sentences
''',

    "max_sentences_checker": '''
def evaluate(response, max_sentences):
    """Verify if response contains less than max_sentences sentences"""
    import nltk
    from nltk import data
    tokenizer = data.load("tokenizers/punkt/english.pickle")
    tokenized_sentences = tokenizer.tokenize(response)  # Fixed variable name
    return len(tokenized_sentences) < max_sentences
''',

    "placeholder_checker": '''
def evaluate(response, min_placeholders):
    """Verify if response contains at least min_placeholders placeholders"""
    import re
    placeholders = re.findall(r"\$\$.*?\$\$", response)  # Fixed regex pattern
    return len(placeholders) >= min_placeholders
''',

    "bullet_list_checker": '''
def evaluate(response, exact_bullets, marker):
    """Verify if response contains exactly exact_bullets bullet points with specified marker"""
    import re
    pattern = r"^\s*" + re.escape(marker) + r"\s+.*$"  # Improved pattern to match bullet points
    bullets = re.findall(pattern, response, flags=re.MULTILINE)
    return len(bullets) == exact_bullets
''',

    "constrained_response_checker": '''
def evaluate(response, allowed_responses):
    """Verify if response matches one of the allowed_responses"""
    response = response.strip()
    return any(allowed in response for allowed in allowed_responses)
''',

    "constrained_start_checker": '''
def evaluate(response, required_start):
    """Verify if response starts with the required_start phrase"""
    return response.strip().startswith(required_start)
''',

    "highlight_section_checker": '''
def evaluate(response, min_highlights, marker):
    """Verify if response contains min_highlights highlighted sections with specified marker"""
    import re
    pattern = re.escape(marker) + r"[^\n" + re.escape(marker) + r"]*" + re.escape(marker)
    highlights = re.findall(pattern, response)
    return len(highlights) >= min_highlights
''',

    "section_checker": '''
def evaluate(response, section_marker, min_sections):
    """Verify if response contains at least min_sections sections marked with section_marker"""
    import re
    pattern = r"\s?" + section_marker + r"\s?\d+\s?"
    sections = re.split(pattern, response)
    return len(sections) - 1 >= min_sections
''',

    "paragraph_checker": '''
def evaluate(response, exact_paragraphs):
    """Verify if response contains exactly exact_paragraphs paragraphs"""
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    valid_paragraphs = [p for p in paragraphs if p.strip()]
    return len(valid_paragraphs) == exact_paragraphs
''',

    "min_paragraph_checker": '''
def evaluate(response, min_paragraphs):
    """Verify if response contains at least min_paragraphs paragraphs"""
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    valid_paragraphs = [p for p in paragraphs if p.strip()]
    return len(valid_paragraphs) >= min_paragraphs
''',

    "max_paragraph_checker": '''
def evaluate(response, max_paragraphs):
    """Verify if response contains less than max_paragraphs paragraphs"""
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    valid_paragraphs = [p for p in paragraphs if p.strip()]
    return len(valid_paragraphs) < max_paragraphs
''',

    "postscript_checker": '''
def evaluate(response, postscript_marker):
    """Verify if response ends with postscript_marker"""
    import re
    if postscript_marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + postscript_marker.lower() + r".*$"
    return bool(re.search(pattern, response.lower(), flags=re.MULTILINE))
''',

    "table_format_checker": '''
def evaluate(response, min_rows):
    """Verify if response contains a properly formatted table with at least min_rows"""
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    table_lines = sum(1 for line in lines if '|' in line and '-' not in line)
    return table_lines >= min_rows
''',

    "xml_format_checker": '''
def evaluate(response):
    """Verify if response is well-formed XML"""
    from xml.etree import ElementTree
    try:
        ElementTree.fromstring(response)
        return True
    except ElementTree.ParseError:
        return False
''',

    "yaml_format_checker": '''
def evaluate(response):
    """Verify if response is valid YAML"""
    import yaml
    try:
        yaml.safe_load(response)
        return True
    except yaml.YAMLError:
        return False
''',

    "header_hierarchy_checker": '''
def evaluate(response):
    """Verify if Markdown headers follow proper hierarchy (no H3 without H2, etc.)"""
    import re
    headers = re.findall(r'^(#+)\s', response, flags=re.MULTILINE)
    if not headers:
        return True
    levels = [len(h) for h in headers]
    return all(levels[i] <= levels[i-1]+1 for i in range(1, len(levels)))
''',

    "min_line_length_checker": '''
def evaluate(response, min_chars):
    """Verify if all lines have at least min_chars characters"""
    return all(len(line.strip()) >= min_chars 
             for line in response.split('\n') if line.strip())
''',

    "max_line_length_checker": '''
def evaluate(response, max_chars):
    """Verify if no line exceeds max_chars characters"""
    return all(len(line) <= max_chars for line in response.split('\n'))
''',

    "min_line_word_count": '''
def evaluate(response, min_words):
    """Verify if all lines contain at least min_words words"""
    import nltk
    return all(len(nltk.word_tokenize(line)) >= min_words 
              for line in response.split('\n') if line.strip())
''',

    "max_line_word_count": '''
def evaluate(response, max_words):
    """Verify if no line exceeds max_words words"""
    import nltk
    return all(len(nltk.word_tokenize(line)) <= max_words 
              for line in response.split('\n'))
''',

    "min_paragraph_length_checker": '''
def evaluate(response, min_paragraphs, min_lines):
    """Verify if at least min_paragraphs paragraphs have min_lines lines each"""
    paragraphs = [p for p in response.split('\n\n') if p.strip()]
    valid_paragraphs = sum(1 for p in paragraphs 
                          if len([l for l in p.split('\n') if l.strip()]) >= min_lines)
    return valid_paragraphs >= min_paragraphs
''',

    "keyword_presence_checker": '''
def evaluate(response, required_keywords):
    """Verify if response contains all required_keywords"""
    import re
    return all(re.search(r"\b" + kw + r"\b", response, flags=re.IGNORECASE) 
              for kw in required_keywords)
''',

    "min_keyword_frequency": '''
def evaluate(response, keyword, min_frequency):
    """Verify if keyword appears at least min_frequency times"""
    import re
    return len(re.findall(keyword, response, flags=re.IGNORECASE)) >= min_frequency
''',

    "max_keyword_frequency": '''
def evaluate(response, keyword, max_frequency):
    """Verify if keyword appears less than max_frequency times"""
    import re
    return len(re.findall(keyword, response, flags=re.IGNORECASE)) < max_frequency
''',

    "min_word_count": '''
def evaluate(response, min_words):
    """Verify if response contains at least min_words words"""
    import nltk
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(response)
    num_words = len(tokens)
    return num_words >= min_words
''',

    "max_word_count": '''
def evaluate(response, max_words):
    """Verify if response contains less than max_words words"""
    import nltk
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(response)
    num_words = len(tokens)
    return num_words < max_words
''',

    "json_format_checker": '''
def evaluate(response):
    """Verify if response is valid JSON"""
    import json
    try:
        json.loads(response.strip().removeprefix("```json").removesuffix("```"))
        return True
    except ValueError:
        return False
''',

    "paragraph_first_word_checker": '''
def evaluate(response, paragraph_num, first_word):
    """Verify if specified paragraph starts with first_word"""
    paragraphs = [p.strip() for p in re.split(r"\n\n", response) if p.strip()]
    if len(paragraphs) < paragraph_num:
        return False
    paragraph = paragraphs[paragraph_num-1]
    first = paragraph.split()[0].strip(\'\'\'\"\'.,?!\'\'\').lower()
    return first == first_word.lower()
''',

    "key_sentences_checker": '''
def evaluate(response, required_sentences, exact_count):
    """Verify if response contains exactly exact_count of required_sentences"""
    import re
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(response)
    return sum(1 for req in required_sentences 
              if any(req.lower() in sent.lower() for sent in sentences)) == exact_count
''',

    "forbidden_words_checker": '''
def evaluate(response, forbidden_words):
    """Verify if response contains none of the forbidden_words"""
    import re
    return not any(re.search(r"\b" + word + r"\b", response, flags=re.IGNORECASE)
                 for word in forbidden_words)
''',

    "min_letter_frequency": '''
def evaluate(response, letter, min_frequency):
    """Verify if letter appears at least min_frequency times"""
    return response.lower().count(letter.lower()) >= min_frequency
''',

    "max_letter_frequency": '''
def evaluate(response, letter, max_frequency):
    """Verify if letter appears less than max_frequency times"""
    return response.lower().count(letter.lower()) < max_frequency
''',

    "all_capital_english_checker": '''
def evaluate(response):
    """Verify if response is in English and all capital letters"""
    import langdetect
    try:
        return response.isupper() and langdetect.detect(response) == "en"
    except langdetect.LangDetectException:
        return True
''',

    "all_lowercase_english_checker": '''
def evaluate(response):
    """Verify if response is in English and all lowercase letters"""
    import langdetect
    try:
        return response.islower() and langdetect.detect(response) == "en"
    except langdetect.LangDetectException:
        return True
''',

    "no_commas_checker": '''
def evaluate(response):
    """Verify if response contains no commas"""
    return "," not in response
''',

    "min_capital_words": '''
def evaluate(response, min_capital_words):
    """Verify if response contains at least min_capital_words all-caps words"""
    import nltk
    words = nltk.word_tokenize(response)
    return sum(1 for word in words if word.isupper()) >= min_capital_words
''',

    "max_capital_words": '''
def evaluate(response, max_capital_words):
    """Verify if response contains less than max_capital_words all-caps words"""
    import nltk
    words = nltk.word_tokenize(response)
    return sum(1 for word in words if word.isupper()) < max_capital_words
''',

    "quotation_wrapped_checker": '''
def evaluate(response):
    """Verify if response is wrapped in double quotes"""
    response = response.strip()
    return len(response) > 1 and response[0] == '"' and response[-1] == '"'
''',

    "two_responses_checker": '''
def evaluate(response):
    """Verify if response contains exactly two distinct responses separated by **"""
    parts = [p.strip() for p in response.split("**") if p.strip()]
    return len(parts) == 2 and parts[0] != parts[1]
''',

    "repeat_prompt_checker": '''
def evaluate(response, prompt):
    """Verify if response starts by repeating the exact prompt"""
    return response.strip().lower().startswith(prompt.strip().lower())
''',

    "ending_phrase_checker": '''
def evaluate(response, ending_phrase):
    """Verify if response ends with the exact ending_phrase"""
    return response.strip().strip('"').lower().endswith(ending_phrase.strip().lower())
''',

    "title_checker": '''
def evaluate(response, start_marker, end_marker):
    """Verify if response contains a title between specified markers"""
    import re
    pattern = re.escape(start_marker) + r"[^\n]+?" + re.escape(end_marker)
    titles = re.findall(pattern, response)
    return any(t[len(start_marker):-len(end_marker)].strip() for t in titles)
''',

    "json_nested_level": '''
def evaluate(response, max_level):
    """Verify if the JSON in response exceeds a specified nesting level"""
    import json
    try:
        data = json.loads(response)
        
        def check_level(obj, current_level=0):
            if current_level > max_level:
                return True
            if isinstance(obj, dict):
                return any(check_level(v, current_level + 1) for v in obj.values())
            elif isinstance(obj, list):
                return any(check_level(item, current_level + 1) for item in obj)
            return False
        
        return check_level(data)
    except json.JSONDecodeError:
        return False
''',

    "markdown_link_checker": '''
def evaluate(response, min_links):
    """Verify if response contains at least min_links markdown-style links"""
    import re
    links = re.findall(r"$$.*?$$$.*?$", response)
    return len(links) >= min_links
''',

    "code_block_count_checker": '''
def evaluate(response, num_blocks):
    """Verify if response contains exactly num_blocks code block markers (```)"""
    marker_count = response.count('```')
    complete_blocks = marker_count // 2
    return complete_blocks == num_blocks
''',

    "emoji_checker": '''
def evaluate(response, min_emojis):
    """Verify if response contains at least min_emojis emoji characters"""
    import emoji
    return sum(1 for char in response if char in emoji.EMOJI_DATA) >= min_emojis
''',

    "sentence_length_checker": '''
def evaluate(response, max_words_per_sentence):
    """Verify if no sentence exceeds max_words_per_sentence"""
    from nltk.tokenize import sent_tokenize, word_tokenize
    sentences = sent_tokenize(response)
    return all(len(word_tokenize(s)) <= max_words_per_sentence for s in sentences)
''',

    "html_tag_checker": '''
def evaluate(response, required_tags):
    """Verify if response contains all required HTML tags"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response, 'html.parser')
    return all(soup.find(tag) is not None for tag in required_tags)
''',

    "min_unique_words": '''
def evaluate(response, min_unique):
    """Verify if response contains at least min_unique distinct words"""
    import nltk
    words = nltk.word_tokenize(response.lower())
    return len(set(words)) >= min_unique
''',

    "max_unique_words": '''
def evaluate(response, max_unique):
    """Verify if response contains less than max_unique distinct words"""
    import nltk
    words = nltk.word_tokenize(response.lower())
    return len(set(words)) < max_unique
''',

    "min_uppercase_words": '''
def evaluate(response, min_uppercase):
    """Verify if response contains at least min_uppercase words starting with capital letter"""
    import re
    uppercase_words = re.findall(r'\b[A-Z][a-z]*\b', response)
    return len(uppercase_words) >= min_uppercase
''',

    "min_paragraph_word_count": '''
def evaluate(response, min_paragraphs, min_words):
    """Verify if at least min_paragraphs paragraphs have min_words words each"""
    import nltk
    paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
    valid_paragraphs = sum(1 for p in paragraphs 
                          if len(nltk.word_tokenize(p)) >= min_words)
    return valid_paragraphs >= min_paragraphs
''',

    "min_sentence_word_count": '''
def evaluate(response, min_sentences, min_words):
    """Verify if at least min_sentences sentences have min_words words each"""
    from nltk.tokenize import sent_tokenize, word_tokenize
    sentences = sent_tokenize(response)
    valid_sentences = sum(1 for s in sentences 
                         if len(word_tokenize(s)) >= min_words)
    return valid_sentences >= min_sentences
''',

    "word_count_range": '''
def evaluate(response, min_words, max_words):
    """Verify if response contains between min_words and max_words words (inclusive)"""
    import nltk
    word_count = len(nltk.word_tokenize(response))
    return min_words <= word_count <= max_words
''',

    "sentence_count_range": '''
def evaluate(response, min_sentences, max_sentences):
    """Verify if response contains between min_sentences and max_sentences sentences (inclusive)"""
    from nltk.tokenize import sent_tokenize
    sentence_count = len(sent_tokenize(response))
    return min_sentences <= sentence_count <= max_sentences
''',

    "paragraph_count_range": '''
def evaluate(response, min_paragraphs, max_paragraphs):
    """Verify if response contains between min_paragraphs and max_paragraphs paragraphs (inclusive)"""
    paragraphs = [p for p in response.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)
    return min_paragraphs <= paragraph_count <= max_paragraphs
''',

    "line_length_range": '''
def evaluate(response, min_chars, max_chars):
    """Verify if all lines contain between min_chars and max_chars characters (inclusive)"""
    lines = [line for line in response.split('\n') if line.strip()]
    return all(min_chars <= len(line) <= max_chars for line in lines)
''',

    "line_word_count_range": '''
def evaluate(response, min_words, max_words):
    """Verify if all lines contain between min_words and max_words words (inclusive)"""
    import nltk
    lines = [line for line in response.split('\n') if line.strip()]
    return all(min_words <= len(nltk.word_tokenize(line)) <= max_words for line in lines)
''',

    "capital_word_count_range": '''
def evaluate(response, min_capitals, max_capitals):
    """Verify if response contains between min_capitals and max_capitals capitalized words (inclusive)"""
    import re
    capital_words = re.findall(r'\b[A-Z][a-z]*\b', response)
    return min_capitals <= len(capital_words) <= max_capitals
''',

    "csv_format_checker": '''
def evaluate(response):
    """Verify if response is valid CSV format"""
    import csv
    from io import StringIO
    try:
        # Try reading the CSV with different dialects
        for dialect in (csv.excel, csv.excel_tab, csv.unix_dialect):
            try:
                reader = csv.reader(StringIO(response), dialect)
                next(reader)  # Try reading first row
                return True
            except csv.Error:
                continue
        return False
    except Exception:
        return False
''',
}

def load_persona(persona_file_path):
    with open(persona_file_path, "r") as f:
        personas = [line.strip() for line in f if line.strip()]
    return personas

def load_questions(question_file_path):
    with open(question_file_path, 'r') as f:
        data = json.load(f)["llm_based"]
    return [d["verification_question"] for d in data]


def select_constraints(all_functions, all_questions, min_total=4, max_total=8):
    # 检查输入是否非空
    if not all_functions or not all_questions:
        raise ValueError("Both all_functions and all_questions must be non-empty")

    # 计算最小可能的总数（至少1func + 1ques = 2）
    actual_min = max(min_total, 4)  # 确保总数至少为2
    total_n = random.randint(actual_min, max_total)

    # 确保 functions 和 questions 都至少有1个
    n_func = random.randint(1, total_n - 1)  # functions至少1个，且给questions留至少1个
    n_ques = total_n - n_func               # questions数量 = 总数 - n_func

    # 安全抽样（避免k超过列表长度）
    selected_funcs = random.sample(all_functions, min(n_func, len(all_functions)))
    selected_questions = random.sample(all_questions, min(n_ques, len(all_questions)))

    return selected_funcs, selected_questions

def select_persona(all_personas):
    return random.choice(all_personas)

def extract_instruction_and_code(response):
    try:
        # Extract Instruction section (everything before first code block)
        instruction_match = re.search(r'^(.*?)(?=```python)', response, re.DOTALL)
        instruction = instruction_match.group(1).strip() if instruction_match else None
        # Extract all code blocks
        code_blocks = re.findall(r'```python\s*\n(.*?)\n```', response, re.DOTALL)

        # Separate functions and questions
        functions = []
        questions = []
        for block in code_blocks:
            if "def evaluate" in block:
                functions.append(block.strip())
            else:
                # This is a question (remove any remaining markdown formatting)
                question = re.sub(r'^\s*-\s*', '', block.strip())  # Remove bullet points if present
                questions.append(question)
        
        if "Instruction:" in instruction:
            instruction = instruction.split("Instruction:")[1].strip()
    except:
        instruction = ""

    return {
        "instruction": instruction,
        "functions": functions,
        "questions": questions,
    }

def construct_conflict_check_prompt(functions, questions):
    func_str = "\n".join([f"{func}" for func in functions])
    ques_str = "\n".join([f"{q}" for q in questions])
    return CONFLICT_CHECK_PROMPT.format(
        functions=func_str,
        questions=ques_str
    )

def parse_yes_no(output: str) -> bool:
    output = output.strip().lower()
    if "yes" in output:
        return True
    elif "no" in output:
        return False
    else:
        print(f"Warning: unexpected output '{output}', defaulting to False")
        return False


if __name__ == "__main__":
    persona_file_path = "data/meta_personas.txt"
    question_file_path = "data/llm_based_constraints.json"
    model_name_or_path = "../models/qwen3-8b"
    decoding_dict = {
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    tp = 8

    # 加载资源
    personas = load_persona(persona_file_path)
    all_questions = load_questions(question_file_path)
    all_functions = list(functions.values())

    # 初始化模型
    vllm_engine = vLLM_Engine(
        model_name_or_path=model_name_or_path,
        decoding_dict=decoding_dict,
        tp=tp,
    )

    total_num = 100000
    # 1. 首先生成所有候选组合
    candidate_combinations = []
    for _ in range(total_num):
        persona = select_persona(personas)
        selected_funcs, selected_questions = select_constraints(all_functions, all_questions)
        candidate_combinations.append((persona, selected_funcs, selected_questions))

    # 2. 为所有候选组合生成冲突检查prompt
    conflict_prompts = []
    for persona, selected_funcs, selected_questions in candidate_combinations:
        conflict_prompt = construct_conflict_check_prompt(selected_funcs, selected_questions)
        conflict_prompts.append({"user": conflict_prompt})

    # 3. 批量执行冲突检查
    conflict_results = vllm_engine.generate(conflict_prompts)

    # 4. 筛选出无冲突的组合
    valid_combinations = []
    for i, result in enumerate(conflict_results):
        if not parse_yes_no(result):
            valid_combinations.append(candidate_combinations[i])

    print(f"经过冲突检查后，剩余有效组合数量: {len(valid_combinations)}")

    # 5. 为有效组合生成指令生成prompt
    instruction_prompts = []
    for persona, selected_funcs, selected_questions in valid_combinations:
        funcs_str = "\n".join([f"```python\n{func}\n```" for func in selected_funcs])
        questions_str = "\n".join([f"```python\n{q.strip()}\n```" for q in selected_questions])

        prompt_text = Prompt.format(
            persona=persona,
            functions=funcs_str,
            questions=questions_str
        )
        instruction_prompts.append({"user": prompt_text})

    # 6. 批量生成指令
    instruction_results = vllm_engine.generate(instruction_prompts)

    # 7. 解析所有结果
    results = []
    for resp in instruction_results:
        results.append(extract_instruction_and_code(resp))
    
    final_results = []
    for final in results:
        if final["instruction"]:
            final_results.append(final)

    # 8. 保存结果
    with open("data/instructions.json", "w") as f:
        json.dump(final_results, f, indent=4)
