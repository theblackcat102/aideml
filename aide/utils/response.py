import os
import json
import re
import black
from openai import OpenAI

fixer_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    """Extract all JSON objects from the text. Caveat: This function cannot handle nested JSON objects."""
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    # Sometimes chatgpt-turbo forget the last curly bracket, so we try to add it back when no json is found
    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string, threshold=5100, k=2500):
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


def extract_code(text):
    """Extract python code blocks from the text."""
    parsed_codes = []

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    formatted_code = format_code("\n\n".join(valid_code_blocks))
    # we need to check again to ensure nothings wrong
    prompt = "Read this following python code which is placed between <code> .. CODE .. </code> carefully, determine if it can be execute as it is, if it can be executed return ANSWER: YES, if it has some natural language in front which wasn't wrapped inside a comment block or the main code was wrapped in a python code return ANSWER: NO. Always make sure you end your reply in ANSWER: Yes/No"
    content = prompt+'\n<code>\n'+formatted_code +'\n</code>'
    response = fixer_client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[{'role': 'user', 'content': content}],
        temperature=0.1,
        max_tokens=1024,
        top_p=0.95,
        logprobs=False
    )
    res_text = response.choices[0].message.content
    res_info = {
        "input": content,
        "output": res_text,
        "num_input_tokens": response.usage.prompt_tokens,
        "num_output_tokens": response.usage.completion_tokens
    }
    if 'LOG_RESPONSE' in os.environ and os.environ["LOG_RESPONSE"] == 'True':
        with open('fixer_gpt-4o-2024-08-06.jsonl', 'a') as fout:
            fout.write(json.dumps(res_info)+'\n')
    answer = res_text.split('ANSWER:', maxsplit=1)[-1]
    if '\n' in answer:
        answer = answer.split('\n')[0]
    answer = answer.strip().lower()
    if answer == 'no':
        prompt = "Read this following python code which is placed between <code> .. CODE .. </code> carefully, this following code wasn't able to be executed due to format issues, there might be some syntax problem such as text in front which wasn't wrapped in comment block or the code block ``` token wasn't added placed correctly which cause the code cannot be executed immediately when placed inside a .py code. Please extract the main code from the natural response, do not include other natural text or irrelevant result code block which wasn't part of the main code. This main code must be executable when f write into a python file, so DO NOT wrap it in ``` code block as well.\n"
        content = prompt+'\n<code>\n'+text +'\n</code>'
        response = fixer_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{'role': 'user', 'content': content}],
            temperature=0.1,
            max_tokens=9182,
            top_p=0.95,
            logprobs=False
        )
        res_text = response.choices[0].message.content
        res_info = {
            "input": content,
            "output": res_text,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens
        }
        if 'LOG_RESPONSE' in os.environ and os.environ["LOG_RESPONSE"] == 'True':
            with open('fixer-rewriter_gpt-4o-2024-08-06.jsonl', 'a') as fout:
                fout.write(json.dumps(res_info)+'\n')
        return res_text

    return formatted_code

def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code
