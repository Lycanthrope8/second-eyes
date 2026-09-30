"""What a prompt's scene and answer look like, shared by the PC references and the headset's jobs (A1.8b, D55).

A prompt's user text lists the objects one per line after "Objects:", then the user's pose ("User at ...") and the
command. The scene part (system message and object list) is the same for every command in a layout, so it is what gets
cached: everything up to and including the last object line. A candidate's score is the log-probability of its ID and
the closing '"}' after ANSWER_PREFIX, the start of the answer the prompt asks for.
"""
from __future__ import annotations

import re

ANSWER_PREFIX = '{"action": "INSPECT", "target": "'
CANDIDATE_SUFFIX = '"}'
USER_LINE = "\nUser at "
OBJECT_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_[0-9]+) ", re.M)


def candidates(user_text: str) -> list:
    """The object IDs a prompt's scene lists, in order, e.g. ['table_1', 'box_1', ...]."""
    head = user_text.split("Objects:", 1)[-1].split(USER_LINE.strip(), 1)[0]
    return OBJECT_LINE.findall(head)


def formatted(template: str, system: str, user: str) -> str:
    """The prompt as the model sees it: the model description's chat template, {0} the user text, {1} the system."""
    return template.replace("{1}", system).replace("{0}", user)


def scene_chars(formatted_prompt: str) -> int:
    """Where the cacheable scene ends in the formatted prompt: just after the last object line's newline."""
    at = formatted_prompt.find(USER_LINE)
    if at < 0:
        raise ValueError("the prompt has no 'User at' line, so its scene can't be told apart from the command")
    return at + 1
