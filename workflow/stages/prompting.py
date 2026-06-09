def format_prompt(title: str, sections: list[tuple[str, str]]) -> str:
    parts = [title.strip()]

    for heading, content in sections:
        text = content.strip()
        if text == "":
            continue
        parts.append(f"{heading}:\n{text}")

    return "\n\n".join(parts)
