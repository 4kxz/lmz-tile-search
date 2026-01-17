import json
import os
import sys


def build_html(
    template_path: str, json_paths: list[str], output_path: str = None
) -> str:
    combined_data = {}
    for path in json_paths:
        with open(path, "r") as f:
            data = json.load(f)
            combined_data.update(data)

    with open(template_path, "r") as t:
        template = t.read()

    # Inject the data directly into the load function call
    json_str = json.dumps(combined_data)
    html = template.replace(
        "fetch('../dist/data.json').then(response => response.json()).then(load);",
        f"load({json_str})",
    )

    if output_path:
        with open(output_path, "w") as f:
            f.write(html)

    return html


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python template.py <data.json> [output.html]")
        sys.exit(1)

    json_input = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output.html"

    # Check if input is a single file or potentially a list
    build_html("template.html", [json_input], output_file)
    print(f"Generated {output_file}")
