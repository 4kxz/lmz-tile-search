if __name__ == "__main__":
    with open("dist/data.json", "r") as r:
        data = r.read()
    with open("template.html", "r") as t:
        template = t.read()
    html = template.replace(
        "fetch('../dist/data.json').then(response => response.json()).then(load);",
        f"load({data})",
    )
    print(html)
