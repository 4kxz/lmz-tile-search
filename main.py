import argparse
import os
import sys
from process import process_collections, COLLECTIONS
from template import build_html


def main():
    parser = argparse.ArgumentParser(description="TileSet Search Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Process command
    process_parser = subparsers.add_parser(
        "process", help="Process tileset images and generate metadata JSONs"
    )
    process_parser.add_argument(
        "--dist", default="dist", help="Path to the dist directory"
    )
    process_parser.add_argument(
        "--limit", type=int, help="Limit the number of pairs to search (for testing)"
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate HTML search file from metadata JSONs"
    )
    generate_parser.add_argument(
        "json_files", nargs="+", help="Metadata JSON files to include"
    )
    generate_parser.add_argument(
        "-o", "--output", default="output.html", help="Output HTML file"
    )
    generate_parser.add_argument(
        "-t", "--template", default="template.html", help="Template HTML file"
    )

    # All command
    all_parser = subparsers.add_parser(
        "all", help="Process everything and generate default HTML files"
    )
    all_parser.add_argument("--dist", default="dist", help="Path to the dist directory")

    args = parser.parse_args()

    if args.command == "process":
        process_collections(args.dist, limit=args.limit)

    elif args.command == "generate":
        build_html(args.template, args.json_files, args.output)
        print(f"Generated {args.output} from {', '.join(args.json_files)}")

    elif args.command == "all":
        # 1. Process everything
        process_collections(args.dist)

        # 2. Generate standard combinations
        dist = args.dist

        # All-in-one
        all_jsons = [os.path.join(dist, f"{name}.json") for name in COLLECTIONS.keys()]
        all_jsons = [f for f in all_jsons if os.path.exists(f)]
        if all_jsons:
            build_html("template.html", all_jsons, "index.html")
            print("Generated index.html (All tilesets)")

        # Exteriors + Interiors
        ext_int = [
            os.path.join(dist, "modernexteriors-win.json"),
            os.path.join(dist, "moderninteriors-win.json"),
        ]
        ext_int = [f for f in ext_int if os.path.exists(f)]
        if len(ext_int) == 2:
            build_html("template.html", ext_int, "exteriors_interiors.html")
            print("Generated exteriors_interiors.html")

        # Individuals
        for name in COLLECTIONS.keys():
            json_path = os.path.join(dist, f"{name}.json")
            if os.path.exists(json_path):
                build_html("template.html", [json_path], f"{name}.html")
                print(f"Generated {name}.html")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
