# Tileset Search File

This repo contains a couple of Python scripts that parse LimeZu's Modern [Exteriors](https://limezu.itch.io/modernexteriors)/[Interiors](https://limezu.itch.io/moderninteriors) tilesets and generate an HTML file that can be used to preview, explore and search all the tiles in the tileset.

![Tileset Search](media/screenshot2.png)

If you want to try it, you can find the most recent version on [itch.io](https://4kxz.itch.io/limezu-tileset-search). The files here are used to generate an up-to-date version.

## How to update the file

Before running the commands below, create a directory named `dist` at the root of this project and copy the uncompressed `modernexteriors-win` and `moderninteriors-win` directories into it. In order to run the scripts you will need to install [uv](https://docs.astral.sh/uv/).

### Quick Start
To process everything and generate all standard HTML files:
```bash
uv run python main.py all
```

### Advanced Usage

**1. Process images into metadata JSONs:**
```bash
uv run python main.py process
```

**2. Generate a custom HTML combination:**
```bash
uv run python main.py generate dist/modernexteriors-win.json dist/moderninteriors-win.json -o custom.html
```

The `notebooks` directory contains a few notebooks I used to debug issues with the tile-matching code, jupyter is not actually required to update the files.

## Thanks

To everybody who reported issues and helped out: mariorez, Snoopiisz, Areinu, LegendarySwordsman...
