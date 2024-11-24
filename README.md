# Tileset Search File

This repo contains a couple of Python scripts that parse LimeZu's Modern [Exteriors](https://limezu.itch.io/modernexteriors)/[Interiors](https://limezu.itch.io/moderninteriors) tilesets and generate an HTML file that can be used to preview, explore and search all the tiles in the tileset.

![Tileset Search](media/screenshot2.png)

If you want to try it, you can find the most recent version on [itch.io](https://4kxz.itch.io/limezu-tileset-search). The files here are used to generate an up-to-date version.

## How to update the file

Before running the commands below, create a directory named `dist` at the root of this project and copy the uncompressed `modernexteriors-win` and `moderninteriors-win` directories into it. In order to run the scripts you will need to install Python 3 and the script dependencies. This can be done with the following commands:

```bash
pip install -r requirements.txt
python process.py
python compile.py
```

It can take a while to run and hog the CPU, as it does a brute force search.

The `notebooks` directory contains a few notebooks I used to debug issues with the tile-matching code, jupyter is not actually required to update the files.

## Thanks

To everybody who reported issues and helped out: mariorez, Snoopiisz, Areinu, LegendarySwordsman...
