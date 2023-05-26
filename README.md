# Tileset Search File

This repo contains a couple of Python scripts that parse LimeZu's Modern [Exteriors](https://limezu.itch.io/modernexteriors)/[Interiors](https://limezu.itch.io/moderninteriors) tilesets and generate an HTML file. The resulting file can be used to search and preview all the tiles in the tileset.

Before running the commands below, create a directory named `dist` at the root of this project and copy the uncompressed `modernexteriors-win` directory into it.

In order to run the scripts you will need to install Python 3 and the script dependencies. This can be done with the following commands:

```bash
pip install -r requirements.txt
python process.py
python compile.py
```

It can take a while to run, as it searches for matching tiles using brute force and it is not efficient.
