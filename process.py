from collections import defaultdict
from typing import Any, Optional
import cv2
import json
import numpy as np
import os
import re
import time


PATH = "dist"
ASSET_PACKS = (
    "modernexteriors-win",
    "Modern_Interiors_v41.3.4",
)
TILE_SIZE = 16
SIMILARITY_THRESHOLD = 0.09
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
EXCLUDED_TAGS = (
    "modernexteriors",
    "moderninteriors",
    "modern",
    "sorter",
    "animated",
    "animation",
    "single",
    "tileset",
    "character",
    "gifs",
    "16x16",
    "32x32",
    "48x48",
    "and",
    "the",
    "win",
)


class BaseImg:
    def __init__(self, path: str):
        self.path: str = path
        self.kind: str = self.__class__.__name__.lower()
        self.tags: tuple[str] = self.__parse_tags(path, self.kind)
        self.duplicates: list[str] = []

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "tags": self.tags,
            "duplicates": self.duplicates,
        }

    def add_duplicate(self, path: str) -> None:
        self.duplicates.append(path)

    @staticmethod
    def __parse_tags(path: str, kind: str) -> tuple[str]:
        string = os.path.splitext(path)[0].lower()
        string = re.sub(r"[^a-z0-9]+", " ", string).strip()
        unique = [kind + "s"]
        for word in string.split():
            if (
                len(word) > 2
                and not word.isdigit()
                and word not in unique
                and word not in EXCLUDED_TAGS
            ):
                unique.append(word)
        return tuple(reversed(unique))

    def __str__(self):
        return f"{self.__class__.__name__}({self.path})"

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseImg):
            return NotImplemented
        return self.path == other.path


class BaseCV2Img(BaseImg):
    def __init__(self, path: str):
        super().__init__(path)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.cv2: cv2 = self.__convert_to_monochrome(image)
        self.checksum: float = self.__calculate_checksum(self.cv2)
        self.shape: tuple[int, int] = image.shape[1], image.shape[0]

    def to_json(self) -> dict[str, Any]:
        as_json = super().to_json()
        as_json["shape"] = self.shape
        return as_json

    def is_identical(self, other: "BaseCV2Img") -> float:
        return (self.shape == other.shape) and (self.cv2 == other.cv2).all()

    @staticmethod
    def __convert_to_monochrome(image: cv2) -> cv2:
        return image[:, :, 0:3].sum(axis=2) * image[:, :, 3]

    @staticmethod
    def __calculate_checksum(image: cv2) -> float:
        # pre-compute the sum of top-left tile
        # this is used to speed up the search
        return image[:TILE_SIZE, :TILE_SIZE].sum()


class Character(BaseImg):
    pass


class Animation(BaseImg):
    pass


class Single(BaseCV2Img):
    def __init__(self, path: str):
        super().__init__(path)
        self.tilesets: list[Tileset] = []

    def to_json(self) -> dict[str, Any]:
        as_json = super().to_json()
        as_json["tilesets"] = [tileset.path for tileset in self.tilesets]
        return as_json

    def add_tileset(self, tileset: "Tileset") -> None:
        self.tilesets.append(tileset)


Score = float
Coord = tuple[int, int]
Tile = tuple[Single, Score]


class Tileset(BaseCV2Img):
    def __init__(self, path: str):
        super().__init__(path)
        self.tiles: dict[Coord, Tile] = {}
        self.checksums: np.array = self.__calculate_checksums(self.cv2)

    def to_json(self) -> dict[str, Any]:
        as_json = super().to_json()
        as_json["tiles"] = {
            single.path: (coord, score) for coord, [single, score] in self.tiles.items()
        }
        return as_json

    def search(self, single: Single) -> Optional[tuple[Coord, Score]]:
        # check if the single image is in the tileset
        sh, sw = single.cv2.shape[:2]
        th, tw = self.cv2.shape[:2]
        for i in range(0, th - sh + 1, TILE_SIZE):
            for j in range(0, tw - sw + 1, TILE_SIZE):
                if single.checksum != self.checksums[i // TILE_SIZE, j // TILE_SIZE]:
                    continue
                tile = self.cv2[i : i + sh, j : j + sw]
                diff = tile != single.cv2
                score = np.count_nonzero(diff) / diff.size
                if score > SIMILARITY_THRESHOLD:
                    continue
                return (j, i), score
        return None

    def add_tile(self, coord: Coord, tile: Tile) -> None:
        if coord in self.tiles:
            raise ValueError(f"Tile already exists at {coord}")
        self.tiles[coord] = tile

    @staticmethod
    def __calculate_checksums(image: cv2) -> int:
        # pre-compute the sum of each tile
        # this is used to speed up the search
        h = image.shape[0] // TILE_SIZE
        w = image.shape[1] // TILE_SIZE
        sums = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                tile = image[
                    i * TILE_SIZE : (i + 1) * TILE_SIZE,
                    j * TILE_SIZE : (j + 1) * TILE_SIZE,
                ]
                sums[i, j] = tile.sum()
        return sums


def load_image(path: str) -> Optional[BaseImg]:
    lower = path.lower()
    ext = os.path.splitext(path)[1]
    if ext not in IMAGE_EXTENSIONS:
        return None
    if "32x32" in lower or "48x48" in lower:
        return None
    elif "animated" in lower or "animation" in lower:
        if ext != ".gif":
            return None
        return Animation(path)
    elif "character" in lower:
        return Character(path)
    elif "single" in lower:
        return Single(path)
    else:
        return Tileset(path)


if __name__ == "__main__":
    os.chdir(PATH)
    singles = []
    tilesets = []
    all_images = []
    visited = defaultdict(list)

    # iterate over all files in the asset pack
    for asset_pack in ASSET_PACKS:
        for root, dirs, files in os.walk(asset_pack):
            print(f"Loading {root}...")
            for file in files:
                path = os.path.join(root, file)
                image = load_image(path)
                if image is None:
                    continue
                if not isinstance(image, BaseCV2Img):
                    all_images.append(image)
                else:
                    # check if the image is a duplicate
                    for other in visited[image.checksum, image.shape]:
                        if image.is_identical(other):
                            other.add_duplicate(path)
                            break
                    else:
                        visited[image.checksum, image.shape].append(image)
                        all_images.append(image)
                        if isinstance(image, Single):
                            singles.append(image)
                        elif isinstance(image, Tileset):
                            tilesets.append(image)
    print(f"Loaded {len(singles)} singles and {len(tilesets)} tilesets.")

    # iterate over all singles and search for them in the tilesets
    n = len(singles)
    start = time.time()
    results: dict[tuple[Tileset, Coord], list[tuple[Single, Score]]] = defaultdict(list)
    for i, single in enumerate(singles, start=1):
        # check if the single is in any of the tilesets
        for tileset in tilesets:
            result = tileset.search(single)
            if result:
                coord, score = result
                results[tileset, coord].append((single, score))
        if i % 20 == 0:
            left = time.gmtime((time.time() - start) / i * (n - i + 1))
            print(f"Processed {i}/{n}, {left.tm_min}m {left.tm_sec}s left...")
    print(f"Done in {time.time() - start:.2f}s.")

    # add the results to the tilesets
    for (tileset, coord), match_list in results.items():
        best_match = min(match_list, key=lambda x: x[1])
        tileset.add_tile(coord, best_match)
        best_match[0].add_tileset(tileset)

    with open("data.json", "w") as f:
        json.dump({image.path: image.to_json() for image in all_images}, f)
