from collections import defaultdict
from itertools import count
from multiprocessing import Pool
from random import sample
from tqdm import tqdm
from typing import Any, Optional, Iterator
import cv2
import json
import numpy as np
import os
import re
import sys


PATH = "dist"
PATH_DIR = os.path.dirname(os.path.abspath(__file__))
TAGS_FILE = os.path.join(PATH_DIR, "notebooks/tags.json")
TILE_SIZE = 16
SIMILARITY = 0.9
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
EXCLUDED_TAGS = (
    "modernexteriors",
    "moderninteriors",
    "modern",
    "theme",
    "sorter",
    "complete",
    "animated",
    "animation",
    "single",
    "tileset",
    "character",
    "gifs",
    "16x16",
    "24x24",
    "32x32",
    "48x48",
    "and",
    "the",
    "win",
    "v41",
)

Id = str
Region = tuple[int, int, int, int]
Tile = tuple[Region, Region, float]


class BaseImg:
    id_counter = count(0)
    tag_data = json.load(open(TAGS_FILE, "r"))

    def __init__(self, path: str):
        self.path: str = path
        self.kind: str = self.__class__.__name__.lower()
        self.tags: list[str] = self.__parse_tags(path, self.kind)
        self.id: Id = f"#{next(self.id_counter)}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseImg):
            return NotImplemented
        return self.path == other.path

    def __hash__(self) -> int:
        return hash(self.path)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "tags": tuple(self.tags),
        }

    @staticmethod
    def __parse_tags(path: str, kind: str) -> list[str]:
        base_name = os.path.basename(path)
        unique = [kind + "s"] + BaseImg.tag_data.get(base_name, [])
        clean_path = os.path.splitext(path)[0].lower()
        clean_path = re.sub(r"[^a-z0-9]+", " ", clean_path).strip()
        for word in clean_path.split():
            if (
                len(word) > 2
                and not word.isdigit()
                and word not in unique
                and word not in EXCLUDED_TAGS
            ):
                unique.append(word)
        unique.reverse()
        return unique


class BaseCV2Img(BaseImg):
    def __init__(self, path: str):
        super().__init__(path)
        self.cv2: cv2 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.shape: tuple[int, int] = self.cv2.shape[1], self.cv2.shape[0]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseCV2Img):
            return NotImplemented
        return (self.shape == other.shape) and (self.cv2 == other.cv2).all()

    def __hash__(self) -> int:
        return hash((self.shape, self.cv2.tobytes()))

    def to_json(self) -> dict[str, Any]:
        as_json = super().to_json()
        as_json["shape"] = self.shape
        return as_json


class Character(BaseImg):
    pass


class Vehicle(BaseImg):
    pass


class Animation(BaseImg):
    pass


class Single(BaseCV2Img):
    def __init__(self, path: str):
        super().__init__(path)
        self.tilesets: set[Id] = set()

    def add_tileset(self, tileset: "Tileset") -> None:
        self.tilesets.add(tileset.id)

    def to_json(self) -> dict[str, Any]:
        as_json = super().to_json()
        as_json["tilesets"] = tuple(self.tilesets)
        return as_json


class Tileset(BaseCV2Img):
    def __init__(self, path: str):
        super().__init__(path)
        self.tiles: dict[Id, list[Tile]] = defaultdict(list)

    def add_tile(self, single: Single, tile: Tile) -> None:
        self.tiles[single.id].append(tile)

    def search(
        self, single: Single, threshold: float=SIMILARITY, size: int=TILE_SIZE
    ) -> Iterator[Tile]:
        # check if the single image is in the tileset
        alpha_mask = single.cv2[:, :, 3] == 255
        if np.count_nonzero(alpha_mask) == 0:
            return
        th, tw, _ = self.cv2.shape
        sh, sw, _ = single.cv2.shape
        for j in range(size - sh, th, size):
            for i in range(size - sw, tw, size):
                ty0, ty1 = max(0, j), min(th, j + sh)
                tx0, tx1 = max(0, i), min(tw, i + sw)
                t = self.cv2[ty0:ty1, tx0:tx1]
                sy0, sy1 = max(0, -j), min(sh, th - j)
                sx0, sx1 = max(0, -i), min(sw, tw - i)
                s = single.cv2[sy0:sy1, sx0:sx1]
                am = alpha_mask[sy0:sy1, sx0:sx1]
                score = self._jaccard_index(s, t, am)
                if score > threshold:
                    t_region = tx0, ty0, tx1 - tx0, ty1 - ty0
                    s_region = sx0, sy0, sx1 - sx0, sy1 - sy0
                    yield t_region, s_region, (score - threshold) / (1 - threshold)

    def to_json(self) -> dict[str, Any]:
        as_json = super().to_json()
        as_json["tiles"] = self.tiles
        return as_json

    @staticmethod
    def _jaccard_index(a: cv2, b: cv2, am: cv2) -> float:
        """Return an index of how much a matches b,
        where 0.0 is no match, and 1.0 is an exact match."""
        if denominator := np.count_nonzero(am):
            comp = (a == b).min(axis=2)
            return np.count_nonzero(comp & am) / denominator
        return 0.0

    @classmethod
    def _compute_score(cls, a: cv2, b: cv2) -> float:
        """Used for debugging."""
        alpha_mask = a[:, :, 3] == 255
        return cls._jaccard_index(a, b, alpha_mask)


class BaseLoader:
    def __init__(self):
        self.indices = defaultdict(set)

    def load(self, path: str) -> Iterator[BaseImg]:
        # iterate over all files in the asset pack
        for root, _, files in os.walk(path):
            for file in files:
                if image := self._load_image(os.path.join(root, file)):
                    self.indices[image.kind].add(image)
        n = len(self.indices["single"])
        m = len(self.indices["tileset"])
        print(f"Loaded {n} singles and {m} tilesets from {path}.")

    def search(self):
        # iterate over all singles and search for them in the tilesets
        args = list(self._get_search_pairs())
        if len(sys.argv) > 1:
            args = sample(args, int(sys.argv[1]))
        print(f"Searching {len(args)} pairs...")
        with Pool() as pool:
            results = pool.imap_unordered(self._search, args, 32)
            results = list(tqdm(results, total=len(args)))
        # update the tilesets and singles with the results
        tileset_index = {image.id: image for image in self.indices["tileset"]}
        single_index = {image.id: image for image in self.indices["single"]}
        found = 0
        for tileset_id, single_id, tiles in results:
            if not tiles:
                continue
            tileset = tileset_index[tileset_id]
            single = single_index[single_id]
            single.add_tileset(tileset)
            for tile in tiles:
                tileset.add_tile(single, tile)
                found += 1
        print(f"Found {found} matches.")

    def to_json(self) -> dict[str, Any]:
        return {
            image.id: image.to_json()
            for index in self.indices.values()
            for image in index
        }

    def _load_image(self, path: str) -> Optional[BaseImg]:
        lower = path.lower()
        ext = os.path.splitext(path)[1]
        if ext not in IMAGE_EXTENSIONS:
            # ingore unwanted files like txt, zip, ini, etc.
            return None
        if "palette" in lower:
            # ignore palette images
            return None
        if "32x32" in lower or "48x48" in lower or "24x24" in lower:
            # these are duplicates of the 16x16 images
            return None
        if "animated" in lower or "animation" in lower:
            if ext != ".gif":
                # character sheets are ignored for now
                return None
            return Animation(path)
        if "character" in lower:
            return Character(path)
        if "single" in lower:
            return Single(path)
        return Tileset(path)

    def _get_search_pairs(self) -> Iterator[tuple[Tileset, Single]]:
        for tileset in self.indices["tileset"]:
            for single in self.indices["single"]:
                yield tileset, single

    @staticmethod
    def _search(args: tuple[Tileset, Single]) -> tuple[Id, Id, list[Tile]]:
        tileset, single = args
        results = tileset.search(single)
        return tileset.id, single.id, list(results)


class ModernExteriorsLoader(BaseLoader):
    def _load_image(self, path: str) -> Optional[BaseImg]:
        if "old_sorting" in path.lower():
            # these are old tilesets
            return None
        if "complete_singles" in path.lower():
            # these are duplicates of Theme Sorter
            return None
        return super()._load_image(path)

    def _get_search_pairs(self) -> Iterator[tuple[Tileset, Single]]:
        for tileset, single in super()._get_search_pairs():
            if (
                "theme_sorter" in tileset.path.lower()
                and "singles" in single.path.lower()
            ):
                # we can skip the search if the names don't match
                single_dir = os.path.dirname(single.path)
                _, *single_theme, __ = os.path.basename(single_dir).split("_")
                _, *tileset_theme, __ = os.path.basename(tileset.path).split("_")
                if single_theme[:-1] != tileset_theme:
                    continue
            yield tileset, single


class ModernInteriorsLoader(BaseLoader):
    def _load_image(self, path: str) -> Optional[BaseImg]:
        if "old stuff" in path.lower():
            # these are old tilesets
            return None
        if "black_shadow" in path.lower():
            # these are duplicates of the normal tiles
            return None
        if "shadowless" in path.lower():
            # these are duplicates of the normal tiles
            return None
        if "user_interface" in path.lower():
            # TODO: these are not supported for now
            return None
        if "home_designs" in path.lower():
            # TODO: these are not supported for now
            return None
        if "room_builder_subfiles" in path.lower():
            # these are not singles
            return None
        return super()._load_image(path)

    def _get_search_pairs(self) -> Iterator[tuple[Tileset, Single]]:
        for tileset, single in super()._get_search_pairs():
            if (
                "theme_sorter" in tileset.path.lower()
                and "theme_sorter_singles" in single.path.lower()
            ):
                # we can skip the search if the names don't match
                single_dir = os.path.dirname(single.path)
                _, *single_theme, __ = os.path.basename(single_dir).split("_")
                _, *tileset_theme, __ = os.path.basename(tileset.path).split("_")
                if single_theme != tileset_theme:
                    continue
            yield tileset, single


class ModernFarmLoader(BaseLoader):
    
    def _load_image(self, path: str) -> Optional[BaseImg]:
        path_lower = path.lower()
        if "animals" in path_lower:
            return Character(path)
        if "vehicle" in path_lower:  # TODO: apply to other tilesets
            return Vehicle(path)
        if "farmer_generator" in path_lower:
            # these are used in the character generator
            return None
        if "rpg_maker_mv" in path_lower:
            # these are for RPG Maker MV
            return None
        if "autotiles" in path_lower:
            # these are not singles
            return None
        if "complete_tileset_singles" in path_lower:
            # these are duplicates of the normal singles
            return None
        return super()._load_image(path)

    def _get_search_pairs(self) -> Iterator[tuple[Tileset, Single]]:
        for tileset, single in super()._get_search_pairs():
            if (
                "modern_farm_v1.0/16x16" == os.path.dirname(tileset.path).lower()
                and "complete_tileset" not in tileset.path.lower()
                and "single_files" in single.path.lower()
            ):
                # we can skip the search if the names don't match
                single_dir = os.path.dirname(single.path)
                single_theme = os.path.basename(single_dir).removesuffix("_16x16")
                tileset_theme = os.path.basename(tileset.path).split("_", 1)[1].removesuffix("_16x16.png")
                if single_theme != tileset_theme:
                    continue
            yield tileset, single


class ModernOfficeLoader(BaseLoader):

    def _load_image(self, path: str) -> Optional[BaseImg]:
        path_lower = path.lower()
        if "room_builder" in path_lower:
            # these are not singles
            return None
        if "rpg_maker" in path_lower:
            # these are for RPG Maker
            return None
        if "black_shadow" in path_lower:
            # these are duplicates of the normal tiles
            return None
        if "shadowless" in path_lower:
            # these are duplicates of the normal tiles
            return None
        if "office_designs" in path_lower:
            # these are previews
            return None
        if "previous_version" in path_lower:
            # these are old tiles
            return None
        return super()._load_image(path)
        

if __name__ == "__main__":
    os.chdir(PATH)
    collections = {
        "Modern_Office_Revamped_v1.2": ModernOfficeLoader,
        "Modern_Farm_v1.0": ModernFarmLoader,
        "modernexteriors-win": ModernExteriorsLoader,
        "moderninteriors-win": ModernInteriorsLoader,
    }
    output = {}
    for path, loader_class in collections.items():
        loader = loader_class()
        loader.load(path)
        loader.search()
        data = loader.to_json()
        output.update(data)
        with open(f"{path}.json", "w") as f:
            json.dump(data, f)
    with open("data.json", "w") as f:
        json.dump(output, f)
    print("Saved data.json.")
