"""Resolve data paths for local files now and s3a:// later.

Bucket and prefix stay TODO until the AWS setup exists. No credentials
are read here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False) -> dict:
    if isinstance(node, yaml.MappingNode):
        loader.flatten_mapping(node)
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise SystemExit(f"Duplicate key in YAML: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=UniqueKeyLoader)
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a YAML mapping")
    return data


def load_config(name: str) -> dict:
    return load_yaml(CONFIG_DIR / name)


def _missing(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip().strip("\"'")
    return text == "" or text == "TODO"


@dataclass(frozen=True)
class DataLocation:
    """A dataset path. local_path is set only when backend is local."""

    backend: str
    uri: str
    local_path: Path | None

    def require_local(self) -> Path:
        if self.local_path is None:
            raise SystemExit(
                "S3 backend is only a placeholder. "
                f"Refusing to write {self.uri}. "
                "Set backend: local in config/storage.yaml."
            )
        root = REPO_ROOT.resolve()
        resolved = self.local_path.resolve()
        if resolved != root and root not in resolved.parents:
            raise SystemExit(f"Refusing to write outside the repo: {resolved}")
        return resolved


def locate(relative: str) -> DataLocation:
    """Return a local path or an s3a:// URI for a path under the data root.

    relative is inside the data root, for example processed/dim_intent.parquet.
    """

    cfg = load_config("storage.yaml")
    backend = str(cfg.get("backend", "local")).strip()
    relative = relative.lstrip("/")
    if backend == "local":
        local_root = REPO_ROOT / str(cfg.get("local_root", "data"))
        path = local_root / relative
        return DataLocation(backend="local", uri=path.as_posix(), local_path=path)
    if backend != "s3":
        raise SystemExit(f"Unknown storage backend: {backend!r}. Use local or s3.")

    bucket = os.environ.get("S3_BUCKET", cfg.get("s3_bucket"))
    prefix = os.environ.get("S3_PREFIX", cfg.get("s3_prefix"))
    if _missing(bucket) or _missing(prefix):
        raise SystemExit(
            "S3 backend is selected, but s3_bucket or s3_prefix is still TODO. "
            "Set them in config/storage.yaml or export S3_BUCKET and S3_PREFIX. "
            "Nothing was uploaded."
        )
    bucket_text = str(bucket).strip().strip("\"'")
    prefix_text = str(prefix).strip().strip("\"'").strip("/")
    uri = f"s3a://{bucket_text}/{prefix_text}/{relative}"
    return DataLocation(backend="s3", uri=uri, local_path=None)
