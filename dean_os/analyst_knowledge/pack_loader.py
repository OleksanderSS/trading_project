from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.analyst_knowledge.schemas import KnowledgeItem, KnowledgePack


def load_knowledge_pack(path: str | Path) -> KnowledgePack:
    """Load a KnowledgePack from JSON file or directory.

    Supported:
    - `pack.json`
    - `knowledge_pack.json`
    - a direct JSON file
    - optional `items/*.md` files in a directory, converted to source_note items
    """

    path = Path(path)
    if path.is_dir():
        pack_path = _find_pack_file(path)
        data = _load_json(pack_path)
        markdown_items = _load_markdown_items(path, domain_id=data.get("domain_id", "unknown"))
        if markdown_items:
            data.setdefault("items", [])
            data["items"].extend([item.model_dump(mode="json") for item in markdown_items])
        return KnowledgePack(**data)

    if path.suffix.lower() == ".json":
        return KnowledgePack(**_load_json(path))

    if path.suffix.lower() in {".yaml", ".yml"}:
        return KnowledgePack(**_load_yaml(path))

    raise ValueError(f"Unsupported knowledge pack path: {path}")


def save_knowledge_pack(pack: KnowledgePack, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pack.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _find_pack_file(directory: Path) -> Path:
    for name in ("pack.json", "knowledge_pack.json", "pack.yaml", "pack.yml", "knowledge_pack.yaml", "knowledge_pack.yml"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No pack.json/knowledge_pack.json found in {directory}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("YAML support requires PyYAML. Use JSON if PyYAML is unavailable.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_markdown_items(directory: Path, domain_id: str) -> list[KnowledgeItem]:
    items_dir = directory / "items"
    if not items_dir.exists():
        return []

    items: list[KnowledgeItem] = []
    for path in sorted(items_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = _title_from_markdown(text) or path.stem.replace("_", " ").title()
        tags = [part.lower() for part in path.stem.split("_") if part]
        items.append(
            KnowledgeItem(
                item_id=f"md_{path.stem}",
                domain_id=domain_id,
                item_type="source_note",
                title=title,
                body=text,
                tags=tags,
                confidence=0.45,
                importance=2,
                metadata={"source_file": str(path)},
            )
        )
    return items


def _title_from_markdown(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None
