import hashlib
import html
import json
import logging
import random
import socket
import time
from functools import lru_cache
from http import HTTPStatus
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "recipes_with_images.json"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
# GitHub media resolves Git LFS objects; jsDelivr serves pointer text instead.
GITHUB_MEDIA_BASE = "https://media.githubusercontent.com/media/FutureUnreal/HowToCook/master/"

DEFAULT_QUICK_QUESTIONS = [
    "今天晚餐吃什么好？",
    "有什么简单易做的家常菜？",
    "适合减肥的低卡菜谱",
    "30分钟内能做完的菜",
    "适合新手的烘焙食谱",
    "下饭的家常菜推荐",
]


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: List[ChatMessage] = Field(default_factory=list)


class GraphRAGWebService:
    def __init__(self) -> None:
        self._rag_system: Optional[Any] = None
        self._rag_init_error: Optional[str] = None
        self._rag_lock = Lock()

        self.recipes: List[Dict[str, Any]] = self._load_recipes()
        self.recipe_by_key: Dict[str, Dict[str, Any]] = {x["key"]: x for x in self.recipes}
        self.recipe_by_name: Dict[str, Dict[str, Any]] = {}
        for recipe in self.recipes:
            self.recipe_by_name[self._normalize_name(recipe.get("name", ""))] = recipe

        self.categories: List[str] = sorted(
            {str(recipe.get("category", "未知")).strip() for recipe in self.recipes}
        )

        rich_pool = []
        for recipe in self.recipes:
            image_info = self.resolve_recipe_image(recipe)
            if image_info["type"] in {"local", "remote"}:
                rich_pool.append(recipe)
        self.recommend_pool: List[Dict[str, Any]] = rich_pool or self.recipes

    def _load_recipes(self) -> List[Dict[str, Any]]:
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

        raw_data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        recipes: List[Dict[str, Any]] = []
        for item in raw_data:
            file_path = str(item.get("file_path", "")).strip()
            name = str(item.get("name", "")).strip() or "未知菜品"
            key_seed = f"{file_path}|{name}"
            recipe_key = hashlib.sha1(key_seed.encode("utf-8")).hexdigest()[:16]

            recipes.append(
                {
                    "key": recipe_key,
                    "name": name,
                    "category": str(item.get("category", "未知")).strip() or "未知",
                    "description": str(item.get("description", "")).strip(),
                    # Legacy cooking_time values may describe just one cooking step.
                    # Only expose a duration with a reviewed total-time label and source.
                    "cooking_time": item.get("cooking_time") if (
                        item.get("cooking_time_source") and item.get("cooking_time_label")
                    ) else None,
                    "cooking_time_label": item.get("cooking_time_label") if (
                        item.get("cooking_time_source") and item.get("cooking_time_label")
                    ) else "耗时待确认",
                    "cooking_time_source": item.get("cooking_time_source", ""),
                    "tags": item.get("tags", []),
                    "file_path": file_path,
                    "image_url": str(item.get("image_url", "")).strip(),
                }
            )
        return recipes

    def _ensure_rag_ready(self) -> Any:
        if self._rag_system is not None:
            return self._rag_system

        with self._rag_lock:
            if self._rag_system is not None:
                return self._rag_system
            if self._rag_init_error:
                raise RuntimeError(self._rag_init_error)

            try:
                logger.info("Initializing GraphRAG system at service startup...")
                from main import AdvancedGraphRAGSystem

                rag = AdvancedGraphRAGSystem()
                rag.initialize_system()
                rag.build_knowledge_base()
                self._rag_system = rag
                logger.info("GraphRAG system initialized")
                return rag
            except Exception as exc:
                self._rag_init_error = str(exc)
                logger.exception("GraphRAG init failed")
                raise RuntimeError(self._rag_init_error) from exc

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name).strip().lower()

    @staticmethod
    def _sanitize_answer(text: str) -> str:
        return str(text or "").replace("*", "")

    def _build_card(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        image_api = f"/api/recipe-image/{recipe['key']}"
        return {
            "key": recipe["key"],
            "name": recipe["name"],
            "category": recipe["category"],
            "description": recipe["description"],
            "cooking_time": recipe.get("cooking_time"),
            "cooking_time_label": recipe.get("cooking_time_label", "耗时待确认"),
            "cooking_time_source": recipe.get("cooking_time_source", ""),
            "tags": recipe.get("tags", []),
            "image_url": image_api,
            "imageUrl": image_api,
        }

    def get_recommendations(self, batch: int = 0, size: int = 6) -> List[Dict[str, Any]]:
        if not self.recommend_pool:
            return []

        clamped_size = max(1, min(size, 24))
        indices = list(range(len(self.recommend_pool)))
        random.Random(batch).shuffle(indices)
        selected = [self.recommend_pool[i] for i in indices[:clamped_size]]
        return [self._build_card(recipe) for recipe in selected]

    def get_home_payload(self, batch: int = 0, size: int = 6) -> Dict[str, Any]:
        return {
            "quick_questions": DEFAULT_QUICK_QUESTIONS,
            "recommendations": self.get_recommendations(batch=batch, size=size),
            "categories": self.categories,
        }

    def get_recipe_by_key(self, recipe_key: str) -> Optional[Dict[str, Any]]:
        return self.recipe_by_key.get(recipe_key)

    def _safe_resolve(self, path: Path) -> Optional[Path]:
        try:
            resolved = path.resolve()
            resolved.relative_to(PROJECT_ROOT.resolve())
            return resolved
        except Exception:
            return None

    def _build_github_image_url(self, recipe: Dict[str, Any]) -> Optional[str]:
        raw_img = str(recipe.get("image_url", "")).strip()
        file_path = str(recipe.get("file_path", "")).strip().replace("\\", "/")
        if not raw_img or not file_path:
            return None
        if raw_img.startswith("http://") or raw_img.startswith("https://"):
            return self._normalize_remote_url(raw_img)

        rel_img = raw_img[2:] if raw_img.startswith("./") else raw_img
        dishes_idx = file_path.find("dishes/")
        if dishes_idx == -1:
            return None

        md_rel = file_path[dishes_idx:]
        dish_dir = "/".join(md_rel.split("/")[:-1])
        if not dish_dir:
            return None

        # HTTP headers and clients require ASCII-safe URLs; encode non-ASCII path segments.
        path = f"{dish_dir}/{rel_img}".replace("\\", "/").lstrip("/")
        encoded_path = quote(path, safe="/-_.~")
        return f"{GITHUB_MEDIA_BASE}{encoded_path}"

    @staticmethod
    def _is_lfs_pointer_file(path: Path) -> bool:
        """
        Detect Git LFS pointer files (text metadata) that masquerade as images.
        """
        try:
            if not path.exists() or not path.is_file():
                return False
            if path.stat().st_size > 4096:
                return False
            with path.open("rb") as f:
                head = f.read(256)
            return head.startswith(b"version https://git-lfs.github.com/spec/v1")
        except Exception:
            return False

    @staticmethod
    def _normalize_remote_url(url: str) -> Optional[str]:
        raw = str(url or "").strip()
        if not raw:
            return None

        parts = urlsplit(raw)
        if not parts.scheme or not parts.netloc:
            return None

        encoded_path = quote(parts.path, safe="/%-_.~")
        return urlunsplit((parts.scheme, parts.netloc, encoded_path, parts.query, parts.fragment))

    def resolve_recipe_image(self, recipe: Dict[str, Any]) -> Dict[str, str]:
        raw_url = str(recipe.get("image_url", "")).strip()
        if raw_url.startswith("http://") or raw_url.startswith("https://"):
            normalized = self._normalize_remote_url(raw_url)
            if normalized:
                return {"type": "remote", "url": normalized}

        file_path = str(recipe.get("file_path", "")).strip().replace("\\", "/")
        markdown_path = self._safe_resolve(PROJECT_ROOT / file_path)
        if markdown_path:
            rel_img = raw_url[2:] if raw_url.startswith("./") else raw_url
            candidate = self._safe_resolve(markdown_path.parent / rel_img)
            if candidate and candidate.exists() and candidate.is_file():
                if not self._is_lfs_pointer_file(candidate):
                    return {"type": "local", "path": str(candidate)}
                logger.info("Local image is an LFS pointer, fallback to remote: %s", candidate)

        github_url = self._build_github_image_url(recipe)
        if github_url:
            return {"type": "remote", "url": github_url}

        return {"type": "placeholder"}

    @staticmethod
    def _probe_remote_url(url: str, timeout_seconds: float = 6.0) -> Dict[str, Any]:
        if not url:
            return {"ok": False, "method": "none", "status": None, "error": "empty_url"}

        try:
            req = Request(url, method="HEAD")
            with urlopen(req, timeout=timeout_seconds) as resp:
                return {"ok": True, "method": "HEAD", "status": int(getattr(resp, "status", 200))}
        except HTTPError as e:
            # Some CDNs block HEAD. Retry with GET when method is not allowed.
            if e.code in {HTTPStatus.METHOD_NOT_ALLOWED, HTTPStatus.FORBIDDEN}:
                try:
                    req = Request(url, method="GET", headers={"Range": "bytes=0-0"})
                    with urlopen(req, timeout=timeout_seconds) as resp:
                        return {
                            "ok": True,
                            "method": "GET_RANGE",
                            "status": int(getattr(resp, "status", 200)),
                        }
                except Exception as inner:
                    return {
                        "ok": False,
                        "method": "GET_RANGE",
                        "status": None,
                        "error": f"{type(inner).__name__}: {inner}",
                    }
            return {"ok": False, "method": "HEAD", "status": int(e.code), "error": str(e)}
        except (URLError, socket.timeout, TimeoutError) as e:
            return {"ok": False, "method": "HEAD", "status": None, "error": f"{type(e).__name__}: {e}"}
        except Exception as e:
            return {"ok": False, "method": "HEAD", "status": None, "error": f"{type(e).__name__}: {e}"}

    def debug_recipe_image(
        self, recipe: Dict[str, Any], probe_remote: bool = False, timeout_seconds: float = 6.0
    ) -> Dict[str, Any]:
        raw_url = str(recipe.get("image_url", "")).strip()
        file_path = str(recipe.get("file_path", "")).strip().replace("\\", "/")
        markdown_abs = self._safe_resolve(PROJECT_ROOT / file_path) if file_path else None
        rel_img = raw_url[2:] if raw_url.startswith("./") else raw_url
        candidate_abs = None
        candidate_exists = False
        if markdown_abs and rel_img:
            candidate_abs = self._safe_resolve(markdown_abs.parent / rel_img)
            candidate_exists = bool(candidate_abs and candidate_abs.exists() and candidate_abs.is_file())

        resolved = self.resolve_recipe_image(recipe)
        info: Dict[str, Any] = {
            "key": recipe.get("key", ""),
            "name": recipe.get("name", ""),
            "category": recipe.get("category", ""),
            "raw_image_url": raw_url,
            "file_path": file_path,
            "markdown_abs": str(markdown_abs) if markdown_abs else None,
            "candidate_image_abs": str(candidate_abs) if candidate_abs else None,
            "candidate_image_exists": candidate_exists,
            "candidate_image_is_lfs_pointer": (
                self._is_lfs_pointer_file(candidate_abs) if candidate_exists and candidate_abs else False
            ),
            "resolved_type": resolved.get("type"),
            "resolved_url": resolved.get("url"),
            "resolved_path": resolved.get("path"),
        }

        if resolved.get("type") == "remote" and probe_remote:
            info["remote_probe"] = self._probe_remote_url(
                str(resolved.get("url", "")), timeout_seconds=timeout_seconds
            )
        return info

    def _rewrite_question_with_history(
        self, rag: Any, message: str, history: List[Dict[str, str]]
    ) -> str:
        if not history:
            return message

        valid_history: List[Dict[str, str]] = []
        for item in history[-8:]:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                valid_history.append({"role": role, "content": content})
        if not valid_history:
            return message

        formatted = "\n".join(
            f"{'用户' if x['role'] == 'user' else '助手'}: {x['content']}" for x in valid_history
        )
        prompt = (
            "你是对话改写助手。请根据历史对话把当前问题改写为可独立检索的问题。"
            "保持原意，不回答问题，只输出改写结果。\n\n"
            f"历史对话:\n{formatted}\n\n当前问题: {message}\n改写结果:"
        )
        try:
            response = rag.generation_module.client.chat.completions.create(
                model=rag.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
                extra_body={"thinking": {"type": "disabled"}},
            )
            rewritten = (response.choices[0].message.content or "").strip()
            return rewritten or message
        except Exception:
            logger.exception("Rewrite failed, fallback to raw query")
            return message

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _normalize_sources(self, documents: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for doc in documents:
            metadata = getattr(doc, "metadata", {}) or {}
            recipe_name = (
                metadata.get("recipe_name")
                or metadata.get("name")
                or metadata.get("entity_name")
                or "未知菜品"
            )
            search_type = (
                metadata.get("search_type")
                or metadata.get("route_strategy")
                or metadata.get("search_method")
                or "unknown"
            )
            score = self._safe_float(
                metadata.get("final_score", metadata.get("relevance_score", metadata.get("score", 0.0)))
            )

            recipe = self.recipe_by_name.get(self._normalize_name(recipe_name))
            category = metadata.get("category", "")
            if (not category) and recipe:
                category = recipe.get("category", "未知")

            source = {
                "recipe_name": str(recipe_name),
                "category": str(category or "未知"),
                "search_type": str(search_type),
                "score": round(score, 4),
                "retrieval_sources": list(metadata.get("retrieval_sources", [])),
                "source_ranks": dict(metadata.get("source_ranks", {})),
                "source_contributions": {
                    str(key): round(self._safe_float(value), 6)
                    for key, value in dict(metadata.get("source_contributions", {})).items()
                },
                "rrf_score": round(self._safe_float(metadata.get("rrf_score", 0.0)), 6),
                "evidence": list(metadata.get("evidence", []))[:5],
                "reasoning_chains": list(metadata.get("reasoning_chains", []))[:5],
            }
            if recipe:
                source["recipe_key"] = recipe["key"]
                source["description"] = recipe.get("description", "")
                source["image_url"] = f"/api/recipe-image/{recipe['key']}"

            dedup_key = (source["recipe_name"], source["search_type"])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            normalized.append(source)
        return normalized[:6]

    def chat(self, message: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        rag = self._ensure_rag_ready()
        rewritten_message = self._rewrite_question_with_history(rag, message, history)

        started = time.perf_counter()
        documents, analysis = rag.query_router.route_query(rewritten_message, rag.config.top_k)
        if documents:
            answer_raw = rag.generation_module.generate_adaptive_answer(rewritten_message, documents)
        else:
            answer_raw = "抱歉，没有检索到足够的菜谱信息，请尝试换一种问法。"
        answer = self._sanitize_answer(answer_raw)
        if not answer:
            raise ValueError("模型未生成有效回答")
        latency_ms = int((time.perf_counter() - started) * 1000)

        route_summary = {
            "strategy": (
                analysis.recommended_strategy.value
                if analysis and getattr(analysis, "recommended_strategy", None)
                else "unknown"
            ),
            "complexity": round(self._safe_float(getattr(analysis, "query_complexity", 0.0)), 4),
            "confidence": round(self._safe_float(getattr(analysis, "confidence", 0.0)), 4),
            "fallback_used": bool(getattr(analysis, "fallback_used", False)),
            "fallback_reason": str(getattr(analysis, "fallback_reason", "")),
        }
        return {
            "answer": answer,
            "route": route_summary,
            "sources": self._normalize_sources(documents),
            "latency_ms": latency_ms,
        }

    def close(self) -> None:
        if self._rag_system is not None:
            try:
                self._rag_system._cleanup()
            except Exception:
                logger.exception("Failed to close RAG system")
            finally:
                self._rag_system = None

    def startup_preload(self) -> None:
        try:
            self._ensure_rag_ready()
        except Exception as exc:
            logger.error("Startup preload failed: %s", exc)


def model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    raise TypeError("Unsupported model type")


@lru_cache
def get_service() -> GraphRAGWebService:
    return GraphRAGWebService()


app = FastAPI(title="菜谱 GraphRAG 可视化对话服务")
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.on_event("startup")
def startup_event() -> None:
    service = get_service()
    service.startup_preload()


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index_path)


@app.get("/api/home")
def home(batch: int = Query(0, ge=0), size: int = Query(6, ge=1, le=24)) -> Dict[str, Any]:
    service = get_service()
    return service.get_home_payload(batch=batch, size=size)


@app.get("/api/recommendations")
def recommendations(batch: int = Query(0, ge=0), size: int = Query(6, ge=1, le=24)) -> Dict[str, Any]:
    service = get_service()
    return {"recommendations": service.get_recommendations(batch=batch, size=size)}


@app.post("/api/chat")
def chat(payload: ChatRequest) -> Dict[str, Any]:
    service = get_service()
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    history = [model_to_dict(item) for item in payload.history][-12:]
    try:
        return service.chat(message=message, history=history)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"RAG 服务不可用: {exc}") from exc
    except Exception as exc:
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail=f"处理问题时出现错误: {exc}") from exc


@lru_cache(maxsize=32)
def fetch_recipe_image(url: str):
    """Cache successful external images; never serve LFS pointers as images."""
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=10) as response:
        content = response.read(20 * 1024 * 1024 + 1)
        media_type = response.headers.get_content_type()
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Image exceeds 20 MiB")
    valid_signature = (
        content.startswith(b"\xff\xd8\xff")
        or content.startswith(b"\x89PNG\r\n\x1a\n")
        or content.startswith((b"GIF87a", b"GIF89a"))
        or (content.startswith(b"RIFF") and content[8:12] == b"WEBP")
    )
    if not media_type.startswith("image/") or not valid_signature:
        raise ValueError("Remote response is not a supported image")
    return content, media_type


@app.get("/api/recipe-image/{recipe_key}")
def recipe_image(recipe_key: str):
    service = get_service()
    recipe = service.get_recipe_by_key(recipe_key)
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")

    resolved = service.resolve_recipe_image(recipe)
    if resolved["type"] == "remote":
        try:
            content, media_type = fetch_recipe_image(resolved["url"])
            return Response(content=content, media_type=media_type)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            logger.warning("Recipe image unavailable for %s: %s", recipe_key, exc)
            # Use the existing placeholder when an external image disappears.
    if resolved["type"] == "local":
        return FileResponse(path=resolved["path"])

    name = html.escape(recipe.get("name", "未知菜品"))
    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#d9f1ff"/>
      <stop offset="100%" stop-color="#eef7ff"/>
    </linearGradient>
  </defs>
  <rect width="640" height="360" fill="url(#g)"/>
  <text x="50%" y="46%" text-anchor="middle" fill="#2b4b6f"
        font-family="KaiTi, STKaiti, Kaiti SC, serif" font-size="28">暂无菜品图片</text>
  <text x="50%" y="58%" text-anchor="middle" fill="#4f6d8f"
        font-family="KaiTi, STKaiti, Kaiti SC, serif" font-size="20">{name}</text>
</svg>
""".strip()
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/debug/image/{recipe_key}")
def debug_one_image(
    recipe_key: str,
    probe: bool = Query(True),
    timeout: float = Query(6.0, ge=1.0, le=30.0),
) -> Dict[str, Any]:
    service = get_service()
    recipe = service.get_recipe_by_key(recipe_key)
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return service.debug_recipe_image(recipe, probe_remote=probe, timeout_seconds=timeout)


@app.get("/api/debug/images")
def debug_images(
    batch: int = Query(0, ge=0),
    size: int = Query(6, ge=1, le=24),
    probe: bool = Query(True),
    timeout: float = Query(6.0, ge=1.0, le=30.0),
) -> Dict[str, Any]:
    service = get_service()
    cards = service.get_recommendations(batch=batch, size=size)
    rows: List[Dict[str, Any]] = []
    for card in cards:
        key = str(card.get("key", ""))
        recipe = service.get_recipe_by_key(key)
        if not recipe:
            rows.append({"key": key, "error": "recipe_not_found"})
            continue
        rows.append(service.debug_recipe_image(recipe, probe_remote=probe, timeout_seconds=timeout))
    return {"batch": batch, "size": size, "probe": probe, "items": rows}


@app.get("/api/health")
def health() -> Dict[str, Any]:
    service = get_service()
    return {
        "status": "ok",
        "rag_ready": service._rag_system is not None,
        "rag_init_error": service._rag_init_error,
    }


@app.on_event("shutdown")
def shutdown_event() -> None:
    if get_service.cache_info().currsize > 0:
        service = get_service()
        service.close()
