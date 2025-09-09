import os
import re
import shutil
import string
import urllib.parse
import subprocess
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field
from loguru import logger
from openai import OpenAI  

# ========================
# 常量与目录
# ========================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_BASE = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("OR_MODEL", "google/gemini-2.5-pro")  # 选择支持 structured outputs 的模型

DEFAULT_HEADERS = {
    "HTTP-Referer": os.environ.get("OR_HTTP_REFERER", "http://localhost"),
    "X-Title": os.environ.get("OR_X_TITLE", "Markdown-to-JSON Converter"),
}

BASE_DIR = Path(__file__).resolve().parent
PDFS_DIR = BASE_DIR / "pdfs"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
PDFS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# arXiv 下载模板
ARXIV_PDF_TMPL = "https://arxiv.org/pdf/{arxiv_id}.pdf"

# 缺省执行参数（可被请求覆盖）
CONDA_ENV_DEFAULT = os.environ.get("MINERU_CONDA_ENV", "minerU")  # 可设为空字符串表示不用 conda run
MINERU_BIN_DEFAULT = os.environ.get("MINERU_BIN", "mineru")       # 也可设为绝对路径
CONDA_SH_DEFAULT = os.environ.get("CONDA_SH", "")

SAFE_ENV_DEFAULTS = {
    "MINERU_MODEL_SOURCE": "modelscope",
    "OMP_NUM_THREADS": "1",
    "PYTHONNOUSERSITE": "1",
    "QT_QPA_PLATFORM": "offscreen",
    "MALLOC_ARENA_MAX": "2",
}

# ========================
# 通用工具
# ========================
def sanitize_filename(name: str) -> str:
    """简单清理文件名（移除不可见字符、保留常见符号）。"""
    valid = f"-_.() {string.ascii_letters}{string.digits}"
    name = "".join(c for c in name if c in valid)
    return name.strip() or "paper"

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# ========================
# 下载 arXiv PDF
# ========================
def download_arxiv_pdf(arxiv_id: str, pdfs_dir: Path = PDFS_DIR, timeout: int = 30) -> Path:
    """
    根据 arxiv_id 下载 PDF，保存到 pdfs_dir/<clean_name>.pdf
    返回保存路径。
    """
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    clean_name = sanitize_filename(arxiv_id)
    url = ARXIV_PDF_TMPL.format(arxiv_id=arxiv_id)
    save_path = pdfs_dir / f"{clean_name}.pdf"

    logger.info(f"Downloading arXiv PDF: {url} -> {save_path}")
    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Failed to download: {resp.status_code}")
            with open(save_path, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Download error: {e}")

    if save_path.stat().st_size < 1000:
        raise HTTPException(status_code=502, detail="Downloaded file too small, likely an error page.")
    logger.info(f"Downloaded: {save_path} ({save_path.stat().st_size} bytes)")
    return save_path

# ========================
# 调用 mineru CLI 解析（4 种模式）
# ========================
def _merged_env(extra_env: Optional[Dict[str, str]]) -> Dict[str, str]:
    env = os.environ.copy()
    for k, v in SAFE_ENV_DEFAULTS.items():
        env.setdefault(k, v)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    return env

def run_mineru_cli(
    pdf_path: Path,
    output_dir: Path,
    exec_mode: str = "conda_run",      # "conda_run" | "bash_activate" | "zsh_activate" | "direct_bin"
    conda_env: Optional[str] = CONDA_ENV_DEFAULT,
    mineru_bin: str = MINERU_BIN_DEFAULT,
    conda_sh: str = CONDA_SH_DEFAULT,
    timeout: int = 7200,               # ← 可长一些（首跑下模型很慢）
    extra_args: Optional[List[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
    log_prefix: str = "",
    poll_interval_secs: float = 1.0,   # 每秒刷一次日志
    wait_md_secs_after_exit: int = 300 # 进程结束后，最多再等 5 分钟找 *.md
) -> Path:
    """
    以流式方式运行 mineru CLI，并把 stdout/stderr 实时落盘。
    成功则返回该论文输出根目录：OUTPUT_DIR/<stem>
    """
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_path}")

    stem = pdf_path.stem
    paper_output_dir = output_dir / stem
    paper_output_dir.mkdir(parents=True, exist_ok=True)

    base_args = ["-p", str(pdf_path), "-o", str(paper_output_dir)]
    if extra_args is None:
        extra_args = ["--backend", "pipeline", "--method", "auto", "--lang", "en"]

    env = _merged_env(extra_env)

    prefix = log_prefix or f"{stem}"
    stdout_file = LOG_DIR / f"{prefix}.mineru.stdout.log"
    stderr_file = LOG_DIR / f"{prefix}.mineru.stderr.log"
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    sf = open(stdout_file, "a", encoding="utf-8")
    ef = open(stderr_file, "a", encoding="utf-8")

    if exec_mode == "conda_run":
        if not conda_env:
            raise HTTPException(status_code=400, detail="conda_run mode requires `conda_env`.")
        cmd = ["conda", "run", "-n", conda_env, "--no-capture-output", mineru_bin] + base_args + extra_args
    elif exec_mode == "bash_activate":
        if not conda_env or not conda_sh:
            raise HTTPException(status_code=400, detail="bash_activate mode requires `conda_env` and `conda_sh`.")
        inner = " ".join([mineru_bin] + [str(x) for x in (base_args + extra_args)])
        cmd = ["bash", "-lc", f"source '{conda_sh}' && conda activate {conda_env} && {inner}"]
    elif exec_mode == "zsh_activate":
        if not conda_env or not conda_sh:
            raise HTTPException(status_code=400, detail="zsh_activate mode requires `conda_env` and `conda_sh`.")
        inner = " ".join([mineru_bin] + [str(x) for x in (base_args + extra_args)])
        cmd = ["zsh", "-lc", f"source '{conda_sh}' && conda activate {conda_env} && {inner}"]
    elif exec_mode == "direct_bin":
        cmd = [mineru_bin] + base_args + extra_args
    else:
        raise HTTPException(status_code=400, detail=f"Unknown exec_mode: {exec_mode}")

    logger.info(f"[mineru] mode={exec_mode} cmd={' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),fer
            env=env
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,          
        )
    except FileNotFoundError as e:
        sf.close(); ef.close()
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Failed to exec CLI. Not found: {e}")

    start = time.time()
    try:
        while True:
            line_out = proc.stdout.readline() if proc.stdout else ""
            if line_out:
                sf.write(line_out); sf.flush()
            line_err = proc.stderr.readline() if proc.stderr else ""
            if line_err:
                ef.write(line_err); ef.flush()

            if proc.poll() is not None:  # 进程已退出
                break

            if time.time() - start > timeout:
                proc.kill()
                sf.write("\n[runner] KILLED: timeout reached\n"); sf.flush()
                ef.write("\n[runner] KILLED: timeout reached\n"); ef.flush()
                raise HTTPException(status_code=504, detail="mineru CLI timeout")

            time.sleep(poll_interval_secs)

        try:
            rest_out, rest_err = proc.communicate(timeout=1)
        except Exception:
            rest_out, rest_err = "", ""
        if rest_out:
            sf.write(rest_out); sf.flush()
        if rest_err:
            ef.write(rest_err); ef.flush()

        rc = proc.returncode
        if rc != 0:
            tail = (rest_err or "")[-300:]
            logger.error(f"[mineru] returncode={rc}, stderr_tail:\n{tail}")
            raise HTTPException(status_code=500, detail=f"mineru CLI failed (code {rc}). stderr_tail: {tail}")

    finally:
        sf.close(); ef.close()

    if wait_md_secs_after_exit > 0:
        deadline = time.time() + wait_md_secs_after_exit
        while time.time() < deadline:
            mds = list(paper_output_dir.rglob("*.md"))
            if mds:
                break
            time.sleep(0.5)

    return paper_output_dir

def locate_output_dirs(paper_output_dir: Path, wait_secs: int = 0) -> Tuple[Path, Path]:
    """
    在给定目录下定位 *.md。可选再等待 wait_secs 秒（通常 0，因上一步已经等过）。
    """
    def try_once() -> Tuple[Optional[Path], Optional[Path]]:
        preferred = [paper_output_dir / "auto", paper_output_dir / "vlm", paper_output_dir / "txt", paper_output_dir]
        for d in preferred:
            if d.exists():
                mds = sorted(d.glob("*.md"))
                if mds:
                    return d, mds[0]
        mds = sorted(paper_output_dir.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if mds:
            return mds[0].parent, mds[0]
        return None, None

    pd, md = try_once()
    if pd and md:
        return pd, md
    deadline = time.time() + max(0, wait_secs)
    while time.time() < deadline:
        time.sleep(0.5)
        pd, md = try_once()
        if pd and md:
            return pd, md

    items = []
    for p in sorted(paper_output_dir.rglob("*"))[:100]:
        kind = "D" if p.is_dir() else "F"
        try:
            size = p.stat().st_size if p.is_file() else "-"
        except Exception:
            size = "?"
        items.append(f"{kind}\t{p.relative_to(paper_output_dir)}\t{size}")
    logger.error("Markdown not found. Listing under paper_output_dir:\n" + "\n".join(items))
    raise HTTPException(status_code=500, detail=f"Markdown not found under: {paper_output_dir}")

def build_messages_for_markdown(markdown_text: str,
                                paper_id: str,
                                title: Optional[str] = None,
                                authors: Optional[List[str]] = None,
                                img_base_url: Optional[str] = None) -> List[Dict[str, str]]:
    """
    生成 system + user 的双消息：
    - system：放入规则
    - user：放入元信息 JSON + Markdown 正文
    """
    SYSTEM_PROMPT = """
# 任务
你是一个严格的 Markdown→JSON 结构化转换器。输入是一篇完整论文的 Markdown（来自 PDF→Markdown 的抽取）。请将其转换为**单一 JSON 对象**，用于前端进行细粒度渲染（章节导航、图片/表格/公式编号、交叉引用、锚点等）。

# 绝对要求
- **仅输出 JSON**，不要任何额外文本、注释或反引号。
- **字段必须完整、类型正确**；缺失信息填 `null` 或空数组。
- 给所有可导航实体分配稳定 `id`（小写短横线，如 `sec-4-experiments`、`fig-6`、`tbl-2`、`eq-3`）。
- Markdown 中的本地图片（如 `images/img_001.jpg`）原样放入 `src`；**不要下载或转 base64**。
- “Figure X: … / Table Y: …” 的说明进入 `caption`；表格尽量给出 `rows`（二维数组），若有 HTML 片段写入 `html`。
- 公式：`$$...$$`/`\\[...\\]` 为块级→`equations[]`；`$...$` 为行内→存入段落的 `inline_math[]`（含相对索引）。
- 参考文献解析到 `references[]`，正文中的 `[12]` 等引用放入 `citations[]` 并与 `references[].id` 对齐。
- 交叉引用（“Figure 6”“Table 2”“Eq. (3)”等）建立到对应 `id` 的映射，放入 `crossref.by_label`。
- 若无法确定，字段仍要保留（填空或 `null`），并在 `diagnostics.warnings[]` 记录中文原因。
- 请结合你自己的知识，对明显错误的表格/公式/拼写进行纠正（AI 论文领域请注意专有名词）。
# 输入与输出
- 输入：元信息 JSON + Markdown 正文（图片路径已标准化，如 `images/img_001.jpg`）
- 输出：**严格遵循调用方提供的 JSON Schema（strict 模式）** 的单一 JSON 对象。
    """

    meta = {
        "paper_id": paper_id,
        "title": title,
        "authors": authors or [],
        "img_base_url": img_base_url,
        "source": "arxiv",
    }
    user_content = (
        "下面是论文的元信息和Markdown正文（JSON + Markdown）。\n\n"
        "【元信息 JSON】\n"
        + json.dumps(meta, ensure_ascii=False)
        + "\n\n【Markdown 正文】\n"
        + markdown_text
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

def markdown_to_json_with_openrouter(
    markdown_text: str,
    paper_id: str,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    img_base_url: Optional[str] = None,
    model: str = MODEL,
    temperature: float = 0.0,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    使用 openai SDK 直连 OpenRouter，启用 Structured Outputs（JSON Schema）。
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("请设置 OPENROUTER_API_KEY 环境变量或在代码中传入。")

    client = OpenAI(
        base_url=API_BASE,
        api_key=OPENROUTER_API_KEY,
        default_headers=DEFAULT_HEADERS,
    )

    messages = build_messages_for_markdown(
        markdown_text=markdown_text,
        paper_id=paper_id,
        title=title,
        authors=authors,
        img_base_url=img_base_url,
    )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "paper_struct",
            "strict": True,
            "schema": PAPER_JSON_SCHEMA,
        },
    }

    if stream:
        chunks: List[str] = []
        with client.chat.completions.stream(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        ) as s:
            for event in s:
                if event.type == "message.delta":
                    if event.delta and event.delta.get("content"):
                        chunks.append(event.delta["content"])
                elif event.type == "message.completed":
                    pass
                elif event.type == "error":
                    raise RuntimeError(f"OpenRouter stream error: {event.error}")
        content = "".join(chunks).strip()
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        content = resp.choices[0].message.content

    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        first = content.find("{")
        last = content.rfind("}")
        if first == -1 or last == -1 or last < first:
            raise RuntimeError("模型未返回有效 JSON。")
        obj = json.loads(content[first:last + 1])
    return obj
    
# ========================
# 正则扫描 Markdown，重命名图片并写回
# ========================
IMG_MD_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")

class RenameResult(BaseModel):
    old_path: str
    new_path: str
    renamed: bool

def shorten_markdown_image_paths(
    md_path: Path,
    images_dir: Path,
    prefix: str = "img_",
    start_index: int = 1,
    keep_extension: bool = True,
    dry_run: bool = False,
) -> Tuple[str, List[RenameResult]]:
    """
    读取 md 文件，查找所有形如 ![](images/xxx.jpg) 的路径：
    - 将 images_dir 下对应文件重命名为短名（如 img_001.jpg）
    - 替换 md 文本中的路径
    - 返回更新后的 md 文本（字符串）和重命名映射列表
    """
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"Markdown not found: {md_path}")
    if not images_dir.exists():
        raise HTTPException(status_code=404, detail=f"Images dir not found: {images_dir}")

    original_md = md_path.read_text(encoding="utf-8", errors="ignore")

    matches = list(IMG_MD_PATTERN.finditer(original_md))
    if not matches:
        logger.info("No images found in markdown.")
        return original_md, []

    seen: Dict[str, str] = {}
    mapping: List[RenameResult] = []
    counter = start_index

    def make_new_name(ext: str) -> str:
        nonlocal counter
        new_name = f"{prefix}{counter:03d}{ext}"
        counter += 1
        return new_name

    for m in matches:
        rel_path_raw = m.group("path").strip()
        rel_path = urllib.parse.unquote(rel_path_raw)
        if re.match(r"^[a-zA-Z]+://", rel_path):
            continue

        rel_path_norm = rel_path
        if rel_path_norm.startswith("./"):
            rel_path_norm = rel_path_norm[2:]

        if not (rel_path_norm.startswith("images/") or rel_path_norm.startswith("images\\")):
            bare = Path(rel_path_norm).name
            candidate = images_dir / bare
            if candidate.exists():
                rel_path_norm = f"images/{bare}"
            else:
                continue

        if rel_path_norm in seen:
            continue

        old_file = images_dir / Path(rel_path_norm).name
        if not old_file.exists():
            candidate = images_dir / Path(rel_path_norm).name
            if candidate.exists():
                old_file = candidate
            else:
                logger.warning(f"Image file missing for path in MD: {rel_path_norm}")
                mapping.append(RenameResult(old_path=rel_path_raw, new_path=rel_path_raw, renamed=False))
                seen[rel_path_norm] = rel_path_norm
                continue

        ext = old_file.suffix if keep_extension else ".jpg"
        new_name = make_new_name(ext)
        new_rel_path = f"images/{new_name}"

        seen[rel_path_norm] = new_rel_path
        mapping.append(RenameResult(old_path=rel_path_raw, new_path=new_rel_path, renamed=True))

    if not dry_run:
        for res in mapping:
            if not res.renamed:
                continue
            new_file = images_dir / Path(res.new_path).name
            if new_file.exists():
                i = 1
                stem = new_file.stem
                ext = new_file.suffix
                while new_file.exists():
                    new_file = images_dir / f"{stem}_{i:02d}{ext}"
                    i += 1
                res.new_path = f"images/{new_file.name}"

        old_to_new_files: Dict[str, Tuple[Path, Path]] = {}
        for res in mapping:
            if not res.renamed:
                continue
            old_norm = urllib.parse.unquote(res.old_path).lstrip("./")
            if not (old_norm.startswith("images/") or old_norm.startswith("images\\")):
                old_norm = f"images/{Path(old_norm).name}"
            old_file = images_dir / Path(old_norm).name
            new_file = images_dir / Path(res.new_path).name
            old_to_new_files[old_file.as_posix()] = (old_file, new_file)

        tmp_files = []
        for old_file, new_file in old_to_new_files.values():
            if old_file.resolve() == new_file.resolve():
                continue
            tmp_file = old_file.with_suffix(old_file.suffix + ".renaming_tmp")
            if tmp_file.exists():
                tmp_file.unlink()
            shutil.move(old_file, tmp_file)
            tmp_files.append((tmp_file, new_file))

        for tmp_file, new_file in tmp_files:
            shutil.move(tmp_file, new_file)

    def replace_path(m: re.Match) -> str:
        alt = m.group("alt")
        rel_path_raw = m.group("path").strip()
        rel_path_norm = urllib.parse.unquote(rel_path_raw)
        if rel_path_norm.startswith("./"):
            rel_path_norm = rel_path_norm[2:]
        if not (rel_path_norm.startswith("images/") or rel_path_norm.startswith("images\\")):
            bare = Path(rel_path_norm).name
            if (images_dir / bare).exists():
                rel_path_norm = f"images/{bare}"
            else:
                return m.group(0)

        new_rel = seen.get(rel_path_norm)
        if not new_rel:
            return m.group(0)
        return f"![{alt}]({new_rel})"

    new_md = IMG_MD_PATTERN.sub(replace_path, original_md)

    if not dry_run and new_md != original_md:
        backup = md_path.with_suffix(md_path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(original_md, encoding="utf-8")
        md_path.write_text(new_md, encoding="utf-8")
        logger.info(f"Markdown updated and written back: {md_path}")

    return new_md, mapping

# ========================
# FastAPI 模型
# ========================
class ParseReq(BaseModel):
    exec_mode: str = Field(default="conda_run", description="conda_run | bash_activate | zsh_activate | direct_bin")
    conda_env: Optional[str] = CONDA_ENV_DEFAULT
    mineru_bin: str = MINERU_BIN_DEFAULT
    conda_sh: str = CONDA_SH_DEFAULT
    timeout: int = 7200
    extra_args: Optional[List[str]] = None
    extra_env: Optional[Dict[str, str]] = None
    log_prefix: Optional[str] = None
    poll_interval_secs: float = 1.0
    wait_md_secs_after_exit: int = 300

class ProcessMdReq(BaseModel):
    md_rel_path: Optional[str] = None
    images_subdir: str = "images"
    prefix: str = "img_"
    start_index: int = 1
    keep_extension: bool = True
    dry_run: bool = False

class LlmJsonReq(BaseModel):
    model: str = MODEL
    temperature: float = 0.0
    stream: bool = False
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    img_base_url: Optional[str] = None
    markdown: Optional[str] = None       
    md_rel_path: Optional[str] = None    

class IngestReq(ParseReq, ProcessMdReq):
    pass

# ========================
# FastAPI 应用 & 路由
# ========================
app = FastAPI(title="arXiv PDF → mineru CLI（多模式）→ Markdown 图片规范化 + LLM JSON API")

@app.post("/api/papers/{arxiv_id}/download")
def api_download(arxiv_id: str):
    pdf_path = download_arxiv_pdf(arxiv_id, PDFS_DIR)
    return {"arxiv_id": arxiv_id, "pdf_path": str(pdf_path)}

@app.post("/api/papers/{arxiv_id}/parse")
def api_parse(arxiv_id: str, req: ParseReq = Body(default=ParseReq())):
    pdf_path = PDFS_DIR / f"{sanitize_filename(arxiv_id)}.pdf"
    if not pdf_path.exists():
        pdf_path = download_arxiv_pdf(arxiv_id, PDFS_DIR)

    paper_output_dir = run_mineru_cli(
        pdf_path=pdf_path,
        output_dir=OUTPUT_DIR,
        exec_mode=req.exec_mode,
        conda_env=req.conda_env,
        mineru_bin=req.mineru_bin,
        conda_sh=req.conda_sh,
        timeout=req.timeout,
        extra_args=req.extra_args,
        extra_env=req.extra_env,
        log_prefix=req.log_prefix or f"{arxiv_id}.{req.exec_mode}",
        poll_interval_secs=req.poll_interval_secs,
        wait_md_secs_after_exit=req.wait_md_secs_after_exit
    )

    paper_dir, md_path = locate_output_dirs(paper_output_dir, wait_secs=0)
    return {"arxiv_id": arxiv_id, "output_dir": str(paper_dir), "md_path": str(md_path), "exists_md": md_path.exists()}

@app.post("/api/papers/{arxiv_id}/ingest")
def api_ingest(arxiv_id: str, req: IngestReq = Body(default=IngestReq())):
    os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
    stem = sanitize_filename(arxiv_id)
    pdf_path = PDFS_DIR / f"{stem}.pdf"
    if not pdf_path.exists():
        pdf_path = download_arxiv_pdf(arxiv_id, PDFS_DIR)

    paper_output_dir = run_mineru_cli(
        pdf_path=pdf_path,
        output_dir=OUTPUT_DIR,
        exec_mode=req.exec_mode,
        conda_env=req.conda_env,
        mineru_bin=req.mineru_bin,
        conda_sh=req.conda_sh,
        timeout=req.timeout,
        extra_args=req.extra_args,
        extra_env=req.extra_env,
        log_prefix=req.log_prefix or f"{arxiv_id}.{req.exec_mode}",
        poll_interval_secs=req.poll_interval_secs,
        wait_md_secs_after_exit=req.wait_md_secs_after_exit
    )

    paper_dir, md_path_auto = locate_output_dirs(paper_output_dir, wait_secs=0)
    md_path = Path(req.md_rel_path) if req.md_rel_path else md_path_auto
    images_dir = md_path.parent / req.images_subdir

    new_md, mapping = shorten_markdown_image_paths(
        md_path=md_path,
        images_dir=images_dir,
        prefix=req.prefix,
        start_index=req.start_index,
        keep_extension=req.keep_extension,
        dry_run=req.dry_run,
    )
    return {
        "arxiv_id": arxiv_id,
        "pdf_path": str(pdf_path),
        "output_dir": str(paper_dir),
        "md_path": str(md_path),
        "images_dir": str(images_dir),
        "mapping": [m.dict() for m in mapping],
        "md_preview_head": new_md[:1000]
    }

@app.post("/api/papers/{arxiv_id}/process_markdown")
def api_process_markdown(arxiv_id: str, req: ProcessMdReq = Body(default=ProcessMdReq())):
    """
    仅做“重命名 images 并回写 md”
    """
    stem = sanitize_filename(arxiv_id)
    if req.md_rel_path:
        md_path = Path(req.md_rel_path)
        images_dir = md_path.parent / req.images_subdir
    else:
        paper_output_dir = OUTPUT_DIR / stem
        if not paper_output_dir.exists():
            raise HTTPException(status_code=404, detail=f"Output dir not found for {arxiv_id}. Run /parse or /ingest first.")
        _, md_path = locate_output_dirs(paper_output_dir, wait_secs=0)
        images_dir = md_path.parent / req.images_subdir

    new_md, mapping = shorten_markdown_image_paths(
        md_path=md_path,
        images_dir=images_dir,
        prefix=req.prefix,
        start_index=req.start_index,
        keep_extension=req.keep_extension,
        dry_run=req.dry_run,
    )
    return {
        "arxiv_id": arxiv_id,
        "md_path": str(md_path),
        "images_dir": str(images_dir),
        "updated": not req.dry_run,
        "mapping": [m.dict() for m in mapping],
        "md_preview_head": new_md[:800]
    }

PAPER_JSON_SCHEMA: Dict[str, Any] = { 
    "type": "object",
    "properties": {
        "paper": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": ["string", "null"]},
                "authors": {"type": "array", "items": {"type": "string"}},
                "abstract": {"type": ["string", "null"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "source": {"type": ["string", "null"]},
                        "arxiv_id": {"type": ["string", "null"]},
                        "created_at": {"type": ["string", "null"]},
                    },
                    "required": ["source", "arxiv_id", "created_at"],
                    "additionalProperties": False,
                },
            },
            "required": ["id", "title", "authors", "abstract", "meta"],
            "additionalProperties": False,
        },
        "toc": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "level": {"type": "integer", "minimum": 1, "maximum": 6},
                    "children": {"type": "array", "items": {"$ref": "#/properties/toc/items"}},
                },
                "required": ["id", "title", "level", "children"],
                "additionalProperties": False,
            },
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "level": {"type": "integer", "minimum": 1, "maximum": 6},
                    "content": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {"const": "paragraph"},
                                        "id": {"type": "string"},
                                        "text": {"type": "string"},
                                        "inline_math": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "latex": {"type": "string"},
                                                    "start": {"type": "integer"},
                                                    "end": {"type": "integer"},
                                                },
                                                "required": ["latex", "start", "end"],
                                                "additionalProperties": False,
                                            },
                                        },
                                        "citations": {"type": "array", "items": {"type": "string"}},
                                    },
                                    "required": ["type", "id", "text", "inline_math", "citations"],
                                    "additionalProperties": False,
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {"const": "figure"},
                                        "ref_id": {"type": "string"},
                                        "caption": {"type": ["string", "null"]},
                                        "src": {"type": "string"},
                                        "width": {"type": ["number", "null"]},
                                        "height": {"type": ["number", "null"]},
                                        "page_hint": {"type": ["integer", "null"]},
                                    },
                                    "required": ["type", "ref_id", "caption", "src", "width", "height", "page_hint"],
                                    "additionalProperties": False,
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {"const": "table"},
                                        "ref_id": {"type": "string"},
                                        "caption": {"type": ["string", "null"]},
                                        "html": {"type": ["string", "null"]},
                                        "rows": {
                                            "type": "array",
                                            "items": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            }
                                        },
                                    },
                                    "required": ["type", "ref_id", "caption", "html", "rows"],
                                    "additionalProperties": False,
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {"const": "equation"},
                                        "ref_id": {"type": "string"},
                                        "latex": {"type": "string"},
                                        "number": {"type": ["string", "null"]},
                                    },
                                    "required": ["type", "ref_id", "latex", "number"],
                                    "additionalProperties": False,
                                },
                            ]
                        },
                    },
                },
                "required": ["id", "title", "level", "content"],
                "additionalProperties": False,
            },
        },
        "equations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "latex": {"type": "string"},
                    "number": {"type": ["string", "null"]},
                    "section": {"type": "string"},
                },
                "required": ["id", "latex", "number", "section"],
                "additionalProperties": False,
            },
        },
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "caption": {"type": ["string", "null"]},
                    "src": {"type": "string"},
                    "section": {"type": "string"},
                },
                "required": ["id", "caption", "src", "section"],
                "additionalProperties": False,
            },
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "caption": {"type": ["string", "null"]},
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "section": {"type": "string"},
                },
                "required": ["id", "caption", "rows", "section"],
                "additionalProperties": False,
            },
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "marker": {"type": "string"},
                    "ref": {"type": "string"},
                    "section": {"type": "string"},
                },
                "required": ["marker", "ref", "section"],
                "additionalProperties": False,
            },
        },
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "title": {"type": ["string", "null"]},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "venue": {"type": ["string", "null"]},
                    "year": {"type": ["integer", "null"]},
                    "doi": {"type": ["string", "null"]},
                    "arxiv": {"type": ["string", "null"]},
                    "url": {"type": ["string", "null"]},
                },
                "required": ["id", "text", "title", "authors", "venue", "year", "doi", "arxiv", "url"],
                "additionalProperties": False,
            },
        },
        "footnotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "section": {"type": "string"},
                },
                "required": ["id", "text", "section"],
                "additionalProperties": False,
            },
        },
        "crossref": {
            "type": "object",
            "properties": {
                "by_label": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            },
            "required": ["by_label"],
            "additionalProperties": False,
        },
        "diagnostics": {
            "type": "object",
            "properties": {
                "warnings": {"type": "array", "items": {"type": "string"}},
                "stats": {
                    "type": "object",
                    "properties": {
                        "sections": {"type": "integer"},
                        "figures": {"type": "integer"},
                        "tables": {"type": "integer"},
                        "equations": {"type": "integer"},
                        "citations": {"type": "integer"},
                        "references": {"type": "integer"},
                    },
                    "required": ["sections", "figures", "tables", "equations", "citations", "references"],
                    "additionalProperties": False,
                },
            },
            "required": ["warnings", "stats"],
            "additionalProperties": False,
        },
    },
    "required": [
        "paper", "toc", "sections", "equations", "figures", "tables",
        "citations", "references", "footnotes", "crossref", "diagnostics"
    ],
    "additionalProperties": False,
}


@app.post("/api/papers/{arxiv_id}/to_json")
def api_paper_to_json(arxiv_id: str, req: LlmJsonReq = Body(default=LlmJsonReq())):
    stem = sanitize_filename(arxiv_id)

    # 定位 markdown
    if req.md_rel_path:
        md_path = Path(req.md_rel_path)
        if not md_path.exists():
            raise HTTPException(status_code=404, detail=f"Markdown not found: {md_path}")
    else:
        paper_output_dir = OUTPUT_DIR / stem
        if not paper_output_dir.exists():
            raise HTTPException(status_code=404, detail=f"Parse output dir not found: {paper_output_dir}")
        _, md_path = locate_output_dirs(paper_output_dir, wait_secs=0)

    md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    result = markdown_to_json_with_openrouter(
        markdown_text=md_text,
        paper_id=stem,
        title=req.title,
        authors=req.authors,
        img_base_url=req.img_base_url,
        model=req.model,
        temperature=req.temperature,
        stream=req.stream,
    )
    return result

# —— 新增路由：直接传 Markdown 文本转 JSON ——
@app.post("/api/markdown/to_json")
def api_markdown_to_json(req: LlmJsonReq = Body(default=LlmJsonReq())):
    if not req.markdown:
        raise HTTPException(status_code=400, detail="`markdown` 不能为空。")
    result = markdown_to_json_with_openrouter(
        markdown_text=req.markdown,
        paper_id="adhoc",
        title=req.title,
        authors=req.authors,
        img_base_url=req.img_base_url,
        model=req.model,
        temperature=req.temperature,
        stream=req.stream,
    )
    return result

@app.get("/api/debug/openrouter/ping")
def api_debug_openrouter_ping(model: str = Query(MODEL)):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY 未设置。")
    try:
        client = OpenAI(base_url=API_BASE, api_key=OPENROUTER_API_KEY, default_headers=DEFAULT_HEADERS)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            response_format={"type": "json_schema", "json_schema": {"name": "pong", "strict": True, "schema": {"type": "object", "properties": {"pong": {"type": "string"}}, "required": ["pong"], "additionalProperties": False}}},
            temperature=0.0,
        )
        return {"ok": True, "model": model, "content": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenRouter ping failed: {e}")

# 运行：uvicorn app:app --host 0.0.0.0 --port 8000
