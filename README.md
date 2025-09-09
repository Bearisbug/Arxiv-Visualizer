
<img src="https://info.arxiv.org/brand/images/brand-logo-primary.jpg" alt="arXiv 标志" width="200" />

# Arxiv - Visualizer

本服务把 **arXiv PDF** 自动化转换为**结构化论文 JSON**，供任意前端精细渲染（目录、锚点、行内/块级数学、图表、交叉引用等）。

---

## 1) 流程

```

arXiv PDF
└─(mineru)→ Markdown(+images)
└─(规范化图片路径)→ 标准 Markdown
└─(OpenRouter Structured Outputs)→ 结构化 JSON（PAPER_JSON_SCHEMA）

````

---

## 2) 运行
需要自行安装 [MinerU](https://github.com/opendatalab/MinerU)
```bash
pip install fastapi uvicorn httpx loguru "openai>=1.40.0" pydantic
# 确保 mineru 可用（建议放入 Conda 环境）
export OPENROUTER_API_KEY=sk-or-xxx

uvicorn app:app --host 0.0.0.0 --port 8000
````

可选环境变量：

* `MINERU_CONDA_ENV`（默认 `minerU`）、`MINERU_BIN`（默认 `mineru`）、`CONDA_SH`（conda 激活脚本）
* `OR_MODEL`（支持 Structured Outputs 的模型，如 `google/gemini-2.5-pro`）
* `OR_HTTP_REFERER`、`OR_X_TITLE`（OpenRouter 标头）

---

## 3) 关键接口

1. **下载 PDF**：`POST /api/papers/{arxiv_id}/download`
2. **解析 Markdown**：`POST /api/papers/{arxiv_id}/parse`
3. **（可选）图片重命名并写回 md**：`POST /api/papers/{arxiv_id}/process_markdown`
4. **Markdown → 结构化 JSON**：`POST /api/papers/{arxiv_id}/to_json`  ← 最终用于前端

> 一步到位：`POST /api/papers/{arxiv_id}/ingest`（下载+解析+图片规范化）

简例：

```bash
curl -X POST http://localhost:8000/api/papers/2506.17368/download
curl -X POST http://localhost:8000/api/papers/2506.17368/parse
curl -X POST http://localhost:8000/api/papers/2506.17368/process_markdown
curl -X POST http://localhost:8000/api/papers/2506.17368/to_json \
  -H "Content-Type: application/json" \
  -d '{"model":"google/gemini-2.5-pro","temperature":0.0}'
```

---

## 4) 前端渲染逻辑

最终返回对象满足 `PAPER_JSON_SCHEMA`。下面是**渲染契约**与**解析逻辑**，前端只需遵守这些步骤即可获得完整的论文页面行为。

### 4.1 全局原则

* **一切可导航元素都有稳定 `id`**（如 `sec-4-experiments`、`fig-2`、`tbl-1`、`eq-3`）。
* **图片路径不修改内容本身**：字段 `src` 是相对路径（如 `images/img_001.jpg`）；前端仅做**前缀拼接**（如 `/static/<paperId>/` + `src`）。
* **交叉引用**依赖 `crossref.by_label`：例如 `"Figure 6" -> "fig-6"`，用于把正文文本中的“Figure 6”替换为锚点链接。
* **数学公式**：

  * 行内：正文的 `$...$`（已由 LLM抽取到 `inline_math` 的索引区间，但前端可直接用 `$...$` 识别）
  * 块级：`equation` 节点或 `equations[]` 列表中的 `latex`，配 `number`（编号）

### 4.2 页面结构与滚动行为

* **目录（TOC）**：使用 `toc` 数组生成层级树；每个节点的 `id` 与正文对应。
* **滚动高亮**：用 IntersectionObserver 观察 `sections[].id` 对应的元素可见性，取最靠前者作为当前高亮。
* **锚点偏移**：正文节点设置 `scroll-margin-top`，避免被固定头部遮挡。

### 4.3 主体渲染顺序（逐节解析）

对 `sections` 数组，从上到下渲染：

1. **标题**：使用 `section.title`，DOM 节点 `id=section.id`，供锚点跳转。
2. **内容流**：遍历 `section.content[]`，根据 `type` 选择渲染分支：

   * `paragraph`：

     * 文本字段：`text`。
     * **行内数学**：识别 `$...$` 片段并渲染为行内公式。
     * **交叉引用替换**：用 `crossref.by_label` 把文本中的 “Figure 6”“Eq. (3)” 等，替换为 `<a href="#{mappedId}">...</a>`。
     * **文献引用**：`citations` 是如 `["[12]","[3]"]` 的列表；以上标方式显示，或在末尾合并展示。
   * `figure`：

     * 图片地址：`finalUrl = imgBaseUrl + "/" + node.src`（仅拼前缀）。
     * 显示 `caption` 为图注；DOM `id=node.ref_id` 作为锚点。
     * 建议提供点击放大（对话框/Lightbox）。
   * `table`：

     * **优先**使用 `rows: string[][]` 渲染表格；
     * 若 `rows` 为空且 `html` 不为空，则**白名单过滤**后以 `innerHTML` 回退渲染；
     * DOM `id=node.ref_id`；表注使用 `caption`。
   * `equation`：

     * 渲染 `latex` 为**块级**公式（display mode），右侧/旁边显示 `number`（如 `(3)`）；
     * DOM `id=node.ref_id` 供交叉引用跳转。

### 4.4 参考文献与脚注

* `references[]`：每条包含 `id、text、title、authors、venue、year、doi、arxiv、url`。

  * 正文引用的 `[12]` 应当能在此列表中找到 `id="12"`（示例）。
* `footnotes[]`：每条 `id/text/section`；可在文末或对应 `section` 下集中显示。

### 4.5 搜索与定位（建议）

* 建立简单索引：收集以下文本源参与搜索匹配：

  * `sections[].title`
  * `paragraph.text`（移除 HTML 标记后）
  * `figure.caption`、`table.caption`
  * `equation.latex`（可降噪处理）
* 点击搜索结果：`scrollIntoView` 到对应 DOM `id`，并临时添加高亮类（如 2s 褪去的背景色）。

### 4.6 诊断信息

* `diagnostics.warnings[]`：显示在页面顶部的警告区域（例如字段缺失、解析不确定等）。
* `diagnostics.stats`：可用于展示“共 N 节 / N 图 / N 表 / N 式 / N 引用”。

---

## 5) JSON 字段对照速查

| 字段路径                                     | 含义     | 用途                                |
| ---------------------------------------- | ------ | --------------------------------- |
| `paper.id/title/authors/abstract/meta.*` | 论文元信息  | 顶部信息区                             |
| `toc[]`（含 `id/title/level/children[]`）   | 目录树    | 左侧 TOC，生成锚点链接                     |
| `sections[].id/title/level`              | 章节元    | 标题 + 锚点                           |
| `sections[].content[].type`              | 节内节点类型 | `paragraph/figure/table/equation` |
| `paragraph.text/inline_math/citations[]` | 段落内容   | 文本 + 行内数学 + 文献上标                  |
| `figure.ref_id/src/caption`              | 图      | 图片 + 图注 + 锚点                      |
| `table.ref_id/rows/html/caption`         | 表      | 二维数组优先；无则用 `html`                 |
| `equation.ref_id/latex/number`           | 块级公式   | KaTeX display + 编号                |
| `crossref.by_label`                      | 交叉引用映射 | “Figure 6” → `#fig-6`             |
| `references[]`                           | 参考文献表  | 与 `citations[]` 对应                |
| `footnotes[]`                            | 脚注     | 章节或文末呈现                           |
| `diagnostics.*`                          | 警告/统计  | 体验与排障                             |
