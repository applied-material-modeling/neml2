# NEML2 docs chatbot

The "Ask AI" page on the deployed docs is a retrieval-augmented chatbot. The neml2 repo owns only the page UI and its Doxygen wiring. Everything else — the Cloudflare Worker that serves `POST /chat`, the indexer that populates Vectorize, and all credentialed CI — lives in the [`neml2-chat-worker`](https://github.com/applied-material-modeling/neml2-chat-worker) repo.

```
chatbot.html (Doxygen-rendered page in this repo)
    │   chat.js mounts a messenger UI inside #neml2-chatbot-root
    ▼
Cloudflare Worker  ─►  Vectorize    (top-k retrieval, populated by the worker repo's ingest)
   (separate repo)  └─►  Workers AI   (embed user query + stream LLM tokens)
```

What lives where:

| Component | Repo | Lifecycle |
|---|---|---|
| `page/chat.js`, `page/chat.css` — chat UI | this repo (`doc/chatbot/page/`) | Built into the docs by Doxygen via `HTML_EXTRA_FILES` / `HTML_EXTRA_STYLESHEET` |
| `chatbot.md` — the page itself | this repo (`doc/content/chatbot.md`) | Registered as the top-level "Ask AI" tab in `DoxygenLayout(.xml,Python.xml)` |
| Cloudflare Worker (`POST /chat`, RAG + LLM streaming) | `neml2-chat-worker` (`src/`) | Maintainer-owned; deploys independently |
| Indexer that populates Vectorize | `neml2-chat-worker` (`ingest/`) | Maintainer-owned; runs on a schedule (typically weekly) |
| Cloudflare account, Vectorize index, AI Gateway, secrets | The maintainer's Cloudflare account | Provisioned once per `BRING-UP.md` in the worker repo |

The split exists because the worker + indexer share a tech stack (Cloudflare bindings, secrets, scheduled CI) that's unrelated to NEML2's C++/Python core. Regular contributors only need to read this page; only maintainers touch the worker repo. The neml2 docs CI does not need any Cloudflare secrets — re-indexing happens on the worker repo's schedule, against the latest neml2 doc build at that time.

## Integration contract with neml2-chat-worker

Two assumptions must hold across both repos. Any change to either requires coordinated PRs.

**Vectorize record metadata.** The indexer writes these keys; the worker reads them; the page renders the `n,url,title,ref` shape on the `event: sources` line as citation links.
- `text` — chunk's raw text.
- `url` — fully-qualified citation URL.
- `title` — page title (used in prompt + citation footer).
- `ref` — Doxygen page slug.
- `anchor` — sub-heading anchor (may be empty).

**Wire format on `POST /chat`** (browser ↔ worker, served as SSE):
- `data: {"type":"token","text":"..."}` per LLM token (no `event:` line).
- `event: sources\ndata: {"sources":[{n,url,title,ref}, ...]}` once.
- `event: done\ndata: {}` as the terminator.
- `event: error\ndata: {"message":"..."}` on stream failure.

`page/chat.js` here consumes that format. The worker's CORS allowlist (`ALLOWED_ORIGINS`) must include the deployed doc origin (`https://applied-material-modeling.github.io`).

The contract derives ultimately from how the indexer reads neml2's preprocessed markdown — every page must start with `# Title {#ref}` so the indexer can extract the page slug and build the citation URL `<base>/<ref>.html[#anchor]`. `doc/scripts/genhtml.py` + `doc/scripts/preprocess.py` already guarantee this; if a doc PR ever broke it, the indexer would silently skip those pages.

## Working on the page

`chat.js` mounts the messenger UI into `#neml2-chatbot-root` and consumes the worker's SSE stream. Styling in `chat.css` reads doxygen-awesome's CSS custom properties so light/dark mode tracks the doc theme.

To test page changes against either the deployed worker or a local `wrangler dev` (run from the worker repo), serve the built docs over HTTP — **not `file://`**:

> **Why:** opening `chatbot.html` by double-click sets the browser's `Origin` header to `null`, which can never match the worker's CORS allowlist. Fetches fail with "Failed to fetch" with no further detail.

```
# Build first (see /build-docs skill or doc/scripts/genhtml.py)
cd build/doc/build/html && python3 -m http.server 8000
# Then open http://localhost:8000/chatbot.html
```

`chat.js` detects `localhost`/`127.0.0.1` and automatically points to `http://localhost:8787/chat`, so a local `wrangler dev` (started in the worker repo) is picked up without further config.

## Operating notes

**Index drift.** The chatbot answers from whatever the worker repo's last scheduled reindex captured. Doc changes here become visible to the bot after the next reindex (typically up to a week). If the bot gives a stale answer, that's almost always why.

For cost ceilings, hallucination guardrails, provider-swap notes, and the indexer's design, see the [worker repo](https://github.com/applied-material-modeling/neml2-chat-worker)'s `README.md`.
