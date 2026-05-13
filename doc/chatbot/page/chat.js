// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

(function () {
  "use strict";

  const STORAGE_KEY = "neml2-chat-history";
  const MAX_HISTORY = 8;
  const PLACEHOLDER = "Ask about NEML2 — models, tensors, input files, build, …";

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k.startsWith("on") && typeof v === "function") {
          node.addEventListener(k.slice(2), v);
        } else if (v !== false && v != null) {
          node.setAttribute(k, v);
        }
      }
    }
    for (const c of children) {
      if (c == null) continue;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }

  function resolveEndpoint(root) {
    const attr = root.getAttribute("data-endpoint");
    const isLocal = ["localhost", "127.0.0.1"].includes(location.hostname);
    if (isLocal) return "http://localhost:8787/chat";
    return attr || "";
  }

  function loadHistory() {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const arr = JSON.parse(raw);
      return Array.isArray(arr) ? arr : [];
    } catch (_) {
      return [];
    }
  }

  function saveHistory(history) {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (_) {
      // sessionStorage full or disabled; ignore
    }
  }

  // Configure DOMPurify to open every link in a new tab. Done once at script
  // load; subsequent sanitize() calls all benefit. The hook is a no-op when
  // DOMPurify isn't available.
  if (typeof DOMPurify !== "undefined") {
    DOMPurify.addHook("afterSanitizeAttributes", function (node) {
      if (node.tagName === "A") {
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noopener noreferrer");
      }
    });
  }

  // Convert Doxygen math delimiters to MathJax delimiters, then double-escape
  // every `\(`, `\)`, `\[`, `\]` so marked doesn't strip the backslash via
  // its standard escape rule. After marked runs, `\\(` becomes `\(` in the
  // HTML output, where MathJax (loaded by Doxygen on every page) picks it up
  // when we call typesetMath() on the bubble.
  function preprocessMath(text) {
    // \f$ delimiters are symmetric — alternate between open and close.
    let inMath = false;
    text = text.replace(/\\f\$/g, function () {
      inMath = !inMath;
      return inMath ? "\\(" : "\\)";
    });
    // \f[ ... \f] is asymmetric (opening vs closing).
    text = text.replace(/\\f\[/g, "\\[").replace(/\\f\]/g, "\\]");
    // Protect MathJax delimiters from marked's `\X` -> `X` escape rule.
    text = text.replace(/\\([()[\]])/g, "\\\\$1");
    return text;
  }

  // Render assistant text as markdown -> sanitized HTML. Returns null when
  // the marked/DOMPurify libs failed to load, so the caller can fall back to
  // textContent and the chat still works (just without formatting).
  function renderMarkdown(text) {
    if (typeof marked === "undefined" || typeof DOMPurify === "undefined") return null;
    try {
      const html = marked.parse(text, { breaks: true, gfm: true });
      return DOMPurify.sanitize(html);
    } catch (_) {
      return null;
    }
  }

  // Run MathJax over an element. No-op when MathJax isn't available (e.g.
  // because the parent Doxygen page didn't load it). Call after the stream
  // completes — typesetting per-token would be expensive and re-flow the
  // bubble noisily.
  function typesetMath(element) {
    if (typeof MathJax === "undefined" || typeof MathJax.typesetPromise !== "function") return;
    MathJax.typesetPromise([element]).catch(function () {
      // Swallow typeset errors (malformed math) — we'd rather leave raw text
      // visible than break the whole render.
    });
  }

  // Set a bubble's content, honoring the role:
  //   - assistant: preprocess math + parse as markdown when libs are
  //     available, else textContent.
  //   - user: always textContent (don't render user input as HTML).
  function setBubbleContent(body, role, text) {
    if (role === "assistant") {
      const html = renderMarkdown(preprocessMath(text));
      if (html !== null) {
        body.innerHTML = html;
        return;
      }
    }
    body.textContent = text;
  }

  function bubble(role, text) {
    const wrap = el("div", { class: `neml2-chat-bubble neml2-chat-${role}` });
    const body = el("div", { class: "neml2-chat-bubble-body" });
    setBubbleContent(body, role, text);
    wrap.appendChild(body);
    return { wrap, body };
  }

  function citationFooter(sources) {
    const list = el("div", { class: "neml2-chat-citations" });
    list.appendChild(el("span", { class: "neml2-chat-citations-label", text: "Sources: " }));
    sources.forEach((s, i) => {
      if (i > 0) list.appendChild(document.createTextNode(" · "));
      list.appendChild(el("a", { href: s.url, target: "_blank", rel: "noopener" }, `[${s.n}] ${s.title || s.ref}`));
    });
    return list;
  }

  function parseSseStream(stream, onToken, onSources, onDone, onError) {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let event = null;
    function flushEvent(dataStr) {
      if (!dataStr) return;
      let payload;
      try {
        payload = JSON.parse(dataStr);
      } catch (_) {
        return;
      }
      if (event === "sources") onSources(payload.sources || []);
      else if (event === "done") onDone();
      else if (event === "error") onError(new Error(payload.message || "stream_error"));
      else if (payload && payload.type === "token") onToken(payload.text || "");
      event = null;
    }
    return reader.read().then(function pump({ value, done }) {
      if (done) {
        if (buffer.length > 0) {
          // process trailing event without newline
          buffer.split("\n").forEach((line) => {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) flushEvent(line.slice(5).trim());
          });
        }
        return;
      }
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        let dataStr = "";
        block.split("\n").forEach((line) => {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
        });
        flushEvent(dataStr);
      }
      return reader.read().then(pump);
    });
  }

  function mount(root) {
    const endpoint = resolveEndpoint(root);
    if (!endpoint) {
      root.appendChild(
        el(
          "p",
          { class: "neml2-chat-error" },
          "Chatbot endpoint is not configured. Set the `data-endpoint` attribute on the root element."
        )
      );
      return;
    }

    const history = loadHistory();
    const messages = el("div", { class: "neml2-chat-messages" });
    const input = el("textarea", {
      class: "neml2-chat-input",
      placeholder: PLACEHOLDER,
      rows: "2",
      "aria-label": "Ask a question",
    });
    const sendBtn = el("button", { class: "neml2-chat-send", type: "button", text: "Send" });
    const clearBtn = el("button", { class: "neml2-chat-clear", type: "button", text: "Clear" });
    const composer = el(
      "div",
      { class: "neml2-chat-composer" },
      input,
      el("div", { class: "neml2-chat-controls" }, sendBtn, clearBtn)
    );
    const container = el("div", { class: "neml2-chat-container" }, messages, composer);
    root.appendChild(container);

    function appendHistoryRender() {
      messages.innerHTML = "";
      for (const m of history) {
        const { wrap } = bubble(m.role, m.content);
        if (m.role === "assistant" && m.sources && m.sources.length) {
          wrap.appendChild(citationFooter(m.sources));
        }
        messages.appendChild(wrap);
      }
      // Typeset math across the whole rendered history in one MathJax pass.
      typesetMath(messages);
      messages.scrollTop = messages.scrollHeight;
    }

    appendHistoryRender();

    function setBusy(busy) {
      input.disabled = busy;
      sendBtn.disabled = busy;
      sendBtn.textContent = busy ? "Sending…" : "Send";
    }

    async function send() {
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      const userMsg = { role: "user", content: text };
      history.push(userMsg);
      const { wrap: userWrap } = bubble("user", text);
      messages.appendChild(userWrap);

      const { wrap: aWrap, body: aBody } = bubble("assistant", "");
      aBody.textContent = "…";
      messages.appendChild(aWrap);
      messages.scrollTop = messages.scrollHeight;

      setBusy(true);
      let acc = "";
      let sources = [];
      try {
        const trimmedHistory = history.slice(-MAX_HISTORY).map((m) => ({
          role: m.role,
          content: m.content,
        }));
        const resp = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: trimmedHistory }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        if (!resp.body) throw new Error("no response body");

        await parseSseStream(
          resp.body,
          (tok) => {
            acc += tok;
            // Re-render on every token. Markdown libs handle partial input
            // gracefully; the user sees text appear in formatted form, with
            // partial-link/code-fence flicker that resolves once the token
            // finishes streaming.
            setBubbleContent(aBody, "assistant", acc);
            messages.scrollTop = messages.scrollHeight;
          },
          (srcs) => {
            sources = srcs;
            if (srcs.length) aWrap.appendChild(citationFooter(srcs));
          },
          () => {
            history.push({ role: "assistant", content: acc, sources });
            saveHistory(history);
            // Math is left as raw `\(...\)` during streaming (typesetting per
            // token would be expensive and visually noisy). Now that the
            // stream is complete, run MathJax over this bubble once.
            typesetMath(aBody);
          },
          (err) => {
            aBody.textContent = `Error: ${err.message}`;
          }
        );
      } catch (err) {
        aBody.textContent = `Error: ${err && err.message ? err.message : "request failed"}`;
      } finally {
        setBusy(false);
        input.focus();
      }
    }

    sendBtn.addEventListener("click", send);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });
    clearBtn.addEventListener("click", () => {
      history.length = 0;
      saveHistory(history);
      messages.innerHTML = "";
      input.focus();
    });
    input.focus();
  }

  ready(() => {
    const root = document.getElementById("neml2-chatbot-root");
    if (root) mount(root);
  });
})();
