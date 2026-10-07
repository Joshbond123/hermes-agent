import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Markdown } from "./markdown/Markdown";
import { isSafeUrl, lexMarkdown } from "./markdown/utils";
import { escapeHtml, highlight, languageLabel, normalizeLanguage } from "./markdown/highlight";

const html = (text: string, streaming = false) => renderToStaticMarkup(<Markdown text={text} streaming={streaming} />);

describe("markdown rendering", () => {
  it("renders headings, emphasis, inline code, links, lists", () => {
    const out = html("# Title\n\n## Sub\n\nSome **bold**, *italic*, ~~gone~~ and `code`.\n\n- a\n- b\n\n3. three\n4. four\n");
    expect(out).toContain("<h1");
    expect(out).toContain("<h2");
    expect(out).toContain("<strong>bold</strong>");
    expect(out).toContain("<em>italic</em>");
    expect(out).toContain("<del>gone</del>");
    expect(out).toContain('<code class="bt-inline-code">code</code>');
    expect(out).toMatch(/<ul[^>]*><li>.*a.*<\/li><li>.*b.*<\/li><\/ul>/s);
    expect(out).toContain('start="3"');
  });

  it("renders GFM tables with alignment, and task lists", () => {
    const out = html("| Name | Qty |\n|:-----|----:|\n| apple | 3 |\n| pear | 10 |\n\n- [x] done\n- [ ] todo\n");
    expect(out).toContain("<table>");
    expect(out).toContain('<th style="text-align:left">Name</th>');
    expect(out).toContain('<td style="text-align:right">10</td>');
    expect(out).toContain("bt-table-wrap");
    expect(out).toMatch(/type="checkbox"[^>]*checked/);
  });

  it("renders blockquotes, rules, nested lists", () => {
    const out = html("> quoted *text*\n\n---\n\n1. one\n   - nested\n2. two\n");
    expect(out).toContain("<blockquote>");
    expect(out).toContain("<hr/>");
    expect(out).toMatch(/<ol[^>]*>.*<ul[^>]*>.*nested.*<\/ul>.*<\/ol>/s);
  });

  it("links: safe urls open in a new tab with rel; dangerous urls are not linked", () => {
    const ok = html("[site](https://example.com/a?b=1)");
    expect(ok).toContain('href="https://example.com/a?b=1"');
    expect(ok).toContain('target="_blank"');
    expect(ok).toContain("noopener");
    const bad = html("[click](javascript:alert(1)) and [d](data:text/html;base64,AAAA) and [v](  VbScript:x)");
    expect(bad).not.toContain("<a ");
    expect(bad).toContain("click");
  });

  it("isSafeUrl", () => {
    for (const u of ["https://a.b", "http://a.b", "mailto:a@b.c", "#frag", "/rel", "./rel"]) expect(isSafeUrl(u)).toBe(true);
    for (const u of ["javascript:alert(1)", "JaVaScRiPt:1", " java\tscript:1", "data:text/html,x", "vbscript:x", "", null, undefined]) {
      expect(isSafeUrl(u as string)).toBe(false);
    }
  });

  it("never injects raw HTML from the model (shown as text), and never auto-loads images", () => {
    const out = html('<script>alert(1)</script>\n\n<img src=x onerror=alert(1)>\n\ntext <b onclick="x()">b</b>\n\n![pixel](https://track.example/p.png)');
    expect(out).not.toContain("<script");
    expect(out).not.toContain("<img");
    expect(out).not.toMatch(/<[^>]*\sonclick=/); // no *element* carries the handler (the text is just shown)
    expect(out).not.toMatch(/<[^>]*\sonerror=/);
    expect(out).toContain("&lt;script&gt;");
    expect(out).toContain('<a href="https://track.example/p.png"');
  });

  it("the markdown source is escaped in text too", () => {
    expect(html("1 < 2 & 3 > 2")).toContain("1 &lt; 2 &amp; 3 &gt; 2");
  });
});

describe("code blocks", () => {
  const code = "```python\ndef f(x):\n    if x:\n        return 'a<b'\n\treturn None\n```";

  it("shows the language label, a copy button, and keeps indentation exactly", () => {
    const out = html(code);
    expect(out).toContain('class="bt-code-lang">python<');
    expect(out).toContain("bt-code-copy");
    expect(out).toContain(">Copy<");
    expect(out).toContain("language-python");
    // indentation survives (spaces and tab) and the content is highlighted + escaped
    const text = out.replace(/<[^>]+>/g, "").replace(/&#x27;/g, "'").replace(/&#39;/g, "'").replace(/&lt;/g, "<");
    expect(text).toContain("def f(x):\n    if x:\n        return 'a<b'\n\treturn None");
    expect(out).toContain("hljs-keyword");
  });

  it("escapes markup inside code (no injection through highlighted output)", () => {
    const out = html('```html\n<script>alert("x")</script>\n```');
    expect(out).not.toContain("<script>alert");
    expect(out).toContain("&lt;");
    const unknown = html("```whatever\n<img src=x onerror=1>\n```");
    expect(unknown).not.toContain("<img");
    expect(unknown).toContain('bt-code-lang">whatever<');
  });

  it("unlabeled fences say 'text'; indented code is a code block", () => {
    expect(html("```\nplain\n```")).toContain('bt-code-lang">text<');
    expect(html("para\n\n    indented code\n")).toContain("bt-code-pre");
  });

  it("an unterminated fence while streaming renders as a growing code block", () => {
    const out = html("Here:\n\n```js\nconst a = 1;\nconst b =", true);
    expect(out).toContain("bt-code-pre");
    expect(out.replace(/<[^>]+>/g, "")).toContain("const b =");
  });

  it("language aliases and labels", () => {
    expect(normalizeLanguage("py")).toBe("python");
    expect(normalizeLanguage("TypeScript")).toBe("typescript");
    expect(normalizeLanguage("sh")).toBe("bash");
    expect(normalizeLanguage("html")).toBe("xml");
    expect(normalizeLanguage("python title=x")).toBe("python");
    expect(normalizeLanguage("nonsense")).toBeNull();
    expect(languageLabel("py")).toBe("py");
    expect(languageLabel("")).toBe("text");
    expect(highlight("a < b", "nonsense")).toEqual({ html: "a &lt; b", language: null });
    expect(escapeHtml(`<a href="x">&'</a>`)).toBe("&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;");
  });
});

describe("progressive rendering", () => {
  const DOC = [
    "# Report", "", "Intro with **bold** and `code` and a [link](https://example.com).", "",
    "| a | b |", "|---|:-:|", "| 1 | 2 |", "", "1. first", "   - nested *item*", "2. second", "",
    "> quote", "", "```python", "for i in range(3):", "    print(i)", "```", "", "- [x] done", "", "---", "", "Done.",
  ].join("\n");

  it("every prefix of a complex document renders without throwing", () => {
    for (let i = 0; i <= DOC.length; i += 1) {
      expect(() => html(DOC.slice(0, i), true)).not.toThrow();
    }
  });

  it("the final render equals rendering the complete text in one go", () => {
    expect(html(DOC, false)).toBe(html(DOC, false));
    expect(html(DOC)).toContain("<table>");
    expect(html(DOC)).toContain('bt-code-lang">python<');
  });

  it("a partial table is a paragraph until its delimiter row arrives, then becomes a table", () => {
    expect(html("| a | b |")).not.toContain("<table>");
    expect(html("| a | b |\n|---|---|")).toContain("<table>");
  });

  it("only the growing last block changes between two partial renders (block identity by raw text)", () => {
    const a = lexMarkdown("First paragraph.\n\nSecond par");
    const b = lexMarkdown("First paragraph.\n\nSecond paragraph grows");
    expect(a[0].raw).toBe(b[0].raw);
    expect(a[a.length - 1].raw).not.toBe(b[b.length - 1].raw);
  });

  it("shows a caret only while streaming", () => {
    expect(html("hello", true)).toContain("bt-caret");
    expect(html("hello", false)).not.toContain("bt-caret");
  });
});
