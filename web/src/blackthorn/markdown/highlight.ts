/**
 * Syntax highlighting (highlight.js core + a fixed language set, bundled — no CDN).
 *
 * `highlight()` always returns **escaped** HTML: either highlight.js output (which escapes the
 * source) or a manual escape for unknown languages. Nothing from the model is ever injected raw.
 */
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import c from "highlight.js/lib/languages/c";
import cpp from "highlight.js/lib/languages/cpp";
import csharp from "highlight.js/lib/languages/csharp";
import css from "highlight.js/lib/languages/css";
import diff from "highlight.js/lib/languages/diff";
import dockerfile from "highlight.js/lib/languages/dockerfile";
import go from "highlight.js/lib/languages/go";
import ini from "highlight.js/lib/languages/ini";
import java from "highlight.js/lib/languages/java";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import kotlin from "highlight.js/lib/languages/kotlin";
import lua from "highlight.js/lib/languages/lua";
import makefile from "highlight.js/lib/languages/makefile";
import markdown from "highlight.js/lib/languages/markdown";
import php from "highlight.js/lib/languages/php";
import powershell from "highlight.js/lib/languages/powershell";
import python from "highlight.js/lib/languages/python";
import r from "highlight.js/lib/languages/r";
import ruby from "highlight.js/lib/languages/ruby";
import rust from "highlight.js/lib/languages/rust";
import sql from "highlight.js/lib/languages/sql";
import swift from "highlight.js/lib/languages/swift";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

const LANGUAGES: Record<string, Parameters<typeof hljs.registerLanguage>[1]> = {
  bash, c, cpp, csharp, css, diff, dockerfile, go, ini, java, javascript, json, kotlin, lua, makefile,
  markdown, php, powershell, python, r, ruby, rust, sql, swift, typescript, xml, yaml,
};
for (const [name, def] of Object.entries(LANGUAGES)) hljs.registerLanguage(name, def);

const ALIASES: Record<string, string> = {
  js: "javascript", jsx: "javascript", mjs: "javascript", cjs: "javascript", node: "javascript",
  ts: "typescript", tsx: "typescript",
  py: "python", python3: "python", py3: "python",
  sh: "bash", shell: "bash", zsh: "bash", console: "bash", terminal: "bash", shellscript: "bash",
  html: "xml", htm: "xml", svg: "xml", xhtml: "xml", vue: "xml",
  yml: "yaml", "c++": "cpp", cc: "cpp", cxx: "cpp", hpp: "cpp", h: "c",
  cs: "csharp", "c#": "csharp", rb: "ruby", rs: "rust", golang: "go", kt: "kotlin",
  md: "markdown", ps1: "powershell", pwsh: "powershell", toml: "ini", conf: "ini", cfg: "ini",
  jsonc: "json", json5: "json", patch: "diff", docker: "dockerfile", make: "makefile", mysql: "sql",
  postgres: "sql", postgresql: "sql", sqlite: "sql", plsql: "sql",
};

/** Map a fence info string (``python title=x``) to a registered language, or null. */
export function normalizeLanguage(raw: string | undefined | null): string | null {
  const first = (raw || "").trim().split(/\s+/)[0].toLowerCase().replace(/^\{?\.?/, "").replace(/[}:].*$/, "");
  if (!first) return null;
  const name = ALIASES[first] ?? first;
  return hljs.getLanguage(name) ? name : null;
}

/** The label shown on the code block: what the author wrote, or "text". */
export function languageLabel(raw: string | undefined | null): string {
  const first = (raw || "").trim().split(/\s+/)[0];
  return first ? first.replace(/^\{?\.?/, "").replace(/[}:].*$/, "") || "text" : "text";
}

export function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

export function highlight(code: string, lang?: string | null): { html: string; language: string | null } {
  const language = normalizeLanguage(lang);
  if (!language) return { html: escapeHtml(code), language: null };
  try {
    return { html: hljs.highlight(code, { language, ignoreIllegals: true }).value, language };
  } catch {
    return { html: escapeHtml(code), language: null };
  }
}
