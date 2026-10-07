import { marked, type Token } from "marked";

const LEX_OPTIONS = { gfm: true, breaks: false } as const;

/** Only http(s), mailto and in-page/relative links are ever turned into <a href>. */
export function isSafeUrl(href: string | undefined | null): boolean {
  if (!href) return false;
  // eslint-disable-next-line no-control-regex
  const cleaned = href.replace(/[\u0000-\u001f\u007f\s]+/g, "").toLowerCase();
  if (cleaned.startsWith("#") || cleaned.startsWith("/") || cleaned.startsWith("./") || cleaned.startsWith("../")) return true;
  return /^(https?:|mailto:)/.test(cleaned);
}

export function lexMarkdown(text: string): Token[] {
  return marked.lexer(text, LEX_OPTIONS) as Token[];
}
