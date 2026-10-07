import { memo, useEffect, useRef, useState } from "react";
import { copyTextToClipboard } from "@/lib/clipboard";
import { highlight, languageLabel } from "./highlight";

/**
 * A fenced code block: language label, working copy button, horizontal scroll, indentation
 * preserved (`white-space: pre`). While the block is still streaming it simply grows.
 */
function CodeBlockImpl({ code, lang }: { code: string; lang?: string | null }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  const { html, language } = highlight(code, lang);

  const onCopy = async () => {
    const ok = await copyTextToClipboard(code);
    setCopied(ok);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className="bt-code" data-language={language ?? "text"}>
      <div className="bt-code-head">
        <span className="bt-code-lang">{languageLabel(lang)}</span>
        <button type="button" className="bt-code-copy" onClick={onCopy} aria-label="Copy code to clipboard">
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="bt-code-pre" tabIndex={0}>
        <code className={`hljs${language ? ` language-${language}` : ""}`} dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </div>
  );
}

export const CodeBlock = memo(CodeBlockImpl, (a, b) => a.code === b.code && a.lang === b.lang);
