import { memo, useMemo, type ReactNode } from "react";
import { marked, type Token, type Tokens } from "marked";
import { CodeBlock } from "./CodeBlock";

/**
 * Markdown -> React elements (never `dangerouslySetInnerHTML`, never raw HTML from the model).
 *
 * Progressive: the full text is re-lexed on every update, but each top-level block is a memoized
 * component keyed by its raw text, so only the block that is still growing re-renders while
 * tokens stream in. An unterminated ``` fence renders as a growing code block; a table appears the
 * moment its delimiter row arrives.
 */
const LEX_OPTIONS = { gfm: true, breaks: false } as const;

export function isSafeUrl(href: string | undefined | null): boolean {
  if (!href) return false;
  // eslint-disable-next-line no-control-regex
  const cleaned = href.replace(/[\u0000-\u001f\u007f\s]+/g, "").toLowerCase();
  if (cleaned.startsWith("#") || cleaned.startsWith("/") || cleaned.startsWith("./") || cleaned.startsWith("../")) return true;
  return /^(https?:|mailto:)/.test(cleaned);
}

function Inline({ tokens }: { tokens: Token[] | undefined }): ReactNode {
  if (!tokens) return null;
  return tokens.map((t, i) => <InlineToken key={i} token={t} />);
}

function InlineToken({ token }: { token: Token }): ReactNode {
  switch (token.type) {
    case "text": {
      const t = token as Tokens.Text;
      return t.tokens && t.tokens.length ? <Inline tokens={t.tokens} /> : t.text;
    }
    case "escape":
      return (token as Tokens.Escape).text;
    case "strong":
      return <strong><Inline tokens={(token as Tokens.Strong).tokens} /></strong>;
    case "em":
      return <em><Inline tokens={(token as Tokens.Em).tokens} /></em>;
    case "del":
      return <del><Inline tokens={(token as Tokens.Del).tokens} /></del>;
    case "codespan":
      return <code className="bt-inline-code">{(token as Tokens.Codespan).text}</code>;
    case "br":
      return <br />;
    case "link": {
      const t = token as Tokens.Link;
      if (!isSafeUrl(t.href)) return <Inline tokens={t.tokens} />;
      return (
        <a href={t.href} title={t.title || undefined} target="_blank" rel="noopener noreferrer nofollow">
          <Inline tokens={t.tokens} />
        </a>
      );
    }
    case "image": {
      // Never auto-load model-chosen images (tracking pixels, mixed content). Show a link instead.
      const t = token as Tokens.Image;
      return isSafeUrl(t.href) ? (
        <a href={t.href} target="_blank" rel="noopener noreferrer nofollow">{t.text || t.href}</a>
      ) : (
        <>{t.text}</>
      );
    }
    case "html":
    case "tag":
      return (token as Tokens.HTML).text; // inline HTML is shown as text, not interpreted
    default:
      return "raw" in token ? (token as { raw: string }).raw : null;
  }
}

function Caret({ show }: { show: boolean }): ReactNode {
  return show ? <span className="bt-caret" aria-hidden="true" /> : null;
}

function ListBlock({ token, caret }: { token: Tokens.List; caret: boolean }): ReactNode {
  const Tag = token.ordered ? "ol" : "ul";
  return (
    <Tag className="bt-list" start={token.ordered && token.start !== "" && token.start !== 1 ? Number(token.start) : undefined}>
      {token.items.map((item, i) => (
        <li key={i} className={item.task ? "bt-task" : undefined}>
          {item.task ? <input type="checkbox" checked={Boolean(item.checked)} readOnly disabled aria-label="task" /> : null}
          {item.tokens.map((t, j) => {
            const last = i === token.items.length - 1 && j === item.tokens.length - 1;
            return t.type === "text" ? (
              <span key={j}>
                <Inline tokens={(t as Tokens.Text).tokens ?? [t]} />
                <Caret show={caret && last} />
              </span>
            ) : (
              <BlockToken key={j} token={t} caret={caret && last} />
            );
          })}
        </li>
      ))}
    </Tag>
  );
}

function BlockToken({ token, caret }: { token: Token; caret: boolean }): ReactNode {
  switch (token.type) {
    case "space":
    case "def":
      return null;
    case "heading": {
      const t = token as Tokens.Heading;
      const Tag = (`h${Math.min(Math.max(t.depth, 1), 6)}`) as "h1";
      return <Tag className="bt-h"><Inline tokens={t.tokens} /><Caret show={caret} /></Tag>;
    }
    case "paragraph":
      return <p><Inline tokens={(token as Tokens.Paragraph).tokens} /><Caret show={caret} /></p>;
    case "text": {
      const t = token as Tokens.Text;
      return <p>{t.tokens && t.tokens.length ? <Inline tokens={t.tokens} /> : t.text}<Caret show={caret} /></p>;
    }
    case "blockquote":
      return (
        <blockquote>
          {(token as Tokens.Blockquote).tokens.map((t, i, a) => (
            <BlockToken key={i} token={t} caret={caret && i === a.length - 1} />
          ))}
        </blockquote>
      );
    case "list":
      return <ListBlock token={token as Tokens.List} caret={caret} />;
    case "code": {
      const t = token as Tokens.Code;
      return <CodeBlock code={t.text} lang={t.lang} />;
    }
    case "hr":
      return <hr />;
    case "table": {
      const t = token as Tokens.Table;
      return (
        <div className="bt-table-wrap">
          <table>
            <thead>
              <tr>
                {t.header.map((cell, i) => (
                  <th key={i} style={cell.align ? { textAlign: cell.align } : undefined}><Inline tokens={cell.tokens} /></th>
                ))}
              </tr>
            </thead>
            <tbody>
              {t.rows.map((row, r) => (
                <tr key={r}>
                  {row.map((cell, i) => (
                    <td key={i} style={cell.align ? { textAlign: cell.align } : undefined}><Inline tokens={cell.tokens} /></td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    case "html":
      return <p className="bt-raw-html">{(token as Tokens.HTML).text}</p>;
    default:
      return "raw" in token && (token as { raw: string }).raw.trim() ? <p>{(token as { raw: string }).raw}</p> : null;
  }
}

const Block = memo(
  function Block({ token, caret }: { token: Token; caret: boolean }) {
    return <BlockToken token={token} caret={caret} />;
  },
  (a, b) => a.token.raw === b.token.raw && a.caret === b.caret,
);

export function lexMarkdown(text: string): Token[] {
  return marked.lexer(text, LEX_OPTIONS) as Token[];
}

export const Markdown = memo(function Markdown({ text, streaming = false }: { text: string; streaming?: boolean }) {
  const tokens = useMemo(() => lexMarkdown(text), [text]);
  const visible = tokens.filter((t) => t.type !== "space");
  return (
    <div className="bt-md">
      {visible.map((t, i) => (
        <Block key={i} token={t} caret={streaming && i === visible.length - 1} />
      ))}
      {streaming && visible.length === 0 ? <Caret show /> : null}
    </div>
  );
});
