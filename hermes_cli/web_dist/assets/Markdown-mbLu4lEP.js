import{n as e}from"./rolldown-runtime-CbXtAM7H.js";import{h as t,n,t as r}from"./react-vendor-BoVnYuL4.js";var i=n(),a=e(t(),1),o=r();

/** Token-based syntax highlighting via highlight.js (window.hljs). */
function highlightCode(lang, code) {
  try {
    var hl = typeof window !== "undefined" ? window.hljs : null;
    if (!hl || !code) return null;
    if (lang && hl.getLanguage && hl.getLanguage(lang)) {
      return hl.highlight(code, { language: lang, ignoreIllegals: true }).value;
    }
    if (hl.highlightAuto) {
      return hl.highlightAuto(code).value;
    }
  } catch (_) {}
  return null;
}

function s({ content: e, highlightTerms: t, streaming: n, codeCopy: r = !1 }) {
  let i = (0, a.useMemo)(() => l(e), [e]);
  let s = n ? (0, o.jsx)(c, {}) : null;
  return (0, o.jsxs)(`div`, {
    className: `bt-md text-sm text-foreground leading-relaxed space-y-3`,
    children: [
      i.map((e, n) =>
        (0, o.jsx)(
          u,
          { block: e, highlightTerms: t, caret: s && n === i.length - 1 ? s : null, codeCopy: r },
          n
        )
      ),
      i.length === 0 && s,
    ],
  });
}

function c() {
  let e = (0, i.c)(1),
    t;
  return e[0] === Symbol.for(`react.memo_cache_sentinel`)
    ? ((t = (0, o.jsx)(`span`, {
        "aria-hidden": !0,
        className: `inline-block w-[0.5em] h-[1em] ml-0.5 align-[-0.15em] bg-foreground/50 animate-pulse`,
      })),
      (e[0] = t))
    : (t = e[0]),
    t;
}

function l(e) {
  let t = String(e || ``).split(`\n`),
    n = [],
    r = 0;
  for (; r < t.length; ) {
    let e = t[r],
      i = e.match(/^```(\w*)/);
    if (i) {
      let e = i[1] || ``,
        a = [];
      for (r++; r < t.length && !t[r].startsWith("```"); ) a.push(t[r]), r++;
      r++, n.push({ type: `code`, lang: e, content: a.join(`\n`) });
      continue;
    }
    let a = e.match(/^(#{1,4})\s+(.+)/);
    if (a) {
      n.push({ type: `heading`, level: a[1].length, content: a[2] }), r++;
      continue;
    }
    if (/^[-*_]{3,}\s*$/.test(e)) {
      n.push({ type: `hr` }), r++;
      continue;
    }
    if (/^>\s?/.test(e)) {
      let e = [];
      for (; r < t.length && /^>\s?/.test(t[r]); ) e.push(t[r].replace(/^>\s?/, ``)), r++;
      n.push({ type: `blockquote`, content: e.join(`\n`) });
      continue;
    }
    if (/^\|/.test(e) && r + 1 < t.length && /^\|?\s*[-:| ]+\|/.test(t[r + 1])) {
      let parseRow = (line) =>
        line
          .replace(/^\|/, ``)
          .replace(/\|$/, ``)
          .split(`|`)
          .map((c) => c.trim());
      let rows = [parseRow(e)];
      r += 2;
      for (; r < t.length && /^\|/.test(t[r]); ) {
        rows.push(parseRow(t[r]));
        r++;
      }
      n.push({ type: `table`, rows });
      continue;
    }
    if (/^[-*+]\s/.test(e)) {
      let e = [];
      for (; r < t.length && /^[-*+]\s/.test(t[r]); ) e.push(t[r].replace(/^[-*+]\s/, ``)), r++;
      n.push({ type: `list`, ordered: !1, items: e });
      continue;
    }
    if (/^\d+[.)]\s/.test(e)) {
      let e = [];
      for (; r < t.length && /^\d+[.)]\s/.test(t[r]); ) e.push(t[r].replace(/^\d+[.)]\s/, ``)), r++;
      n.push({ type: `list`, ordered: !0, items: e });
      continue;
    }
    if (e.trim() === ``) {
      r++;
      continue;
    }
    let o = [];
    for (
      ;
      r < t.length &&
      t[r].trim() !== `` &&
      !t[r].match(/^```/) &&
      !t[r].match(/^#{1,4}\s/) &&
      !t[r].match(/^[-*+]\s/) &&
      !t[r].match(/^\d+[.)]\s/) &&
      !t[r].match(/^[-*_]{3,}\s*$/) &&
      !t[r].match(/^>\s?/) &&
      !(t[r].match(/^\|/) && r + 1 < t.length && /^\|?\s*[-:| ]+\|/.test(t[r + 1]));

    )
      o.push(t[r]), r++;
    o.length > 0 && n.push({ type: `paragraph`, content: o.join(`\n`) });
  }
  return n;
}

function m(e) {
  return e.replace(/[.*+?^${}()|[\]\\]/g, `\\$&`);
}

function inlineFormat(text, highlightTerms) {
  let parts = [];
  let s = String(text || ``);
  let re = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[([^\]]+)\]\(([^)]+)\)|https?:\/\/[^\s<]+)/g;
  let last = 0,
    m2;
  while ((m2 = re.exec(s))) {
    if (m2.index > last) parts.push({ t: `text`, v: s.slice(last, m2.index) });
    let g = m2[0];
    if (g.startsWith("`")) parts.push({ t: `code`, v: g.slice(1, -1) });
    else if (g.startsWith("**")) parts.push({ t: `bold`, v: g.slice(2, -2) });
    else if (g.startsWith("*")) parts.push({ t: `em`, v: g.slice(1, -1) });
    else if (g.startsWith("[")) parts.push({ t: `link`, v: m2[2], href: m2[3] });
    else parts.push({ t: `link`, v: g, href: g });
    last = m2.index + g.length;
  }
  if (last < s.length) parts.push({ t: `text`, v: s.slice(last) });
  if (!parts.length) parts = [{ t: `text`, v: s }];

  return parts.map((p, idx) => {
    if (p.t === `code`)
      return (0, o.jsx)(
        `code`,
        {
          className: `bt-inline-code rounded px-1 py-0.5 text-[0.85em] font-mono bg-secondary/80 border border-border/60`,
          children: p.v,
        },
        idx
      );
    if (p.t === `bold`) return (0, o.jsx)(`strong`, { children: p.v }, idx);
    if (p.t === `em`) return (0, o.jsx)(`em`, { children: p.v }, idx);
    if (p.t === `link`)
      return (0, o.jsx)(
        `a`,
        {
          href: p.href,
          target: `_blank`,
          rel: `noopener noreferrer`,
          className: `text-primary underline underline-offset-2 hover:opacity-90`,
          children: p.v,
        },
        idx
      );
    if (highlightTerms && highlightTerms.length) {
      let terms = highlightTerms.filter(Boolean);
      if (terms.length) {
        let rx = new RegExp(`(${terms.map(m).join(`|`)})`, `gi`);
        let segs = p.v.split(rx);
        return (0, o.jsx)(
          `span`,
          {
            children: segs.map((seg, j) =>
              rx.test(seg)
                ? (0, o.jsx)(
                    `mark`,
                    { className: `bg-warning/30 text-warning px-0.5`, children: seg },
                    j
                  )
                : (0, o.jsx)(`span`, { children: seg }, j)
            ),
          },
          idx
        );
      }
    }
    return (0, o.jsx)(`span`, { children: p.v }, idx);
  });
}

function CodeBlock({ lang, content, caret, codeCopy }) {
  let html = (0, a.useMemo)(() => highlightCode(lang, content), [lang, content]);
  let label = (lang || `text`).toLowerCase();
  let onCopy = () => {
    try {
      navigator.clipboard && navigator.clipboard.writeText(content);
    } catch (_) {}
  };

  return (0, o.jsxs)(`div`, {
    className: `bt-code-block group/code relative my-2 min-w-0 overflow-hidden rounded-lg border border-border bg-[#0d1117]`,
    children: [
      (0, o.jsxs)(`div`, {
        className: `bt-code-header flex items-center justify-between gap-2 border-b border-border/80 bg-[#161b22] px-3 py-1.5`,
        children: [
          (0, o.jsx)(`span`, {
            className: `text-[11px] font-medium uppercase tracking-wide text-zinc-400`,
            children: label,
          }),
          codeCopy
            ? (0, o.jsx)(`button`, {
                type: `button`,
                onClick: onCopy,
                className: `rounded-md border border-border/60 bg-transparent px-2 py-0.5 text-[11px] text-zinc-400 hover:text-zinc-100 hover:bg-white/5`,
                "aria-label": `Copy code`,
                children: `Copy`,
              })
            : null,
        ],
      }),
      (0, o.jsx)(`div`, {
        className: `bt-code-body overflow-x-auto`,
        children: (0, o.jsxs)(`pre`, {
          className: `m-0 px-0 py-2 text-[12.5px] leading-[1.55] font-mono`,
          children: [
            html
              ? (0, o.jsx)(`code`, {
                  className: `hljs language-${label} block px-3`,
                  dangerouslySetInnerHTML: { __html: html },
                })
              : (0, o.jsx)(`code`, {
                  className: `block px-3 whitespace-pre text-zinc-200`,
                  children: content,
                }),
            caret,
          ],
        }),
      }),
    ],
  });
}

function u(e) {
  let { block: n, highlightTerms: r, caret: a, codeCopy: s } = e;
  switch (n.type) {
    case `code`:
      return (0, o.jsx)(CodeBlock, {
        lang: n.lang,
        content: n.content,
        caret: a,
        codeCopy: s,
      });
    case `heading`: {
      let tag = `h${Math.min(n.level, 4)}`;
      let cls = {
        h1: `text-base font-bold mt-3 mb-1`,
        h2: `text-sm font-bold mt-3 mb-1`,
        h3: `text-sm font-semibold mt-2 mb-1`,
        h4: `text-sm font-medium mt-2 mb-0.5`,
      }[tag];
      return (0, o.jsx)(tag, {
        className: cls,
        children: inlineFormat(n.content, r),
      });
    }
    case `hr`:
      return (0, o.jsx)(`hr`, { className: `border-border my-3` });
    case `blockquote`:
      return (0, o.jsx)(`blockquote`, {
        className: `border-l-2 border-border pl-3 text-muted-foreground italic`,
        children: inlineFormat(n.content, r),
      });
    case `table`:
      return (0, o.jsx)(`div`, {
        className: `overflow-x-auto my-2`,
        children: (0, o.jsxs)(`table`, {
          className: `w-full text-left text-xs border-collapse`,
          children: [
            (0, o.jsx)(`thead`, {
              children: (0, o.jsx)(`tr`, {
                children: (n.rows[0] || []).map((c, i) =>
                  (0, o.jsx)(
                    `th`,
                    {
                      className: `border-b border-border px-2 py-1.5 font-semibold text-muted-foreground`,
                      children: inlineFormat(c, r),
                    },
                    i
                  )
                ),
              }),
            }),
            (0, o.jsx)(`tbody`, {
              children: (n.rows.slice(1) || []).map((row, ri) =>
                (0, o.jsx)(
                  `tr`,
                  {
                    className: `border-b border-border/50`,
                    children: row.map((c, ci) =>
                      (0, o.jsx)(
                        `td`,
                        { className: `px-2 py-1.5 align-top`, children: inlineFormat(c, r) },
                        ci
                      )
                    ),
                  },
                  ri
                )
              ),
            }),
          ],
        }),
      });
    case `list`:
      return (0, o.jsx)(n.ordered ? `ol` : `ul`, {
        className: n.ordered
          ? `list-decimal pl-5 space-y-1`
          : `list-disc pl-5 space-y-1`,
        children: n.items.map((item, idx) =>
          (0, o.jsx)(`li`, { children: inlineFormat(item, r) }, idx)
        ),
      });
    case `paragraph`:
    default:
      return (0, o.jsxs)(`p`, {
        className: `whitespace-pre-wrap break-words`,
        children: [inlineFormat(n.content, r), a],
      });
  }
}

export { s as t };
