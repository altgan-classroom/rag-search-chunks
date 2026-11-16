import { memo, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "../lib/citations";

/** `[1]`, `[[1]]` and legacy `[1](url)` all become an in-page cite link. */
const toCiteLinks = (s: string) => s.replace(/\[\[?(\d+)\]\]?(\([^)]*\))?/g, (_m, n) => `[${n}](#cite-${n})`);

interface Props {
  text: string;
  sources?: Source[];
  streaming?: boolean;
}

function MarkdownImpl({ text, sources = [], streaming }: Props) {
  const md = useMemo(() => toCiteLinks(text), [text]);
  return (
    <div className={`prose-orca ${streaming ? "is-streaming" : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children, ...rest }) => {
            const m = href?.match(/^#cite-(\d+)$/);
            if (m) {
              const src = sources[Number(m[1]) - 1];
              return src ? (
                <a className="cite" href={src.url} target="_blank" rel="noopener noreferrer" title={`${src.docName}${src.page ? ` · p.${src.page}` : ""}`}>
                  {m[1]}
                </a>
              ) : (
                <span className="cite opacity-60">{m[1]}</span>
              );
            }
            return <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>{children}</a>;
          },
          table: ({ children }) => (
            <div className="table-wrap">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {md}
      </ReactMarkdown>
    </div>
  );
}

export const Markdown = memo(MarkdownImpl);
