import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, BookOpenText, FileText, RefreshCw } from 'lucide-react';
import { PageHeader, PlaceholderBadge } from './PageHeader';

interface MarkdownDocSummary {
  id: string;
  title: string;
  description: string;
  path: string;
  updated_at?: string | null;
  size_bytes?: number | null;
  available: boolean;
}

interface MarkdownDoc extends MarkdownDocSummary {
  absolute_path: string;
  content: string;
}

const docsOrder = ['runbook', 'experiment-log'];
const bytes = new Intl.NumberFormat();

const formatTimestamp = (value?: string | null) => {
  if (!value) return 'not available';
  return new Date(value).toLocaleString();
};

const isSpecialMarkdownStart = (line: string) =>
  /^#{1,6}\s+/.test(line) ||
  /^```/.test(line) ||
  /^-\s+/.test(line) ||
  /^\d+\.\s+/.test(line) ||
  /^>\s?/.test(line) ||
  /^---+$/.test(line.trim());

const renderInline = (text: string, keyPrefix: string): React.ReactNode[] => {
  const parts: React.ReactNode[] = [];
  const pattern = /(\[[^\]]+\]\([^)]+\)|`[^`]+`|\*\*[^*]+\*\*)/g;
  let cursor = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text))) {
    if (match.index > cursor) {
      parts.push(text.slice(cursor, match.index));
    }
    const token = match[0];
    const key = `${keyPrefix}-${parts.length}`;
    const link = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link) {
      const [, label, href] = link;
      parts.push(
        <a key={key} href={href} target={href.startsWith('http') ? '_blank' : undefined} rel={href.startsWith('http') ? 'noreferrer' : undefined}>
          {label}
        </a>,
      );
    } else if (token.startsWith('`')) {
      parts.push(<code key={key}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith('**')) {
      parts.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else {
      parts.push(token);
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) {
    parts.push(text.slice(cursor));
  }
  return parts;
};

const renderMarkdown = (content: string) => {
  const lines = content.replace(/\r\n/g, '\n').split('\n');
  const blocks: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();
    const key = `md-${i}`;

    if (!trimmed) {
      i += 1;
      continue;
    }

    const fence = trimmed.match(/^```(.*)$/);
    if (fence) {
      const language = fence[1]?.trim();
      const codeLines: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      blocks.push(
        <pre key={key} className="markdown-code-block">
          {language ? <span className="markdown-code-language">{language}</span> : null}
          <code>{codeLines.join('\n')}</code>
        </pre>,
      );
      continue;
    }

    const heading = trimmed.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const level = Math.min(heading[1].length, 4);
      const Tag = `h${level}` as keyof JSX.IntrinsicElements;
      blocks.push(<Tag key={key}>{renderInline(heading[2], key)}</Tag>);
      i += 1;
      continue;
    }

    if (/^---+$/.test(trimmed)) {
      blocks.push(<hr key={key} />);
      i += 1;
      continue;
    }

    if (/^-\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^-\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^-\s+/, ''));
        i += 1;
      }
      blocks.push(
        <ul key={key}>
          {items.map((item, index) => (
            <li key={`${key}-li-${index}`}>{renderInline(item, `${key}-li-${index}`)}</li>
          ))}
        </ul>,
      );
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+\.\s+/, ''));
        i += 1;
      }
      blocks.push(
        <ol key={key}>
          {items.map((item, index) => (
            <li key={`${key}-li-${index}`}>{renderInline(item, `${key}-li-${index}`)}</li>
          ))}
        </ol>,
      );
      continue;
    }

    if (/^>\s?/.test(line)) {
      const quoteLines: string[] = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) {
        quoteLines.push(lines[i].replace(/^>\s?/, ''));
        i += 1;
      }
      blocks.push(<blockquote key={key}>{renderInline(quoteLines.join(' '), key)}</blockquote>);
      continue;
    }

    const paragraphLines = [trimmed];
    i += 1;
    while (i < lines.length && lines[i].trim() && !isSpecialMarkdownStart(lines[i])) {
      paragraphLines.push(lines[i].trim());
      i += 1;
    }
    blocks.push(<p key={key}>{renderInline(paragraphLines.join(' '), key)}</p>);
  }

  return blocks;
};

export const DocumentationPage: React.FC = () => {
  const [docs, setDocs] = useState<MarkdownDocSummary[]>([]);
  const [activeDocId, setActiveDocId] = useState<string>('runbook');
  const [doc, setDoc] = useState<MarkdownDoc | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestSeq = useRef(0);

  const sortedDocs = useMemo(
    () => [...docs].sort((a, b) => docsOrder.indexOf(a.id) - docsOrder.indexOf(b.id)),
    [docs],
  );

  const loadIndex = async () => {
    const res = await fetch('/api/docs/markdown');
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail ?? 'Documentation index is unavailable');
    setDocs(json.docs ?? []);
  };

  const loadDoc = async (docId: string = activeDocId) => {
    const seq = requestSeq.current + 1;
    requestSeq.current = seq;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/docs/markdown/${encodeURIComponent(docId)}`);
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Documentation file is unavailable');
      if (requestSeq.current === seq) setDoc(json);
    } catch (err) {
      if (requestSeq.current === seq) {
        setError(err instanceof Error ? err.message : 'Documentation file is unavailable');
        setDoc(null);
      }
    } finally {
      if (requestSeq.current === seq) setLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const bootstrap = async () => {
      try {
        const res = await fetch('/api/docs/markdown');
        const json = await res.json();
        if (!res.ok) throw new Error(json.detail ?? 'Documentation index is unavailable');
        if (cancelled) return;
        const nextDocs = json.docs ?? [];
        setDocs(nextDocs);
        const firstAvailable = docsOrder.find((id) => nextDocs.some((item: MarkdownDocSummary) => item.id === id && item.available)) ?? 'runbook';
        setActiveDocId(firstAvailable);
        await loadDoc(firstAvailable);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Documentation is unavailable');
      }
    };
    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSelect = (docId: string) => {
    setActiveDocId(docId);
    void loadDoc(docId);
  };

  const handleRefresh = async () => {
    try {
      await loadIndex();
      await loadDoc(activeDocId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Documentation refresh failed');
    }
  };

  return (
    <>
      <PageHeader
        kicker="Documentation"
        title={
          <>
            Runbook & <em>experiment log</em>
          </>
        }
        description={
          <>
            Read the same markdown files that are being updated as the methodology evolves. Refresh pulls the
            current file contents from disk.
          </>
        }
        actions={<PlaceholderBadge label="Live markdown" />}
      />

      <section className="docs-layout">
        <aside className="card docs-index">
          <div className="card-header">
            <div className="card-title">Markdown Files</div>
            <div className="card-meta">{sortedDocs.length || 2} tracked</div>
          </div>
          <div className="docs-tabs">
            {sortedDocs.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`docs-tab${item.id === activeDocId ? ' is-active' : ''}`}
                onClick={() => handleSelect(item.id)}
                disabled={!item.available}
              >
                <FileText size={16} />
                <span>
                  <strong>{item.title}</strong>
                  <span>{item.path}</span>
                </span>
              </button>
            ))}
          </div>
          <button className="btn btn-ghost docs-refresh" type="button" onClick={() => void handleRefresh()} disabled={loading}>
            <RefreshCw size={16} className={loading ? 'spin' : ''} /> Refresh Markdown
          </button>
          {doc ? (
            <div className="paths" style={{ marginTop: 16 }}>
              <span>File: <span className="mono">{doc.path}</span></span>
              <span>Updated: <span className="mono">{formatTimestamp(doc.updated_at)}</span></span>
              <span>Size: <span className="mono">{bytes.format(doc.size_bytes ?? 0)} bytes</span></span>
            </div>
          ) : null}
        </aside>

        <article className="card docs-reader">
          <div className="card-header">
            <div className="card-title">{doc?.title ?? 'Documentation'}</div>
            <div className="card-meta">{loading ? 'loading' : doc ? formatTimestamp(doc.updated_at) : 'not loaded'}</div>
          </div>
          {error ? (
            <div className="pool-empty-state">
              <AlertTriangle size={18} /> {error}
            </div>
          ) : null}
          {!error && !doc ? (
            <div className="pool-empty-state">
              <BookOpenText size={18} /> Loading documentation...
            </div>
          ) : null}
          {doc ? (
            <>
              <div className="pool-empty-state" style={{ marginBottom: 18 }}>
                <BookOpenText size={18} /> {doc.description}
              </div>
              <div className="markdown-doc">{renderMarkdown(doc.content)}</div>
            </>
          ) : null}
        </article>
      </section>
    </>
  );
};
