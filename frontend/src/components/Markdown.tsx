import type { ReactNode } from 'react';

/**
 * Minimal Markdown for chat answers: fenced code, headings, lists, paragraphs,
 * inline code, bold and http(s) links. Builds React elements (never innerHTML),
 * so model output can't inject markup.
 */
export function Markdown({ text }: { text: string }) {
  return <div className="md">{renderBlocks(text)}</div>;
}

function renderBlocks(text: string): ReactNode[] {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const nodes: ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    const fence = line.match(/^\s*```\s*([\w#+.-]*)/);
    if (fence) {
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) body.push(lines[i++]);
      i += 1; // closing fence (or end of text)
      // Models sometimes put the "owner/repo · path:L1-L9" source label inside
      // the fence; lift it out into a caption.
      const caption = body.length > 1 && SOURCE_LABEL.test(body[0]) ? body.shift()!.replace(/`/g, '').trim() : null;
      nodes.push(
        <div key={nodes.length} className="md-code-block">
          {caption && <div className="md-code-caption">{caption}</div>}
          <pre className="md-code" data-lang={fence[1] || undefined}>
            <code>{body.join('\n')}</code>
          </pre>
        </div>,
      );
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      nodes.push(
        <div key={nodes.length} className={`md-h md-h${heading[1].length}`}>
          {renderInline(heading[2])}
        </div>,
      );
      i += 1;
      continue;
    }

    const listItem = /^\s*(?:[-*•]|\d+[.)])\s+/;
    if (listItem.test(line)) {
      const ordered = /^\s*\d+[.)]/.test(line);
      const items: string[] = [];
      while (i < lines.length && listItem.test(lines[i])) {
        let item = lines[i].replace(listItem, '');
        i += 1;
        // Indented continuation lines belong to the same item.
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !listItem.test(lines[i])) item += ` ${lines[i++].trim()}`;
        items.push(item);
      }
      const children = items.map((item, idx) => <li key={idx}>{renderInline(item)}</li>);
      nodes.push(ordered ? <ol key={nodes.length}>{children}</ol> : <ul key={nodes.length}>{children}</ul>);
      continue;
    }

    if (!line.trim()) {
      i += 1;
      continue;
    }

    const paragraph: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^\s*```/.test(lines[i]) &&
      !/^#{1,4}\s/.test(lines[i]) &&
      !listItem.test(lines[i])
    ) {
      paragraph.push(lines[i++]);
    }
    nodes.push(<p key={nodes.length}>{renderInline(paragraph.join(' '))}</p>);
  }
  return nodes;
}

const INLINE = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\(https?:\/\/[^)\s]+\))/g;
const SOURCE_LABEL = /^\s*`?[\w.-]+\/[\w.-]+\s+·\s+\S+:L\d+/;

function renderInline(text: string): ReactNode[] {
  return text.split(INLINE).map((part, idx) => {
    if (part.startsWith('`') && part.endsWith('`') && part.length > 1) {
      return <code key={idx} className="md-inline-code">{part.slice(1, -1)}</code>;
    }
    if (part.startsWith('**') && part.endsWith('**') && part.length > 3) {
      return <strong key={idx}>{part.slice(2, -2)}</strong>;
    }
    const link = part.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);
    if (link) {
      return (
        <a key={idx} href={link[2]} target="_blank" rel="noreferrer noopener">
          {link[1]}
        </a>
      );
    }
    return part;
  });
}
