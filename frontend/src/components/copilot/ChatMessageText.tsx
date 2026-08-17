import { Fragment, type ReactNode } from 'react'

type TextBlock =
  | { type: 'paragraph'; text: string }
  | { type: 'list'; items: string[][] }
  | { type: 'table'; headers: string[]; rows: string[][] }

function parseTableRow(line: string): string[] {
  return line
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim())
}

function isTableSeparator(line: string): boolean {
  return /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(line.trim())
}

function parseText(text: string): TextBlock[] {
  const blocks: TextBlock[] = []
  const lines = text.split(/\r?\n/)
  let listItems: string[][] = []

  const flushList = () => {
    if (listItems.length) {
      blocks.push({ type: 'list', items: listItems })
      listItems = []
    }
  }

  for (let index = 0; index < lines.length; index += 1) {
    const rawLine = lines[index]
    const line = rawLine.trimEnd()
    const nextLine = lines[index + 1]?.trimEnd() ?? ''
    if (line.trim().startsWith('|') && isTableSeparator(nextLine)) {
      flushList()
      const headers = parseTableRow(line)
      const rows: string[][] = []
      let cursor = index + 2
      while (cursor < lines.length && lines[cursor].trim().startsWith('|')) {
        rows.push(parseTableRow(lines[cursor]))
        cursor += 1
      }
      blocks.push({ type: 'table', headers, rows })
      index = cursor - 1
      continue
    }
    if (!line.trim()) {
      flushList()
      continue
    }
    if (line.startsWith('- ')) {
      listItems.push([line.slice(2).trim()])
      continue
    }
    if (/^\s{2,}\S/.test(rawLine) && listItems.length) {
      listItems[listItems.length - 1].push(line.trim())
      continue
    }

    flushList()
    if (line.trim()) blocks.push({ type: 'paragraph', text: line.trim() })
  }

  flushList()
  return blocks
}

const inlinePattern = /(\*\*[^*]+\*\*|`[^`]+`)/g

function renderInline(text: string): ReactNode {
  return text.split(inlinePattern).map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={`${part}-${index}`}>{part.slice(1, -1)}</code>
    }
    return <Fragment key={`${part}-${index}`}>{part}</Fragment>
  })
}

function renderLines(lines: string[]): ReactNode {
  return lines.map((line, index) => (
    <Fragment key={`${line}-${index}`}>
      {index > 0 && <br />}
      {renderInline(line)}
    </Fragment>
  ))
}

export function ChatMessageText({ text }: { text: string }) {
  return (
    <div className="chat-message-text">
      {parseText(text).map((block, index) =>
        block.type === 'paragraph' ? (
          <p key={`${block.text}-${index}`}>{renderInline(block.text)}</p>
        ) : block.type === 'table' ? (
          <div className="chat-comparison-table-wrap" key={`table-${index}`}>
            <table className="chat-comparison-table">
              <thead>
                <tr>
                  {block.headers.map((header, headerIndex) => (
                    <th key={`${header}-${headerIndex}`}>{renderInline(header)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {block.rows.map((row, rowIndex) => (
                  <tr key={`row-${rowIndex}`}>
                    {row.map((cell, cellIndex) => (
                      <td key={`${cell}-${cellIndex}`}>{renderInline(cell)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <ul key={`list-${index}`}>
            {block.items.map((item, itemIndex) => (
              <li key={`${item[0]}-${itemIndex}`}>{renderLines(item)}</li>
            ))}
          </ul>
        ),
      )}
    </div>
  )
}
