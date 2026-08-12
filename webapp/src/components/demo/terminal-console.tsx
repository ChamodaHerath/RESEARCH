"use client";

import * as React from "react";
import { TerminalSquare, Circle } from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { formatTimestamp } from "@/lib/utils";

interface LogEntry {
  text: string;
  timestamp: string;
}

export function TerminalConsole({
  logs,
  runId,
  onComplete,
}: {
  logs: string[];
  runId: number;
  onComplete?: () => void;
}) {
  const [entries, setEntries] = React.useState<LogEntry[]>([]);
  const [typingText, setTypingText] = React.useState("");
  const [isTyping, setIsTyping] = React.useState(false);
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const completedRef = React.useRef(false);

  React.useEffect(() => {
    completedRef.current = false;
    if (logs.length === 0) {
      return;
    }

    let cancelled = false;
    let lineIndex = 0;

    function typeLine() {
      if (cancelled) return;
      if (lineIndex >= logs.length) {
        setIsTyping(false);
        if (!completedRef.current) {
          completedRef.current = true;
          onComplete?.();
        }
        return;
      }
      setIsTyping(true);
      const line = logs[lineIndex];
      let charIndex = 0;
      const isSeparator = line.startsWith("=====");

      function typeChar() {
        if (cancelled) return;
        charIndex += isSeparator ? line.length : 2;
        setTypingText(line.slice(0, charIndex));
        if (charIndex < line.length) {
          window.setTimeout(typeChar, isSeparator ? 0 : 6 + Math.random() * 10);
        } else {
          setEntries((prev) => [...prev, { text: line, timestamp: formatTimestamp() }]);
          setTypingText("");
          lineIndex += 1;
          const pause = isSeparator ? 80 : 45 + Math.random() * 130;
          window.setTimeout(typeLine, pause);
        }
      }
      typeChar();
    }

    const startDelay = window.setTimeout(typeLine, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(startDelay);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, logs.length]);

  React.useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries, typingText]);

  return (
    <GlassCard hover={false} className="flex h-full flex-col overflow-hidden !bg-black/70 !p-0">
      <div className="flex items-center gap-2 border-b border-white/10 bg-white/[0.03] px-4 py-3">
        <div className="flex items-center gap-1.5">
          <Circle className="h-2.5 w-2.5 fill-red-500 text-red-500" />
          <Circle className="h-2.5 w-2.5 fill-amber-400 text-amber-400" />
          <Circle className="h-2.5 w-2.5 fill-emerald-500 text-emerald-500" />
        </div>
        <div className="ml-2 flex items-center gap-1.5 text-xs text-slate-400">
          <TerminalSquare className="h-3.5 w-3.5" />
          forensic@edf-pipeline: ~/inference
        </div>
        <div className="ml-auto flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider">
          <span className={`h-1.5 w-1.5 rounded-full ${isTyping ? "bg-emerald-400 animate-pulse" : "bg-slate-600"}`} />
          <span className={isTyping ? "text-emerald-400" : "text-slate-500"}>
            {isTyping ? "Live" : "Idle"}
          </span>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="terminal-font flex-1 overflow-y-auto px-4 py-4 text-[12.5px] leading-relaxed text-emerald-400 sm:text-[13px]"
        style={{ minHeight: 240, maxHeight: 420 }}
      >
        {entries.length === 0 && !typingText && (
          <p className="text-slate-600">
            &gt; Awaiting input. Upload a media file and click{" "}
            <span className="text-emerald-500">Run Detection</span> to begin the forensic
            analysis pipeline...
          </p>
        )}
        {entries.map((entry, i) => (
          <TerminalLine key={i} entry={entry} />
        ))}
        {typingText && (
          <div className="flex gap-3">
            <span className="shrink-0 text-slate-600">[{formatTimestamp()}]</span>
            <span className={typingText.startsWith("=====") ? "text-blue-400" : ""}>
              {typingText}
              <span className="terminal-cursor" />
            </span>
          </div>
        )}
        {entries.length > 0 && !typingText && !isTyping && (
          <div className="mt-2 flex items-center gap-2 text-emerald-500">
            <span>$</span>
            <span className="terminal-cursor" />
          </div>
        )}
      </div>
    </GlassCard>
  );
}

function TerminalLine({ entry }: { entry: LogEntry }) {
  const isSeparator = entry.text.startsWith("=====");
  const isCompleted = entry.text === "Completed.";
  const isFrame = /^Frame \d+\/\d+/.test(entry.text);
  return (
    <div className="flex gap-3">
      <span className="shrink-0 text-slate-600">[{entry.timestamp}]</span>
      <span
        className={
          isSeparator
            ? "text-blue-400"
            : isCompleted
            ? "font-semibold text-emerald-300"
            : isFrame
            ? "text-slate-400"
            : ""
        }
      >
        {entry.text}
      </span>
    </div>
  );
}
