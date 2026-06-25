import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import {
  Play, Code, Clock, Edit3, GitBranch,
  Loader2, AlertCircle, ChevronDown, ChevronRight,
  X, RefreshCw, Hash, Layers,
} from 'lucide-react';

// ── types ────────────────────────────────────────────────────

type Span = {
  span_id: string;
  name: string;
  start_time: number;
  end_time: number;
  attributes: Record<string, any>;
};

type Checkpoint = {
  checkpoint_id: string;
  next: string[];
  has_messages: boolean;
};

type BranchGroup = {
  branch_id: string;
  is_original: boolean;
  spans: Span[];
  meta: { span_count: number; fork_point: string | null };
};

// ── palette ──────────────────────────────────────────────────
// Token names map straight to CSS vars from index.css.

type C = {
  fg: string;       // text colour on dark
  bg: string;       // chip background
  border: string;   // chip border
  soft: string;     // row hover / faint
  rail: string;     // SVG rail stroke
};

const PAL: C[] = [
  // 0 = original (warm amber)
  { fg: 'var(--color-orig)', bg: 'color-mix(in srgb, var(--color-orig) 14%, transparent)', border: 'var(--color-orig)', soft: 'color-mix(in srgb, var(--color-orig) 6%, transparent)', rail: 'var(--color-orig)' },
  // 1+ = branches
  { fg: 'var(--color-b0)', bg: 'color-mix(in srgb, var(--color-b0) 14%, transparent)', border: 'var(--color-b0)', soft: 'color-mix(in srgb, var(--color-b0) 6%, transparent)', rail: 'var(--color-b0)' },
  { fg: 'var(--color-b1)', bg: 'color-mix(in srgb, var(--color-b1) 14%, transparent)', border: 'var(--color-b1)', soft: 'color-mix(in srgb, var(--color-b1) 6%, transparent)', rail: 'var(--color-b1)' },
  { fg: 'var(--color-b2)', bg: 'color-mix(in srgb, var(--color-b2) 14%, transparent)', border: 'var(--color-b2)', soft: 'color-mix(in srgb, var(--color-b2) 6%, transparent)', rail: 'var(--color-b2)' },
  { fg: 'var(--color-b3)', bg: 'color-mix(in srgb, var(--color-b3) 14%, transparent)', border: 'var(--color-b3)', soft: 'color-mix(in srgb, var(--color-b3) 6%, transparent)', rail: 'var(--color-b3)' },
  { fg: 'var(--color-b4)', bg: 'color-mix(in srgb, var(--color-b4) 14%, transparent)', border: 'var(--color-b4)', soft: 'color-mix(in srgb, var(--color-b4) 6%, transparent)', rail: 'var(--color-b4)' },
];

function c(i: number): C { return PAL[i % PAL.length]; }

// ── helpers ──────────────────────────────────────────────────

function getNodeIcon(name: string) {
  if (name === 'tool_call') return <Code size={12} />;
  return <Play size={12} />;
}

function findCheckpointForSpan(
  span: Span,
  checkpoints: Checkpoint[],
): Checkpoint | null {
  if (span.name === 'tool_call') {
    return checkpoints.find((cp) => cp.next.includes('tools')) ?? checkpoints[0] ?? null;
  }
  if (span.name === 'llm_call') {
    return checkpoints.find((cp) => cp.next.includes('agent'))
      ?? checkpoints[checkpoints.length - 1]
      ?? null;
  }
  return checkpoints[0] ?? null;
}

function formatDuration(start: number, end: number): string {
  const ms = (end - start) / 1_000_000;
  if (ms < 1) return `${(ms * 1000).toFixed(0)}µs`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function spanLabel(span: Span): string {
  if (span.name === 'tool_call') return span.attributes?.['gen_ai.tool.name'] ?? '(tool)';
  return span.attributes?.['gen_ai.completion']
    ? String(span.attributes['gen_ai.completion']).slice(0, 88)
    : span.attributes?.['gen_ai.system'] ?? '(llm)';
}

// "orig", "b0", "b1" — the small caps chip that replaces icons in the row
function branchChipLabel(index: number, isOriginal: boolean) {
  if (isOriginal) return 'orig';
  return `b${index - 1}`;
}

// Small caps eyebrow. Used as the visual signature of "section title".
function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="text-[10px] font-semibold tracking-[0.12em] uppercase mono"
      style={{ color: 'var(--color-mute)' }}
    >
      {children}
    </div>
  );
}

// Squared colored chip (the recurring element from the reference).
function Chip({
  children, color, selected, onClick,
}: {
  children: React.ReactNode;
  color: C;
  selected?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="mono text-[10.5px] uppercase tracking-wider px-1.5 h-[18px] leading-[18px] font-semibold cursor-pointer"
      style={{
        color: color.fg,
        background: selected ? color.bg : 'transparent',
        border: `1px solid ${selected ? color.border : 'var(--color-ink-3)'}`,
      }}
    >
      {children}
    </button>
  );
}

// ── main component ───────────────────────────────────────────

export default function App() {
  const [threads, setThreads] = useState<string[]>([]);
  const [activeThread, setActiveThread] = useState<string | null>(null);
  const [branches, setBranches] = useState<BranchGroup[]>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedSpan, setSelectedSpan] = useState<Span | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [threadBranchCounts, setThreadBranchCounts] = useState<Record<string, number>>({});

  // inspector focus: which branch is the selected span on?
  const [focusBranchId, setFocusBranchId] = useState<string | null>(null);

  // branch modal
  const [showBranchModal, setShowBranchModal] = useState(false);
  const [replayOutput, setReplayOutput] = useState('');
  const [branching, setBranching] = useState(false);
  const [branchError, setBranchError] = useState<string | null>(null);
  const [branchResult, setBranchResult] = useState<string | null>(null);

  const timelineRef = useRef<HTMLDivElement>(null);

  // ── fetch threads ─────────────────────────────────────────
  const fetchThreads = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch('/api/threads');
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const data = await r.json();
      setThreads(data.threads);
      if (data.threads.length && !activeThread) setActiveThread(data.threads[0]);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load threads');
    } finally {
      setLoading(false);
    }
  }, [activeThread]);

  useEffect(() => { fetchThreads(); }, []);

  // ── fetch traces ──────────────────────────────────────────
  const fetchTraces = useCallback(async (threadId: string) => {
    setLoading(true);
    setError(null);
    try {
      const [traceR, cpR] = await Promise.all([
        fetch(`/api/traces/${threadId}`),
        fetch(`/api/traces/${threadId}/checkpoints`),
      ]);
      if (!traceR.ok) throw new Error(`${traceR.status} ${traceR.statusText}`);
      const trace = await traceR.json();
      const check = cpR.ok ? (await cpR.json()).checkpoints : [];
      setBranches(trace.branches ?? []);
      setCheckpoints(check);
      setSelectedSpan(null);
      setFocusBranchId(null);
      const branchCount = (trace.branches ?? []).filter((b: BranchGroup) => !b.is_original).length;
      setThreadBranchCounts(prev => ({ ...prev, [threadId]: branchCount }));
    } catch (e: any) {
      setError(e.message ?? 'Failed to load trace');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeThread) fetchTraces(activeThread);
  }, [activeThread, fetchTraces]);

  // ── branch replay ─────────────────────────────────────────
  const handleRunBranch = async () => {
    if (!selectedSpan || !activeThread) return;
    const matchedCheckpoint = findCheckpointForSpan(selectedSpan, checkpoints);
    if (!matchedCheckpoint) {
      setBranchError('No matching checkpoint found for this span');
      return;
    }
    const nodeName = selectedSpan.name === 'tool_call' ? 'tools' : 'agent';
    const toolName = selectedSpan.attributes?.['gen_ai.tool.name'] ?? '';

    setBranching(true);
    setBranchError(null);
    setBranchResult(null);
    try {
      const r = await fetch('/api/branch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          thread_id: activeThread,
          checkpoint_id: matchedCheckpoint.checkpoint_id,
          node_name: nodeName,
          span_type: selectedSpan.name,
          tool_call_id: toolName,
          new_output: replayOutput,
        }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail ?? `${r.status}`);
      const newBid = data.branch_id;
      setBranchResult(`branch replay ok · ${newBid.slice(0, 12)}…`);
      await Promise.all([fetchThreads(), fetchTraces(activeThread)]);
      setCollapsed(prev => { const n = new Set(prev); n.delete(newBid); return n; });
      // scroll the new branch into view after render
      requestAnimationFrame(() => {
        const el = document.getElementById(`branch-${newBid}`);
        el?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    } catch (e: any) {
      setBranchError(e.message ?? 'Branch replay failed');
    } finally {
      setBranching(false);
    }
  };

  const openBranchModal = () => {
    if (!selectedSpan) return;
    const initialOutput =
      selectedSpan.attributes?.['gen_ai.tool.output']
      ?? selectedSpan.attributes?.['gen_ai.completion']
      ?? '';
    setReplayOutput(initialOutput);
    setBranchError(null);
    setBranchResult(null);
    setShowBranchModal(true);
  };

  const toggleCollapse = (branchId: string) => {
    setCollapsed(prev => {
      const next = new Set(prev);
      next.has(branchId) ? next.delete(branchId) : next.add(branchId);
      return next;
    });
  };

  // Derived: index of the currently focused branch (for inspector breadcrumb chip color)
  const focusedIndex = useMemo(() => {
    if (!focusBranchId) return 0;
    const idx = branches.findIndex(b => b.branch_id === focusBranchId);
    return idx >= 0 ? idx : 0;
  }, [focusBranchId, branches]);

  // Span counter for inspector breadcrumb
  const selectedStepIndex = useMemo(() => {
    if (!selectedSpan) return null;
    let i = 0;
    for (const b of branches) {
      const found = b.spans.findIndex(s => s.span_id === selectedSpan.span_id);
      if (found >= 0) return { step: found + 1, total: b.spans.length, branchIdx: branches.indexOf(b) };
      i += b.spans.length;
    }
    return null;
  }, [selectedSpan, branches]);

  // ── render ────────────────────────────────────────────────
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--color-ink)' }}>
      {/* ─────── Sidebar ─────────────────────────────── */}
      <aside
        className="w-[220px] flex flex-col shrink-0 border-r"
        style={{ background: 'var(--color-ink-2)', borderColor: 'var(--color-ink-3)' }}
      >
        <div className="px-4 h-12 flex items-center gap-2 border-b" style={{ borderColor: 'var(--color-ink-3)' }}>
          <GitBranch size={15} style={{ color: 'var(--color-orig)' }} />
          <span className="mono text-[12px] font-semibold tracking-wide" style={{ color: 'var(--color-text)' }}>
            agentstep
          </span>
        </div>
        <div className="px-4 pt-4 pb-2"><Eyebrow>Threads</Eyebrow></div>
        <div className="flex-1 overflow-y-auto px-2 pb-2">
          {threads.map((t) => {
            const bc = threadBranchCounts[t];
            const active = activeThread === t;
            return (
              <button
                key={t}
                onClick={() => setActiveThread(t)}
                className="w-full text-left px-3 py-1.5 mono text-[12px] flex items-center justify-between transition-colors"
                style={{
                  color: active ? 'var(--color-text)' : 'var(--color-text-dim)',
                  background: active ? 'var(--color-ink-3)' : 'transparent',
                  borderLeft: `2px solid ${active ? 'var(--color-orig)' : 'transparent'}`,
                }}
              >
                <span className="truncate">{t}</span>
                {bc !== undefined && bc > 0 && (
                  <span
                    className="mono text-[10px] px-1 ml-2 shrink-0"
                    style={{
                      background: 'var(--color-ink-3)',
                      color: 'var(--color-mute)',
                      border: '1px solid var(--color-ink-4)',
                    }}
                  >
                    +{bc}
                  </span>
                )}
              </button>
            );
          })}
          {!loading && threads.length === 0 && (
            <div className="px-3 text-[12px] italic" style={{ color: 'var(--color-ink-5)' }}>
              no threads
            </div>
          )}
        </div>
        <div
          className="px-4 py-2 border-t mono text-[10.5px] flex items-center gap-1.5"
          style={{ borderColor: 'var(--color-ink-3)', color: 'var(--color-ink-5)' }}
        >
          <span
            className="w-1.5 h-1.5"
            style={{ background: 'var(--color-b1)', display: 'inline-block' }}
          />
          connected · v0.1
        </div>
      </aside>

      {/* ─────── Main ─────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* breadcrumb / branch toggle bar */}
        <header
          className="h-12 flex items-center justify-between px-5 border-b shrink-0"
          style={{ borderColor: 'var(--color-ink-3)', background: 'var(--color-ink)' }}
        >
          <div className="flex items-center gap-2 mono text-[12.5px]">
            <Hash size={12} style={{ color: 'var(--color-mute)' }} />
            <span style={{ color: 'var(--color-mute)' }}>thread</span>
            <span style={{ color: 'var(--color-text)' }}>{activeThread ?? '—'}</span>
            {selectedStepIndex && (
              <>
                <span style={{ color: 'var(--color-ink-5)' }}>/</span>
                <span style={{ color: 'var(--color-mute)' }}>step</span>
                <span style={{ color: 'var(--color-text)' }}>
                  {selectedStepIndex.step}
                  <span style={{ color: 'var(--color-ink-5)' }}>/{selectedStepIndex.total}</span>
                </span>
              </>
            )}
          </div>

          {/* branch toggle group */}
          {branches.length > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="mono text-[10px] uppercase tracking-wider mr-2" style={{ color: 'var(--color-mute)' }}>
                view
              </span>
              {branches.map((b, i) => {
                const color = c(i);
                const selected = focusBranchId === b.branch_id || (focusBranchId === null && i === 0);
                return (
                  <Chip
                    key={b.branch_id}
                    color={color}
                    selected={selected}
                    onClick={() => setFocusBranchId(b.branch_id)}
                  >
                    {branchChipLabel(i, b.is_original)}
                  </Chip>
                );
              })}
              <button
                onClick={() => fetchTraces(activeThread!)}
                className="ml-2 p-1 transition-colors"
                style={{ color: 'var(--color-mute)' }}
                title="refresh"
              >
                <RefreshCw size={12} />
              </button>
            </div>
          )}
        </header>

        {/* ─────── Body: timeline + inspector ─────────── */}
        <div className="flex-1 flex overflow-hidden">
          {/* Timeline */}
          <div
            ref={timelineRef}
            className="flex-1 overflow-y-auto"
            style={{ background: 'var(--color-ink)' }}
          >
            <div className="px-8 py-6 max-w-[860px]">
              {loading && (
                <div className="flex items-center gap-2 py-12 justify-center" style={{ color: 'var(--color-mute)' }}>
                  <Loader2 size={14} className="animate-spin" />
                  <span className="mono text-[12px]">loading trace…</span>
                </div>
              )}
              {error && (
                <div className="flex items-center gap-2 py-12 justify-center" style={{ color: 'var(--color-orig)' }}>
                  <AlertCircle size={14} />
                  <span className="mono text-[12px]">{error}</span>
                </div>
              )}
              {!loading && !error && branches.length === 0 && (
                <div className="mono text-[12px] text-center mt-16 italic" style={{ color: 'var(--color-ink-5)' }}>
                  no spans for this thread
                </div>
              )}

              <div className="space-y-7">
                {branches.map((branch, branchIdx) => {
                  const color = c(branchIdx);
                  const isCollapsed = collapsed.has(branch.branch_id);
                  const isOriginal = branch.is_original;

                  return (
                    <BranchBlock
                      key={branch.branch_id}
                      branch={branch}
                      index={branchIdx}
                      color={color}
                      isOriginal={isOriginal}
                      isCollapsed={isCollapsed}
                      selectedSpanId={selectedSpan?.span_id ?? null}
                      onToggleCollapse={() => toggleCollapse(branch.branch_id)}
                      onSelectSpan={(s) => {
                        setSelectedSpan(s);
                        setFocusBranchId(branch.branch_id);
                      }}
                    />
                  );
                })}
              </div>
            </div>
          </div>

          {/* ─────── Inspector ─────────────────────────── */}
          <aside
            className="w-[400px] shrink-0 overflow-y-auto border-l"
            style={{ borderColor: 'var(--color-ink-3)', background: 'var(--color-ink-2)' }}
          >
            {!selectedSpan ? (
              <div className="h-full flex flex-col items-center justify-center gap-2 mono text-[12px]" style={{ color: 'var(--color-ink-5)' }}>
                <Layers size={20} />
                select a span to inspect
              </div>
            ) : (
              <Inspector
                span={selectedSpan}
                branchIndex={selectedStepIndex?.branchIdx ?? 0}
                stepNumber={selectedStepIndex?.step ?? 0}
                stepTotal={selectedStepIndex?.total ?? 0}
                checkpoint={findCheckpointForSpan(selectedSpan, checkpoints)}
                onBranch={openBranchModal}
              />
            )}
          </aside>
        </div>
      </div>

      {/* ─────── Branch modal ─────────────────────────── */}
      {showBranchModal && (
        <div
          className="fixed inset-0 flex items-center justify-center z-50 p-4"
          style={{ background: 'rgba(0,0,0,0.7)' }}
        >
          <div
            className="w-full max-w-[560px] flex flex-col"
            style={{ background: 'var(--color-ink-2)', border: '1px solid var(--color-ink-4)' }}
          >
            <div
              className="px-4 h-10 flex items-center justify-between border-b"
              style={{ borderColor: 'var(--color-ink-3)' }}
            >
              <div className="flex items-center gap-2">
                <GitBranch size={13} style={{ color: 'var(--color-orig)' }} />
                <span className="mono text-[12px] font-semibold">branch_replay</span>
              </div>
              <button onClick={() => setShowBranchModal(false)} style={{ color: 'var(--color-mute)' }}>
                <X size={14} />
              </button>
            </div>

            <div className="p-4 space-y-3">
              <Eyebrow>Override Output</Eyebrow>
              <textarea
                value={replayOutput}
                onChange={(e) => setReplayOutput(e.target.value)}
                className="mono w-full h-56 text-[12px] p-3 outline-none resize-none"
                style={{
                  background: 'var(--color-ink)',
                  color: 'var(--color-text)',
                  border: '1px solid var(--color-ink-3)',
                }}
                placeholder="// new tool output or llm response…"
              />

              {branchError && (
                <div
                  className="mono text-[11px] p-2"
                  style={{ color: 'var(--color-orig)', background: 'color-mix(in srgb, var(--color-orig) 8%, transparent)', border: '1px solid color-mix(in srgb, var(--color-orig) 30%, transparent)' }}
                >
                  ! {branchError}
                </div>
              )}
              {branchResult && (
                <div
                  className="mono text-[11px] p-2"
                  style={{ color: 'var(--color-b1)', background: 'color-mix(in srgb, var(--color-b1) 8%, transparent)', border: '1px solid color-mix(in srgb, var(--color-b1) 30%, transparent)' }}
                >
                  ✓ {branchResult}
                </div>
              )}
            </div>

            <div
              className="px-4 h-12 flex items-center justify-end gap-2 border-t"
              style={{ borderColor: 'var(--color-ink-3)' }}
            >
              <button
                onClick={() => setShowBranchModal(false)}
                className="mono text-[11.5px] px-3 h-7 uppercase tracking-wider"
                style={{ color: 'var(--color-mute)', border: '1px solid var(--color-ink-3)' }}
              >
                close
              </button>
              <button
                onClick={handleRunBranch}
                disabled={branching}
                className="mono text-[11.5px] px-3 h-7 uppercase tracking-wider flex items-center gap-1.5"
                style={{
                  background: 'var(--color-orig)',
                  color: 'var(--color-ink)',
                  opacity: branching ? 0.6 : 1,
                }}
              >
                {branching ? <Loader2 size={11} className="animate-spin" /> : <Play size={11} />}
                run_branch
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── BranchBlock: timeline section for one branch ─────────────

function BranchBlock({
  branch, index, color, isOriginal, isCollapsed,
  selectedSpanId, onToggleCollapse, onSelectSpan,
}: {
  branch: BranchGroup;
  index: number;
  color: C;
  isOriginal: boolean;
  isCollapsed: boolean;
  selectedSpanId: string | null;
  onToggleCollapse: () => void;
  onSelectSpan: (s: Span) => void;
}) {
  const branchId = `branch-${branch.branch_id}`;

  // Position of the fork span — its row offset from the top of this block.
  // If the fork point isn't in this branch's spans, anchor at the end.
  const forkSpanId = branch.meta.fork_point;
  const forkIdx = forkSpanId
    ? branch.spans.findIndex(s => s.span_id === forkSpanId)
    : -1;

  return (
    <div id={branchId}>
      {/* ── Branch header eyebrow ── */}
      <div
        className="flex items-center gap-2 mb-2 cursor-pointer select-none"
        onClick={onToggleCollapse}
      >
        <span style={{ color: 'var(--color-mute)' }}>
          {isCollapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
        </span>
        <span className="mono text-[10px] font-semibold tracking-[0.14em] uppercase"
              style={{ color: color.fg }}>
          {isOriginal ? 'Original Trace' : `Branch ${index - 1}`}
        </span>
        <span className="mono text-[10px]" style={{ color: 'var(--color-ink-5)' }}>
          · {branch.spans.length} span{branch.spans.length !== 1 ? 's' : ''}
        </span>
        {!isOriginal && forkSpanId && (
          <span className="mono text-[10px]" style={{ color: 'var(--color-ink-5)' }}>
            · forked from <span style={{ color: 'var(--color-mute)' }}>{forkSpanId.slice(0, 8)}</span>
          </span>
        )}
        {/* chip on the far right of the header */}
        <div className="ml-auto">
          <Chip color={color} selected>{branchChipLabel(index, isOriginal)}</Chip>
        </div>
      </div>

      {/* ── Spans ── */}
      {!isCollapsed && (
        <div className="relative pl-6">
          {/* The vertical rail running through this branch's spans */}
          <div
            className="absolute left-[7px] top-0 bottom-0 w-px"
            style={{ background: 'var(--color-ink-3)' }}
          />

          <div className="space-y-px">
            {branch.spans.map((span, i) => {
              const selected = span.span_id === selectedSpanId;
              const isForkAnchor = i === forkIdx;

              return (
                <div key={span.span_id} className="relative">
                  {/* fork marker — small caps label sitting on the rail */}
                  {isForkAnchor && (
                    <div className="absolute -left-[24px] top-0 z-10">
                      <span
                        className="mono text-[9.5px] uppercase tracking-wider px-1.5 h-[16px] leading-[16px] font-semibold inline-block"
                        style={{
                          color: color.fg,
                          background: 'var(--color-ink)',
                          border: `1px solid ${color.border}`,
                        }}
                      >
                        {isOriginal ? 'root' : `${branchChipLabel(index, false)} forks here`}
                      </span>
                    </div>
                  )}

                  <button
                    id={span.span_id}
                    onClick={() => onSelectSpan(span)}
                    className="relative w-full text-left flex items-center gap-3 py-2 px-2 transition-colors"
                    style={{
                      background: selected ? color.bg : 'transparent',
                      border: `1px solid ${selected ? color.border : 'transparent'}`,
                      marginLeft: '-7px', // pulls row over the rail
                      paddingLeft: '24px',
                    }}
                  >
                    {/* rail dot */}
                    <span
                      className="absolute"
                      style={{
                        left: '3px',
                        top: '50%',
                        transform: 'translateY(-50%)',
                        width: '9px',
                        height: '9px',
                        background: color.fg,
                      }}
                    />

                    {/* icon */}
                    <span
                      className="mono text-[10px] w-3 flex items-center justify-center"
                      style={{ color: 'var(--color-mute)' }}
                    >
                      {getNodeIcon(span.name)}
                    </span>

                    {/* name + label */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span
                          className="mono text-[11px] font-semibold uppercase tracking-wide"
                          style={{ color: span.name === 'tool_call' ? 'var(--color-b4)' : 'var(--color-b0)' }}
                        >
                          {span.name === 'tool_call' ? 'Tool Call' : 'LLM Call'}
                        </span>
                        <span
                          className="mono text-[10px] px-1"
                          style={{
                            color: 'var(--color-mute)',
                            border: '1px solid var(--color-ink-3)',
                          }}
                        >
                          {span.name === 'tool_call' ? 'tool' : 'agent'}
                        </span>
                      </div>
                      <div
                        className="mono text-[11px] truncate mt-0.5"
                        style={{ color: 'var(--color-text-dim)' }}
                      >
                        {spanLabel(span)}
                      </div>
                    </div>

                    {/* duration */}
                    <span
                      className="mono text-[11px] flex items-center gap-1 shrink-0"
                      style={{ color: 'var(--color-mute)' }}
                    >
                      <Clock size={10} />
                      {formatDuration(span.start_time, span.end_time)}
                    </span>
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Inspector panel ──────────────────────────────────────────

function Inspector({
  span, branchIndex, stepNumber, stepTotal, checkpoint, onBranch,
}: {
  span: Span;
  branchIndex: number;
  stepNumber: number;
  stepTotal: number;
  checkpoint: Checkpoint | null;
  onBranch: () => void;
}) {
  const color = c(branchIndex);
  const isTool = span.name === 'tool_call';
  const toolName = span.attributes?.['gen_ai.tool.name'];
  const completion = span.attributes?.['gen_ai.completion'];
  const output = span.attributes?.['gen_ai.tool.output'];

  return (
    <div className="p-5 space-y-5">
      {/* breadcrumb row */}
      <div className="flex items-center gap-2 mono text-[11px]">
        <span style={{ color: 'var(--color-mute)' }}>step</span>
        <span style={{ color: 'var(--color-text)' }}>#{stepNumber}</span>
        <span style={{ color: 'var(--color-ink-5)' }}>/ {stepTotal}</span>
        <span style={{ color: 'var(--color-ink-5)' }}>·</span>
        <Chip color={color} selected>
          {branchChipLabel(branchIndex, branchIndex === 0)}
        </Chip>
      </div>

      <div className="flex items-baseline gap-2">
        <span
          className="mono text-[14px] font-semibold uppercase tracking-wide"
          style={{ color: isTool ? 'var(--color-b4)' : 'var(--color-b0)' }}
        >
          {isTool ? 'tool_call' : 'llm_call'}
        </span>
        {isTool && toolName && (
          <span className="mono text-[12px]" style={{ color: 'var(--color-text)' }}>
            {toolName}
          </span>
        )}
      </div>

      <button
        onClick={onBranch}
        className="w-full mono text-[11.5px] uppercase tracking-wider h-8 flex items-center justify-center gap-2"
        style={{
          background: 'var(--color-orig)',
          color: 'var(--color-ink)',
        }}
      >
        <Edit3 size={12} />
        branch from here ↗
      </button>

      {/* Checkpoint */}
      <section className="space-y-2">
        <Eyebrow>Checkpoint</Eyebrow>
        <div
          className="p-3 mono text-[11px] space-y-1"
          style={{ background: 'var(--color-ink)', border: '1px solid var(--color-ink-3)' }}
        >
          {checkpoint ? (
            <>
              <div className="flex gap-2">
                <span style={{ color: 'var(--color-mute)' }}>thread</span>
                <span className="truncate" style={{ color: 'var(--color-text)' }}>
                  {checkpoint.checkpoint_id}
                </span>
              </div>
              <div className="flex gap-2">
                <span style={{ color: 'var(--color-mute)' }}>step_id</span>
                <span style={{ color: 'var(--color-text)' }}>{span.span_id.slice(0, 12)}…</span>
              </div>
              <div className="flex gap-2">
                <span style={{ color: 'var(--color-mute)' }}>next</span>
                <span style={{ color: 'var(--color-text)' }}>[{checkpoint.next.join(', ')}]</span>
              </div>
            </>
          ) : (
            <div style={{ color: 'var(--color-ink-5)' }}>— no checkpoint —</div>
          )}
        </div>
      </section>

      {/* Attributes */}
      <section className="space-y-2">
        <Eyebrow>Attributes</Eyebrow>
        <div
          className="p-3 mono text-[11px] leading-relaxed"
          style={{ background: 'var(--color-ink)', border: '1px solid var(--color-ink-3)' }}
        >
          {Object.entries(span.attributes ?? {}).length === 0 ? (
            <span style={{ color: 'var(--color-ink-5)' }}>// empty</span>
          ) : (
            <table className="w-full">
              <tbody>
                {Object.entries(span.attributes).map(([k, v]) => (
                  <tr key={k}>
                    <td className="align-top pr-3 whitespace-nowrap" style={{ color: 'var(--color-mute)' }}>
                      {k}
                    </td>
                    <td className="align-top break-all" style={{ color: 'var(--color-text)' }}>
                      {typeof v === 'string' && v.length > 120
                        ? v.slice(0, 120) + '…'
                        : String(v)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>

      {/* Output preview */}
      {(output || completion) && (
        <section className="space-y-2">
          <Eyebrow>{isTool ? 'Tool Output' : 'Completion'}</Eyebrow>
          <pre
            className="mono text-[11px] leading-relaxed p-3 whitespace-pre-wrap break-words max-h-64 overflow-y-auto"
            style={{
              background: 'var(--color-ink)',
              color: 'var(--color-text)',
              border: '1px solid var(--color-ink-3)',
            }}
          >
            {(isTool ? output : completion) as string}
          </pre>
        </section>
      )}

      {/* Fork info (only meaningful if this span is a fork point) */}
      <section className="space-y-2">
        <Eyebrow>Fork Info</Eyebrow>
        <div
          className="p-3 mono text-[11px]"
          style={{ background: 'var(--color-ink)', border: '1px solid var(--color-ink-3)', color: 'var(--color-text-dim)' }}
        >
          {branchIndex === 0 ? (
            <span style={{ color: 'var(--color-ink-5)' }}>// root span</span>
          ) : (
            <span>branch diverged from this step · see header for fork label</span>
          )}
        </div>
      </section>
    </div>
  );
}
