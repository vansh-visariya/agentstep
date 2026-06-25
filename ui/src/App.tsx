import { useEffect, useState, useCallback } from 'react';
import {
  Play, Code, Clock, Edit3, GitBranch,
  ArrowDown, Loader2, AlertCircle,
} from 'lucide-react';

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

// ── helpers ───────────────────────────────────────────────────

function getNodeColor(name: string): string {
  if (name === 'tool_call') return 'bg-amber-400';
  if (name === 'llm_call') return 'bg-purple-400';
  return 'bg-slate-400';
}

function getNodeIcon(name: string) {
  if (name === 'tool_call') return <Code size={14} className="text-amber-300" />;
  return <Play size={14} className="text-purple-300" />;
}

function findCheckpointForSpan(
  span: Span,
  checkpoints: Checkpoint[],
): Checkpoint | null {
  if (span.name === 'tool_call') {
    return (
      checkpoints.find((cp) => cp.next.includes('tools')) ?? checkpoints[0] ?? null
    );
  }
  // for llm_call that follows a tool result, next should be "agent"
  if (span.name === 'llm_call') {
    return (
      checkpoints.find((cp) => cp.next.includes('agent'))
      ?? checkpoints[checkpoints.length - 1]
      ?? null
    );
  }
  return checkpoints[0] ?? null;
}

// ── component ─────────────────────────────────────────────────

export default function App() {
  const [threads, setThreads] = useState<string[]>([]);
  const [activeThread, setActiveThread] = useState<string | null>(null);
  const [spans, setSpans] = useState<Span[]>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedSpan, setSelectedSpan] = useState<Span | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // branch modal
  const [showBranchModal, setShowBranchModal] = useState(false);
  const [replayOutput, setReplayOutput] = useState('');
  const [branching, setBranching] = useState(false);
  const [branchError, setBranchError] = useState<string | null>(null);
  const [branchResult, setBranchResult] = useState<string | null>(null);

  // ── fetch threads ───────────────────────────────────────────
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

  // ── fetch traces ────────────────────────────────────────────
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
      setSpans(trace.spans);
      setCheckpoints(check);
      setSelectedSpan(null);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load trace');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeThread) fetchTraces(activeThread);
  }, [activeThread, fetchTraces]);

  // ── branch replay ───────────────────────────────────────────
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
      setBranchResult(`Branch replay succeeded! Thread: ${data.thread_id}`);
      // Refresh the thread list and trace data so the new branch's
      // spans appear in the UI immediately.
      fetchThreads();
      if (activeThread) fetchTraces(activeThread);
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

  const formatDuration = (start: number, end: number): string => {
    const ms = (end - start) / 1_000_000;
    return ms < 1 ? `${(ms * 1000).toFixed(0)}µs` : `${ms.toFixed(2)}ms`;
  };

  // ── render ──────────────────────────────────────────────────
  return (
    <div className="flex h-screen bg-slate-900 text-white overflow-hidden font-sans">
      {/* ── Sidebar ───────────────────────────────────────── */}
      <div className="w-64 bg-slate-800/80 border-r border-slate-700 flex flex-col shrink-0">
        <div className="p-4 border-b border-slate-700 font-bold text-lg flex items-center gap-2">
          <GitBranch className="text-blue-400" size={20} />
          Agent Replay
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-2 pb-1">
            Threads
          </div>
          {threads.map((t) => (
            <button
              key={t}
              onClick={() => setActiveThread(t)}
              className={`w-full text-left px-3 py-2 rounded-md text-sm truncate ${
                activeThread === t
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-300 hover:bg-slate-700'
              }`}
            >
              {t}
            </button>
          ))}
          {!loading && threads.length === 0 && (
            <div className="px-2 text-sm text-slate-500 italic">No threads found</div>
          )}
        </div>
      </div>

      {/* ── Main ──────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* header */}
        <header className="h-14 border-b border-slate-700 bg-slate-800/50 flex items-center px-6 justify-between shrink-0">
          <h2 className="font-semibold text-slate-200">Trace Timeline</h2>
          {activeThread && (
            <span className="text-xs text-slate-400 font-mono">{activeThread}</span>
          )}
        </header>

        {/* body */}
        <div className="flex-1 flex overflow-hidden">
          {/* ── Timeline ──────────────────────────────────── */}
          <div className="w-1/2 border-r border-slate-700 overflow-y-auto p-4 bg-slate-900/50">
            {loading && (
              <div className="flex items-center justify-center h-32 gap-2 text-slate-400">
                <Loader2 size={18} className="animate-spin" />
                Loading…
              </div>
            )}
            {error && (
              <div className="flex items-center justify-center h-32 gap-2 text-red-400">
                <AlertCircle size={18} />
                {error}
              </div>
            )}

            {!loading && !error && spans.length === 0 && (
              <div className="text-slate-500 text-center mt-16 italic">No spans for this thread</div>
            )}

            <div className="space-y-0">
              {spans.map((span, i) => (
                <div key={span.span_id}>
                  {/* arrow connector */}
                  {i > 0 && (
                    <div className="flex justify-center py-0.5">
                      <ArrowDown size={14} className="text-slate-600" />
                    </div>
                  )}

                  <div
                    onClick={() => setSelectedSpan(span)}
                    className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                      selectedSpan?.span_id === span.span_id
                        ? 'border-blue-500 bg-slate-800'
                        : 'border-slate-700/50 bg-slate-800/30 hover:bg-slate-800/60'
                    }`}
                  >
                    {/* dot */}
                    <div className="flex flex-col items-center mt-0.5">
                      <div className={`w-2.5 h-2.5 rounded-full ${getNodeColor(span.name)}`} />
                    </div>

                    {/* content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex justify-between items-center mb-0.5">
                        <span className="font-medium text-sm flex items-center gap-1.5">
                          {getNodeIcon(span.name)}
                          <span className={
                            span.name === 'tool_call' ? 'text-amber-300' : 'text-purple-300'
                          }>
                            {span.name === 'tool_call' ? 'Tool Call' : 'LLM Call'}
                          </span>
                        </span>
                        <span className="text-xs text-slate-500 flex items-center gap-1 font-mono">
                          <Clock size={11} />
                          {formatDuration(span.start_time, span.end_time)}
                        </span>
                      </div>
                      <div className="text-xs text-slate-400 truncate">
                        {span.name === 'tool_call'
                          ? span.attributes?.['gen_ai.tool.name'] ?? '(tool)'
                          : span.attributes?.['gen_ai.completion']
                            ? (span.attributes['gen_ai.completion'] as string).slice(0, 80)
                            : span.attributes?.['gen_ai.system'] ?? '(llm)'}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ── Detail ────────────────────────────────────── */}
          <div className="w-1/2 overflow-y-auto bg-slate-900 p-6">
            {selectedSpan ? (
              <div className="space-y-5">
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                      <span className={
                        selectedSpan.name === 'tool_call'
                          ? 'text-amber-300' : 'text-purple-300'
                      }>
                        {selectedSpan.name === 'tool_call' ? 'Tool Call' : 'LLM Call'}
                      </span>
                    </h3>
                    <p className="text-xs text-slate-500 font-mono mt-0.5">
                      {selectedSpan.span_id}
                    </p>
                  </div>
                  <button
                    onClick={openBranchModal}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-md text-sm font-medium flex items-center gap-2 transition-colors"
                  >
                    <Edit3 size={15} />
                    Branch from here
                  </button>
                </div>

                {/* attributes */}
                <div className="bg-slate-800/60 rounded-lg p-4 border border-slate-700/50">
                  <h4 className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">
                    Attributes
                  </h4>
                  <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap break-words max-h-96 overflow-y-auto">
                    {JSON.stringify(selectedSpan.attributes, null, 2) || '(empty)'}
                  </pre>
                </div>

                {/* matching checkpoint info */}
                {(() => {
                  const cp = findCheckpointForSpan(selectedSpan, checkpoints);
                  if (!cp) return null;
                  return (
                    <div className="bg-blue-950/40 rounded-lg p-3 border border-blue-900/50">
                      <span className="text-xs text-blue-300 font-mono">
                        Checkpoint: {cp.checkpoint_id.slice(0, 16)}… | next: [{cp.next.join(', ')}]
                      </span>
                    </div>
                  );
                })()}
              </div>
            ) : (
              <div className="flex h-full items-center justify-center text-slate-500">
                Select a span to inspect
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Branch modal ──────────────────────────────────── */}
      {showBranchModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-800 rounded-xl shadow-2xl w-full max-w-2xl border border-slate-700 flex flex-col overflow-hidden">
            {/* header */}
            <div className="p-4 border-b border-slate-700 bg-slate-800/50 flex justify-between items-center">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <GitBranch className="text-blue-400" size={19} />
                Branch Replay
              </h3>
              <button
                onClick={() => setShowBranchModal(false)}
                className="text-slate-400 hover:text-white text-xl leading-none"
              >
                &times;
              </button>
            </div>

            {/* body */}
            <div className="p-4 space-y-4">
              <label className="block text-sm font-medium text-slate-300">
                Override output for <span className="font-mono text-blue-300">
                  {selectedSpan?.name === 'tool_call'
                    ? selectedSpan?.attributes?.['gen_ai.tool.name'] ?? 'tool'
                    : 'LLM response'}
                </span>
              </label>
              <textarea
                value={replayOutput}
                onChange={(e) => setReplayOutput(e.target.value)}
                className="w-full h-56 bg-slate-900 text-slate-200 border border-slate-700 rounded-lg p-3 font-mono text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none"
              />

              {branchError && (
                <div className="flex items-start gap-2 text-red-400 text-sm bg-red-950/40 rounded-lg p-3 border border-red-900/50">
                  <AlertCircle size={16} className="mt-0.5 shrink-0" />
                  {branchError}
                </div>
              )}

              {branchResult && (
                <div className="text-green-400 text-sm bg-green-950/40 rounded-lg p-3 border border-green-900/50">
                  {branchResult}
                </div>
              )}
            </div>

            {/* footer */}
            <div className="p-4 border-t border-slate-700 bg-slate-800/50 flex justify-end gap-3">
              <button
                onClick={() => setShowBranchModal(false)}
                className="px-4 py-2 rounded-md hover:bg-slate-700 text-slate-300 transition-colors text-sm"
              >
                Close
              </button>
              <button
                onClick={handleRunBranch}
                disabled={branching}
                className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed px-4 py-2 rounded-md font-medium flex items-center gap-2 transition-colors text-sm"
              >
                {branching ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Play size={16} />
                )}
                {branching ? 'Running…' : 'Run Branch'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
