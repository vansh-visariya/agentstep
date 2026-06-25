import { useEffect, useState } from 'react';
import { Play, Code, Clock, Save, Edit3, GitBranch } from 'lucide-react';
import { format } from 'date-fns';

type Span = {
  span_id: string;
  name: string;
  start_time: number;
  end_time: number;
  attributes: Record<string, any>;
  events: any[];
};

export default function App() {
  const [threads, setThreads] = useState<string[]>([]);
  const [activeThread, setActiveThread] = useState<string | null>(null);
  const [spans, setSpans] = useState<Span[]>([]);
  const [checkpoints, setCheckpoints] = useState<any[]>([]);
  const [selectedSpan, setSelectedSpan] = useState<Span | null>(null);
  const [isReplayModalOpen, setIsReplayModalOpen] = useState(false);
  const [replayOutput, setReplayOutput] = useState("");

  useEffect(() => {
    fetch('/api/threads')
      .then(r => r.json())
      .then(data => {
        setThreads(data.threads);
        if (data.threads.length > 0) setActiveThread(data.threads[0]);
      });
  }, []);

  useEffect(() => {
    if (!activeThread) return;
    fetch(`/api/traces/${activeThread}`)
      .then(r => r.json())
      .then(data => {
        setSpans(data.spans);
        setCheckpoints(data.checkpoints);
      });
  }, [activeThread]);

  const handleRunBranch = async () => {
    if (!selectedSpan || !activeThread) return;
    const checkpointId = checkpoints[0]?.checkpoint_id; // Simulating getting the closest checkpoint
    const toolCallId = selectedSpan.attributes['gen_ai.tool.name']; // Fallback placeholder logic
    
    try {
      const res = await fetch('/api/branch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          thread_id: activeThread,
          checkpoint_id: checkpointId,
          node_name: 'tools', // Name of the node in the LangGraph that processes tools
          tool_call_id: toolCallId,
          new_output: replayOutput
        })
      });
      const data = await res.json();
      if (data.status === 'success') {
        alert("Branch created! Thread: " + data.new_thread_id);
        setIsReplayModalOpen(false);
      }
    } catch (err) {
      alert("Failed to create branch");
    }
  };

  return (
    <div className="flex h-screen bg-slate-900 text-white overflow-hidden font-sans">
      {/* Sidebar */}
      <div className="w-64 bg-slate-800 border-r border-slate-700 flex flex-col">
        <div className="p-4 border-b border-slate-700 font-bold text-lg flex items-center gap-2">
          <GitBranch className="text-blue-400" />
          Agent Replay
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider p-2">Threads</div>
          {threads.map(t => (
            <button
              key={t}
              onClick={() => setActiveThread(t)}
              className={`w-full text-left px-3 py-2 rounded-md text-sm truncate ${activeThread === t ? 'bg-blue-600 text-white' : 'text-slate-300 hover:bg-slate-700'}`}
            >
              {t}
            </button>
          ))}
          {threads.length === 0 && <div className="p-2 text-sm text-slate-500">No threads found</div>}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-14 border-b border-slate-700 bg-slate-800 flex items-center px-6 justify-between">
          <h2 className="font-semibold text-slate-200">Trace Timeline</h2>
          {activeThread && <span className="text-xs text-slate-400 font-mono">{activeThread}</span>}
        </header>

        <div className="flex-1 flex overflow-hidden">
          {/* Timeline */}
          <div className="w-1/2 border-r border-slate-700 overflow-y-auto p-4 space-y-3 bg-slate-900">
            {spans.map(span => (
              <div 
                key={span.span_id} 
                onClick={() => setSelectedSpan(span)}
                className={`p-3 rounded-lg border cursor-pointer transition-colors ${selectedSpan?.span_id === span.span_id ? 'border-blue-500 bg-slate-800' : 'border-slate-700 bg-slate-800/50 hover:bg-slate-800'}`}
              >
                <div className="flex justify-between items-center mb-1">
                  <span className="font-medium text-blue-400 flex items-center gap-2">
                    {span.name === 'tool_call' ? <Code size={16} /> : <Play size={16} className="text-purple-400" />}
                    {span.name}
                  </span>
                  <span className="text-xs text-slate-500 flex items-center gap-1">
                    <Clock size={12} />
                    {((span.end_time - span.start_time) / 1000000).toFixed(2)}ms
                  </span>
                </div>
                <div className="text-sm text-slate-300 font-mono truncate">
                  {span.name === 'tool_call' ? span.attributes['gen_ai.tool.name'] : span.attributes['gen_ai.system']}
                </div>
              </div>
            ))}
            {spans.length === 0 && <div className="text-slate-500 text-center mt-10">No spans to display</div>}
          </div>

          {/* Details / Context Viewer */}
          <div className="w-1/2 overflow-y-auto bg-slate-900 p-6">
            {selectedSpan ? (
              <div className="space-y-6">
                <div className="flex justify-between items-center">
                  <h3 className="text-xl font-semibold">Span Details</h3>
                  {selectedSpan.name === 'tool_call' && (
                    <button 
                      onClick={() => {
                        setReplayOutput(selectedSpan.attributes['gen_ai.tool.output'] || '');
                        setIsReplayModalOpen(true);
                      }}
                      className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-md text-sm font-medium flex items-center gap-2 transition-colors"
                    >
                      <Edit3 size={16} />
                      Branch from here
                    </button>
                  )}
                </div>

                <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                  <h4 className="text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">Attributes</h4>
                  <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap break-words overflow-x-auto">
                    {JSON.stringify(selectedSpan.attributes, null, 2)}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="flex h-full items-center justify-center text-slate-500">
                Select a span to view details
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Replay Modal */}
      {isReplayModalOpen && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-800 rounded-xl shadow-2xl w-full max-w-2xl border border-slate-700 flex flex-col overflow-hidden">
            <div className="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-800/50">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <GitBranch className="text-blue-400" /> Branch Replay
              </h3>
              <button onClick={() => setIsReplayModalOpen(false)} className="text-slate-400 hover:text-white">&times;</button>
            </div>
            <div className="p-4 flex-1">
              <label className="block text-sm font-medium text-slate-300 mb-2">Override Tool Output</label>
              <textarea 
                value={replayOutput}
                onChange={e => setReplayOutput(e.target.value)}
                className="w-full h-64 bg-slate-900 text-slate-200 border border-slate-700 rounded-lg p-3 font-mono text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none"
              />
            </div>
            <div className="p-4 border-t border-slate-700 bg-slate-800/50 flex justify-end gap-3">
              <button onClick={() => setIsReplayModalOpen(false)} className="px-4 py-2 rounded-md hover:bg-slate-700 text-slate-300 transition-colors">Cancel</button>
              <button onClick={handleRunBranch} className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-md font-medium flex items-center gap-2 transition-colors">
                <Play size={16} /> Run Branch
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
