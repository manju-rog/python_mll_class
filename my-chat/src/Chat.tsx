import { useState } from "react";

export default function Chat() {
  const [msgs, setMsgs] = useState<{role:'user'|'assistant', text:string}[]>([]);
  const [input, setInput] = useState('');

  async function send() {
    const text = input.trim();
    if (!text) return;
    setMsgs(m => [...m, {role:'user', text}]);
    setInput('');
    const res = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message:text})});
    const data = await res.json();
    if (data.type === 'ROUTED') {
      const actionMsg = data?.action?.result?.message as string | undefined;
      const base = actionMsg || data.message;
      setMsgs(m => [...m, {role:'assistant', text: `${base}\n→ intent: ${data.intent}  · confidence: ${data.confidence}`}]);
    } else {
      setMsgs(m => [...m, {role:'assistant', text: data.message}]);
    }
  }

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-3">
      <div className="border rounded p-3 h-96 overflow-auto">
        {msgs.map((m,i)=> (
          <div key={i} className={m.role==='user'?'text-right':'text-left'}>
            <div className={'inline-block my-1 px-3 py-2 rounded ' + (m.role==='user'?'bg-blue-100':'bg-gray-100')}>
              {m.text}
            </div>
          </div>
        ))}
      </div>
      <div className="flex gap-2">
        <input className="flex-1 border rounded px-3 py-2" value={input} onChange={e=>setInput(e.target.value)} placeholder="Ask me anything…" />
        <button onClick={send} className="px-4 py-2 rounded bg-black text-white">Send</button>
      </div>
    </div>
  );
}
