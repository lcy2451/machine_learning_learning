import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

type Result = { ok: true; fare: number; miles: number } | { ok: false; error: string };
declare global {
  interface Window { taxi: {
    status(): Promise<{ ready: boolean; error?: string }>;
    predict(input: { distance: number; unit: 'mi' | 'km' }): Promise<Result>;
  } }
}

function App() {
  const [distance, setDistance] = useState('');
  const [unit, setUnit] = useState<'mi' | 'km'>('mi');
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Extract<Result, { ok: true }> | null>(null);
  const pending = useRef(false);
  const revision = useRef(0);
  const valid = distance.trim() !== '' && Number.isFinite(Number(distance)) && Number(distance) >= 0;

  useEffect(() => {
    window.taxi.status().then(state => {
      setStatus(state.ready ? 'ready' : 'error');
      if (!state.ready) setError(state.error || '模型加载失败。');
    }).catch(() => { setStatus('error'); setError('无法连接本地模型，请重新打开应用。'); });
  }, []);

  function clearResult() {
    revision.current++;
    setResult(null);
    if (status !== 'error') setError('');
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!valid || status !== 'ready' || pending.current) return;
    pending.current = true;
    setBusy(true); setError(''); setResult(null);
    const requestRevision = revision.current;
    try {
      const response = await window.taxi.predict({ distance: Number(distance), unit });
      // 预测期间若修改了输入，旧请求的结果不能覆盖新输入。
      if (requestRevision === revision.current) {
        if (response.ok) setResult(response); else setError(response.error);
      }
    } catch {
      if (requestRevision === revision.current) setError('预测失败，请重新尝试。');
    } finally { pending.current = false; setBusy(false); }
  }

  return <main>
    <header><div className="brand"><span className="brand-icon" aria-hidden="true">↗</span><span>里程 · 车费</span></div>
      <span className={`status ${status}`}><i />{status === 'ready' ? '本地模型已就绪' : status === 'loading' ? '正在加载模型' : '模型不可用'}</span></header>
    <section className="intro"><p className="eyebrow">每段行程，从这里开始</p><h1>这一程，预计多少钱？</h1><p>输入行程里程，让线性回归模型为你估算车费。</p></section>
    <section className="card">
      <form onSubmit={submit} noValidate>
        <div className="field-heading"><label htmlFor="distance">行程里程</label><div className="units" role="group" aria-label="里程单位">
          {(['mi', 'km'] as const).map(value => <button key={value} type="button" aria-pressed={unit === value} onClick={() => { if (unit !== value) { setUnit(value); clearResult(); } }}>{value === 'mi' ? '英里' : '公里'}</button>)}
        </div></div>
        <div className="input-wrap"><input id="distance" type="number" min="0" step="any" inputMode="decimal" placeholder="例如 3" value={distance} aria-describedby="input-hint" onChange={event => { setDistance(event.target.value); clearResult(); }} /><span>{unit === 'mi' ? 'mi' : 'km'}</span></div>
        <p className="hint" id="input-hint">{distance !== '' && !valid ? '请输入大于或等于零的有效数字。' : unit === 'km' ? '公里将自动换算为英里后预测。' : '支持小数，0 英里也可以预测。'}</p>
        <button className="predict" disabled={!valid || status !== 'ready' || busy}>{busy ? '正在计算…' : '预测车费'}<span aria-hidden="true">→</span></button>
      </form>
      <div className="result" aria-live="polite" aria-atomic="true"><span className="result-label">预计车费 · USD</span>
        <div className={`amount ${result ? 'has-result' : ''}`}>{result ? <><span>$</span>{result.fare.toFixed(2)}</> : '— —'}</div>
        <p>{result ? `按 ${result.miles.toLocaleString('zh-CN', { maximumFractionDigits: 4 })} 英里计算` : '填写里程，查看本次预测'}</p>
      </div>
    </section>
    {error && <div className="error" role="alert">{error}</div>}
    <footer><span>基于芝加哥出租车数据的学习模型</span><span>离线计算 · 仅供学习参考</span></footer>
  </main>;
}
createRoot(document.getElementById('root')!).render(<App />);
