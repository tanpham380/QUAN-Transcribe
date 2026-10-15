"""A tiny single-file UI for smoke-testing the server.

Served at ``/``. It covers file upload (with streaming SSE) and live microphone
capture through the WebSocket endpoint, which is the quickest way to verify a
GPUStack deployment without wiring up a client.
"""

from __future__ import annotations

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>QUAN-ASR playground</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
         background: #0b0f17; color: #e6edf3; }
  main { max-width: 880px; margin: 0 auto; padding: 32px 20px 80px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  p.sub { color: #8b949e; margin: 0 0 24px; font-size: 13px; }
  section { background: #111827; border: 1px solid #1f2937; border-radius: 12px;
            padding: 18px; margin-bottom: 18px; }
  h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .08em;
       color: #8b949e; margin: 0 0 14px; }
  label { display: block; font-size: 12px; color: #8b949e; margin-bottom: 4px; }
  input[type=file], input[type=text], select {
    width: 100%; padding: 9px 10px; background: #0b0f17; color: inherit;
    border: 1px solid #30363d; border-radius: 8px; font: inherit; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
  .row > div { flex: 1 1 160px; }
  button { padding: 9px 16px; border-radius: 8px; border: 1px solid #30363d;
           background: #21262d; color: #e6edf3; font: inherit; cursor: pointer; }
  button.primary { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  pre { white-space: pre-wrap; word-break: break-word; background: #0b0f17;
        border: 1px solid #1f2937; border-radius: 8px; padding: 12px;
        min-height: 90px; margin: 14px 0 0; font-size: 13px; }
  .status { font-size: 12px; color: #8b949e; margin-top: 8px; min-height: 16px; }
  .pill { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px;
          background: #21262d; color: #8b949e; margin-left: 8px; }
  .warn { color: #f0883e; }
</style>
</head>
<body>
<main>
  <h1>QUAN-ASR playground <span class="pill" id="health">…</span></h1>
  <p class="sub">OpenAI-compatible Vietnamese speech recognition on QUAN-Transcribe-1.0.</p>

  <section>
    <h2>Transcribe a file</h2>
    <div class="row">
      <div style="flex:2 1 320px">
        <label for="file">Audio file</label>
        <input id="file" type="file" accept="audio/*" />
      </div>
      <div>
        <label for="format">Response format</label>
        <select id="format">
          <option value="json">json</option>
          <option value="verbose_json">verbose_json</option>
          <option value="text">text</option>
          <option value="srt">srt</option>
          <option value="vtt">vtt</option>
        </select>
      </div>
      <div>
        <label for="step">Stream step (s)</label>
        <input id="step" type="number" value="0.48" step="0.08" min="0.08" />
      </div>
      <button class="primary" id="send">Transcribe</button>
      <button id="sendStream" disabled>Stream it</button>
    </div>
    <div class="status" id="fileStatus"></div>
    <pre id="fileOut">—</pre>
  </section>

  <section>
    <h2>Live microphone</h2>
    <div class="row">
      <button class="primary" id="micStart">Start</button>
      <button id="micStop" disabled>Stop</button>
    </div>
    <div class="status" id="micStatus"></div>
    <pre id="micOut">—</pre>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const RESAMPLE_RATE = 16000;

async function refreshHealth() {
  try {
    const r = await fetch('/health');
    const j = await r.json();
    $('health').textContent = j.loaded ? j.device : (j.status || 'loading');
    $('health').className = j.loaded ? 'pill' : 'pill warn';
  } catch { $('health').textContent = 'offline'; }
}
refreshHealth();
setInterval(refreshHealth, 5000);

const fileInput = $('file');
fileInput.addEventListener('change', () => {
  const has = fileInput.files.length > 0;
  $('sendStream').disabled = !has;
  $('fileStatus').textContent = has
    ? `${fileInput.files[0].name} (${(fileInput.files[0].size / 1048576).toFixed(2)} MB)`
    : '';
});

$('send').onclick = async () => {
  const f = fileInput.files[0];
  if (!f) return;
  const body = new FormData();
  body.append('file', f);
  body.append('response_format', $('format').value);
  $('send').disabled = true;
  $('fileStatus').textContent = 'transcribing…';
  const t0 = performance.now();
  try {
    const r = await fetch('/v1/audio/transcriptions', { method: 'POST', body });
    const text = await r.text();
    $('fileOut').textContent = text;
    $('fileStatus').textContent = r.ok
      ? `done in ${((performance.now() - t0) / 1000).toFixed(2)}s`
      : `error ${r.status}`;
  } catch (e) {
    $('fileStatus').textContent = 'request failed: ' + e;
  } finally {
    $('send').disabled = false;
  }
};

$('sendStream').onclick = async () => {
  const f = fileInput.files[0];
  if (!f) return;
  const body = new FormData();
  body.append('file', f);
  body.append('stream', 'true');
  body.append('stream_partials', 'true');
  body.append('left_chunks', '0');
  $('sendStream').disabled = true;
  $('fileOut').textContent = '';
  $('fileStatus').textContent = 'streaming…';
  try {
    const r = await fetch('/v1/audio/transcriptions', { method: 'POST', body });
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\\n\\n');
      buffer = frames.pop();
      for (const frame of frames) {
        const line = frame.split('\\n').find((l) => l.startsWith('data: '));
        if (!line) continue;
        const event = JSON.parse(line.slice(6));
        if (event.type === 'transcript.text.delta') {
          $('fileOut').textContent = '[in progress] ' + event.text;
        } else if (event.type === 'transcript.text.segment') {
          $('fileOut').textContent += (($('fileOut').textContent.startsWith('[')) ? '' : '\\n') + event.text;
        } else if (event.type === 'transcript.text.done' && event.text) {
          $('fileOut').textContent = event.text;
          $('fileStatus').textContent = `done in ${event.elapsed}s`;
        } else if (event.type === 'error') {
          $('fileStatus').textContent = 'error: ' + event.error.message;
        }
      }
    }
  } catch (e) {
    $('fileStatus').textContent = 'stream failed: ' + e;
  } finally {
    $('sendStream').disabled = false;
  }
};

let ws, ctx, processor, source, micStream;
$('micStart').onclick = async () => {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  ctx = new AudioContext();
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([`
    class Tap extends AudioWorkletProcessor {
      process(inputs) {
        const ch = inputs[0][0];
        if (ch) this.port.postMessage(ch.slice());
        return true;
      }
    }
    registerProcessor('tap', Tap);
  `], { type: 'application/javascript' })));

  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${scheme}://${location.host}/v1/audio/transcriptions/stream?format=f32`);
  const finals = [];
  ws.onmessage = (ev) => {
    const event = JSON.parse(ev.data);
    if (event.type === 'transcript.text.delta') {
      $('micOut').textContent = finals.join(' ') + '\\n[in progress] ' + event.text;
    } else if (event.type === 'transcript.text.segment') {
      finals.push(event.text);
      $('micOut').textContent = finals.join('\\n');
    } else if (event.type === 'transcript.text.done' && event.final && event.text) {
      finals.push(event.text);
      $('micOut').textContent = finals.join('\\n');
    } else if (event.type === 'error') {
      $('micStatus').textContent = 'error: ' + event.error.message;
    }
  };
  ws.onclose = () => { $('micStatus').textContent = 'disconnected'; };

  ws.onopen = () => {
    $('micStatus').textContent = 'listening… speak Vietnamese';
    $('micStart').disabled = true;
    $('micStop').disabled = false;
    source = ctx.createMediaStreamSource(stream);
    processor = new AudioWorkletNode(ctx, 'tap');
    processor.port.onmessage = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const input = e.data;
      const ratio = ctx.sampleRate / RESAMPLE_RATE;
      const out = new Float32Array(Math.floor(input.length / ratio));
      for (let i = 0; i < out.length; i++) {
        const idx = i * ratio;
        const lo = Math.floor(idx), hi = Math.min(lo + 1, input.length - 1);
        out[i] = input[lo] + (input[hi] - input[lo]) * (idx - lo);
      }
      ws.send(out.buffer);
    };
    source.connect(processor);
  };
};

$('micStop').onclick = () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send('stop');
  processor?.disconnect(); source?.disconnect();
  micStream?.getTracks().forEach((t) => t.stop());
  ctx?.close();
  $('micStart').disabled = false;
  $('micStop').disabled = true;
  setTimeout(() => ws?.close(), 400);
};
</script>
</body>
</html>
"""
