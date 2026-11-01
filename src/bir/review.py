"""Local-only review UI with bounded video streaming and separate corrections."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
import webbrowser

from .manifest import canonical_json, load_manifest
from .workflow import generated_records, records, save_review


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FLORA label review</title><style>
:root{font:16px system-ui;color:#172433;background:#f3f5f7}body{max-width:1120px;margin:auto;padding:24px}
h1{margin-bottom:4px}p{line-height:1.5}.muted{color:#526273}#banner{padding:12px;background:#e7eef8;border-radius:6px}
main{display:grid;grid-template-columns:270px 1fr;gap:24px;margin-top:24px}aside{max-height:75vh;overflow:auto}
button,input,textarea{font:inherit}button{cursor:pointer;padding:9px 14px;border:1px solid #aab7c6;background:white;border-radius:5px}
button.primary{background:#195db3;color:white;border-color:#195db3}button:disabled{opacity:.5;cursor:default}
.item{display:block;width:100%;text-align:left;margin-bottom:7px}.item[aria-current=true]{border:2px solid #195db3}
video{width:100%;max-height:430px;background:#111;border-radius:6px}label{display:block;margin-top:14px;font-weight:600}
textarea{box-sizing:border-box;width:100%;min-height:95px;margin:6px 0;padding:10px}input{padding:8px;width:240px}
#generated{background:white;padding:12px;white-space:pre-wrap;border:1px solid #ccd4dd;border-radius:5px}
#notice{white-space:pre-wrap;color:#8a2300;min-height:24px}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}
code{font-size:13px;overflow-wrap:anywhere}#provenance{font-size:13px;overflow-wrap:anywhere}a{color:#195db3}
@media(max-width:720px){main{grid-template-columns:1fr}aside{max-height:180px}body{padding:12px}}
</style><h1>FLORA label review</h1><p class="muted">Inspect the video before you use a label. Corrections never replace generated text.</p>
<p id="banner">Loading saved results…</p><main><aside id="items" aria-label="Action intervals"></aside><section>
<h2 id="heading">Select an interval</h2><video id="video" controls preload="metadata"></video>
<div class="row"><button id="play">Play interval</button><button id="previous">Previous</button><button id="next">Next</button></div>
<p id="notice" role="status"></p><label>Generated caption (unchanged)</label><div id="generated"></div>
<label for="reviewed">Reviewed caption</label><textarea id="reviewed" placeholder="Edit after checking the video"></textarea>
<label for="reviewer">Reviewer name or research identifier</label><input id="reviewer" maxlength="120" autocomplete="off">
<div class="row"><button class="primary" id="save">Save correction</button><button id="restore">Restore generated label</button><a href="export">Download JSONL</a></div>
<p class="muted">Export includes original text, corrections, source hashes, timestamps, and failures.</p>
<details><summary>Source and review provenance</summary><pre id="provenance"></pre></details>
</section></main><script>
let rows=[],selected=0,playingInterval=false;
const $=id=>document.getElementById(id), base=location.pathname;
async function request(path,options={}){const response=await fetch(base+path,options);const data=await response.json();if(!response.ok)throw Error(data.error||'Request failed');return data}
function show(index){if(!rows.length)return;selected=Math.max(0,Math.min(rows.length-1,index));const row=rows[selected];
 document.querySelectorAll('.item').forEach((el,i)=>el.setAttribute('aria-current',String(i===selected)));
 $('heading').textContent=row.episode_id+' · '+(row.start_s===null?'No completed interval':row.start_s.toFixed(2)+'–'+row.end_s.toFixed(2)+' s');
 $('generated').textContent=row.generated_caption??'No generated caption';$('reviewed').value=row.reviewed_caption??row.generated_caption??'';
 $('notice').textContent=row.failure?(row.failure.error+'\n'+row.failure.action):'Check source video. Model agreement does not prove label accuracy.';
 $('provenance').textContent=JSON.stringify({source_sha256:row.source_sha256,request_id:row.request_id,record_id:row.record_id,inference_status:row.inference_status,generation:row.generation,review:row.review},null,2);
 const source=base+'video/'+row.episode_id_key; if($('video').getAttribute('src')!==source)$('video').src=source;
 playingInterval=false;$('video').pause();$('play').disabled=row.start_s===null;$('save').disabled=row.start_s===null;$('restore').disabled=!row.review;
 $('previous').disabled=selected===0;$('next').disabled=selected===rows.length-1;
}
function play(){const row=rows[selected];if(!row||row.start_s===null)return;const video=$('video');
 const start=()=>{video.currentTime=row.start_s;playingInterval=true;video.play().catch(e=>{$('notice').textContent=e.message;playingInterval=false})};
 if(video.readyState>=1)start();else video.addEventListener('loadedmetadata',start,{once:true});}
$('video').addEventListener('timeupdate',()=>{const row=rows[selected];if(playingInterval&&row&&$('video').currentTime>=row.end_s){$('video').pause();playingInterval=false}});
$('video').addEventListener('error',()=>{$('notice').textContent='This browser cannot play the source codec. Keep the original file; use a compatible browser/player. No inference data was changed.'});
$('play').onclick=play;$('previous').onclick=()=>show(selected-1);$('next').onclick=()=>show(selected+1);
async function save(restore){const row=rows[selected];try{await request('review',{method:'POST',headers:{'Content-Type':'application/json','X-BIR-Review':base.split('/')[1]},body:JSON.stringify({record_id:row.record_id,reviewed_caption:restore?null:$('reviewed').value,reviewer:$('reviewer').value,expected_revision:row.review_revision})});await load(selected);$('notice').textContent='Saved. Generated text and inference status remain unchanged.'}catch(e){$('notice').textContent=e.message}}
$('save').onclick=()=>save(false);$('restore').onclick=()=>save(true);
async function load(index=0){try{const data=await request('data');rows=data.records;$('banner').textContent=(data.precomputed?'PRECOMPUTED DEMONSTRATION — no model ran here. ':'Saved inference results — ')+rows.length+' records. '+data.failed_or_pending+' failed or pending.';
 $('items').replaceChildren();rows.forEach((row,i)=>{let button=document.createElement('button');button.className='item';button.textContent=row.episode_id+' · '+(row.start_s===null?row.inference_status:row.start_s.toFixed(2)+'–'+row.end_s.toFixed(2)+' s')+(row.reviewed_caption!==null?' · reviewed':'');button.onclick=()=>show(i);$('items').append(button)});show(index)
}catch(e){$('banner').textContent=e.message}}
load();</script></html>'''


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, directory, *, manifest=None, port=8765):
        self.directory = Path(directory).expanduser().resolve(strict=True)
        self.token = secrets.token_urlsafe(24)
        raw = generated_records(self.directory)
        if not raw:
            raise ValueError('There are no saved records. Inspect bir status before opening review.')
        jobs = load_manifest(manifest or self.directory / 'inputs.jsonl', contract={'purpose': 'local_review'})
        indexed = {job.id: job for job in jobs}
        self.videos = {}
        for row in raw:
            job = indexed.get(row['episode_id'])
            if job is None or dict(job.source_sha256) != row['source_sha256'] or job.profile != row['profile'].lower():
                raise ValueError('Review video identity differs from saved inference. Supply the exact source manifest.')
            if job.left_wrist is not None or job.profile != 'human':
                raise ValueError('The review view currently supports human single-view video only.')
            key = row['source_sha256']['main']
            self.videos[key] = job.video
        self.video_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino)
                            for path in self.videos.values()}
        super().__init__(('127.0.0.1', port), Handler)

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server_port}/{self.token}/'

    def view_data(self):
        values = records(self.directory)
        return dict(records=[dict(row, episode_id_key=row['source_sha256']['main']) for row in values],
                    precomputed=any(row['generation']['precomputed'] for row in values),
                    failed_or_pending=sum(row['inference_status'] != 'complete' for row in values))


class Handler(BaseHTTPRequestHandler):
    server: ReviewServer

    def log_message(self, *_):
        pass  # Do not echo the capability token or private paths in access logs.

    def _route(self):
        prefix = '/' + self.server.token + '/'
        if not self.path.startswith(prefix):
            return None
        return self.path[len(prefix):].split('?', 1)[0]

    def _send(self, code, data, content_type='application/json; charset=utf-8', *, attachment=False):
        if not isinstance(data, bytes):
            data = canonical_json(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        if attachment:
            self.send_header('Content-Disposition', 'attachment; filename="labels.jsonl"')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        route = self._route()
        try:
            if route == '':
                self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
            elif route == 'data':
                self._send(200, self.server.view_data())
            elif route == 'export':
                data = ''.join(canonical_json(row) + '\n' for row in records(self.server.directory)).encode()
                self._send(200, data, 'application/x-ndjson; charset=utf-8', attachment=True)
            elif route and route.startswith('video/') and route[6:] in self.server.videos:
                self._video(self.server.videos[route[6:]])
            else:
                self._send(404, {'error': 'Not found'})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self._send(400, {'error': str(error)})

    def _video(self, path):
        info = path.stat()
        if (info.st_size, info.st_mtime_ns, info.st_ino) != self.server.video_stats[path]:
            raise ValueError('The source video changed after verification. Restart review with the original file.')
        size = info.st_size
        start, end = 0, size - 1
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()):
                self._send(416, {'error': 'A single byte range is required.'})
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                length = int(last)
                start = max(0, size - length)
            if not 0 <= start <= end < size:
                self._send(416, {'error': 'Range is outside the video.'})
                return
        self.send_response(206 if requested else 200)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Cache-Control', 'no-store')
        if requested:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if self.command == 'HEAD':
            return
        with path.open('rb') as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def do_POST(self):
        if self._route() != 'review' or self.headers.get('X-BIR-Review') != self.server.token:
            self._send(403, {'error': 'Use the local review page to save corrections.'})
            return
        origin = self.headers.get('Origin')
        if origin and origin != f'http://127.0.0.1:{self.server.server_port}':
            self._send(403, {'error': 'Cross-origin requests are not allowed.'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 65536 or self.headers.get_content_type() != 'application/json':
                raise ValueError('Send a bounded JSON correction.')
            data = json.loads(self.rfile.read(size))
            self._send(200, save_review(self.server.directory, **data))
        except Exception as error:
            self._send(400, {'error': str(error)})


def serve(directory, *, manifest=None, port=8765, open_browser=False):
    with ReviewServer(directory, manifest=manifest, port=port) as server:
        print(f'Local review: {server.url}\nPress Ctrl-C to stop. Generated results remain unchanged.', flush=True)
        if open_browser:
            webbrowser.open(server.url)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
    return dict(status='review_closed'), 0
