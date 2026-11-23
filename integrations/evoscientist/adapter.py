"""Local-only EvoScientist generation with strict gateway call budget + offline fallback."""
import contextlib
from datetime import datetime, timezone
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
LOCAL = os.environ.get('CUSTOM_OPENAI_BASE_URL', '').rstrip('/')


def consume_budget(path,cap):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with (path.parent/(path.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(path.read_text()) if path.exists() else dict(calls=0)
        if state['calls']>=cap:raise ValueError('Local LLM call cap exhausted')
        state['calls']+=1
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state));temporary.replace(path)
        return state['calls']


def _endpoint():
    endpoint=os.environ.get('CUSTOM_OPENAI_BASE_URL',LOCAL).rstrip('/')
    u=urllib.parse.urlparse(endpoint)
    if u.scheme!='https' or not u.hostname or u.port not in (None,443) or u.path!='/v1' or u.username or u.password or u.query or u.fragment:
        raise ValueError('Configure an explicit HTTPS OpenAI-compatible /v1 endpoint')
    return endpoint


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('Redirect rejected: local endpoint only')


@contextlib.contextmanager
def limited_gateway(endpoint,key,budget,cap,transcript=None,force_model=None,compact_agent=False):
    """Counted forwarding gateway. ``transcript`` (optional folder) receives each
    counted request body and raw response; the bearer key is never written.
    ``force_model`` replaces the request's model id, so the client cannot select
    another model and a client-side alias cannot change the approved one.
    Upstream HTTP errors keep their status and body (a 4xx is not turned into a
    retryable 5xx)."""
    opener=urllib.request.build_opener(NoRedirect)
    if transcript is not None:transcript=Path(transcript);transcript.mkdir(parents=True,exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.0'
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                if self.path not in ('/v1/chat/completions','/v1/responses'):raise ValueError('Route not allowed')
                number=consume_budget(budget,cap)
                length=int(self.headers.get('Content-Length','0'))
                if length>2*1024*1024:raise ValueError('Request too large')
                body=self.rfile.read(length)
                if force_model is not None or compact_agent:
                    payload=json.loads(body)
                    if force_model is not None:payload['model']=force_model
                    if compact_agent:
                        # EvoScientist's generic research system prompt and 17
                        # tool descriptions dominate the local model's prefill.
                        # Keep its actual tool loop, with only the four tools
                        # this bounded evaluation run is permitted to call.
                        payload['tools']=[tool for tool in payload.get('tools',[])
                                          if tool.get('function',{}).get('name') in
                                          ('read_file','write_file','execute','run_in_background')]
                        payload['tool_choice']='required'
                        payload['max_tokens']=4096
                        messages=payload.get('messages',[])
                        user_positions=[i for i,message in enumerate(messages)
                                        if message.get('role')=='user']
                        if user_positions:
                            # A resumed run gets a new, self-contained user
                            # instruction. Earlier tool transcripts remain in
                            # EvoScientist's checkpoint, but need not be sent
                            # back to the local model for this continuation.
                            payload['messages']=[message for message in messages
                                                 if message.get('role')=='system'][:1]+messages[user_positions[-1]:]
                        for message in payload.get('messages',[]):
                            if message.get('role')=='system':
                                message['content']=(
                                    'You are EvoScientist, an autonomous research agent. '
                                    'Use the provided tools to follow the user request exactly. '
                                    'Do not invent data or results. Treat all reported one-seed '
                                    'scores as provisional. Do not use subagents, web search, '
                                    'todo tools, or paid APIs. File tool root / is your workspace; '
                                    'shell paths must be relative ./ paths. Do not edit evaluation '
                                    'code, protocol, data, seeds, budget, or evaluate.sh. '
                                    'Finish by writing the requested files.')
                    body=json.dumps(payload).encode()
                sink=None
                if transcript is not None:
                    (transcript/f'call_{number:03d}_request.json').write_bytes(body)
                    sink=(transcript/f'call_{number:03d}_response.txt').open('wb')
                request=urllib.request.Request(endpoint+self.path.removeprefix('/v1'),data=body,headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
                response=None
                for attempt in range(1,5):
                    try:
                        response=opener.open(request,timeout=180)
                        break
                    except urllib.error.HTTPError as exc:
                        if exc.code not in (502,503,504) or attempt==4:
                            raise
                        reason=f'HTTP {exc.code}'
                        exc.close()
                    except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
                        if attempt==4:
                            raise
                        reason=type(exc).__name__
                    delay=2**(attempt-1)
                    if transcript is not None:
                        with (transcript/'retries.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                                                         call=number, retry=attempt, reason=reason,
                                                         delay_seconds=delay))+'\n')
                    time.sleep(delay)
                with response:
                    if transcript is not None:(transcript/f'call_{number:03d}_status.txt').write_text(str(response.status))
                    self.send_response(response.status);self.send_header('Content-Type',response.headers.get('Content-Type','application/json'));self.end_headers()
                    while True:
                        # Streaming chat completions may pause between small SSE
                        # events. read(size) waits to fill the buffer (or EOF),
                        # which can stall the client even after HTTP 200. read1
                        # forwards the next available bytes without that wait.
                        chunk=response.read1(4096)
                        if not chunk:break
                        if sink is not None:sink.write(chunk);sink.flush()
                        self.wfile.write(chunk);self.wfile.flush()
                if sink is not None:sink.close()
            except urllib.error.HTTPError as exc:
                detail=exc.read()
                if transcript is not None and 'number' in locals():(transcript/f'call_{number:03d}_status.txt').write_text(str(exc.code))
                if transcript is not None:
                    with (transcript/'errors.log').open('a') as stream:stream.write(f'HTTP {exc.code}: {detail[:500]!r}\n')
                self.send_response(exc.code);self.send_header('Content-Type',exc.headers.get('Content-Type','application/json'));self.end_headers()
                self.wfile.write(detail)
            except Exception as exc:
                if transcript is not None:
                    with (transcript/'errors.log').open('a') as stream:stream.write(f'{type(exc).__name__}: {str(exc)[:300]}\n')
                self.send_response(429 if 'cap' in str(exc) else 502);self.end_headers()
                self.wfile.write(json.dumps({'error':type(exc).__name__}).encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}/v1'
    finally:server.shutdown();server.server_close();thread.join(timeout=5)


def generate(feedback,folder,count,backend,cap,budget,work):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);work=Path(work)
    receipt=work/'generation.json'
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if all(Path(p).is_file() for p in previous['files']):return previous
    prompt=(ROOT/'integrations/evoscientist/prompt_template.md').read_text()
    prompt+='\n\n'+Path(feedback).read_text()+f'\n\nReturn at most {count} candidates. Write generated_candidates.json with candidates:[{{name,code}}]. Each code defines CandidateMixer.\n'
    (work/'next_prompt.md').write_text(prompt)
    reason=None;bundle=None;actual='mock'
    try:
        if backend=='mock':raise RuntimeError('Explicit offline mock mode')
        endpoint=_endpoint();key=os.environ.get('CUSTOM_OPENAI_API_KEY')
        if not key:raise RuntimeError('Local endpoint returned 401 without credentials; CUSTOM_OPENAI_API_KEY is absent')
        executable=os.environ.get('EVO_SCI_BIN') or str(work.parents[1]/'venv/bin/EvoSci')
        if not Path(executable).is_file():raise RuntimeError('EvoSci is not installed in the allowed venv')
        runtime=work/'evoscientist_runtime';runtime.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy()
        for k in list(env):
            if k.endswith('API_KEY'):env.pop(k,None)
        env.update(EVOSCIENTIST_DATA_DIR=str(runtime/'data'),EVOSCIENTIST_WORKSPACE_DIR=str(runtime/'workspace'),
            XDG_CONFIG_HOME=str(runtime/'config'),XDG_CACHE_HOME=str(runtime/'cache'),TMPDIR=str(runtime),PYTHONDONTWRITEBYTECODE='1')
        with limited_gateway(endpoint,key,budget,cap) as gateway:
            env.update(CUSTOM_OPENAI_BASE_URL=gateway,CUSTOM_OPENAI_API_KEY='local-gateway')
            for keyname,value in [('provider','custom-openai'),('model','deepseek-v4-flash')]:
                subprocess.run([executable,'config','set',keyname,value],env=env,cwd=runtime,check=True,capture_output=True,text=True,timeout=30)
            with (work/'evoscientist.log').open('w') as log:
                # No --dangerous: use the framework's workspace mapping. Tools can call our shell adapter.
                process=subprocess.Popen([executable,'--ui','cli','--mode','run','--name','generation','--auto-mode','-p',prompt],env=env,cwd=runtime,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=process.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10);raise RuntimeError('EvoScientist generation timed out')
            if code:raise RuntimeError(f'EvoScientist exited {code}')
            outputs=list(runtime.rglob('generated_candidates.json'))
            if not outputs:raise RuntimeError('EvoScientist did not produce a candidate bundle')
            bundle=json.loads(max(outputs,key=lambda p:p.stat().st_mtime).read_text());actual='evoscientist'
    except Exception as exc:
        reason=f'{type(exc).__name__}: {exc}'
        if backend=='evoscientist':raise
    files=[]
    if bundle is None:
        response=json.loads((ROOT/'integrations/evoscientist/mock_response.json').read_text())
        (work/'mock_response.json').write_text(json.dumps(response,indent=2))
        candidates=[dict(name=r['name'],code=(ROOT/'examples/research_mixers.py').read_text()+f'\n# Fixed mock response, not an AI-generated improvement.\nCandidateMixer = {r["class"]}\n') for r in response['candidates'][:count]]
    else:candidates=bundle['candidates'][:count]
    for i,item in enumerate(candidates):
        if not isinstance(item.get('code'),str) or len(item['code'])>100000:raise ValueError('Invalid candidate code size')
        path=folder/f'{i+1}_candidate.py';path.write_text(item['code']);files.append(str(path))
    result=dict(backend=actual,fallback_reason=reason,files=files,feedback=str(feedback),llm_calls=json.loads(Path(budget).read_text())['calls'] if Path(budget).exists() else 0)
    receipt.write_text(json.dumps(result,indent=2));return result
