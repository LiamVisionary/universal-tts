from __future__ import annotations

import io, json, os, sys, time, wave, threading
from pathlib import Path
from typing import Any, Iterator

import numpy as np
try:  # keep /health importable from the lean Universal test venv
    import torch
    import torchaudio
except Exception:  # pragma: no cover - exercised by runtime venv instead
    torch = None  # type: ignore[assignment]
    torchaudio = None  # type: ignore[assignment]
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

SAMPLE_RATE=24000
MODEL_ID=os.environ.get('HIGGS_V3_MODEL_ID','bosonai/higgs-audio-v3-tts-4b')
MODEL_NAME=os.environ.get('HIGGS_V3_MODEL_NAME','higgs-audio-v3-tts-4b')
MODEL_DIR=Path(os.environ.get('HIGGS_V3_MODEL_DIR','/Users/liam/voice-lab/models/TTS/higgs_audio_v3/higgs-audio-v3-tts-4b'))
TTS_AUDIO_SUITE=Path(os.environ.get('TTS_AUDIO_SUITE_REPO','/Users/liam/voice-lab/TTS-Audio-Suite'))
ENGINE_DIR=TTS_AUDIO_SUITE/'engines'/'higgs_audio_v3'
PROJECT_ROOT=TTS_AUDIO_SUITE
for p in (str(PROJECT_ROOT), str(ENGINE_DIR.parent.parent)):
    if p not in sys.path: sys.path.insert(0,p)

app=FastAPI(title='Universal TTS Higgs Audio v3 Sidecar', version='0.1.0')
_bundle=None
_lock=threading.Lock()
_last_error=None

class SpeechRequest(BaseModel):
    model: str|None=None
    input: str=Field(..., min_length=1)
    voice: str|None=None
    response_format: str='wav'
    speed: float=1.0
    ref_audio: str|None=None
    reference_audio: str|None=None
    reference_audio_path: str|None=None
    ref_text: str|None=None
    reference_text: str|None=None
    reference_transcript: str|None=None
    transcript: str|None=None
    max_new_tokens: int=512
    temperature: float=0.8
    top_p: float=0.95
    top_k: int=50
    seed: int=0
    stream_commit_tokens: int=16
    stream_overlap_tokens: int=4
    max_reference_seconds: float=30.0
    trim_reference_audio: bool=True

class VoiceConversionRequest(BaseModel):
    source_audio: str
    ref_audio: str

REQ=['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json','chat_template.jinja','LICENSE']

def _missing():
    miss=[x for x in REQ if not (MODEL_DIR/x).exists()]
    if not list(MODEL_DIR.glob('*.safetensors')): miss.append('*.safetensors')
    return miss

def _download():
    if not _missing(): return
    from huggingface_hub import snapshot_download
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_DIR), local_dir_use_symlinks=False, resume_download=True)

def _load():
    global _bundle,_last_error
    if _bundle is not None: return _bundle
    with _lock:
        if _bundle is not None: return _bundle
        try:
            if torch is None:
                raise RuntimeError('torch is not installed in this environment; use the Higgs sidecar venv')
            _download()
            from engines.higgs_audio_v3.native import build_native_model, load_native_weights, load_tokenizer, read_config, HiggsAudioCodec
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dtype=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
            config=read_config(MODEL_DIR)
            model=build_native_model(config, dtype, 'sdpa')
            load_native_weights(model, MODEL_DIR, device, dtype)
            codec=HiggsAudioCodec.from_pretrained(MODEL_DIR, device=device, dtype=dtype)
            tok=load_tokenizer(MODEL_DIR)
            class B: pass
            b=B(); b.model=model; b.codec=codec; b.tokenizer=tok; b.model_dir=MODEL_DIR; b.device=device; b.torch_dtype=dtype; b.dtype_name='auto'; b.attention='sdpa'
            _bundle=b; _last_error=None; return b
        except Exception as e:
            _last_error=f'{type(e).__name__}: {e}'
            raise

def _audio_to_tensor(path:str, max_seconds:float=30.0):
    if torchaudio is None:
        raise RuntimeError('torchaudio is not installed in this environment; use the Higgs sidecar venv')
    wav,sr=torchaudio.load(path)
    wav=wav.float()
    if wav.ndim==2 and wav.shape[0]>1: wav=wav.mean(0, keepdim=True)
    elif wav.ndim==1: wav=wav.unsqueeze(0)
    if max_seconds and wav.shape[-1] > int(sr*max_seconds): wav=wav[:,:int(sr*max_seconds)]
    return {'waveform': wav.unsqueeze(0), 'sample_rate': int(sr)}

def _req_ref(req:SpeechRequest):
    p=req.ref_audio or req.reference_audio or req.reference_audio_path
    txt=req.ref_text or req.reference_text or req.reference_transcript or req.transcript or ''
    return (_audio_to_tensor(p, req.max_reference_seconds) if p else None), txt

def _float_to_pcm16(arr):
    if hasattr(arr,'detach'): arr=arr.detach().float().cpu().numpy()
    arr=np.asarray(arr,dtype=np.float32).reshape(-1)
    return np.round(np.clip(arr,-1,1)*32767).astype('<i2').tobytes()

def _wav(audio):
    pcm=_float_to_pcm16(audio); bio=io.BytesIO()
    with wave.open(bio,'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE); w.writeframes(pcm)
    return bio.getvalue()

def _generate_full(req:SpeechRequest):
    from engines.higgs_audio_v3.native import generate_higgs_audio
    b=_load(); ref,txt=_req_ref(req)
    out=generate_higgs_audio(b, text=req.input, reference_audio=ref, reference_text=txt, max_new_tokens=req.max_new_tokens, temperature=req.temperature, top_p=req.top_p, top_k=req.top_k, seed=req.seed, max_reference_seconds=req.max_reference_seconds, trim_reference_audio=req.trim_reference_audio)
    wav=out['waveform']
    if wav.dim()==3: wav=wav.squeeze(0)
    return wav.reshape(-1)

def _stream_prefix(req:SpeechRequest)->Iterator[bytes]:
    # True incremental semantic-token path: generate code rows one by one, decode stable prefixes periodically,
    # and emit only newly available PCM. This is heavier than native decoder callbacks but is not full-generate-then-chunk.
    from engines.higgs_audio_v3.native import comfy_audio_to_tensor, trim_silence_edges, apply_delay_pattern, reverse_delay_pattern, HiggsSamplerState, sampler_step, attention_runtime, manual_seed_all
    b=_load(); ref,txt=_req_ref(req)
    if req.seed: manual_seed_all(req.seed)
    ref_delayed=None
    if ref is not None:
        wav,sr=comfy_audio_to_tensor(ref)
        if req.trim_reference_audio: wav=trim_silence_edges(wav, sr, -42.0)
        if req.max_reference_seconds and wav.numel()>int(sr*req.max_reference_seconds): wav=wav[:int(sr*req.max_reference_seconds)].contiguous()
        raw_codes=b.codec.encode_reference(wav, sr); ref_delayed=apply_delay_pattern(raw_codes)
    prompt_ids=b.tokenizer.build_prompt(req.input.strip(), num_ref_tokens=0 if ref_delayed is None else int(ref_delayed.shape[0]), reference_text=txt.strip() or None)
    device=next(b.model.parameters()).device
    with torch.inference_mode(), attention_runtime(b.attention):
        prompt_embeds=b.model._prompt_embeds(prompt_ids, ref_delayed, device)
        out=b.model.backbone.model(inputs_embeds=prompt_embeds, use_cache=True)
        past=out.past_key_values; hidden=out.last_hidden_state[:,-1,:]
        state=HiggsSamplerState(num_codebooks=b.model.num_codebooks)
        rows=[]; emitted=0; commit=max(1,int(req.stream_commit_tokens)); overlap=max(0,int(req.stream_overlap_tokens))
        for i in range(int(req.max_new_tokens)):
            logits=b.model.modality_head.generate(hidden)[0].to(torch.float32)
            codes=sampler_step(logits,state,temperature=float(req.temperature),top_p=None if req.top_p<=0 or req.top_p>=1 else float(req.top_p),top_k=None if req.top_k<=0 else int(req.top_k))
            if int(codes[0].item())!=-1: rows.append(codes.detach().to('cpu',torch.long))
            if (len(rows)>=b.model.num_codebooks+commit and (len(rows)-b.model.num_codebooks)%commit==0) or state.generation_done:
                delayed=torch.stack(rows,dim=0)
                try:
                    raw=reverse_delay_pattern(delayed)
                    # hold back a few codec steps to avoid unstable prefix tails unless final
                    decode_raw=raw if state.generation_done or overlap<=0 else raw[:-overlap]
                    if decode_raw.shape[0]>0:
                        audio=b.codec.decode(decode_raw)
                        pcm=_float_to_pcm16(audio)
                        new=pcm[emitted:]
                        if new:
                            emitted += len(new); yield new
                except Exception:
                    pass
            if state.generation_done or state.last_codes is None: break
            next_embed=b.model.modality_embedding(state.last_codes.view(1,-1)).view(1,1,-1)
            out=b.model.backbone.model(inputs_embeds=next_embed.to(device=device,dtype=prompt_embeds.dtype), past_key_values=past, use_cache=True)
            past=out.past_key_values; hidden=out.last_hidden_state[:,-1,:]

@app.get('/health')
def health(load:bool=False):
    d={'ok': not _missing(), 'loaded': _bundle is not None, 'model': MODEL_NAME, 'repo': MODEL_ID, 'model_dir': str(MODEL_DIR), 'missing_files': _missing(), 'last_load_error': _last_error, 'sample_rate': SAMPLE_RATE, 'supports_true_streaming': True, 'streaming_mode': 'prefix-incremental-codec-decode'}
    if load:
        try: _load(); d.update({'ok':True,'loaded':True,'missing_files':[],'last_load_error':None})
        except Exception as e: d.update({'ok':False,'last_load_error':f'{type(e).__name__}: {e}'})
    return d

@app.get('/v1/voices')
def voices(): return {'object':'list','data':[{'id':'clone','provider':'higgs-audio-v3','requires':['ref_audio'],'recommended':['ref_text']}]}

@app.post('/v1/audio/speech')
def speech(req:SpeechRequest):
    try:
        audio=_generate_full(req)
        if req.response_format.lower()=='pcm': return Response(_float_to_pcm16(audio), media_type='audio/pcm')
        return Response(_wav(audio), media_type='audio/wav')
    except Exception as e: raise HTTPException(status_code=500, detail=f'Higgs v3 synthesis failed: {type(e).__name__}: {e}')

@app.post('/v1/audio/speech-stream')
def speech_stream(req:SpeechRequest):
    return StreamingResponse(_stream_prefix(req), media_type='audio/pcm', headers={'X-Audio-Sample-Rate':str(SAMPLE_RATE),'X-Audio-Channels':'1','X-Audio-Sample-Format':'pcm16','X-Universal-TTS-Streaming-Mode':'prefix-incremental-codec-decode'})
