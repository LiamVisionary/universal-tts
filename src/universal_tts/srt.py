from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SRTCue:
    index: int
    start_ms: int
    end_ms: int
    text: str


def format_srt_timestamp(ms: int) -> str:
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def split_text_for_srt(text: str, max_chars: int = 72) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    out: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars + 1)
            if cut < max_chars // 2:
                cut = max_chars
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            out.append(sentence)
    return out


def build_srt_cues(
    text: str,
    *,
    total_duration_ms: int | None = None,
    words_per_minute: float = 155.0,
    min_cue_ms: int = 900,
    max_cue_ms: int = 6000,
    gap_ms: int = 80,
    max_chars: int = 72,
) -> list[SRTCue]:
    parts = split_text_for_srt(text, max_chars=max_chars)
    if not parts:
        return []

    weights = [max(1, len(re.findall(r"\w+", part)) or math.ceil(len(part) / 5)) for part in parts]
    if total_duration_ms is None:
        millis_per_word = 60_000.0 / max(1.0, float(words_per_minute))
        durations = [min(max_cue_ms, max(min_cue_ms, int(round(w * millis_per_word)))) for w in weights]
    else:
        available = max(len(parts) * min_cue_ms, int(total_duration_ms) - gap_ms * (len(parts) - 1))
        total_weight = sum(weights)
        durations = [min(max_cue_ms, max(min_cue_ms, int(round(available * w / total_weight)))) for w in weights]

    cues: list[SRTCue] = []
    cursor = 0
    for idx, (part, duration) in enumerate(zip(parts, durations), start=1):
        start = cursor
        end = start + duration
        cues.append(SRTCue(index=idx, start_ms=start, end_ms=end, text=part))
        cursor = end + gap_ms
    return cues


def render_srt(cues: Iterable[SRTCue]) -> str:
    blocks = []
    for cue in cues:
        blocks.append(
            f"{cue.index}\n{format_srt_timestamp(cue.start_ms)} --> {format_srt_timestamp(cue.end_ms)}\n{cue.text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def text_to_srt(text: str, **kwargs) -> str:
    return render_srt(build_srt_cues(text, **kwargs))
