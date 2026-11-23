"""Assemble the Phase 1 demo video: the silent scene clips from record_demo.py plus an
AI voice-over (Piper TTS, offline) -> outputs/demo_video/QC-Assist_PS5_demo.mp4

    .venv-tts/bin/python scripts/make_demo_video.py          # full version, ~4 min
    .venv-tts/bin/python scripts/make_demo_video.py short    # 2-minute submission cut

Needs a separate venv so the main one (ExecuTorch, torch) is untouched:
    uv venv --python 3.12 .venv-tts && uv pip install --python .venv-tts/bin/python piper-tts imageio-ffmpeg
    .venv-tts/bin/python -m piper.download_voices en_US-lessac-high --data-dir .cache/piper

Each scene lasts as long as its narration plus a short pause: a longer clip is sped up (at most
1.6x, then the scene is longer than its narration), a shorter one holds its last frame. The short cut instead plays a chosen window of each clip,
sped up to fit its narration.
"""

import json
import sys
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "outputs" / "demo_video"
WORK = CLIPS / "build"
VOICE = ROOT / ".cache" / "piper" / "en_US-lessac-high.onnx"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
PAUSE = 1.2      # seconds of picture after the narration ends
MAX_SPEEDUP = 1.6

# (clip, narration). Spelled for the TTS voice: "Q C Assist", "A U R O C", "N P U".
SCENES = [
    ("s1_readme", "This is Q C Assist, by team Win from S V N I T Surat: our entry for problem statement five "
     "of the Arm Bharat A I S o C Challenge. Real-time defect detection and a quality-control assistant, "
     "running fully on the device."),
    ("s2_architecture", "Electronics lines inspect every circuit board, often by eye. Cloud A I is often not "
     "an option: factory networks are offline, and product images are confidential. Q C Assist does "
     "everything on an Arm device. There are two pipelines. A vision network finds and classifies defects. "
     "Then Llama three point two, one billion parameters, running through Execu-Torch on the Arm C P U, "
     "turns the findings into advice for the operator. In this video, everything runs on the laptop C P U, "
     "fully offline."),
    ("s3_inspect", "Boards stream past the camera. Good boards pass. Here is a defect. The heat map and the "
     "red outline show where it is. Each defect is rated minor, major or critical, and its type is named: "
     "a missing component, a bent pin, a scratch, or melted solder. The model learned what a good board looks like from defect-free boards only, "
     "so it needs no defect images to start. On the right, the advice. Its first line is the plant rule, "
     "for example, remove the board and keep the line running. That rule is decided by code, never by the "
     "A I. Below it, Llama, running through Execu-Torch on this C P U, explains what to look for, and the "
     "usual cause. Every inspection is written to a local log, with type, severity, location and "
     "confidence."),
    ("s4_ask_report", "Operators can ask questions about the shift in plain language, for example, where are "
     "most defects this shift. The numbers come from the inspection log. "
     "Llama only phrases them, and every answer is checked against the facts before it is shown. A wrong "
     "number or location is replaced by the exact figures. And this is the end-of-shift report for the "
     "supervisor."),
    ("s5_hindi", "Many operators prefer Hindi. With one click, the dashboard switches to Hindi. New advice "
     "from Llama arrives in Hindi, and the shift report is in Hindi too, checked against the same log "
     "figures as the English one."),
    ("s6_teach", "Now a new product arrives on the line, and there is no model for it yet. The dashboard "
     "says so. We point it to a folder of twenty good boards of the new design, and click train. It "
     "trains in about three seconds, on the device, and it is already inspecting the new board. No defect "
     "images, no labelling, and no retraining in the cloud."),
    ("s7_results", "Everything is measured. On the VisA circuit board dataset, the mean image A U R O C is "
     "zero point nine five five. Precision is ninety five percent, recall seventy six percent, and the F1 "
     "score zero point eight four. The vision step takes about forty milliseconds per frame on the laptop. "
     "We compiled the vision network for the Hailo ten H N P U, and in eight-bit integer form the accuracy "
     "is unchanged. Llama is exported with Execu-Torch in four bits. Its first token arrives in about one "
     "second on the laptop, and the whole app, vision and Llama together, peaks at about one point six "
     "gigabytes of memory."),
    ("s8_status", "We also ran the same code on an Android phone's Arm C P U, offline. The accuracy was the "
     "same as on the laptop, vision ran at three frames per second, and Llama generated twelve tokens per "
     "second through Execu-Torch. In phase two, the Raspberry Pi five with the A I HAT plus two moves the "
     "vision network onto the Hailo N P U."),
    ("s3_inspect@end", "Q C Assist: offline, on-device inspection that learns a new board from twenty "
     "samples. Thank you."),
]

# 2-minute cut (the submission guideline): (clip, window start s, window end s, narration).
SHORT = [
    ("s1_readme", 0, 25, "This is Q C Assist, by team Win from S V N I T Surat, for problem statement five of the "
     "Arm Bharat A I S o C Challenge: real-time circuit board defect detection, fully on the device."),
    ("s2_architecture", 0, 34, "Factory networks are often offline, and board images are confidential, so "
     "everything runs on Arm. A vision network finds and classifies defects. Llama three point two, one "
     "billion, running through Execu-Torch on the C P U, advises the operator."),
    ("s3_inspect", 0, 33, "Good boards pass. A defect gets a heat map, a severity, and a type: missing part, "
     "bent pin, scratch, or melted solder. The model learned from good boards only. The first advice line is "
     "a plant rule, decided by code. Llama then explains what to check, and the likely cause."),
    ("s4_ask_report", 2, 24, "Operators can ask about the shift in plain language. The numbers come from the "
     "inspection log, and every Llama answer is checked against them."),
    ("s5_hindi", 22, 36, "Many operators prefer Hindi. One click switches the advice and the shift report to Hindi."),
    ("s6_teach", 0, 38, "A new product arrives, with no model. We point it at twenty good boards and click "
     "train. Three seconds later, on the device, it is inspecting the new board. No defect images, and no "
     "cloud."),
    ("s7_results", 0, 62, "On the VisA circuit board dataset, the mean image A U R O C is zero point nine five "
     "five, with ninety five percent precision. Vision takes about forty milliseconds a frame. In eight-bit "
     "form for the Hailo N P U, accuracy is unchanged. Four-bit Llama answers in about a second, and the whole "
     "app peaks at one point six gigabytes."),
    ("s8_status", 0, 28, "On an Android phone's Arm C P U, offline, the accuracy was the same, and Llama ran at "
     "twelve tokens per second. Q C Assist: offline, on-device inspection. Thank you."),
]


def run(*args):
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", *map(str, args)], check=True)


def duration(path: Path) -> float:
    out = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = out.split("Duration: ")[1].split(",")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def tts(text: str, out: Path) -> float:
    subprocess.run([str(ROOT / ".venv-tts" / "bin" / "python"), "-m", "piper", "-m", str(VOICE),
                    "-f", str(out), "--length-scale", "1.05", "--sentence-silence", "0.35"],
                   input=text, text=True, check=True, capture_output=True)
    with wave.open(str(out)) as w:
        return w.getnframes() / w.getframerate()


def main():
    short = sys.argv[1:] == ["short"]
    work = WORK / "short" if short else WORK
    work.mkdir(parents=True, exist_ok=True)
    parts, log = [], []
    scenes = [(c, a, b, t) for c, a, b, t in SHORT] if short else [(c, None, None, t) for c, t in SCENES]
    for i, (clip, a, b, text) in enumerate(scenes, 1):
        voice = work / f"{i:02d}.wav"
        speech = tts(text, voice)
        name, _, where = clip.partition("@")
        src = CLIPS / f"{name}.mp4"
        length = duration(src)
        if short:  # play the window, sped up to fit the narration
            target = speech + 0.9
            start = a
            length = min(b, length) - a
            speed = max(length / target, 1.0)
            vf = f"setpts=(PTS-STARTPTS)/{speed:.4f}"
            where = "window"
        else:
            target = speech + PAUSE + (0.6 if i > 1 else 0.3)
        if not where:
            target = max(target, length / MAX_SPEEDUP)  # never cut a clip short
        if where == "end":  # closing shot: the last seconds of the clip
            vf, start, speed = "setpts=PTS-STARTPTS", max(length - target, 0), 1.0
            length = min(length, target)
        elif not short:
            speed = min(max(length / target, 1.0), MAX_SPEEDUP)
            vf, start = f"setpts=(PTS-STARTPTS)/{speed:.4f}", 0
        hold = max(target - length / speed, 0)
        out = work / f"{i:02d}.mp4"
        run("-ss", f"{start:.2f}", "-i", src, "-i", voice,
            "-filter_complex",
            f"[0:v]{vf},fps=30,tpad=stop_mode=clone:stop_duration={hold + 0.1:.2f},format=yuv420p[v];"
            f"[1:a]adelay=300:all=1,apad[a]",
            "-map", "[v]", "-map", "[a]", "-t", f"{target:.2f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
            "-ar", "48000", "-ac", "2", out)
        parts.append(out)
        log.append({"scene": i, "clip": clip, "speech_s": round(speech, 1), "scene_s": round(target, 1),
                    "speedup": round(speed, 2), "hold_s": round(hold, 1)})
        print(log[-1])
    listing = work / "list.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    final = CLIPS / ("QC-Assist_PS5_demo_2min.mp4" if short else "QC-Assist_PS5_demo.mp4")
    run("-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", "-movflags", "+faststart", final)
    (work / "timeline.json").write_text(json.dumps(log, indent=1))
    print(f"-> {final} ({duration(final) / 60:.1f} min)")


if __name__ == "__main__":
    main()
