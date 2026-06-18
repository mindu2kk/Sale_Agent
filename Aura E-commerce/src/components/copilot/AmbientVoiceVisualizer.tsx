import { useEffect, useRef, useState } from "react";

type Props = {
  onResult: (text: string) => void;
  onCancel: () => void;
};

const SILENCE_MS = 2000;
const SAMPLES = 48;

type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onerror: ((e: unknown) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

export function AmbientVoiceVisualizer({ onResult, onCancel }: Props) {
  const svgRef = useRef<SVGPathElement>(null);
  const dataRef = useRef<number[]>(new Array(SAMPLES).fill(0));
  const lastSoundRef = useRef<number>(Date.now());
  const finishedRef = useRef(false);
  const transcriptRef = useRef<string>("");
  const [hint, setHint] = useState("Hãy nói câu hỏi của anh/chị…");

  useEffect(() => {
    let raf = 0;
    let stream: MediaStream | null = null;
    let audioCtx: AudioContext | null = null;
    let analyser: AnalyserNode | null = null;
    let recognition: SpeechRecognitionLike | null = null;
    let cancelled = false;

    const finish = (text: string) => {
      if (finishedRef.current) return;
      finishedRef.current = true;
      try { recognition?.stop(); } catch { /* ignore */ }
      onResult(text);
    };

    const tick = () => {
      const arr = dataRef.current;
      let amp = 0;
      if (analyser) {
        const buf = new Uint8Array(analyser.fftSize);
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) {
          const v = (buf[i] - 128) / 128;
          sum += v * v;
        }
        amp = Math.sqrt(sum / buf.length);
      } else {
        amp = (Math.sin(Date.now() / 180) + 1) * 0.25;
      }
      if (amp > 0.04) lastSoundRef.current = Date.now();
      arr.shift();
      arr.push(amp);
      drawPath(arr);
      if (Date.now() - lastSoundRef.current > SILENCE_MS) {
        finish(transcriptRef.current || "");
        return;
      }
      raf = requestAnimationFrame(tick);
    };

    const drawPath = (arr: number[]) => {
      const el = svgRef.current;
      if (!el) return;
      const w = 300;
      const h = 40;
      const mid = h / 2;
      const step = w / (SAMPLES - 1);
      let d = `M 0 ${mid}`;
      for (let i = 1; i < SAMPLES; i++) {
        const x = i * step;
        const y = mid - arr[i] * mid * 1.8;
        const px = (i - 1) * step;
        const py = mid - arr[i - 1] * mid * 1.8;
        const cx = (px + x) / 2;
        d += ` Q ${cx} ${py}, ${cx} ${(py + y) / 2} T ${x} ${y}`;
      }
      el.setAttribute("d", d);
    };

    (async () => {
      try {
        if (navigator.mediaDevices?.getUserMedia) {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          const Ctx = (window.AudioContext ??
            (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext);
          audioCtx = new Ctx();
          const src = audioCtx.createMediaStreamSource(stream);
          analyser = audioCtx.createAnalyser();
          analyser.fftSize = 1024;
          src.connect(analyser);
        }
      } catch {
        // mic denied — fall back to sim
      }
      if (cancelled) return;

      const SR =
        (window as unknown as { SpeechRecognition?: new () => SpeechRecognitionLike })
          .SpeechRecognition ??
        (window as unknown as { webkitSpeechRecognition?: new () => SpeechRecognitionLike })
          .webkitSpeechRecognition;
      if (SR) {
        try {
          recognition = new SR();
          recognition.lang = "vi-VN";
          recognition.continuous = true;
          recognition.interimResults = true;
          recognition.onresult = (e) => {
            let t = "";
            for (let i = 0; i < e.results.length; i++) {
              t += e.results[i][0].transcript;
            }
            transcriptRef.current = t.trim();
            if (transcriptRef.current) setHint(transcriptRef.current);
            lastSoundRef.current = Date.now();
          };
          recognition.onerror = () => { /* ignore */ };
          recognition.start();
        } catch {
          recognition = null;
        }
      }

      lastSoundRef.current = Date.now();
      raf = requestAnimationFrame(tick);
    })();

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      try { recognition?.abort(); } catch { /* ignore */ }
      stream?.getTracks().forEach((t) => t.stop());
      audioCtx?.close().catch(() => {});
    };
  }, [onResult]);

  return (
    <div className="copilot-fade-in flex w-full items-center gap-3 rounded-2xl border border-gray-200 bg-white px-4 py-2.5">
      <svg viewBox="0 0 300 40" preserveAspectRatio="none" className="h-9 flex-1">
        <path
          ref={svgRef}
          d="M 0 20 L 300 20"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          className="text-neutral-800"
        />
      </svg>
      <button
        type="button"
        onClick={onCancel}
        className="text-[11px] font-medium text-neutral-500 transition-colors hover:text-neutral-900"
      >
        Huỷ
      </button>
      <span className="sr-only">{hint}</span>
    </div>
  );
}