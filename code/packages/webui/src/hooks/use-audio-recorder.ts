import { useCallback, useEffect, useRef, useState } from "react";

// Target encoding: WAV PCM 16-bit, mono, 16 kHz → 32 kB/s.
// Capped so a full recording stays under the 5 MB upload limit.
const TARGET_SAMPLE_RATE = 16000;
const BYTES_PER_SECOND = TARGET_SAMPLE_RATE * 2;
export const MAX_RECORDING_SECONDS = 160;

export type RecorderStatus = "idle" | "requesting" | "recording" | "error";

interface AudioContextWindow extends Window {
  webkitAudioContext?: typeof AudioContext;
}

function getAudioContextCtor(): typeof AudioContext | null {
  if (typeof window === "undefined") return null;
  const w = window as AudioContextWindow;
  return window.AudioContext ?? w.webkitAudioContext ?? null;
}

function isRecordingSupported(): boolean {
  return typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia && getAudioContextCtor() !== null;
}

function downsample(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === TARGET_SAMPLE_RATE) return input;
  const ratio = inputRate / TARGET_SAMPLE_RATE;
  const outLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), input.length);
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    output[i] = sum / Math.max(1, end - start);
  }
  return output;
}

function encodeWav(samples: Float32Array): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  const dataSize = samples.length * 2;
  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, TARGET_SAMPLE_RATE, true);
  view.setUint32(28, BYTES_PER_SECOND, true);
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

export function useAudioRecorder() {
  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [supported, setSupported] = useState(() => isRecordingSupported());

  const streamRef = useRef<MediaStream | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const inputRateRef = useRef<number>(TARGET_SAMPLE_RATE);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startedAtRef = useRef<number>(0);
  const stopFnRef = useRef<(() => Promise<File | null>) | null>(null);

  const teardown = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    if (contextRef.current && contextRef.current.state !== "closed") {
      void contextRef.current.close();
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    processorRef.current = null;
    sourceRef.current = null;
    contextRef.current = null;
    streamRef.current = null;
  }, []);

  const stop = useCallback(async (): Promise<File | null> => {
    if (status !== "recording") return null;
    const chunks = chunksRef.current;
    const inputRate = inputRateRef.current;
    teardown();
    setStatus("idle");
    setSeconds(0);
    chunksRef.current = [];

    const total = chunks.reduce((acc, c) => acc + c.length, 0);
    if (total === 0) return null;
    const merged = new Float32Array(total);
    let offset = 0;
    for (const c of chunks) {
      merged.set(c, offset);
      offset += c.length;
    }
    const resampled = downsample(merged, inputRate);
    const blob = encodeWav(resampled);
    return new File([blob], "audio-note.wav", { type: "audio/wav" });
  }, [status, teardown]);

  useEffect(() => {
    stopFnRef.current = stop;
  }, [stop]);

  const start = useCallback(async () => {
    if (!isRecordingSupported()) {
      setSupported(false);
      setError("Audio recording is not supported in this browser.");
      return;
    }
    setError(null);
    setStatus("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      const Ctor = getAudioContextCtor();
      if (!Ctor) throw new Error("unsupported");
      const context = new Ctor();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);

      chunksRef.current = [];
      inputRateRef.current = context.sampleRate;
      processor.onaudioprocess = (e) => {
        chunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(context.destination);

      streamRef.current = stream;
      contextRef.current = context;
      sourceRef.current = source;
      processorRef.current = processor;

      startedAtRef.current = Date.now();
      setSeconds(0);
      setStatus("recording");
      timerRef.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startedAtRef.current) / 1000);
        setSeconds(elapsed);
        if (elapsed >= MAX_RECORDING_SECONDS) void stopFnRef.current?.();
      }, 250);
    } catch (err) {
      teardown();
      setStatus("error");
      const name = err instanceof DOMException ? err.name : "";
      if (name === "NotAllowedError" || name === "SecurityError") {
        setError("Microphone permission denied. Allow access to record audio.");
      } else if (name === "NotFoundError") {
        setError("No microphone found.");
      } else {
        setError("Could not start recording.");
      }
    }
  }, [teardown]);

  const cancel = useCallback(() => {
    teardown();
    chunksRef.current = [];
    setStatus("idle");
    setSeconds(0);
    setError(null);
  }, [teardown]);

  useEffect(() => () => teardown(), [teardown]);

  return { status, seconds, error, supported, start, stop, cancel };
}
