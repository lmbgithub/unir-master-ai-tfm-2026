"use client";

import { useRef, useState, useEffect } from "react";
import { Play, Pause, Loader2 } from "lucide-react";
import { attachmentService } from "@/services/AttachmentService";

interface AudioPlayerProps {
  attachmentId: string;
  size?: "sm" | "md";
  className?: string;
  active?: boolean;
}

export function AudioPlayer({ attachmentId, size = "sm", className = "", active = true }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!active && audioRef.current && playing) {
      audioRef.current.pause();
      setPlaying(false);
    }
  }, [active, playing]);

  async function toggle(e: React.MouseEvent) {
    e.stopPropagation();
    if (!audioRef.current) {
      setLoading(true);
      try {
        const blob = await attachmentService.download(attachmentId);
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => setPlaying(false);
        audioRef.current = audio;
      } catch {
        setLoading(false);
        return;
      }
      setLoading(false);
    }

    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
    } else {
      await audioRef.current.play();
      setPlaying(true);
    }
  }

  const iconCls = size === "sm" ? "w-3 h-3" : "w-4 h-4";
  const btnCls = size === "sm" ? "p-1" : "p-1.5";

  return (
    <button
      type="button"
      onClick={toggle}
      disabled={loading}
      className={`${btnCls} rounded hover:bg-muted transition-colors disabled:opacity-50 shrink-0 ${className}`}
      title={playing ? "Pause" : "Play audio"}
    >
      {loading ? (
        <Loader2 className={`${iconCls} animate-spin`} />
      ) : playing ? (
        <Pause className={iconCls} />
      ) : (
        <Play className={iconCls} />
      )}
    </button>
  );
}
