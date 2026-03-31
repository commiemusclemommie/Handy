import React, { useState, useRef, useEffect, useCallback } from "react";
import { Play, Pause } from "lucide-react";

// Global registry: when one player starts, stop all others
const activePlayers = new Set<() => void>();

interface AudioPlayerProps {
  /** Audio source URL. If not provided, onLoadRequest must be provided. */
  src?: string;
  /** Called when play is clicked and no src is loaded yet. If fallback is true,
   * the caller should prefer an inline/data-based fallback source.
   */
  onLoadRequest?: (fallback?: boolean) => Promise<string | null>;
  className?: string;
  autoPlay?: boolean;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({
  src: initialSrc,
  onLoadRequest,
  className = "",
  autoPlay = false,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [loadedSrc, setLoadedSrc] = useState<string | null>(initialSrc ?? null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const dragTimeRef = useRef<number>(0);
  const attemptedFallbackRef = useRef(false);
  const desiredPlayingRef = useRef(false);
  const playbackConfirmedRef = useRef(false);
  const playWatchdogRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onLoadRequestRef = useRef(onLoadRequest);
  const loadedSrcRef = useRef(loadedSrc);

  useEffect(() => {
    onLoadRequestRef.current = onLoadRequest;
  }, [onLoadRequest]);

  useEffect(() => {
    loadedSrcRef.current = loadedSrc;
  }, [loadedSrc]);

  const clearPlayWatchdog = useCallback(() => {
    if (playWatchdogRef.current) {
      clearTimeout(playWatchdogRef.current);
      playWatchdogRef.current = null;
    }
  }, []);

  const loadSource = useCallback(
    (audio: HTMLAudioElement, newSrc: string) => {
      if (audio.src === newSrc) {
        return;
      }

      clearPlayWatchdog();
      playbackConfirmedRef.current = false;
      audio.pause();
      audio.currentTime = 0;
      setIsPlaying(false);
      setCurrentTime(0);
      setDuration(0);
      setLoadError(null);
      audio.src = newSrc;
      audio.load();
    },
    [clearPlayWatchdog],
  );

  const tryFallbackPlayback = useCallback(
    async (audio: HTMLAudioElement): Promise<boolean> => {
      const loadRequest = onLoadRequestRef.current;
      const currentSrc = loadedSrcRef.current || audio.currentSrc || audio.src;

      if (!loadRequest || attemptedFallbackRef.current || !currentSrc) {
        return false;
      }

      attemptedFallbackRef.current = true;
      setIsLoading(true);

      try {
        const fallbackSrc = await loadRequest(true);
        if (!fallbackSrc || fallbackSrc === currentSrc) {
          return false;
        }

        setLoadedSrc(fallbackSrc);
        loadSource(audio, fallbackSrc);

        if (desiredPlayingRef.current) {
          await audio.play();
        }

        return true;
      } catch (error) {
        console.error("Fallback audio load failed:", error);
        return false;
      } finally {
        setIsLoading(false);
      }
    },
    [loadSource],
  );

  const schedulePlayWatchdog = useCallback(
    (audio: HTMLAudioElement) => {
      clearPlayWatchdog();
      playbackConfirmedRef.current = false;

      playWatchdogRef.current = setTimeout(async () => {
        if (!desiredPlayingRef.current || playbackConfirmedRef.current) {
          return;
        }

        const recovered = await tryFallbackPlayback(audio);
        if (!recovered) {
          setIsPlaying(false);
        }
      }, 1200);
    },
    [clearPlayWatchdog, tryFallbackPlayback],
  );

  useEffect(() => {
    const audio = new Audio();
    audio.preload = "metadata";
    audioRef.current = audio;

    const confirmPlayback = () => {
      playbackConfirmedRef.current = true;
      clearPlayWatchdog();
    };

    const handleLoadedMetadata = () => {
      setDuration(audio.duration || 0);
      setCurrentTime(audio.currentTime || 0);
      setLoadError(null);
    };

    const handleDurationChange = () => {
      if (audio.duration && isFinite(audio.duration)) {
        setDuration(audio.duration);
      }
    };

    const handleCanPlay = () => {
      if (audio.duration && isFinite(audio.duration)) {
        setDuration(audio.duration);
      }
      setLoadError(null);
    };

    const handlePlaying = () => {
      confirmPlayback();
      setIsPlaying(true);
    };

    const handlePlay = () => {
      setIsPlaying(true);
    };

    const handlePause = () => {
      if (!desiredPlayingRef.current) {
        clearPlayWatchdog();
      }
      setIsPlaying(false);
    };

    const handleEnded = () => {
      desiredPlayingRef.current = false;
      clearPlayWatchdog();
      setIsPlaying(false);
      setCurrentTime(audio.duration || 0);
    };

    const handleTimeUpdate = () => {
      if (audio.currentTime > 0) {
        confirmPlayback();
      }
      setCurrentTime(audio.currentTime);
    };

    const handleError = async () => {
      console.error("Audio load/play error:", audio.error);
      const recovered = await tryFallbackPlayback(audio);
      if (!recovered) {
        desiredPlayingRef.current = false;
        clearPlayWatchdog();
        setIsPlaying(false);
        setLoadError(audio.error?.message || "Failed to load audio");
      }
    };

    audio.addEventListener("loadedmetadata", handleLoadedMetadata);
    audio.addEventListener("durationchange", handleDurationChange);
    audio.addEventListener("canplay", handleCanPlay);
    audio.addEventListener("playing", handlePlaying);
    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("error", handleError);

    return () => {
      clearPlayWatchdog();
      audio.pause();
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
      audio.removeEventListener("durationchange", handleDurationChange);
      audio.removeEventListener("canplay", handleCanPlay);
      audio.removeEventListener("playing", handlePlaying);
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("error", handleError);
      audio.src = "";
      audioRef.current = null;
    };
  }, [clearPlayWatchdog, tryFallbackPlayback]);

  useEffect(() => {
    const audio = audioRef.current;
    setLoadedSrc(initialSrc ?? null);
    attemptedFallbackRef.current = false;

    if (!audio) {
      return;
    }

    if (initialSrc) {
      loadSource(audio, initialSrc);

      if (autoPlay) {
        desiredPlayingRef.current = true;
        schedulePlayWatchdog(audio);
        audio.play().catch((error) => {
          console.error("Auto-play failed:", error);
        });
      }
    } else {
      desiredPlayingRef.current = false;
      clearPlayWatchdog();
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
      setIsPlaying(false);
      setDuration(0);
      setCurrentTime(0);
      setLoadError(null);
    }
  }, [
    autoPlay,
    clearPlayWatchdog,
    initialSrc,
    loadSource,
    schedulePlayWatchdog,
  ]);

  // Register stop callback for global "only one player at a time"
  const stopPlayback = useCallback(() => {
    desiredPlayingRef.current = false;
    clearPlayWatchdog();
    audioRef.current?.pause();
  }, [clearPlayWatchdog]);

  useEffect(() => {
    activePlayers.add(stopPlayback);
    return () => {
      activePlayers.delete(stopPlayback);
    };
  }, [stopPlayback]);

  // Global drag handlers
  const handleMouseUp = useCallback(() => {
    if (isDragging) {
      setIsDragging(false);
      if (audioRef.current) {
        audioRef.current.currentTime = dragTimeRef.current;
        setCurrentTime(dragTimeRef.current);
      }
    }
  }, [isDragging]);

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mouseup", handleMouseUp);
      document.addEventListener("touchend", handleMouseUp);
      return () => {
        document.removeEventListener("mouseup", handleMouseUp);
        document.removeEventListener("touchend", handleMouseUp);
      };
    }
  }, [isDragging, handleMouseUp]);

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      if (loadedSrc?.startsWith("blob:")) {
        URL.revokeObjectURL(loadedSrc);
      }
    };
  }, [loadedSrc]);

  const togglePlay = async () => {
    const audio = audioRef.current;
    if (!audio || isLoading) return;

    try {
      if (isPlaying) {
        desiredPlayingRef.current = false;
        clearPlayWatchdog();
        audio.pause();
        return;
      }

      activePlayers.forEach((stop) => {
        if (stop !== stopPlayback) stop();
      });

      desiredPlayingRef.current = true;
      setLoadError(null);

      if (!loadedSrc && onLoadRequest) {
        setIsLoading(true);

        try {
          const newSrc = await onLoadRequest(false);
          if (!newSrc) {
            desiredPlayingRef.current = false;
            return;
          }

          attemptedFallbackRef.current = false;
          setLoadedSrc(newSrc);
          loadSource(audio, newSrc);
        } finally {
          setIsLoading(false);
        }
      }

      schedulePlayWatchdog(audio);
      await audio.play();
    } catch (error) {
      console.error("Playback failed:", error);
      const recovered = await tryFallbackPlayback(audio);
      if (!recovered) {
        desiredPlayingRef.current = false;
        clearPlayWatchdog();
        setIsPlaying(false);
      }
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = parseFloat(e.target.value);
    dragTimeRef.current = newTime;
    setCurrentTime(newTime);

    if (!isDragging && audioRef.current) {
      audioRef.current.currentTime = newTime;
    }
  };

  const handleSliderMouseDown = () => {
    setIsDragging(true);
  };

  const handleSliderTouchStart = () => {
    setIsDragging(true);
  };

  const formatTime = (time: number): string => {
    if (!isFinite(time)) return "0:00";
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  const getProgressPercent = (): number => {
    if (duration <= 0) return 0;
    if (duration - currentTime < 0.1) return 100;
    const percent = (currentTime / duration) * 100;
    return Math.min(100, Math.max(0, percent));
  };

  const progressPercent = getProgressPercent();

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <button
        onClick={togglePlay}
        disabled={isLoading}
        className="transition-colors cursor-pointer text-text hover:text-logo-primary disabled:opacity-50"
        aria-label={isPlaying ? "Pause" : "Play"}
      >
        {isPlaying ? (
          <Pause width={20} height={20} fill="currentColor" />
        ) : (
          <Play width={20} height={20} fill="currentColor" />
        )}
      </button>

      <div className="flex-1 flex items-center gap-2">
        <span className="text-xs text-text/60 min-w-[30px] tabular-nums">
          {formatTime(currentTime)}
        </span>

        <input
          type="range"
          min="0"
          max={duration || 0}
          step="0.01"
          value={currentTime}
          onChange={handleSeek}
          onMouseDown={handleSliderMouseDown}
          onTouchStart={handleSliderTouchStart}
          className={`flex-1 h-1 rounded-lg appearance-none cursor-pointer focus:outline-none focus:ring-1 focus:ring-logo-primary ${progressPercent >= 99.5 ? "[&::-webkit-slider-thumb]:translate-x-0.5 [&::-moz-range-thumb]:translate-x-0.5" : ""}`}
          style={{
            background: `linear-gradient(to right, #FAA2CA 0%, #FAA2CA ${progressPercent}%, rgba(128, 128, 128, 0.2) ${progressPercent}%, rgba(128, 128, 128, 0.2) 100%)`,
          }}
        />

        <span className="text-xs text-text/60 min-w-[30px] tabular-nums">
          {formatTime(duration)}
        </span>
      </div>

      {loadError && (
        <div className="text-xs text-red-500 mt-1">{loadError}</div>
      )}
    </div>
  );
};
