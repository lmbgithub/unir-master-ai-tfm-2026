#!/usr/bin/env python3
"""Benchmark de modelos GGUF locales (llama.cpp) para UrgeNurse.

Script AUTOCONTENIDO: la única dependencia externa es ``llama-cpp-python``
(que trae llama.cpp embebido). Todo lo demás es stdlib. Pensado para correr
en CPU en una Beelink S12 / N100 con 16 GB de RAM.

Qué hace:
  1. Recorre una lista de modelos GGUF (agrupados por familia), uno por uno.
  2. Si el modelo no está descargado, lo baja desde HuggingFace.
  3. Por cada modelo prueba DOS contextos (ctx_min y ctx_max) en un ciclo interno
     y compara. Mide tamaño de archivo, tiempo de carga (ms) y RAM tras cargar.
  4. Ejecuta operaciones async (triage / NER / SBAR), cada una mide tiempo (ms) y
     accuracy. Cada inferencia resetea la sesión/KV-cache (arranca de cero) y usa
     streaming con continuación para no truncar la respuesta (JSON válido).
  5. Tras cada prueba/modelo limpia memoria/contexto; al final libera todo.
  6. Devuelve un pandas.DataFrame (una fila por modelo×contexto) y escribe un
     error__dd_mm_yyyy__hh_mm.json con los problemas detectados por modelo.

Uso:
    pip install llama-cpp-python pandas
    python llm_performance.py          # imprime la tabla de resultados
    # o desde un notebook:  df = await run_benchmark()

Variables de entorno opcionales:
    LLM_PERF_MODELS_DIR   carpeta de modelos (default: ../models junto al script)
    LLM_PERF_OUTPUT_DIR   carpeta para el error__*.json (default: carpeta del script)
    LLM_PERF_DOWNLOAD     "0" para no descargar modelos faltantes (default: "1" = baja todos)
    LLM_PERF_THREADS      nº de hilos (default: os.cpu_count())
    LLM_PERF_CTX          contexto por defecto si el modelo no fija n_ctx (default: 4096)
    LLM_PERF_MAX_RAM_GB   presupuesto de RAM; omite modelos más grandes (default: 5.0)
    LLM_PERF_GPU_LAYERS   "auto" (default) usa GPU si el build de llama.cpp la soporta
                          y hay GPU; o un entero (0=solo CPU, -1=todas las capas)

GPU automática: si hay GPU y el build de llama-cpp-python la soporta (Metal/CUDA),
se hace offload de todas las capas; si no, corre en CPU. Tope de RAM de 5 GB por
defecto: los modelos cuyo footprint estimado supere el presupuesto se omiten.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("LLM_PERF_MODELS_DIR", SCRIPT_DIR.parent / "models"))
OUTPUT_DIR = Path(
    os.environ.get("LLM_PERF_OUTPUT_DIR", SCRIPT_DIR)
)  # solo para error__*.json
DOWNLOAD_MISSING = os.environ.get("LLM_PERF_DOWNLOAD", "1") != "0"


def _physical_cores() -> int:
    """Núcleos físicos (mejor para llama.cpp que los lógicos: evita oversubscription)."""
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:  # noqa: BLE001
        pass
    return os.cpu_count() or 4


N_THREADS = int(os.environ.get("LLM_PERF_THREADS", _physical_cores()))
N_CTX = int(os.environ.get("LLM_PERF_CTX", 4096))
# Salidas JSON son cortas (NER ~80, triage ~150 tokens). 512 evita que un modelo
# que divague gaste minutos generando hasta el tope. Subir solo si hace falta.
MAX_TOKENS = int(os.environ.get("LLM_PERF_MAX_TOKENS", 512))
TEMPERATURE = 0.0

# Streaming OFF por defecto: en CPU añade overhead Python por token (más lento) y
# ensucia la medición. El grammar JSON ya evita truncados. Actívalo con "1" si
# quieres recibir la respuesta en trozos + continuación al cortarse por longitud.
STREAM = os.environ.get("LLM_PERF_STREAM", "0") != "0"
MAX_CONTINUATIONS = int(os.environ.get("LLM_PERF_MAX_CONT", 2))

# GPU: "auto" detecta y USA la GPU si el build de llama.cpp la soporta (Metal/CUDA)
# y hay una GPU real; si no, cae a solo-CPU. Se puede forzar con un entero:
# LLM_PERF_GPU_LAYERS=0 (solo CPU), =-1 (todas las capas a GPU), =N (N capas).
GPU_LAYERS_SETTING = os.environ.get("LLM_PERF_GPU_LAYERS", "auto")

# Presupuesto de RAM. llama.cpp no tiene un tope duro; lo aplicamos saltándonos
# los modelos cuyo footprint estimado (archivo GGUF + overhead) supere el límite,
# para no provocar OOM ni swap en la máquina.
MAX_RAM_GB = float(os.environ.get("LLM_PERF_MAX_RAM_GB", 5.0))
# Overhead aproximado por encima del peso del archivo: KV-cache (≈ proporcional a
# N_CTX) + buffers de cómputo + runtime. Conservador para modelos hasta ~4B Q4.
RAM_OVERHEAD_MB = float(os.environ.get("LLM_PERF_RAM_OVERHEAD_MB", 1024))


# ─────────────────────────────────────────────────────────────────────────────
# Catálogo de modelos
# ─────────────────────────────────────────────────────────────────────────────
# Modelos open-source pequeños, cuantizados Q4_K_M, rápidos en CPU y capaces
# para extracción NER / SBAR. El nombre de archivo se guarda en minúsculas.
# Si el archivo no existe en MODELS_DIR se descarga desde `url`.
#
# Comenta/descomenta para elegir qué modelos correr. Cuidado con el espacio en
# disco: la lista completa son ~12 GB.


@dataclass(frozen=True)
class ModelSpec:
    name: str  # nombre de archivo local, en minúsculas
    url: str  # URL de descarga (HuggingFace resolve)
    ctx_min: int = 2048  # contexto MÍNIMO a probar (suficiente para los prompts)
    ctx_max: int = 8192  # contexto MÁXIMO a probar (capado por RAM, no por train)
    system_mode: str = (
        "system"  # "system" normal | "merge" si el modelo no soporta rol system
    )
    json_object: bool = (
        True  # forzar salida JSON con grammar (evita JSON inválido y "thinking")
    )


# El aviso "n_ctx_seq < n_ctx_train" es informativo, NO un error: usamos menos
# contexto del que el modelo soporta. Para NER/SBAR/triage los prompts son cortos,
# así que probamos cada modelo en DOS contextos (ctx_min y ctx_max) y comparamos:
# más contexto = más KV-cache = más RAM y algo más lento, normalmente sin ganancia
# de calidad. ctx_max se capa por RAM (no por el contexto de entrenamiento) para
# respetar el presupuesto de 5 GB. Train ctx de referencia entre paréntesis.
# Modelos agrupados por familia.
MODELS: list[ModelSpec] = [
    # ── Llama 2 ─────────────────────────────────────────────────────────────
    ModelSpec(
        "llama-2-7b-chat-q4_k_m.gguf",
        "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=4096,  # train 4096; 7B ≈4 GB: capamos a 4096
    ),
    # ── Llama 3.2 ───────────────────────────────────────────────────────────
    ModelSpec(
        "llama-3.2-1b-instruct-q4_k_m.gguf",
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 131072
    ),
    ModelSpec(
        "llama-3.2-3b-instruct-q4_k_m.gguf",
        "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 131072 (3B: capamos por RAM)
    ),
    # ── Phi (Microsoft) ─────────────────────────────────────────────────────
    ModelSpec(
        "phi-2-q4_k_m.gguf",
        "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        ctx_min=1024,
        ctx_max=2048,  # train 2048 (su máximo; base, sin plantilla chat oficial)
        system_mode="merge",
    ),
    ModelSpec(
        "phi-3.5-mini-instruct-q4_k_m.gguf",
        "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=4096,  # train 131072 (capamos por RAM)
    ),
    ModelSpec(
        "phi-4-mini-instruct-q4_k_m.gguf",
        "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=4096,  # train 131072 (≈4.3 GB RSS a 4096: no subir)
    ),
    # ── Qwen 2.5 ────────────────────────────────────────────────────────────
    ModelSpec(
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 32768
    ),
    ModelSpec(
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 32768
    ),
    ModelSpec(
        "qwen2.5-1.5b-instruct-q5_k_m.gguf",
        "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 32768 (variante Q5)
    ),
    ModelSpec(
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 32768 (3B: capamos por RAM)
    ),
    # ── Qwen 3 ──────────────────────────────────────────────────────────────
    ModelSpec(
        "qwen3-1.7b-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 40960 (razonador: json_object evita el "thinking")
    ),
    # ── DeepSeek R1 (distill) ───────────────────────────────────────────────
    ModelSpec(
        "deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf",
        "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 131072 (json_object evita el "thinking")
    ),
    # ── Gemma 2 (Google) — no soporta rol system → merge ────────────────────
    ModelSpec(
        "gemma-2-2b-it-q4_k_m.gguf",
        "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 8192 (su máximo)
        system_mode="merge",
    ),
    ModelSpec(
        "gemma-2-2b-it-q5_k_m.gguf",
        "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q5_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,
        system_mode="merge",
    ),
    # ── Mistral ─────────────────────────────────────────────────────────────
    ModelSpec(
        "mistral-7b-instruct-v0.3-q4_k_m.gguf",
        "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=4096,  # train 32768 (7B: capamos por RAM)
    ),
    # ── SmolLM2 (HuggingFace) ───────────────────────────────────────────────
    ModelSpec(
        "smollm2-1.7b-instruct-q4_k_m.gguf",
        "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 8192 (su máximo)
    ),
    # ── LFM2 (LiquidAI) — arquitectura "edge" pequeña y rápida ──────────────
    ModelSpec(
        "lfm2-700m-q4_k_m.gguf",
        "https://huggingface.co/LiquidAI/LFM2-700M-GGUF/resolve/main/LFM2-700M-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 32768
    ),
    ModelSpec(
        "lfm2-1.2b-q4_k_m.gguf",
        "https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF/resolve/main/LFM2-1.2B-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 32768
    ),
    ModelSpec(
        "lfm2-2.6b-q4_k_m.gguf",
        "https://huggingface.co/LiquidAI/LFM2-2.6B-GGUF/resolve/main/LFM2-2.6B-Q4_K_M.gguf",
        ctx_min=2048,
        ctx_max=8192,  # train 32768
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de sistema (memoria, descarga)
# ─────────────────────────────────────────────────────────────────────────────


def get_rss_mb() -> float:
    """RSS del proceso en MB. Usa psutil si está, si no cae a resource (stdlib)."""
    try:
        import psutil  # opcional; medición más precisa y cross-platform

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reporta KB, macOS/BSD reporta bytes.
        if sys.platform == "darwin":
            return ru / (1024 * 1024)
        return ru / 1024
    except Exception:
        return float("nan")


def human_mb(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Información del computador (best-effort, nunca lanza excepción)
# ─────────────────────────────────────────────────────────────────────────────


def _run(cmd: list[str]) -> str:
    """Ejecuta un comando y devuelve stdout (vacío si falla / no existe)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _cpu_model() -> str:
    s = platform.system()
    if s == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:  # noqa: BLE001
            pass
    elif s == "Darwin":
        v = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if v:
            return v
    return platform.processor() or "unknown"


def _cpu_family() -> str:
    s = platform.system()
    if s == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("cpu family"):
                    return line.split(":", 1)[1].strip()
        except Exception:  # noqa: BLE001
            pass
    elif s == "Darwin":
        v = _run(["sysctl", "-n", "machdep.cpu.family"])
        if v:
            return v
    return "unknown"


def _cpu_clock_mhz() -> dict | None:
    try:
        import psutil

        f = psutil.cpu_freq()
        if f:
            return {
                "current": round(f.current),
                "min": round(f.min),
                "max": round(f.max),
            }
    except Exception:  # noqa: BLE001
        pass
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("cpu mhz"):
                    return {"current": round(float(line.split(":", 1)[1].strip()))}
        except Exception:  # noqa: BLE001
            pass
    elif platform.system() == "Darwin":
        hz = _run(["sysctl", "-n", "hw.cpufrequency_max"]) or _run(
            ["sysctl", "-n", "hw.cpufrequency"]
        )
        if hz.isdigit():
            return {"max": round(int(hz) / 1_000_000)}
    return None


def _memory_type_speed() -> tuple[str | None, str | None]:
    s = platform.system()
    if s == "Linux":
        txt = _run(["dmidecode", "-t", "memory"])  # normalmente requiere root
        if txt:
            mtype = speed = None
            for line in txt.splitlines():
                ls = line.strip()
                if ls.startswith("Type:") and mtype is None:
                    val = ls.split(":", 1)[1].strip()
                    if val not in ("Other", "Unknown", "None", ""):
                        mtype = val
                if ls.startswith("Speed:") and speed is None:
                    val = ls.split(":", 1)[1].strip()
                    if "Unknown" not in val and val:
                        speed = val
            return mtype, speed
    elif s == "Darwin":
        txt = _run(["system_profiler", "SPMemoryDataType"])
        mtype = speed = None
        for line in txt.splitlines():
            ls = line.strip()
            if ls.startswith("Type:") and mtype is None:
                mtype = ls.split(":", 1)[1].strip()
            if ls.startswith("Speed:") and speed is None:
                speed = ls.split(":", 1)[1].strip()
        return mtype, speed
    return None, None


def _memory_info() -> dict:
    out: dict = {"total_mb": None, "available_mb": None, "type": None, "speed": None}
    try:
        import psutil

        vm = psutil.virtual_memory()
        out["total_mb"] = round(vm.total / 1048576)
        out["available_mb"] = round(vm.available / 1048576)
    except Exception:  # noqa: BLE001
        try:
            mi = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, _, v = line.partition(":")
                mi[k.strip()] = v.strip()
            out["total_mb"] = round(int(mi["MemTotal"].split()[0]) / 1024)
            out["available_mb"] = round(
                int(mi.get("MemAvailable", "0").split()[0]) / 1024
            )
        except Exception:  # noqa: BLE001
            pass
        if out["total_mb"] is None and platform.system() == "Darwin":
            mem = _run(["sysctl", "-n", "hw.memsize"])
            if mem.isdigit():
                out["total_mb"] = round(int(mem) / 1048576)
    out["type"], out["speed"] = _memory_type_speed()
    return out


def _swap_info() -> dict:
    try:
        import psutil

        sw = psutil.swap_memory()
        return {
            "total_mb": round(sw.total / 1048576),
            "used_mb": round(sw.used / 1048576),
            "free_mb": round(sw.free / 1048576),
        }
    except Exception:  # noqa: BLE001
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("SwapTotal"):
                    return {"total_mb": round(int(line.split()[1]) / 1024)}
        except Exception:  # noqa: BLE001
            pass
    return {"total_mb": None}


def _gpu_info() -> list[dict]:
    # NVIDIA primero (si hay driver instalado)
    nv = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if nv:
        return [
            {"vendor": "nvidia", "name": ln.strip()}
            for ln in nv.splitlines()
            if ln.strip()
        ]
    gpus: list[dict] = []
    s = platform.system()
    if s == "Linux":
        lp = _run(["lspci"])
        for line in lp.splitlines():
            if any(k in line.lower() for k in ("vga", "3d controller", "display")):
                gpus.append({"name": line.split(":", 2)[-1].strip()})
    elif s == "Darwin":
        txt = _run(["system_profiler", "SPDisplaysDataType"])
        for line in txt.splitlines():
            ls = line.strip()
            if ls.startswith("Chipset Model:"):
                gpus.append({"name": ls.split(":", 1)[1].strip()})
    return gpus or [{"name": "none / integrated"}]


def llama_supports_offload() -> bool | None:
    """True/False si el build de llama-cpp-python soporta offload a GPU.

    None si llama-cpp-python no está instalado.
    """
    try:
        import llama_cpp

        fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        return bool(fn()) if callable(fn) else None
    except Exception:  # noqa: BLE001
        return None


def _has_real_gpu(gpus: list[dict]) -> bool:
    """¿Hay al menos una GPU usable (no el placeholder 'none/integrated')?"""
    return any(
        g.get("name", "").lower() and "none" not in g.get("name", "").lower()
        for g in gpus
    )


def resolve_gpu_layers() -> tuple[int, str]:
    """Decide n_gpu_layers según LLM_PERF_GPU_LAYERS o autodetección.

    Returns (n_gpu_layers, motivo). -1 = todas las capas a GPU; 0 = solo CPU.
    Usa GPU solo si el build de llama.cpp lo soporta Y hay una GPU real.
    """
    if GPU_LAYERS_SETTING != "auto":
        try:
            n = int(GPU_LAYERS_SETTING)
            return n, f"forzado por LLM_PERF_GPU_LAYERS={n}"
        except ValueError:
            pass  # valor inválido -> seguimos en auto

    supports = llama_supports_offload()
    if supports is None:
        return 0, "auto: llama-cpp-python no instalado -> solo CPU (provisional)"
    if not supports:
        return 0, "auto: el build de llama.cpp no soporta GPU -> solo CPU"
    if _has_real_gpu(_gpu_info()):
        return -1, "auto: GPU presente y soportada -> offload de todas las capas"
    return 0, "auto: no se detectó GPU usable -> solo CPU"


def collect_system_info() -> dict:
    """Reúne info del host: OS, CPU, RAM (tipo/velocidad/disp.), swap, GPU y aceleración."""
    cpu = {
        "model": _cpu_model(),
        "arch": platform.machine(),
        "family": _cpu_family(),
        "cores_logical": os.cpu_count(),
        "cores_physical": None,
        "clock_mhz": _cpu_clock_mhz(),
    }
    try:
        import psutil

        cpu["cores_physical"] = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        pass
    n_gpu_layers, gpu_reason = resolve_gpu_layers()
    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "cpu": cpu,
        "memory": _memory_info(),
        "swap": _swap_info(),
        "gpu": _gpu_info(),
        "acceleration": {
            "llama_gpu_offload_supported": llama_supports_offload(),
            "n_gpu_layers": n_gpu_layers,
            "using_gpu": n_gpu_layers != 0,
            "decision": gpu_reason,
        },
    }


def print_system_info(info: dict | None = None) -> dict:
    """Imprime un bloque legible con la info del computador. Devuelve el dict."""
    info = info or collect_system_info()
    os_i, cpu, mem, swap, gpu = (
        info["os"],
        info["cpu"],
        info["memory"],
        info["swap"],
        info["gpu"],
    )
    clk = cpu.get("clock_mhz") or {}
    clk_s = " · ".join(f"{k} {v} MHz" for k, v in clk.items()) if clk else "n/d"
    mem_t = mem.get("type") or "n/d (requiere root/dmidecode)"
    mem_sp = mem.get("speed") or "n/d"
    print("=" * 64)
    print("INFORMACIÓN DEL COMPUTADOR")
    print("=" * 64)
    print(f"OS        : {os_i['system']} {os_i['release']}  ({os_i['platform']})")
    print(f"CPU       : {cpu['model']}")
    print(
        f"            arch={cpu['arch']} · familia={cpu['family']} · "
        f"núcleos={cpu['cores_physical']}f/{cpu['cores_logical']}l"
    )
    print(f"            clock: {clk_s}")
    print(
        f"RAM       : total {mem['total_mb']} MB · disponible {mem['available_mb']} MB"
    )
    print(f"            tipo {mem_t} · velocidad {mem_sp}")
    sw_t = swap.get("total_mb")
    if swap.get("used_mb") is not None:
        print(
            f"Swap      : total {sw_t} MB · usada {swap['used_mb']} MB · libre {swap['free_mb']} MB"
        )
    else:
        print(f"Swap      : total {sw_t} MB")
    print(f"GPU       : {', '.join(g['name'] for g in gpu)}")
    acc = info.get("acceleration", {})
    if acc:
        usando = "GPU" if acc.get("using_gpu") else "solo CPU"
        print(f"Aceleración: {usando}  (n_gpu_layers={acc.get('n_gpu_layers')})")
        print(f"            {acc.get('decision')}")
    print("=" * 64)
    return info


def ensure_model(spec: ModelSpec) -> Path | None:
    """Devuelve la ruta del modelo, descargándolo si falta. None si no se pudo."""
    path = MODELS_DIR / spec.name
    if path.exists() and path.stat().st_size > 0:
        return path

    if not DOWNLOAD_MISSING:
        print(f"  ⨯ {spec.name} no está y la descarga está desactivada — se omite")
        return None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"  ↓ descargando {spec.name}\n    desde {spec.url}")
    try:
        req = urllib.request.Request(
            spec.url, headers={"User-Agent": "urgenurse-bench/1.0"}
        )
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            chunk = 1024 * 1024
            last_pct = -1
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                if total:
                    pct = int(done * 100 / total)
                    if pct != last_pct and pct % 5 == 0:
                        print(
                            f"    {pct:3d}%  ({human_mb(done)} / {human_mb(total)} MB)"
                        )
                        last_pct = pct
        tmp.rename(path)
        print(f"  ✓ descargado {spec.name}")
        return path
    except Exception as exc:  # noqa: BLE001
        print(f"  ⨯ fallo al descargar {spec.name}: {exc}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Capa de inferencia: chat completion + parseo de JSON tolerante
# ─────────────────────────────────────────────────────────────────────────────


def _extract_json_object(text: str) -> str:
    """Extrae el primer objeto JSON balanceado, respetando strings y escapes.

    Tolera texto antes (p. ej. "Here is the JSON:") y DESPUÉS del objeto
    (causa típica de "JSONDecodeError: Extra data" cuando el modelo añade cola).
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]  # JSON sin cerrar: lo intentamos igual


def parse_json_loose(content: str) -> dict:
    """Normaliza la salida del modelo y parsea un objeto JSON.

    Maneja: bloques de "thinking" (<think>…</think>), fences markdown, y texto
    extra antes/después del objeto JSON.
    """
    content = content.strip()
    # Modelos razonadores: quedarse con lo que va tras el cierre del pensamiento.
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()
    # Fences ```json ... ```
    if content.startswith("```"):
        content = content.split("```", 2)[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.rsplit("```", 1)[0].strip()
    content = _extract_json_object(content).strip()
    return json.loads(content)


def reset_llm_state(llm) -> None:
    """Borra sesión / KV-cache / contexto previo del modelo en llama.cpp.

    Así cada inferencia (cada prueba) arranca DESDE CERO: sin reuso de prefijo de
    la KV-cache ni estado de tokens de la operación anterior, lo que mantiene los
    tiempos y resultados comparables entre pruebas.
    """
    try:
        llm.reset()  # n_tokens=0 y limpia el tracking de prefijo de entrada
    except Exception:  # noqa: BLE001
        pass
    ctx = getattr(llm, "_ctx", None)  # llama_cpp.LlamaContext
    if ctx is not None:
        for meth in ("kv_cache_clear", "kv_self_clear"):  # nombre según versión
            fn = getattr(ctx, meth, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass
                break


def _build_messages(system: str, user: str, mode: str) -> list[dict]:
    """Plantilla de mensajes según el modelo.

    - "system": rol system separado (la mayoría).
    - "merge": funde system+user en un único mensaje user (modelos como Gemma,
      cuya plantilla de chat NO admite rol system → "System role not supported").
    """
    if mode == "merge":
        return [{"role": "user", "content": f"{system}\n\n{user}"}]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _stream_once(
    llm, messages: list[dict], max_tokens: int, json_fmt: bool
) -> tuple[str, str | None, int, float | None]:
    """Genera en streaming: une los trozos cortos en un único texto.

    Devuelve (texto, finish_reason, n_tokens, ttft_ms). ttft_ms = tiempo hasta el
    primer token. finish_reason == "length" => se cortó (candidato a continuación).
    """
    kwargs: dict = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": True,
    }
    if json_fmt:
        kwargs["response_format"] = {"type": "json_object"}

    parts: list[str] = []
    finish: str | None = None
    n_tokens = 0
    ttft_ms: float | None = None
    t0 = time.perf_counter()
    for chunk in llm.create_chat_completion(**kwargs):
        choice = chunk["choices"][0]
        piece = choice.get("delta", {}).get("content")
        if piece:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000
            parts.append(piece)
            n_tokens += 1
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
    return "".join(parts), finish, n_tokens, ttft_ms


def _generate(
    llm, system: str, user: str, mode: str, json_fmt: bool, max_tokens: int
) -> tuple[str, dict]:
    """Una prueba completa en una sesión común. Devuelve (content, meta) con el
    desglose temporal por fase (prep / infer / ttft) y conteo de tokens."""
    # ── ANTES de enviar: reset (limpia KV-cache) + construir mensajes ──
    t_prep = time.perf_counter()
    reset_llm_state(llm)  # arranca de cero, sin carryover de la prueba anterior
    messages = _build_messages(system, user, mode)
    stream = getattr(llm, "_bench_stream", STREAM)
    prep_ms = (time.perf_counter() - t_prep) * 1000

    # ── DURANTE: llamada(s) al modelo ──
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens = 0
    t_inf = time.perf_counter()
    if not stream:
        kwargs: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
        }
        if json_fmt:
            kwargs["response_format"] = {"type": "json_object"}
        resp = llm.create_chat_completion(**kwargs)
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens") or 0
    else:
        # Streaming en la MISMA sesión; continúa si se cortó por longitud.
        content, finish, n, ttft_ms = _stream_once(llm, messages, max_tokens, json_fmt)
        completion_tokens = n
        guard = 0
        while finish == "length" and guard < MAX_CONTINUATIONS:
            guard += 1
            cont = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": "Continue exactly from where you stopped. "
                    "Output only the missing remainder, with no repetition and no extra commentary.",
                },
            ]
            more, finish, n2, _ = _stream_once(llm, cont, max_tokens, False)
            if not more:
                break
            content += more
            completion_tokens += n2
    infer_ms = (time.perf_counter() - t_inf) * 1000

    meta = {
        "prep_ms": round(prep_ms, 1),
        "infer_ms": round(infer_ms, 1),
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    return content, meta


def chat_json(
    llm, system: str, user: str, max_tokens: int = MAX_TOKENS
) -> tuple[dict, dict]:
    """Infiere y devuelve (data, meta) con desglose temporal por fase.

    Cada llamada arranca desde cero (reset). Usa plantilla/formato del modelo
    (atributos `_bench_*`), con fallbacks ante errores de rol system o grammar.
    """
    mode = getattr(llm, "_bench_system_mode", "system")
    use_json = getattr(llm, "_bench_json_object", True)

    try:
        content, meta = _generate(llm, system, user, mode, use_json, max_tokens)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        # Si la plantilla no admite rol system, reintenta fundiéndolo en el user.
        retry_mode = (
            "merge"
            if ("system" in msg and ("role" in msg or "support" in msg))
            else mode
        )
        try:
            content, meta = _generate(
                llm, system, user, retry_mode, use_json, max_tokens
            )
        except Exception:  # noqa: BLE001
            content, meta = _generate(
                llm, system, user, retry_mode, False, max_tokens
            )  # sin grammar

    # ── DESPUÉS: parseo del JSON ──
    t_parse = time.perf_counter()
    data = parse_json_loose(content)
    meta["parse_ms"] = round((time.perf_counter() - t_parse) * 1000, 1)
    return data, meta


async def chat_json_async(
    llm, system: str, user: str, max_tokens: int = MAX_TOKENS
) -> tuple[dict, dict]:
    """Wrapper async: corre la inferencia bloqueante en un hilo para no congelar el loop."""
    return await asyncio.to_thread(chat_json, llm, system, user, max_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────


def _norm(text: object) -> str:
    return str(text).strip().lower()


def score_ner(expected: dict[str, str], got: dict) -> float:
    """Fracción de claves esperadas cuyo valor coincide (substring en cualquier dirección)."""
    if not expected:
        return 0.0
    hits = 0
    for key, exp_val in expected.items():
        got_val = None
        for gk, gv in got.items():
            if _norm(gk) == _norm(key):
                got_val = gv
                break
        if got_val is None:
            continue
        ev, gv = _norm(exp_val), _norm(got_val)
        if ev and (ev in gv or gv in ev):
            hits += 1
    return round(hits / len(expected), 3)


def score_sbar(expected: dict[str, list[str]], got: dict) -> float:
    """Fracción de campos SBAR cuyo texto contiene al menos un keyword esperado."""
    if not expected:
        return 0.0
    hits = 0
    for field_name, keywords in expected.items():
        got_val = ""
        for gk, gv in got.items():
            if _norm(gk) == _norm(field_name) or _norm(gk).startswith(
                _norm(field_name)[0]
            ):
                got_val = _norm(gv)
                break
        if any(_norm(kw) in got_val for kw in keywords):
            hits += 1
    return round(hits / len(expected), 3)


def score_esi(expected: int, got: int) -> float:
    """1.0 si acierta el nivel ESI; baja linealmente con la distancia."""
    return round(max(0.0, 1.0 - abs(expected - got) / 4.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Operaciones de benchmark
# ─────────────────────────────────────────────────────────────────────────────
# Cada operación es una función async que recibe el modelo cargado y devuelve un
# OpResult. El runner la cronometra y vuelca sus columnas al CSV:
#   <op_name>_ms   tiempo de ejecución
#   <op_name>_acc  accuracy contra el resultado esperado [0..1]


@dataclass
class OpResult:
    name: str
    time_ms: float = 0.0
    accuracy: float = 0.0
    error: str = ""
    # Desglose por fase (para ver si el coste está antes / durante / después)
    prep_ms: float = 0.0  # ANTES de enviar: reset + construir mensajes
    infer_ms: float = 0.0  # DURANTE: llamada(s) al modelo (generación)
    ttft_ms: float | None = None  # time-to-first-token (solo en streaming)
    parse_ms: float = 0.0  # DESPUÉS: parseo del JSON
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tok_s: float | None = None


Operation = Callable[[object], Awaitable[OpResult]]
OPERATIONS: list[Operation] = []


def operation(name: str) -> Callable[[Operation], Operation]:
    """Registra una operación y le adjunta su nombre de columna."""

    def deco(fn: Operation) -> Operation:
        fn.op_name = name  # type: ignore[attr-defined]
        OPERATIONS.append(fn)
        return fn

    return deco


async def _run_op(llm, name: str, system: str, user: str, scorer) -> OpResult:
    """Helper común: cronometra (con desglose por fase), infiere, puntúa y captura errores."""
    res = OpResult(name=name)
    t0 = time.perf_counter()
    try:
        data, meta = await chat_json_async(llm, system, user)
        res.accuracy = scorer(data)
        res.prep_ms = meta.get("prep_ms", 0.0)
        res.infer_ms = meta.get("infer_ms", 0.0)
        res.ttft_ms = meta.get("ttft_ms")
        res.parse_ms = meta.get("parse_ms", 0.0)
        res.prompt_tokens = meta.get("prompt_tokens")
        res.completion_tokens = meta.get("completion_tokens")
        if res.completion_tokens and res.infer_ms:
            res.tok_s = round(res.completion_tokens / (res.infer_ms / 1000), 1)
    except Exception as exc:  # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"[:200]
        res.accuracy = 0.0
    res.time_ms = round((time.perf_counter() - t0) * 1000, 1)
    return res


# ── Triage ──────────────────────────────────────────────────────────────────
# Prompt copiado de packages/agent-triage/.../llm.py para reproducir el agente.

_TRIAGE_SYSTEM = """\
You are a triage documentation reviewer assisting healthcare staff in an emergency setting. \
Patient care always comes first — your goal is to HELP the team move forward, not to block cases.

Your role is STRICTLY documentary — you do NOT diagnose, prescribe, or make clinical decisions.

You will receive patient demographic data, vital signs, a main complaint, and transcriptions of \
attached documents or audio recordings.

Your task:
1. Assign a preliminary ESI level (1–5) based on documented urgency indicators.
2. Identify any fields that are missing, implausible, or need verification — list them as notes.

ESI levels (documentation reference only):
  1 — Immediate life-saving intervention indicated by documented signs
  2 — High-risk or severe distress documented
  3 — Multiple resources likely needed, stable vitals documented
  4 — One resource needed, no distress documented
  5 — No resources needed, minimal documented complaints

Always respond with a valid JSON object with these exact keys:
- "valid": boolean
- "missing_fields": list of field names
- "esi_level": integer 1–5
- "analysis": 3–5 sentence documentary note

Return ONLY the JSON object, no markdown, no explanation.\
"""

_TRIAGE_USER_TMPL = """\
Review the following triage case documentation:

PATIENT:
{patient}

MAIN COMPLAINT:
{chief_complaint}

ATTACHMENTS:
{attachments}
"""


def _triage_user(patient: dict, chief_complaint: str, attachments: list[dict]) -> str:
    plines = []
    for k, v in patient.items():
        label = k.replace("_", " ").title()
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v) if v else "None"
        plines.append(f"  {label}: {v}")
    aparts = [f"  [{a['name']}]\n  Transcription: {a['content']}" for a in attachments]
    return _TRIAGE_USER_TMPL.format(
        patient="\n".join(plines),
        chief_complaint=chief_complaint,
        attachments="\n\n".join(aparts),
    )


@operation("triage_high_acuity")
async def op_triage_high(llm) -> OpResult:
    """Caso de alta urgencia: dolor torácico + ECG y troponina alterados → ESI 2."""
    patient = {
        "name": "Robert Hayes",
        "gender": "male",
        "date_of_birth": "1958-04-12",
        "id_number": "X1239876",
        "blood_type": "O",
        "blood_rh": True,
        "blood_pressure_systolic": 168,
        "blood_pressure_diastolic": 98,
        "weight": 92,
        "height": 178,
        "pulse": 112,
        "allergies": ["penicillin"],
        "chronic_conditions": ["hypertension", "type 2 diabetes"],
    }
    chief_complaint = (
        "Sudden crushing chest pain radiating to the left arm for the last 40 minutes, "
        "with shortness of breath, sweating and nausea."
    )
    attachments = [
        {
            "name": "ecg_report.pdf",
            "content": "12-lead ECG: ST-segment elevation in leads II, III and aVF. "
            "Sinus tachycardia at 112 bpm. Findings consistent with acute inferior STEMI.",
        },
        {
            "name": "lab_troponin.pdf",
            "content": "High-sensitivity troponin I: 1450 ng/L (reference < 34 ng/L). "
            "Markedly elevated, consistent with acute myocardial injury.",
        },
    ]
    user = _triage_user(patient, chief_complaint, attachments)
    return await _run_op(
        llm,
        "triage_high_acuity",
        _TRIAGE_SYSTEM,
        user,
        lambda d: score_esi(2, max(1, min(5, int(d.get("esi_level", 3))))),
    )


@operation("triage_low_acuity")
async def op_triage_low(llm) -> OpResult:
    """Caso menor: esguince de tobillo, vitales normales → ESI 4."""
    patient = {
        "name": "Lucia Romano",
        "gender": "female",
        "date_of_birth": "1996-09-30",
        "id_number": "Y7781234",
        "blood_type": "A",
        "blood_rh": False,
        "blood_pressure_systolic": 118,
        "blood_pressure_diastolic": 76,
        "weight": 61,
        "height": 165,
        "pulse": 74,
        "allergies": [],
        "chronic_conditions": [],
    }
    chief_complaint = (
        "Twisted right ankle while jogging 2 hours ago. Mild swelling, able to bear weight "
        "with discomfort. No numbness. Pain 4/10."
    )
    attachments = [
        {
            "name": "ankle_xray.pdf",
            "content": "X-ray right ankle, AP and lateral views: no fracture or dislocation. "
            "Soft tissue swelling lateral malleolus. Consistent with grade I sprain.",
        },
        {
            "name": "nurse_note.wav",
            "content": "Patient ambulatory, mild lateral ankle tenderness, neurovascular intact, "
            "no open wound. Stable and comfortable.",
        },
    ]
    user = _triage_user(patient, chief_complaint, attachments)
    return await _run_op(
        llm,
        "triage_low_acuity",
        _TRIAGE_SYSTEM,
        user,
        lambda d: score_esi(4, max(1, min(5, int(d.get("esi_level", 3))))),
    )


# ── NER ───────────────────────────────────────────────────────────────────────

_NER_SYSTEM = """\
You are a clinical information extraction engine. Extract named entities from the clinical text.
Return ONLY a JSON object with these exact keys (use the string "none" if absent):
- "patient_name"
- "age"
- "sex"
- "chief_complaint"
- "medications"
- "allergies"
- "vital_signs"
No markdown, no explanation. Return only the JSON object.\
"""


def _ner_op(name: str, text: str, expected: dict[str, str]) -> Operation:
    @operation(name)
    async def _op(llm, _text=text, _exp=expected, _name=name) -> OpResult:
        return await _run_op(
            llm, _name, _NER_SYSTEM, _text, lambda d: score_ner(_exp, d)
        )

    return _op


_ner_op(
    "ner_case1",
    "Mr. James Carter, a 67-year-old male, presents with severe shortness of breath. "
    "He takes metoprolol and furosemide daily. Allergic to aspirin. BP 150/95, HR 102, SpO2 88%.",
    {
        "patient_name": "James Carter",
        "age": "67",
        "sex": "male",
        "chief_complaint": "shortness of breath",
        "medications": "metoprolol",
        "allergies": "aspirin",
        "vital_signs": "150/95",
    },
)

_ner_op(
    "ner_case2",
    "Patient Maria Lopez, female, 34 years old, came in with a high fever and a productive cough "
    "for three days. Current medication: ibuprofen. No known drug allergies. Temperature 39.2°C, "
    "pulse 98 bpm.",
    {
        "patient_name": "Maria Lopez",
        "age": "34",
        "sex": "female",
        "chief_complaint": "fever",
        "medications": "ibuprofen",
        "allergies": "none",
        "vital_signs": "39.2",
    },
)

_ner_op(
    "ner_case3",
    "An 8-year-old boy, Daniel Kim, was brought by his mother for a peanut allergy reaction with "
    "facial swelling and hives. He uses an albuterol inhaler. Allergic to peanuts and shellfish. "
    "Blood pressure 95/60, heart rate 130.",
    {
        "patient_name": "Daniel Kim",
        "age": "8",
        "sex": "male",
        "chief_complaint": "allergic reaction",
        "medications": "albuterol",
        "allergies": "peanuts",
        "vital_signs": "95/60",
    },
)


# ── SBAR ──────────────────────────────────────────────────────────────────────

_SBAR_SYSTEM = """\
You are a clinical handoff assistant. Convert the clinical note into an SBAR handoff.
Return ONLY a JSON object with these exact keys:
- "situation"
- "background"
- "assessment"
- "recommendation"
No markdown, no explanation. Return only the JSON object.\
"""


def _sbar_op(name: str, text: str, expected: dict[str, list[str]]) -> Operation:
    @operation(name)
    async def _op(llm, _text=text, _exp=expected, _name=name) -> OpResult:
        return await _run_op(
            llm, _name, _SBAR_SYSTEM, _text, lambda d: score_sbar(_exp, d)
        )

    return _op


_sbar_op(
    "sbar_case1",
    "70-year-old man in bed 4 with worsening chest pain over the past hour. History of coronary "
    "artery disease and prior stent. ECG shows ST elevation; troponin is rising. I think he is "
    "having an acute MI and needs urgent cardiology review and cath lab activation.",
    {
        "situation": ["chest pain", "bed 4"],
        "background": ["coronary", "stent"],
        "assessment": ["mi", "myocardial", "st elevation"],
        "recommendation": ["cardiology", "cath"],
    },
)

_sbar_op(
    "sbar_case2",
    "55-year-old woman admitted with community-acquired pneumonia. She has COPD and is on home "
    "oxygen. Her oxygen saturation dropped to 84% and she is increasingly breathless. She likely "
    "has respiratory decompensation; recommend ABG, increase oxygen and consider ICU transfer.",
    {
        "situation": ["pneumonia", "breathless", "saturation"],
        "background": ["copd", "oxygen"],
        "assessment": ["respiratory", "decompensation", "hypoxia"],
        "recommendation": ["abg", "icu", "oxygen"],
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelRow:
    model: str
    n_ctx: int
    file_size_mb: float
    load_time_ms: float
    mem_after_load_mb: float
    mem_delta_mb: float
    load_error: str = ""
    ops: dict[str, OpResult] = field(default_factory=dict)


def result_columns() -> list[str]:
    cols = [
        "model",
        "n_ctx",
        "file_size_mb",
        "load_time_ms",
        "mem_after_load_mb",
        "mem_delta_mb",
        "total_ops_ms",
        "total_prep_ms",
        "total_infer_ms",
        "total_parse_ms",
        "gen_tokens",
        "mean_tok_s",
        "avg_accuracy",
        "load_error",
    ]
    for op in OPERATIONS:
        cols.append(f"{op.op_name}_ms")  # type: ignore[attr-defined]
        cols.append(f"{op.op_name}_acc")
    return cols


def row_to_dict(row: ModelRow) -> dict:
    ops = list(row.ops.values())
    total_ms = round(sum(r.time_ms for r in ops), 1)
    accs = [r.accuracy for r in ops if not r.error]
    avg_acc = round(sum(accs) / len(accs), 3) if accs else 0.0
    tok_s_vals = [r.tok_s for r in ops if r.tok_s]
    out: dict = {
        "model": row.model,
        "n_ctx": row.n_ctx,
        "file_size_mb": row.file_size_mb,
        "load_time_ms": row.load_time_ms,
        "mem_after_load_mb": row.mem_after_load_mb,
        "mem_delta_mb": row.mem_delta_mb,
        "total_ops_ms": total_ms,
        "total_prep_ms": round(sum(r.prep_ms for r in ops), 1),
        "total_infer_ms": round(sum(r.infer_ms for r in ops), 1),
        "total_parse_ms": round(sum(r.parse_ms for r in ops), 1),
        "gen_tokens": sum(r.completion_tokens or 0 for r in ops),
        "mean_tok_s": round(sum(tok_s_vals) / len(tok_s_vals), 1)
        if tok_s_vals
        else 0.0,
        "avg_accuracy": avg_acc,
        "load_error": row.load_error,
    }
    for op in OPERATIONS:
        name = op.op_name  # type: ignore[attr-defined]
        res = row.ops.get(name)
        out[f"{name}_ms"] = res.time_ms if res else ""
        out[f"{name}_acc"] = res.accuracy if res else ""
    return out


async def benchmark_model(
    Llama, spec: ModelSpec, path: Path, n_ctx: int, gpu_layers: int
) -> ModelRow:
    file_size_mb = human_mb(path.stat().st_size)
    print(f"\n=== {spec.name}  ({file_size_mb} MB)  ctx={n_ctx} ===")

    mem_before = get_rss_mb()
    row = ModelRow(
        model=spec.name,
        n_ctx=n_ctx,
        file_size_mb=file_size_mb,
        load_time_ms=0.0,
        mem_after_load_mb=0.0,
        mem_delta_mb=0.0,
    )

    # 0) Comprobar presupuesto de RAM antes de cargar nada. El overhead (KV-cache)
    #    escala con el contexto, así que lo ajustamos según el n_ctx de esta prueba.
    budget_mb = MAX_RAM_GB * 1024
    est_mb = file_size_mb + RAM_OVERHEAD_MB * (n_ctx / 4096)
    if est_mb > budget_mb:
        row.load_error = f"skipped: footprint estimado {round(est_mb)} MB > presupuesto {round(budget_mb)} MB"
        print(f"  ⨯ omitido — {row.load_error}")
        return row

    # 1) Cargar el modelo (solo CPU, sin mlock para que el SO pueda paginar)
    llm = None
    t0 = time.perf_counter()
    try:
        llm = Llama(
            model_path=str(path),
            n_ctx=n_ctx,  # contexto recomendado por modelo
            n_threads=N_THREADS,
            n_gpu_layers=gpu_layers,  # 0 = solo CPU · -1 = todas las capas a GPU
            use_mmap=True,  # mapea el GGUF; el SO pagina bajo presión de RAM
            use_mlock=False,  # no fija páginas en RAM (respeta el presupuesto)
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        row.load_error = f"{type(exc).__name__}: {exc}"[:200]
        print(f"  ⨯ error al cargar: {row.load_error}")
        return row

    # Plantilla/formato por modelo que leerá chat_json en cada inferencia.
    llm._bench_system_mode = spec.system_mode  # type: ignore[attr-defined]
    llm._bench_json_object = spec.json_object  # type: ignore[attr-defined]
    llm._bench_stream = STREAM  # type: ignore[attr-defined]

    row.load_time_ms = round((time.perf_counter() - t0) * 1000, 1)
    row.mem_after_load_mb = round(get_rss_mb(), 1)
    row.mem_delta_mb = round(row.mem_after_load_mb - mem_before, 1)
    over = " ⚠ SUPERA PRESUPUESTO" if row.mem_after_load_mb > budget_mb else ""
    if over:
        row.load_error = f"warning: RAM real {row.mem_after_load_mb} MB > presupuesto {round(budget_mb)} MB"
    accel = "GPU" if gpu_layers != 0 else "CPU"
    print(
        f"  cargado en {row.load_time_ms} ms · ctx {n_ctx} · {accel}(gpl={gpu_layers}) · "
        f"system={spec.system_mode} · RAM proceso {row.mem_after_load_mb} MB "
        f"(Δ {row.mem_delta_mb} MB){over}"
    )

    # 2) Ejecutar operaciones secuencialmente. Cada inferencia resetea la sesión
    #    (ver chat_json -> reset_llm_state), así no hay carryover entre pruebas.
    t_ops = time.perf_counter()
    for op in OPERATIONS:
        name = op.op_name  # type: ignore[attr-defined]
        res = await op(llm)
        row.ops[name] = res
        flag = (
            "✓"
            if (res.accuracy >= 0.5 and not res.error)
            else ("⨯" if res.error else "~")
        )
        detail = res.error or f"acc={res.accuracy}"
        ttft = f" · ttft {res.ttft_ms:.0f}" if res.ttft_ms is not None else ""
        toks = (
            f" · {res.completion_tokens}tok@{res.tok_s}/s"
            if res.completion_tokens
            else ""
        )
        # Desglose: ANTES (prep) · DURANTE (infer) · DESPUÉS (parse)
        print(
            f"  {flag} {name:<20} {res.time_ms:>7.0f}ms  "
            f"[prep {res.prep_ms:>4.0f} · infer {res.infer_ms:>7.0f}{ttft} · parse {res.parse_ms:>4.0f}]"
            f"{toks}   {detail}"
        )
    ops_wall_s = time.perf_counter() - t_ops
    print(
        f"  ⏱  {len(OPERATIONS)} pruebas en {ops_wall_s:.1f}s "
        f"(carga {row.load_time_ms / 1000:.1f}s · total modelo×ctx {ops_wall_s + row.load_time_ms / 1000:.1f}s)"
    )

    # 3) Descargar el modelo de cero: limpia KV-cache/sesión, cierra el contexto
    #    y libera RAM antes de cargar el siguiente modelo.
    try:
        reset_llm_state(llm)
        if hasattr(llm, "close"):
            llm.close()
    except Exception:  # noqa: BLE001
        pass

    llm = None
    gc.collect()
    return row


def _row_issues(row: ModelRow) -> list[str]:
    """Lista legible de problemas de un modelo (carga + por operación)."""
    issues: list[str] = []
    if row.load_error:
        issues.append(f"load: {row.load_error}")
    for name, res in row.ops.items():
        if res.error:
            issues.append(f"{name}: {res.error}")
    return issues


def write_error_report(
    rows: list[ModelRow], models: list[ModelSpec], out_dir: Path
) -> Path:
    """Escribe error__dd_mm_yyyy__hh_mm.json con UNA entrada por modelo.

    Cada entrada agrupa las ejecuciones por contexto (`runs`: ctx_min y ctx_max) y
    lista los problemas detectados (no descargado, error de carga, errores por
    operación) para poder corregir el script.
    """
    runs_by_model: dict[str, list[ModelRow]] = {}
    for r in rows:
        runs_by_model.setdefault(r.model, []).append(r)

    entries: list[dict] = []
    for spec in models:
        model_runs = runs_by_model.get(spec.name)
        if not model_runs:
            entries.append(
                {
                    "model": spec.name,
                    "status": "no_evaluado",
                    "ctx_min": spec.ctx_min,
                    "ctx_max": spec.ctx_max,
                    "system_mode": spec.system_mode,
                    "issues": [
                        "no disponible: no descargado o descarga desactivada/fallida"
                    ],
                    "runs": [],
                }
            )
            continue
        runs = []
        any_issue = False
        for r in sorted(model_runs, key=lambda x: x.n_ctx):
            issues = _row_issues(r)
            any_issue = any_issue or bool(issues)
            runs.append(
                {
                    "n_ctx": r.n_ctx,
                    "status": "ok" if not issues else "con_problemas",
                    "load_error": r.load_error or None,
                    "issues": issues,
                    "operations": {n: (res.error or None) for n, res in r.ops.items()},
                }
            )
        entries.append(
            {
                "model": spec.name,
                "status": "ok" if not any_issue else "con_problemas",
                "ctx_min": spec.ctx_min,
                "ctx_max": spec.ctx_max,
                "system_mode": spec.system_mode,
                "runs": runs,
            }
        )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "host": collect_system_info(),
        "config": {
            "threads": N_THREADS,
            "max_ram_gb": MAX_RAM_GB,
            "gpu_layers": resolve_gpu_layers()[0],
            "stream": STREAM,
        },
        "summary": {
            "total": len(models),
            "con_problemas": sum(1 for e in entries if e["status"] != "ok"),
        },
        "models": entries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"error__{datetime.now().strftime('%d_%m_%Y__%H_%M')}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


async def run_benchmark(models: list[ModelSpec] = MODELS):
    """Ejecuta el benchmark completo y devuelve un pandas.DataFrame.

    Por cada modelo prueba DOS contextos (ctx_min y ctx_max) en un ciclo interno y
    produce una fila por (modelo, contexto). Limpia memoria/sesión entre cada
    prueba y modelo, y al terminar libera todo (gracious cleaning). Escribe un
    error__*.json con los problemas por modelo.

    Returns:
        pandas.DataFrame con una fila por (modelo, n_ctx).
    """
    from llama_cpp import Llama  # se importa aquí para dar error claro si falta
    import pandas as pd

    print_system_info()  # info del host antes de empezar
    gpu_layers, gpu_reason = resolve_gpu_layers()  # GPU si está disponible, si no CPU
    print(f"Modelos en: {MODELS_DIR}")
    print(
        f"Hilos: {N_THREADS} · ctx por defecto: {N_CTX} · n_gpu_layers: {gpu_layers} ({gpu_reason})"
    )
    print(
        f"Presupuesto RAM: {MAX_RAM_GB} GB · streaming: {STREAM} (cont. máx {MAX_CONTINUATIONS})"
    )
    print(f"Operaciones registradas: {[op.op_name for op in OPERATIONS]}")  # type: ignore[attr-defined]

    columns = result_columns()
    rows: list[dict] = []
    model_rows: list[ModelRow] = []
    t_run = time.perf_counter()
    try:
        for spec in models:
            print(f"\n\n ===== {spec.name} ===== ")
            path = ensure_model(spec)
            if path is None:
                continue
            # Ciclo interno: probar en contexto mínimo y máximo (dedup si coinciden)
            for n_ctx in sorted({spec.ctx_min, spec.ctx_max}):
                row = await benchmark_model(Llama, spec, path, n_ctx, gpu_layers)
                model_rows.append(row)
                rows.append(row_to_dict(row))
                gc.collect()  # asegura liberar entre pruebas de contexto
    finally:
        # Gracious cleaning: libera toda la memoria de modelos/pruebas al terminar.
        gc.collect()
        gc.collect()

    run_wall_s = time.perf_counter() - t_run
    error_path = write_error_report(model_rows, models, OUTPUT_DIR)
    print(
        f"\n✓ Benchmark completo: {len(rows)} ejecuciones (modelo × contexto) "
        f"en {run_wall_s / 60:.1f} min"
    )
    print(f"✓ Reporte de errores: {error_path}")
    return pd.DataFrame(rows, columns=columns)


async def main() -> int:
    try:
        df = await run_benchmark()
    except ImportError as exc:
        print(f"ERROR: falta una dependencia ({exc.name}).")
        print("  Instala con: pip install llama-cpp-python pandas")
        return 1

    print("\n" + df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
