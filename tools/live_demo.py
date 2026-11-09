"""Serve the local canvas and a finite or continuous oscillator telemetry simulation."""

import argparse
import asyncio
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from cnge_engine import EngineConfig
from cnge_engine.streaming.asynchronous_orchestrator import EpistemicContinuousOrchestrator
from cnge_engine.visualization.interactive_canvas_bridge import InteractiveCanvasServer


async def run(port=8080, steps=0):
    canvas = await InteractiveCanvasServer(port).start()
    runtime = EpistemicContinuousOrchestrator(EngineConfig(spatial_dim=2), canvas=canvas)
    canvas.on_user_interaction(runtime.inject_user_intervention)
    web_root = Path(__file__).resolve().parents[1] / "web"
    http = ThreadingHTTPServer(
        ("127.0.0.1", port + 1), partial(SimpleHTTPRequestHandler, directory=str(web_root))
    )
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    print(f"Canvas HTTP port: {port + 1}; WebSocket port: {port}", flush=True)

    async def source():
        step = 0
        while steps == 0 or step < steps:
            yield [np.cos(step * 0.01), np.sin(step * 0.01)]
            step += 1
            await asyncio.sleep(0.01)

    try:
        await runtime.start_epistemic_runtime(source())
    finally:
        await runtime.stop()
        await canvas.close()
        await asyncio.to_thread(http.shutdown)
        http.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--steps", type=int, default=0, help="0 runs until interrupted")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.port, args.steps))
    except KeyboardInterrupt:
        pass
