import asyncio
import inspect
import json

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


class InteractiveCanvasServer:
    """Loopback WebSocket transport, bounded messages and latest-frame delivery."""

    def __init__(self, port=8080, host="127.0.0.1"):
        self.port, self.host = port, host
        self.clients = {}
        self.callbacks = []
        self.server = None

    async def start(self):
        if self.server is not None:
            raise RuntimeError("server already started")
        self.server = await serve(
            self._handler,
            self.host,
            self.port,
            max_size=65536,
            max_queue=16,
            origins=[
                None,
                f"http://{self.host}:{self.port + 1}",
                f"http://localhost:{self.port + 1}",
            ],
        )
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def _handler(self, socket):
        queue = asyncio.Queue(maxsize=1)
        self.clients[socket] = queue

        async def send():
            while True:
                await socket.send(await queue.get())

        sender = asyncio.create_task(send())
        try:
            async for message in socket:
                try:
                    packet = json.loads(message)
                    if not isinstance(packet, dict):
                        raise TypeError("interaction must be an object")
                    for callback in self.callbacks:
                        result = callback(packet)
                        if inspect.isawaitable(result):
                            await result
                except (ValueError, TypeError, KeyError, asyncio.QueueFull) as exc:
                    await socket.send(json.dumps({"error": str(exc)}))
        except ConnectionClosed:
            pass
        finally:
            self.clients.pop(socket, None)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    async def broadcast_phase_frame(self, frame_data):
        message = json.dumps(frame_data, allow_nan=False)
        for queue in tuple(self.clients.values()):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(message)

    def on_user_interaction(self, callback):
        self.callbacks.append(callback)

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
