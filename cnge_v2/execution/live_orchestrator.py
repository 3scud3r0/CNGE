import asyncio

from .streaming_pipeline import StreamingEpistemicPipeline


class LiveOrchestrator:
    """Explicit bounded ingress, ordered single-worker updates and propagated errors."""

    def __init__(self, pipeline: StreamingEpistemicPipeline, capacity=64, callback=None):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.pipeline, self.callback = pipeline, callback
        self.capacity = capacity
        self.queue = asyncio.Queue(maxsize=capacity)
        self._running = False

    async def run(self, source):
        if self._running:
            raise RuntimeError("orchestrator already running")
        self._running = True
        self.queue = asyncio.Queue(maxsize=self.capacity)
        sentinel = object()

        async def produce():
            async for observation in source:
                await self.queue.put(observation)
            await self.queue.put(sentinel)

        async def consume():
            while True:
                observation = await self.queue.get()
                try:
                    if observation is sentinel:
                        return
                    telemetry = await self.pipeline.ingest_sensor_tick(observation)
                    if self.callback:
                        result = self.callback(telemetry)
                        if hasattr(result, "__await__"):
                            await result
                    await asyncio.sleep(0)
                finally:
                    self.queue.task_done()

        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(produce(), name="cnge-v2-ingestion")
                group.create_task(consume(), name="cnge-v2-learning")
        finally:
            self._running = False
