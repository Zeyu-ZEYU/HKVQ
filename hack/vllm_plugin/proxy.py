"""HTTP proxy for disaggregated serving (OpenAI completions API).

For every request the proxy
  1. picks the prefill instance and the decode instance with the fewest queued tokens,
  2. runs the prompt on the prefill instance with `max_tokens=1`; the response carries the
     first generated token and the `kv_transfer_params` that locate the staged KV,
  3. sends prompt + first token to the decode instance together with these parameters and
     relays its output, preceded by the first token.
A streamed response starts when the decode instance delivers its first chunk.

    python -m hack.vllm_plugin.proxy --port 8000 \
        --prefill http://127.0.0.1:8100 --decode http://127.0.0.1:8200 --log-path proxy.jsonl
"""

import argparse
import json
import time
import uuid
from dataclasses import dataclass

import aiohttp
from aiohttp import web

REQUEST_ID_PREFIX = "hk"


@dataclass
class Instance:
    url: str
    queued_tokens: int = 0
    served: int = 0


def new_request_id() -> str:
    return REQUEST_ID_PREFIX + uuid.uuid4().hex[:12]


def shortest_queue(instances: list[Instance]) -> Instance:
    return min(instances, key=lambda instance: (instance.queued_tokens, instance.served))


def estimate_prompt_tokens(prompt) -> int:
    if isinstance(prompt, list):
        return len(prompt)
    return max(1, len(str(prompt)) // 4)


class Proxy:
    def __init__(self, prefill_urls: list[str], decode_urls: list[str], log_path: str | None):
        self.prefill = [Instance(url.rstrip("/")) for url in prefill_urls]
        self.decode = [Instance(url.rstrip("/")) for url in decode_urls]
        self.log_path = log_path
        self.session: aiohttp.ClientSession | None = None

    async def start(self, app: web.Application) -> None:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
        # One connection per request: a pooled connection can be closed by the server while it is being reused.
        connector = aiohttp.TCPConnector(limit=0, force_close=True)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def stop(self, app: web.Application) -> None:
        await self.session.close()

    def log(self, record: dict) -> None:
        if self.log_path:
            with open(self.log_path, "a") as stream:
                stream.write(json.dumps(record) + "\n")

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"prefill": [vars(i) for i in self.prefill], "decode": [vars(i) for i in self.decode]})

    async def models(self, request: web.Request) -> web.Response:
        async with self.session.get(self.decode[0].url + "/v1/models") as response:
            return web.json_response(await response.json(), status=response.status)

    async def completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        request_id = request.headers.get("X-Request-Id") or new_request_id()
        record = {"request_id": request_id, "arrival": time.time()}
        prompt_estimate = estimate_prompt_tokens(body.get("prompt"))
        max_tokens = int(body.get("max_tokens") or 16)

        prefill = shortest_queue(self.prefill)
        prefill.queued_tokens += prompt_estimate
        try:
            first = await self._prefill(prefill, body, request_id, record)
        finally:
            prefill.queued_tokens -= prompt_estimate
            prefill.served += 1
        choice = first["choices"][0]
        if max_tokens <= 1 or choice.get("finish_reason") == "stop" or not first.get("kv_transfer_params"):
            record["done"] = time.time()
            self.log(record)
            return web.json_response(first)

        decode = shortest_queue(self.decode)
        load = len(choice["prompt_token_ids"]) + max_tokens
        decode.queued_tokens += load
        try:
            return await self._decode(decode, request, body, first, request_id, record)
        finally:
            decode.queued_tokens -= load
            decode.served += 1
            record["done"] = time.time()
            self.log(record)

    async def _prefill(self, instance: Instance, body: dict, request_id: str, record: dict) -> dict:
        payload = dict(body)
        payload.update(
            max_tokens=1, stream=False, return_token_ids=True, kv_transfer_params={"do_remote_decode": True}
        )
        payload.pop("stream_options", None)
        payload.pop("min_tokens", None)
        record.update(prefill_instance=instance.url, prefill_sent=time.time())
        async with self.session.post(
            instance.url + "/v1/completions", json=payload, headers={"X-Request-Id": request_id}
        ) as response:
            result = await response.json()
            if response.status != 200:
                raise web.HTTPBadGateway(text=json.dumps(result))
        record["prefill_done"] = time.time()
        return result

    async def _decode(
        self, instance: Instance, request: web.Request, body: dict, first: dict, request_id: str, record: dict
    ) -> web.StreamResponse:
        choice = first["choices"][0]
        first_token, first_text = choice["token_ids"][0], choice["text"]
        payload = dict(body)
        payload.update(
            prompt=list(choice["prompt_token_ids"]) + [first_token],
            max_tokens=int(body.get("max_tokens") or 16) - 1,
            kv_transfer_params=first["kv_transfer_params"],
        )
        if "min_tokens" in payload:
            payload["min_tokens"] = max(0, int(payload["min_tokens"]) - 1)
        record.update(decode_instance=instance.url, decode_sent=time.time())
        record["prompt_tokens"] = len(choice["prompt_token_ids"])
        headers = {"X-Request-Id": request_id}
        async with self.session.post(instance.url + "/v1/completions", json=payload, headers=headers) as response:
            if response.status != 200:
                raise web.HTTPBadGateway(text=await response.text())
            if not body.get("stream"):
                result = await response.json()
                record["first_decode_output"] = time.time()
                return web.json_response(self._merge(result, first_token, first_text, body))
            return await self._relay_stream(request, response, first, record)

    @staticmethod
    def _merge(result: dict, first_token: int, first_text: str, body: dict) -> dict:
        choice = result["choices"][0]
        choice["text"] = first_text + choice["text"]
        if choice.get("token_ids") is not None:
            choice["token_ids"] = [first_token] + choice["token_ids"]
        if choice.get("prompt_token_ids") is not None:
            choice["prompt_token_ids"] = choice["prompt_token_ids"][:-1]
        usage = result.get("usage")
        if usage:
            usage["prompt_tokens"] -= 1
            usage["completion_tokens"] += 1
        return result

    async def _relay_stream(
        self, request: web.Request, upstream: aiohttp.ClientResponse, first: dict, record: dict
    ) -> web.StreamResponse:
        downstream = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await downstream.prepare(request)
        choice = first["choices"][0]
        head = {
            "id": first["id"],
            "object": "text_completion",
            "created": first["created"],
            "model": first["model"],
            "choices": [
                {"index": 0, "text": choice["text"], "token_ids": choice["token_ids"], "logprobs": None,
                 "finish_reason": None}
            ],
        }  # fmt: skip
        started = False
        async for line in upstream.content:
            if not started and line.startswith(b"data:"):
                started = True
                record["first_decode_output"] = time.time()
                await downstream.write(b"data: " + json.dumps(head).encode() + b"\n\n")
            await downstream.write(line)
        await downstream.write_eof()
        return downstream


def build_app(prefill_urls: list[str], decode_urls: list[str], log_path: str | None = None) -> web.Application:
    proxy = Proxy(prefill_urls, decode_urls, log_path)
    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.stop)
    app.router.add_post("/v1/completions", proxy.completions)
    app.router.add_get("/v1/models", proxy.models)
    app.router.add_get("/health", proxy.health)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefill", action="append", required=True, help="URL of a prefill instance (repeatable)")
    parser.add_argument("--decode", action="append", required=True, help="URL of a decode instance (repeatable)")
    parser.add_argument("--log-path", default=None, help="JSON-lines file with one record per request")
    args = parser.parse_args()
    web.run_app(build_app(args.prefill, args.decode, args.log_path), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
