#!/usr/bin/env python3
"""Loopback proxy that adds per-case isolation to DeepSeek Responses calls."""

from __future__ import annotations

import hmac
import http.client
import json
import re
import shutil
import socket
import socketserver
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import urlsplit


USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,512}$")
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
UPSTREAM_RESPONSE_HEADER_ALLOWLIST = {
    "content-type",
    "retry-after",
    "request-id",
    "x-request-id",
    "openai-request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never copy the host API credential to a redirected origin."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _open_upstream(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> Any:
    # HTTPS_PROXY is a trusted host-network dependency in the benchmark
    # environment. urllib uses CONNECT for an HTTPS upstream, so the DeepSeek
    # Authorization header remains inside the TLS tunnel. Redirects stay
    # disabled independently below.
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(),
        _NoRedirectHandler(),
    )
    return opener.open(request, timeout=timeout)


def _set_upstream_read_timeout(response: Any, timeout: float) -> None:
    """Best-effort bound for urllib's underlying socket on every stream read."""
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is not None:
        sock.settimeout(max(0.001, timeout))


@dataclass(frozen=True)
class ProxyLimits:
    """Finite per-case resource limits for the model broker."""

    max_requests: int = 64
    max_inbound_requests: int = 128
    max_connection_attempts: int = 128
    max_request_bytes: int = 2 * 1024 * 1024
    max_total_request_bytes: int = 32 * 1024 * 1024
    max_response_bytes: int = 32 * 1024 * 1024
    max_total_response_bytes: int = 256 * 1024 * 1024
    # Codex may briefly overlap the next Responses turn with cleanup of the
    # preceding streamed response.  Keep this finite and per case, but allow
    # that normal two-request handoff instead of manufacturing a local 429.
    max_concurrent_requests: int = 2
    max_connections: int = 4
    max_output_tokens: int = 32_768
    max_total_requested_output_tokens: int = 1_048_576
    max_lifetime_seconds: float = 660.0
    upstream_timeout_seconds: float = 600.0
    client_idle_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"Proxy limit {name} must be positive")
        if self.max_request_bytes > self.max_total_request_bytes:
            raise ValueError("max_request_bytes cannot exceed max_total_request_bytes")
        if self.max_response_bytes > self.max_total_response_bytes:
            raise ValueError("max_response_bytes cannot exceed max_total_response_bytes")
        if self.max_requests > self.max_inbound_requests:
            raise ValueError("max_requests cannot exceed max_inbound_requests")
        if self.max_inbound_requests > self.max_connection_attempts:
            raise ValueError(
                "max_inbound_requests cannot exceed max_connection_attempts"
            )
        if self.max_output_tokens > self.max_total_requested_output_tokens:
            raise ValueError(
                "max_output_tokens cannot exceed max_total_requested_output_tokens"
            )


class _ProxyState:
    def __init__(self, limits: ProxyLimits):
        self.limits = limits
        self.started_monotonic = time.monotonic()
        self.deadline_monotonic = self.started_monotonic + limits.max_lifetime_seconds
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(limits.max_concurrent_requests)
        self.connection_slots = threading.BoundedSemaphore(
            limits.max_connections
        )
        self.stats: dict[str, Any] = {
            "limits": asdict(limits),
            "requests_seen": 0,
            "connection_attempts": 0,
            "requests_accepted": 0,
            "requests_completed": 0,
            "requests_rejected": 0,
            "upstream_requests": 0,
            "upstream_status_counts": {},
            "request_bytes": 0,
            "response_bytes": 0,
            "requested_output_tokens": 0,
            "active_requests": 0,
            "peak_concurrent_requests": 0,
            "response_limit_exceeded": 0,
            "response_deadline_exceeded": 0,
            "upstream_stream_failures": 0,
            "client_stream_aborts": 0,
            "rejections": {},
        }

    def admit_request(self) -> tuple[bool, str | None]:
        with self.lock:
            if time.monotonic() >= self.deadline_monotonic:
                return False, "deadline_exceeded"
            if self.stats["requests_seen"] >= self.limits.max_inbound_requests:
                return False, "inbound_request_limit"
            self.stats["requests_seen"] += 1
        return True, None

    def admit_connection(self) -> tuple[bool, str | None]:
        with self.lock:
            if time.monotonic() >= self.deadline_monotonic:
                return False, "deadline_exceeded"
            if self.stats["connection_attempts"] >= self.limits.max_connection_attempts:
                return False, "connection_attempt_limit"
            self.stats["connection_attempts"] += 1
            return True, None

    def reject(self, reason: str) -> None:
        with self.lock:
            self.stats["requests_rejected"] += 1
            rejections = self.stats["rejections"]
            rejections[reason] = rejections.get(reason, 0) + 1

    def begin(self, length: int) -> tuple[bool, str | None]:
        if not self.slots.acquire(blocking=False):
            return False, "concurrency_limit"
        with self.lock:
            if time.monotonic() >= self.deadline_monotonic:
                self.slots.release()
                return False, "deadline_exceeded"
            if self.stats["requests_accepted"] >= self.limits.max_requests:
                self.slots.release()
                return False, "request_count_limit"
            if self.stats["request_bytes"] + length > self.limits.max_total_request_bytes:
                self.slots.release()
                return False, "total_request_bytes_limit"
            self.stats["requests_accepted"] += 1
            self.stats["request_bytes"] += length
            self.stats["active_requests"] += 1
            self.stats["peak_concurrent_requests"] = max(
                self.stats["peak_concurrent_requests"],
                self.stats["active_requests"],
            )
        return True, None

    def finish(self) -> None:
        with self.lock:
            self.stats["active_requests"] -= 1
            self.stats["requests_completed"] += 1
        self.slots.release()

    def upstream_timeout(self) -> float | None:
        remaining = self.deadline_monotonic - time.monotonic()
        if remaining <= 0:
            return None
        return min(self.limits.upstream_timeout_seconds, remaining)

    def note_upstream_request(self) -> None:
        with self.lock:
            self.stats["upstream_requests"] += 1

    def note_upstream_status(self, status: int) -> None:
        with self.lock:
            counts = self.stats["upstream_status_counts"]
            key = str(status)
            counts[key] = counts.get(key, 0) + 1

    def claim_output_tokens(self, size: int) -> bool:
        with self.lock:
            if (
                self.stats["requested_output_tokens"] + size
                > self.limits.max_total_requested_output_tokens
            ):
                return False
            self.stats["requested_output_tokens"] += size
            return True

    def remaining_lifetime(self) -> float:
        return self.deadline_monotonic - time.monotonic()

    def note_response_deadline(self) -> None:
        with self.lock:
            self.stats["response_deadline_exceeded"] += 1

    def note_upstream_stream_failure(self) -> None:
        with self.lock:
            self.stats["upstream_stream_failures"] += 1

    def note_client_stream_abort(self) -> None:
        with self.lock:
            self.stats["client_stream_aborts"] += 1

    def claim_response_bytes(self, size: int, response_bytes: int) -> bool:
        with self.lock:
            if (
                response_bytes + size > self.limits.max_response_bytes
                or self.stats["response_bytes"] + size
                > self.limits.max_total_response_bytes
            ):
                self.stats["response_limit_exceeded"] += 1
                return False
            self.stats["response_bytes"] += size
            return True

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.stats))


def validate_user_id(user_id: str) -> str:
    if not USER_ID_RE.fullmatch(user_id):
        raise ValueError(
            "DeepSeek user id must match [A-Za-z0-9_-]+ and be at most 512 bytes"
        )
    return user_id


def inject_responses_user(
    body: bytes,
    user_id: str,
    expected_model: str | None = None,
    max_output_tokens: int | None = None,
    upstream_model: str | None = None,
) -> bytes:
    """Return a stateless Responses request tagged for one benchmark case."""
    validate_user_id(user_id)
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Responses request body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Responses request body must be a JSON object")
    if expected_model is not None and payload.get("model") != expected_model:
        raise ValueError(
            f"Responses model must be exactly {expected_model!r}"
        )
    if upstream_model is not None:
        if not isinstance(upstream_model, str) or not upstream_model.strip():
            raise ValueError("Upstream Responses model must be non-empty")
        payload["model"] = upstream_model
    if max_output_tokens is not None:
        requested_tokens = payload.get("max_output_tokens", max_output_tokens)
        if (
            isinstance(requested_tokens, bool)
            or not isinstance(requested_tokens, int)
            or requested_tokens < 1
            or requested_tokens > max_output_tokens
        ):
            raise ValueError(
                f"max_output_tokens must be an integer from 1 to {max_output_tokens}"
            )
        payload["max_output_tokens"] = requested_tokens
    for continuity_field in ("previous_response_id", "conversation"):
        if payload.get(continuity_field) is not None:
            raise ValueError(
                f"Stateful field {continuity_field!r} is forbidden in benchmark runs"
            )
    # Persistence and provider-side cache routing are never caller-controlled
    # in a stateless benchmark.  Force supported booleans off and remove cache
    # affinity fields even when an agent constructs its own loopback request.
    payload["store"] = False
    if "background" in payload:
        payload["background"] = False
    for cache_field in (
        "prompt_cache_key",
        "prompt_cache_retention",
        "cache_key",
    ):
        payload.pop(cache_field, None)
    payload["user"] = user_id
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _configure_proxy_server(
    server: object,
    upstream_base_url: str,
    user_id: str,
    upstream_api_key: str,
    downstream_token: str,
    expected_model: str,
    upstream_model: str | None,
    limits: ProxyLimits,
) -> None:
    parsed = urlsplit(upstream_base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "DeepSeek upstream must be an absolute HTTPS URL without "
            "credentials, query, or fragment"
        )
    if not isinstance(upstream_api_key, str) or not upstream_api_key:
        raise ValueError("DeepSeek upstream API key must be non-empty")
    if not isinstance(downstream_token, str) or not downstream_token:
        raise ValueError("DeepSeek downstream token must be non-empty")
    if not isinstance(expected_model, str) or not expected_model:
        raise ValueError("DeepSeek expected model must be non-empty")
    server.upstream_host = parsed.hostname  # type: ignore[attr-defined]
    server.upstream_port = parsed.port or 443  # type: ignore[attr-defined]
    server.upstream_origin = f"https://{parsed.netloc}"  # type: ignore[attr-defined]
    server.upstream_prefix = parsed.path.rstrip("/")  # type: ignore[attr-defined]
    server.user_id = validate_user_id(user_id)  # type: ignore[attr-defined]
    server.upstream_api_key = upstream_api_key  # type: ignore[attr-defined]
    server.downstream_token = downstream_token  # type: ignore[attr-defined]
    server.expected_model = expected_model  # type: ignore[attr-defined]
    server.upstream_model = upstream_model  # type: ignore[attr-defined]
    server.proxy_state = _ProxyState(limits)  # type: ignore[attr-defined]


class _BoundedServerMixin:
    def process_request(self, request: socket.socket, client_address: Any) -> None:
        state = self.proxy_state  # type: ignore[attr-defined]
        admitted, reason = state.admit_connection()
        if not admitted:
            assert reason is not None
            state.reject(reason)
            self.shutdown_request(request)  # type: ignore[attr-defined]
            self._stop_accepting_after_limit()
            return
        if not state.connection_slots.acquire(blocking=False):
            state.reject("connection_limit")
            self.shutdown_request(request)  # type: ignore[attr-defined]
            return
        try:
            super().process_request(request, client_address)  # type: ignore[misc]
        except BaseException:
            state.connection_slots.release()
            raise

    def _stop_accepting_after_limit(self) -> None:
        lock = getattr(self, "_limit_shutdown_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._limit_shutdown_lock = lock  # type: ignore[attr-defined]
            self._limit_shutdown_started = False  # type: ignore[attr-defined]
        with lock:
            if self._limit_shutdown_started:  # type: ignore[attr-defined]
                return
            self._limit_shutdown_started = True  # type: ignore[attr-defined]
            threading.Thread(
                target=self.shutdown,  # type: ignore[attr-defined]
                name="deepseek-proxy-limit-shutdown",
                daemon=True,
            ).start()

    def process_request_thread(
        self,
        request: socket.socket,
        client_address: Any,
    ) -> None:
        try:
            super().process_request_thread(request, client_address)  # type: ignore[misc]
        finally:
            self.proxy_state.connection_slots.release()  # type: ignore[attr-defined]


class _ProxyServer(_BoundedServerMixin, ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        upstream_base_url: str,
        user_id: str,
        upstream_api_key: str,
        downstream_token: str,
        expected_model: str,
        upstream_model: str | None,
        limits: ProxyLimits,
    ):
        _configure_proxy_server(
            self,
            upstream_base_url,
            user_id,
            upstream_api_key,
            downstream_token,
            expected_model,
            upstream_model,
            limits,
        )
        super().__init__(("127.0.0.1", 0), _ProxyHandler)


class _UnixProxyServer(
    _BoundedServerMixin,
    socketserver.ThreadingMixIn,
    socketserver.UnixStreamServer,
):
    daemon_threads = True

    def __init__(
        self,
        socket_path: str,
        upstream_base_url: str,
        user_id: str,
        upstream_api_key: str,
        downstream_token: str,
        expected_model: str,
        upstream_model: str | None,
        limits: ProxyLimits,
    ):
        _configure_proxy_server(
            self,
            upstream_base_url,
            user_id,
            upstream_api_key,
            downstream_token,
            expected_model,
            upstream_model,
            limits,
        )
        super().__init__(socket_path, _ProxyHandler)


class _ProxyHandler(BaseHTTPRequestHandler):
    server: _ProxyServer | _UnixProxyServer
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        self.request.settimeout(
            self.server.proxy_state.limits.client_idle_timeout_seconds
        )
        super().setup()

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._forward()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        admitted, reason = self.server.proxy_state.admit_request()
        if not admitted:
            assert reason is not None
            self.server.proxy_state.reject(reason)
            status = 408 if reason == "deadline_exceeded" else 429
            self._send_local_error(status, f"Proxy limit exceeded: {reason}")
            return
        self.server.proxy_state.reject("method")
        self._send_local_error(405, "Only POST Responses requests are allowed")

    do_DELETE = do_GET
    do_HEAD = do_GET
    do_OPTIONS = do_GET
    do_PATCH = do_GET
    do_PUT = do_GET

    def _validated_content_length(self) -> int:
        if self.headers.get("Transfer-Encoding") is not None:
            raise ValueError("Transfer-Encoding is not allowed")
        values = self.headers.get_all("Content-Length", failobj=[])
        if len(values) != 1 or re.fullmatch(r"[0-9]+", values[0]) is None:
            raise ValueError("Exactly one valid Content-Length is required")
        length = int(values[0])
        if length > self.server.proxy_state.limits.max_request_bytes:
            raise OverflowError("Request body exceeds the per-request byte limit")
        return length

    def _send_local_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _forward(self) -> None:
        state = self.server.proxy_state
        admitted, admission_reason = state.admit_request()
        if not admitted:
            assert admission_reason is not None
            state.reject(admission_reason)
            status = 408 if admission_reason == "deadline_exceeded" else 429
            self._send_local_error(
                status,
                f"Proxy limit exceeded: {admission_reason}",
            )
            return
        begun = False
        try:
            # Validate all request-line/header policy before consuming any body bytes.
            if self.path != "/responses":
                state.reject("route")
                self._send_local_error(404, "Only the exact /responses endpoint is available")
                return
            authorization_values = self.headers.get_all("Authorization", failobj=[])
            supplied = authorization_values[0] if len(authorization_values) == 1 else ""
            expected = f"Bearer {self.server.downstream_token}"
            if not hmac.compare_digest(supplied, expected):
                state.reject("authorization")
                self._send_local_error(401, "Invalid downstream credential")
                return
            length = self._validated_content_length()
            begun, limit_reason = state.begin(length)
            if not begun:
                assert limit_reason is not None
                state.reject(limit_reason)
                status = 408 if limit_reason == "deadline_exceeded" else 429
                self._send_local_error(status, f"Proxy limit exceeded: {limit_reason}")
                return
            self.connection.settimeout(state.limits.client_idle_timeout_seconds)
            try:
                body = self.rfile.read(length)
            except (OSError, TimeoutError):
                state.reject("client_body_failure")
                try:
                    self._send_local_error(
                        408,
                        "Downstream request body timed out",
                    )
                except OSError:
                    self.close_connection = True
                return
            if len(body) != length:
                raise ValueError("Request body ended before Content-Length bytes arrived")
            body = inject_responses_user(
                body,
                self.server.user_id,
                self.server.expected_model,
                state.limits.max_output_tokens,
                self.server.upstream_model,
            )
            requested_output_tokens = json.loads(body)["max_output_tokens"]
            if not state.claim_output_tokens(requested_output_tokens):
                state.reject("total_output_token_limit")
                self._send_local_error(
                    429,
                    "Proxy limit exceeded: total_output_token_limit",
                )
                return
            upstream_path = f"{self.server.upstream_prefix}/responses"
            headers = {
                "Authorization": f"Bearer {self.server.upstream_api_key}",
                "Content-Length": str(len(body)),
                "Content-Type": "application/json",
            }
            accept_values = self.headers.get_all("Accept", failobj=[])
            if (
                len(accept_values) == 1
                and 0 < len(accept_values[0]) <= 256
            ):
                headers["Accept"] = accept_values[0]
            request = urllib.request.Request(
                f"{self.server.upstream_origin}{upstream_path}",
                data=body,
                headers=headers,
                method="POST",
            )
            timeout = state.upstream_timeout()
            if timeout is None:
                raise TimeoutError("Per-case proxy deadline exceeded")
            state.note_upstream_request()
            try:
                response = _open_upstream(request, timeout=timeout)
            except urllib.error.HTTPError as exc:
                response = exc
            state.note_upstream_status(int(response.status))
        except OverflowError as exc:
            state.reject("request_body_limit")
            self._send_local_error(413, str(exc))
            return
        except ValueError as exc:
            state.reject("invalid_request")
            self._send_local_error(400, str(exc))
            return
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            state.reject("upstream_failure")
            self._send_local_error(502, f"DeepSeek upstream request failed: {type(exc).__name__}")
            return
        finally:
            if begun and "response" not in locals():
                state.finish()

        response_bytes = 0
        response_truncated = False
        try:
            self.send_response(response.status, response.reason)
            for key, value in response.headers.items():
                if key.lower() in UPSTREAM_RESPONSE_HEADER_ALLOWLIST:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.close_connection = True
            read_chunk = getattr(response, "read1", response.read)
            while True:
                remaining_lifetime = state.remaining_lifetime()
                if remaining_lifetime <= 0:
                    state.note_response_deadline()
                    response_truncated = True
                    self.close_connection = True
                    break
                try:
                    _set_upstream_read_timeout(
                        response, remaining_lifetime
                    )
                    chunk = read_chunk(64 * 1024)
                except (OSError, http.client.HTTPException):
                    state.note_upstream_stream_failure()
                    response_truncated = True
                    self.close_connection = True
                    break
                if not chunk:
                    break
                if not state.claim_response_bytes(len(chunk), response_bytes):
                    response_truncated = True
                    self.close_connection = True
                    break
                response_bytes += len(chunk)
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            if not response_truncated:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        except (BrokenPipeError, ConnectionResetError, OSError):
            if state.remaining_lifetime() <= 0:
                state.note_response_deadline()
            else:
                state.note_client_stream_abort()
        finally:
            try:
                response.close()
            finally:
                state.finish()


class DeepSeekResponsesProxy:
    """A quiet, loopback-only proxy with a fixed anonymous case identity."""

    def __init__(
        self,
        upstream_base_url: str,
        user_id: str,
        *,
        upstream_api_key: str,
        downstream_token: str,
        expected_model: str,
        upstream_model: str | None = None,
        unix_socket: bool = False,
        limits: ProxyLimits | None = None,
    ):
        self._broker_dir: str | None = None
        self._socket_path: str | None = None
        if unix_socket:
            self._broker_dir = tempfile.mkdtemp(
                prefix="safeact_model_broker_"
            )
            self._socket_path = f"{self._broker_dir}/responses.sock"
            self._server = _UnixProxyServer(
                self._socket_path,
                upstream_base_url,
                user_id,
                upstream_api_key,
                downstream_token,
                expected_model,
                upstream_model,
                limits or ProxyLimits(),
            )
        else:
            self._server = _ProxyServer(
                upstream_base_url,
                user_id,
                upstream_api_key,
                downstream_token,
                expected_model,
                upstream_model,
                limits or ProxyLimits(),
            )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"deepseek-proxy-{user_id}",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        if self._socket_path is not None:
            raise RuntimeError("Unix-socket proxy has no host TCP base URL")
        host, port = self._server.server_address
        return f"http://{host}:{port}/"

    @property
    def broker_dir(self) -> str | None:
        return self._broker_dir

    @property
    def socket_name(self) -> str | None:
        return "responses.sock" if self._socket_path is not None else None

    @property
    def stats(self) -> dict[str, Any]:
        return self._server.proxy_state.snapshot()

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        if self._broker_dir is not None:
            shutil.rmtree(self._broker_dir, ignore_errors=True)


@contextmanager
def isolated_responses_proxy(
    upstream_base_url: str,
    user_id: str,
    *,
    upstream_api_key: str,
    downstream_token: str,
    expected_model: str,
    upstream_model: str | None = None,
    unix_socket: bool = False,
    limits: ProxyLimits | None = None,
) -> Iterator[DeepSeekResponsesProxy]:
    proxy = DeepSeekResponsesProxy(
        upstream_base_url,
        user_id,
        upstream_api_key=upstream_api_key,
        downstream_token=downstream_token,
        expected_model=expected_model,
        upstream_model=upstream_model,
        unix_socket=unix_socket,
        limits=limits,
    )
    proxy.start()
    try:
        yield proxy
    finally:
        proxy.close()
