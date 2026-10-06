#!/usr/bin/env python3
"""Bridge isolated-loopback HTTP connections to one trusted Unix socket."""

from __future__ import annotations

import argparse
import selectors
import socket
import socketserver
import subprocess
import sys
import threading
import time
from typing import Any


DEFAULT_MAX_CONNECTIONS = 4
DEFAULT_REQUEST_QUEUE_SIZE = 8
DEFAULT_IDLE_TIMEOUT_SECONDS = 30.0
DEFAULT_CONNECTION_LIFETIME_SECONDS = 660.0


class _BridgeHandler(socketserver.BaseRequestHandler):
    unix_socket_path: str

    def handle(self) -> None:
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        selector = selectors.DefaultSelector()
        deadline = (
            time.monotonic()
            + self.server.max_connection_lifetime_seconds  # type: ignore[attr-defined]
        )
        initial_idle_timeout = self.server.idle_timeout_seconds  # type: ignore[attr-defined]
        active_idle_timeout = self.server.active_idle_timeout_seconds  # type: ignore[attr-defined]
        traffic_started = False
        try:
            upstream.settimeout(
                min(
                    initial_idle_timeout,
                    max(0.001, deadline - time.monotonic()),
                )
            )
            upstream.connect(self.server.unix_socket_path)  # type: ignore[attr-defined]
            selector.register(self.request, selectors.EVENT_READ, upstream)
            selector.register(upstream, selectors.EVENT_READ, self.request)
            last_activity = time.monotonic()
            while selector.get_map():
                now = time.monotonic()
                remaining_lifetime = deadline - now
                idle_timeout = (
                    active_idle_timeout
                    if traffic_started
                    else initial_idle_timeout
                )
                remaining_idle = idle_timeout - (now - last_activity)
                wait_for = min(remaining_lifetime, remaining_idle)
                if wait_for <= 0:
                    break
                events = selector.select(timeout=wait_for)
                if not events:
                    break
                for key, _mask in events:
                    source = key.fileobj
                    destination = key.data
                    data = source.recv(64 * 1024)
                    if not data:
                        selector.unregister(source)
                        try:
                            destination.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        continue
                    traffic_started = True
                    last_activity = time.monotonic()
                    remaining_lifetime = deadline - last_activity
                    if remaining_lifetime <= 0:
                        return
                    destination.settimeout(min(idle_timeout, remaining_lifetime))
                    destination.sendall(data)
        except OSError:
            # A reset/timeout is an ordinary end to one untrusted relay.
            pass
        finally:
            selector.close()
            upstream.close()


class _ThreadingBridge(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        unix_socket_path: str,
        *,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        request_queue_size: int = DEFAULT_REQUEST_QUEUE_SIZE,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        active_idle_timeout_seconds: float | None = None,
        max_connection_lifetime_seconds: float = DEFAULT_CONNECTION_LIFETIME_SECONDS,
    ):
        for name, value in (
            ("max_connections", max_connections),
            ("request_queue_size", request_queue_size),
            ("idle_timeout_seconds", idle_timeout_seconds),
            ("max_connection_lifetime_seconds", max_connection_lifetime_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive")
        self.unix_socket_path = unix_socket_path
        self.request_queue_size = int(request_queue_size)
        self.idle_timeout_seconds = float(idle_timeout_seconds)
        self.max_connection_lifetime_seconds = float(
            max_connection_lifetime_seconds
        )
        self.active_idle_timeout_seconds = float(
            active_idle_timeout_seconds
            if active_idle_timeout_seconds is not None
            else max_connection_lifetime_seconds
        )
        if self.active_idle_timeout_seconds <= 0:
            raise ValueError("active_idle_timeout_seconds must be positive")
        self._connection_slots = threading.BoundedSemaphore(int(max_connections))
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = {
            "active_connections": 0,
            "peak_connections": 0,
            "rejected_connections": 0,
            "completed_connections": 0,
        }
        super().__init__(address, _BridgeHandler)

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        if not self._connection_slots.acquire(blocking=False):
            with self._stats_lock:
                self._stats["rejected_connections"] += 1
            self.shutdown_request(request)
            return
        with self._stats_lock:
            self._stats["active_connections"] += 1
            self._stats["peak_connections"] = max(
                self._stats["peak_connections"],
                self._stats["active_connections"],
            )
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._finish_connection()
            raise

    def process_request_thread(
        self,
        request: socket.socket,
        client_address: Any,
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._finish_connection()

    def _finish_connection(self) -> None:
        with self._stats_lock:
            self._stats["active_connections"] -= 1
            self._stats["completed_connections"] += 1
        self._connection_slots.release()

    @property
    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument(
        "--max-connection-lifetime",
        type=float,
        default=DEFAULT_CONNECTION_LIFETIME_SECONDS,
    )
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a child command is required after --")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.max_connection_lifetime <= 0:
        parser.error("--max-connection-lifetime must be positive")
    return args, command


def main() -> int:
    args, command = parse_args()
    server = _ThreadingBridge(
        ("127.0.0.1", args.port),
        args.socket,
        active_idle_timeout_seconds=args.max_connection_lifetime,
        max_connection_lifetime_seconds=args.max_connection_lifetime,
    )
    thread = threading.Thread(
        target=server.serve_forever,
        name="safeact-model-broker-bridge",
        daemon=True,
    )
    thread.start()
    try:
        return subprocess.run(command, check=False).returncode
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
