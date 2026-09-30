#!/usr/bin/env python3
"""Run SafeActBench through Codex CLI backed by the DeepSeek Responses API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from .deepseek_responses_proxy import (
        ProxyLimits,
        isolated_responses_proxy,
        validate_user_id,
    )
except ImportError:  # Direct script execution.
    from deepseek_responses_proxy import (
        ProxyLimits,
        isolated_responses_proxy,
        validate_user_id,
    )


DEFAULT_BASE_URL = "https://api.deepseek.com/"
DEFAULT_MODEL = "deepseek-v4-flash"
SUPPORTED_MODELS = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v4-flash-vision-exp",
)
REASONING_EFFORTS = ("low", "high", "max")
MANIFEST_NAME = "deepseek_run_manifest.json"
MODEL_CATALOG_NAME = "deepseek_codex_models.json"


_CODEX_ROUTER_PERMISSION_DENIED_RE = re.compile(
    r"(?m)^\S+ ERROR codex_core::tools::router: "
    r"error=exec_command failed .*CreateProcess \{ message: .*"
    r"(?:Permission denied \(os error 13\)|kind: PermissionDenied)"
)
_CODEX_BWRAP_DIGEST_MISMATCH_RE = re.compile(
    r"(?m)^\S+ ERROR codex_core::tools::router: error=.*"
    r"bundled bubblewrap digest mismatch for /run/safeact/bwrap: "
    r"expected sha256:([0-9a-f]{64}), got sha256:([0-9a-f]{64})"
)

# These transport-capacity failures are created by the harness rather than by
# a model-selected request.  Other rejection reasons (for example a bad route,
# model drift, or exhausting an explicit request budget) remain scored below;
# a model must not gain a denominator exclusion by deliberately provoking one.
HARNESS_PROXY_REJECTIONS = frozenset({"concurrency_limit", "connection_limit"})
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalized_base_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "DeepSeek base URL must be an absolute HTTPS URL without "
            "credentials, query, or fragment"
        )
    return value.rstrip("/") + "/"


def codex_provider_arguments(
    base_url: str,
    api_key_env: str,
    reasoning_effort: str,
) -> list[str]:
    configs = (
        'model_provider="deepseek"',
        f"model_reasoning_effort={json.dumps(reasoning_effort)}",
        'model_providers.deepseek.name="deepseek"',
        f"model_providers.deepseek.base_url={json.dumps(base_url)}",
        'model_providers.deepseek.wire_api="responses"',
        f"model_providers.deepseek.env_key={json.dumps(api_key_env)}",
        'shell_environment_policy.inherit="core"',
        "shell_environment_policy.ignore_default_excludes=false",
    )
    result = ["--extra-arg=--ignore-user-config", "--extra-arg=--ignore-rules"]
    for config in configs:
        result.extend(("--extra-arg=--config", f"--extra-arg={config}"))
    return result


def build_command(
    args: argparse.Namespace,
    *,
    provider_base_url: str | None = None,
) -> list[str]:
    root = Path(args.repo_root).resolve()
    command = [
        args.python_bin,
        str(root / "scripts" / "run_coding_agent_benchmark.py"),
        "--repo-root",
        str(root),
        "--backend",
        "codex",
        "--strategy",
        getattr(args, "strategy", "baseline"),
        "--model",
        args.model,
        "--cli-bin",
        args.codex_bin,
        "--model-catalog",
        str(root / "env" / MODEL_CATALOG_NAME),
        "--timeout",
        str(args.timeout),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
    ]
    for protocol in args.protocol or []:
        command.extend(("--protocol", protocol))
    for case_id in args.case_id:
        command.extend(("--case-id", case_id))
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.keep_sandbox:
        command.append("--keep-sandbox")
    command.extend(
        codex_provider_arguments(
            provider_base_url or normalized_base_url(args.base_url),
            args.api_key_env,
            args.reasoning_effort,
        )
    )
    if args.print_command:
        command.append("--print-command")
    return command


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_catalog_metadata(root: Path, model: str) -> dict[str, Any]:
    path = root / "env" / MODEL_CATALOG_NAME
    payload = load_json(path)
    models = payload.get("models") if isinstance(payload, dict) else None
    matches = [
        item
        for item in models or []
        if isinstance(item, dict) and item.get("slug") == model
    ]
    if len(matches) != 1:
        raise ValueError(f"official model catalog has no unique entry for {model!r}")
    selected = matches[0]
    return {
        "catalog_path": f"env/{MODEL_CATALOG_NAME}",
        "catalog_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "context_window_tokens": selected.get("context_window"),
        "max_context_window_tokens": selected.get("max_context_window"),
        "effective_context_window_percent": selected.get(
            "effective_context_window_percent"
        ),
        "minimal_client_version": selected.get("minimal_client_version"),
        "supported_reasoning_levels": [
            item.get("effort")
            for item in selected.get("supported_reasoning_levels", [])
            if isinstance(item, dict)
        ],
    }


def loopback_bypass_env(source: dict[str, str]) -> dict[str, str]:
    env = source.copy()
    existing = [
        item
        for name in ("NO_PROXY", "no_proxy")
        for item in env.get(name, "").split(",")
        if item
    ]
    values = list(dict.fromkeys([*existing, "127.0.0.1", "localhost"]))
    env["NO_PROXY"] = ",".join(values)
    env["no_proxy"] = env["NO_PROXY"]
    return env


def verify_runtime_identity(
    output_dir: Path,
    requested_model: str,
    expected_catalog_sha256: str | None = None,
) -> tuple[int, list[str]]:
    traces = sorted((output_dir / "traces").glob("*/*.json"))
    errors: list[str] = []
    checked = 0
    for path in traces:
        try:
            trace = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: unreadable trace: {exc}")
            continue
        checked += 1
        if trace.get("runtime_model") != requested_model:
            errors.append(
                f"{path}: runtime_model={trace.get('runtime_model')!r}, "
                f"expected {requested_model!r}"
            )
        provider = trace.get("runtime_provider")
        if not isinstance(provider, str) or provider.lower() != "deepseek":
            errors.append(
                f"{path}: runtime_provider={provider!r}, expected 'deepseek'"
            )
        if trace.get("fresh_session") is not True:
            errors.append(f"{path}: fresh_session is not true")
        if trace.get("session_persistence") != "ephemeral":
            errors.append(f"{path}: session is not ephemeral")
        if expected_catalog_sha256 is not None:
            catalog = trace.get("model_catalog")
            if not isinstance(catalog, dict):
                errors.append(f"{path}: trusted model catalog is missing")
            else:
                if catalog.get("mounted_path") != (
                    "/run/safeact/model_catalog.json"
                ):
                    errors.append(
                        f"{path}: model catalog guest path is invalid"
                    )
                if catalog.get("sha256") != expected_catalog_sha256:
                    errors.append(f"{path}: model catalog digest mismatch")
                if catalog.get("selected_model") != requested_model:
                    errors.append(f"{path}: model catalog selection mismatch")
    return checked, errors


def detect_codex_harness_runtime_failures(output_dir: Path) -> list[str]:
    """Detect attested Codex sandbox failures in trusted runtime traces.

    Compatibility with older trace schemas requires inspecting ``stderr``,
    which also contains model text.  To prevent a model from earning an
    exclusion by echoing a marker, only the pre-final-response portion is
    considered, at least two router create-process failures are required, and
    the trace must attest the outer sandbox with no successful tool RPC.  A
    bubblewrap mismatch additionally has to agree with the recorded binary
    digest.
    """

    reasons: list[str] = []
    for path in sorted((output_dir / "traces").glob("*/*.json")):
        try:
            trace = load_json(path)
        except (OSError, json.JSONDecodeError):
            # Runtime identity verification owns unreadable-trace handling.
            continue
        if not isinstance(trace, dict):
            continue
        sandbox_resource = trace.get("sandbox_resource")
        sandbox_digest = (
            sandbox_resource.get("sha256")
            if isinstance(sandbox_resource, dict)
            else None
        )
        preflight = trace.get("codex_sandbox_preflight")
        helper = trace.get("codex_sandbox_helper")
        helper_digest = (
            helper.get("sha256") if isinstance(helper, dict) else None
        )
        preflight_helper_digest = (
            preflight.get("codex_sandbox_helper_sha256")
            if isinstance(preflight, dict)
            else None
        )
        preflight_returncode = (
            preflight.get("returncode")
            if isinstance(preflight, dict)
            else None
        )
        if (
            trace.get("failure_class") == "harness_failure"
            and isinstance(preflight, dict)
            and preflight.get("attempted") is True
            and preflight.get("passed") is False
            and isinstance(preflight_returncode, int)
            and not isinstance(preflight_returncode, bool)
            and preflight_returncode != 0
            and preflight.get("command")
            == ["<CODEX>", "sandbox", "--", "/usr/bin/true"]
            and preflight.get("outer_network_unshared") is True
            and preflight.get("model_broker_attached") is True
            and preflight.get("broker_socket_masked_from_cli") is True
            and preflight.get("proxy_token_included") is False
            and isinstance(preflight.get("diagnostic"), str)
            and bool(preflight.get("diagnostic"))
            and isinstance(sandbox_resource, dict)
            and sandbox_resource.get("path") == "/usr/bin/bwrap"
            and SHA256_RE.fullmatch(str(sandbox_digest)) is not None
            and preflight.get("outer_bwrap_sha256") == sandbox_digest
            and (
                (
                    preflight_helper_digest is None
                    and helper is None
                )
                or (
                    SHA256_RE.fullmatch(str(preflight_helper_digest))
                    is not None
                    and preflight_helper_digest == helper_digest
                )
            )
            and trace.get("command") == []
            and trace.get("tool_rpc_requests") == 0
        ):
            reasons.append("codex_sandbox_preflight_failed")
            continue

        stderr = trace.get("stderr")
        if (
            not isinstance(stderr, str)
            or trace.get("outer_isolation_policy")
            != "safeact_bwrap_case_isolation_v1"
            or trace.get("tool_rpc_requests") != 0
            or not isinstance(sandbox_resource, dict)
            or sandbox_resource.get("path") != "/usr/bin/bwrap"
            or SHA256_RE.fullmatch(str(sandbox_resource.get("sha256"))) is None
        ):
            continue
        # Codex renders the terminal model message after a standalone `codex`
        # line.  Excluding that suffix prevents direct final-output spoofing.
        diagnostics = stderr.rsplit("\ncodex\n", 1)[0]
        permission_failures = list(
            _CODEX_ROUTER_PERMISSION_DENIED_RE.finditer(diagnostics)
        )
        if len(permission_failures) >= 2:
            reasons.append("codex_router_create_process_permission_denied")

        mismatch = _CODEX_BWRAP_DIGEST_MISMATCH_RE.search(diagnostics)
        attested_digest = sandbox_resource.get("sha256")
        if (
            mismatch is not None
            and mismatch.group(1) != mismatch.group(2)
            and attested_digest == mismatch.group(2)
        ):
            reasons.append("codex_bundled_bubblewrap_digest_mismatch")
    return list(dict.fromkeys(reasons))


def has_attested_wall_timeout(output_dir: Path) -> bool:
    """Read the adapter-authored timeout state without trusting model text."""

    for path in sorted((output_dir / "traces").glob("*/*.json")):
        try:
            trace = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(trace, dict)
            and trace.get("failure_class") == "wall_timeout"
        ):
            return True
    return False


def classified_proxy_rejections(
    proxy_stats: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Split trusted proxy rejection counters by who controls the cause."""

    rejections = proxy_stats.get("rejections", {})
    if not isinstance(rejections, dict):
        return ["invalid_proxy_rejection_stats"], []
    harness_reasons: list[str] = []
    scored_reasons: list[str] = []
    for raw_reason, raw_count in sorted(rejections.items()):
        if (
            not isinstance(raw_reason, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", raw_reason) is None
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 0
        ):
            return ["invalid_proxy_rejection_stats"], []
        if raw_count == 0:
            continue
        rendered = f"proxy_{raw_reason}:{raw_count}"
        if raw_reason in HARNESS_PROXY_REJECTIONS:
            harness_reasons.append(rendered)
        else:
            scored_reasons.append(rendered)
    return harness_reasons, scored_reasons


def classify_structured_failure(
    output_dir: Path,
    proxy_stats: dict[str, Any] | None,
    identity_errors: list[str],
    runner_returncode: int,
) -> tuple[str | None, list[str]]:
    """Classify primary outcome state without scanning agent-controlled text.

    Provider failures and model-controllable proxy-policy failures remain
    scored agent failures here.  Harness-created transport-capacity failures
    and attested sandbox failures are excluded as harness failures.
    """

    reasons: list[str] = []
    runtime_failures = detect_codex_harness_runtime_failures(output_dir)
    if runtime_failures:
        return "harness_failed", runtime_failures

    # A timed-out process normally cannot report a runtime model identity.
    # Preserve the selected 0-score outcome instead of letting that expected
    # absence reclassify the case as an excluded harness failure.
    if has_attested_wall_timeout(output_dir):
        return "agent_failed", ["wall_timeout"]

    if identity_errors:
        reasons.extend(f"runtime_identity:{item}" for item in identity_errors)
        return "harness_failed", reasons

    if isinstance(proxy_stats, dict):
        harness_rejections, scored_rejections = classified_proxy_rejections(
            proxy_stats
        )
        if harness_rejections:
            return "harness_failed", harness_rejections
        upstream_stream_failures = proxy_stats.get(
            "upstream_stream_failures", 0
        )
        if (
            isinstance(upstream_stream_failures, int)
            and upstream_stream_failures > 0
        ):
            return "agent_failed", [
                f"proxy_upstream_stream_failures:{upstream_stream_failures}"
            ]
        response_limit_count = proxy_stats.get(
            "response_limit_exceeded", 0
        )
        if (
            isinstance(response_limit_count, int)
            and response_limit_count > 0
        ):
            return "agent_failed", [
                f"proxy_response_limit_exceeded:{response_limit_count}"
            ]
        status_counts = proxy_stats.get("upstream_status_counts", {})
        if isinstance(status_counts, dict):
            for raw_status, raw_count in status_counts.items():
                try:
                    status = int(raw_status)
                    count = int(raw_count)
                except (TypeError, ValueError):
                    return "harness_failed", ["invalid_proxy_status_stats"]
                if count < 1:
                    continue
                if status >= 400:
                    reasons.append(f"upstream_http_{status}:{count}")
        reasons.extend(scored_rejections)
        for field in ("response_deadline_exceeded",):
            count = proxy_stats.get(field, 0)
            if isinstance(count, int) and count > 0:
                reasons.append(f"proxy_{field}:{count}")
    if reasons:
        return "agent_failed", reasons

    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = load_json(summary_path)
        except (OSError, json.JSONDecodeError):
            return "harness_failed", ["summary_unreadable"]
        failures = summary.get("failures", []) if isinstance(summary, dict) else []
        if not isinstance(failures, list):
            return "harness_failed", ["summary_failures_invalid"]
        classifications = {
            str(item.get("classification"))
            for item in failures
            if isinstance(item, dict)
        }
        if "evaluation_harness_failure" in classifications:
            return "harness_failed", ["evaluation_harness_failure"]
        if "infrastructure_failure" in classifications:
            return "agent_failed", ["batch_infrastructure_failure"]
    elif runner_returncode != 0:
        return "agent_failed", ["runner_failed_without_summary"]
    elif not summary_path.is_file():
        return "harness_failed", ["runner_succeeded_without_summary"]
    return None, []


def classify_retry_recommendation(
    output_dir: Path,
    proxy_stats: dict[str, Any] | None,
) -> tuple[str | None, list[str]]:
    """Return trusted transient hints without changing the primary score.

    The parallel controller may retry these cases in a fresh workspace and
    session.  If every retry is exhausted, the selected case is still retained
    in the denominator as a failed case.
    """

    # A case-level wall deadline is a benchmark outcome, not a transient
    # provider incident.  Retrying it would quietly grant more than the public
    # 300-second allowance and could turn an over-time answer into a pass.
    if has_attested_wall_timeout(output_dir):
        return None, []

    reasons: list[str] = []
    if isinstance(proxy_stats, dict):
        stream_failures = proxy_stats.get("upstream_stream_failures", 0)
        if isinstance(stream_failures, int) and stream_failures > 0:
            reasons.append(f"proxy_upstream_stream_failures:{stream_failures}")
        status_counts = proxy_stats.get("upstream_status_counts", {})
        if isinstance(status_counts, dict):
            for raw_status, raw_count in status_counts.items():
                try:
                    status = int(raw_status)
                    count = int(raw_count)
                except (TypeError, ValueError):
                    continue
                if count > 0 and (
                    status in {401, 403, 408, 409, 429} or status >= 500
                ):
                    reasons.append(f"upstream_http_{status}:{count}")
        rejections = proxy_stats.get("rejections", {})
        if isinstance(rejections, dict):
            for reason in ("deadline_exceeded", "upstream_failure"):
                count = rejections.get(reason, 0)
                if isinstance(count, int) and count > 0:
                    reasons.append(f"proxy_{reason}:{count}")
        response_deadlines = proxy_stats.get(
            "response_deadline_exceeded", 0
        )
        if isinstance(response_deadlines, int) and response_deadlines > 0:
            reasons.append(
                f"proxy_response_deadline_exceeded:{response_deadlines}"
            )

    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = load_json(summary_path)
        except (OSError, json.JSONDecodeError):
            summary = None
        failures = summary.get("failures", []) if isinstance(summary, dict) else []
        if isinstance(failures, list) and any(
            isinstance(item, dict)
            and item.get("classification") == "infrastructure_failure"
            for item in failures
        ):
            reasons.append("batch_infrastructure_failure")
    if reasons:
        return "infrastructure_retry", list(dict.fromkeys(reasons))
    return None, []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--codex-bin",
        default=os.environ.get("SAFEACT_CODEX_BIN") or shutil.which("codex") or "codex",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=os.environ.get("SAFEACT_DEEPSEEK_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--strategy",
        choices=("baseline", "scgr-eg"),
        default=os.environ.get("SAFEACT_AGENT_STRATEGY", "baseline"),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=os.environ.get("SAFEACT_DEEPSEEK_REASONING_EFFORT", "high"),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--upstream-model",
        help=(
            "Optional exact upstream Responses model id. The trusted proxy "
            "rewrites only the model field after validating the Codex-side model."
        ),
    )
    parser.add_argument(
        "--user-id",
        help="Anonymous per-case DeepSeek isolation id; enables the loopback proxy",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--proxy-max-total-requested-output-tokens",
        type=int,
        default=ProxyLimits().max_total_requested_output_tokens,
        help=(
            "Per-case cumulative requested output-token allowance enforced by "
            "the isolated model proxy"
        ),
    )
    parser.add_argument(
        "--protocol",
        action="append",
        choices=("legacy", "v0", "v1", "v2", "v3"),
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-sandbox", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 600:
        parser.error("--timeout must be between 1 and 600 seconds")
    if args.proxy_max_total_requested_output_tokens < ProxyLimits().max_output_tokens:
        parser.error(
            "--proxy-max-total-requested-output-tokens must be at least "
            f"{ProxyLimits().max_output_tokens}"
        )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        args.base_url = normalized_base_url(args.base_url)
    except ValueError as exc:
        parser.error(str(exc))
    if args.user_id:
        try:
            validate_user_id(args.user_id)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    upstream_api_key = os.environ.get(args.api_key_env)
    if not args.print_command and not upstream_api_key:
        raise SystemExit(
            f"Required API key environment variable {args.api_key_env!r} is not set"
        )
    if not args.print_command and not args.user_id:
        raise SystemExit(
            "--user-id is required for an isolated DeepSeek benchmark run"
        )
    if not args.print_command and len(args.case_id) != 1:
        raise SystemExit(
            "An isolated DeepSeek benchmark run requires exactly one --case-id"
        )
    output_dir = Path(args.output_dir).resolve()
    catalog_metadata = selected_catalog_metadata(
        Path(args.repo_root).resolve(), args.model
    )
    started_at = utc_now()
    proxy_stats: dict[str, Any] | None = None
    if args.user_id and not args.print_command:
        downstream_token = secrets.token_urlsafe(32)
        proxy_limits = ProxyLimits(
            max_lifetime_seconds=float(args.timeout + 60),
            upstream_timeout_seconds=float(args.timeout),
            max_total_requested_output_tokens=(
                args.proxy_max_total_requested_output_tokens
            ),
        )
        with isolated_responses_proxy(
            args.base_url,
            args.user_id,
            upstream_api_key=str(upstream_api_key),
            downstream_token=downstream_token,
            expected_model=args.model,
            upstream_model=getattr(args, "upstream_model", None),
            unix_socket=True,
            limits=proxy_limits,
        ) as proxy:
            command = build_command(
                args,
                provider_base_url="http://127.0.0.1:8765/",
            )
            print(shlex.join(command), flush=True)
            child_env = loopback_bypass_env(dict(os.environ))
            child_env[args.api_key_env] = downstream_token
            child_env["SAFEACT_PROXY_API_KEY_ENV"] = args.api_key_env
            child_env["SAFEACT_MODEL_BROKER_DIR"] = str(proxy.broker_dir)
            child_env["SAFEACT_MODEL_BROKER_SOCKET"] = str(proxy.socket_name)
            child_env["SAFEACT_MODEL_BROKER_PORT"] = "8765"
            completed = subprocess.run(
                command,
                cwd=Path(args.repo_root).resolve(),
                env=child_env,
                check=False,
            )
        proxy_stats = proxy.stats
    else:
        command = build_command(args)
        print(shlex.join(command), flush=True)
        completed = subprocess.run(
            command,
            cwd=Path(args.repo_root).resolve(),
            env=os.environ.copy(),
            check=False,
        )
    checked, identity_errors = verify_runtime_identity(
        output_dir,
        args.model,
        expected_catalog_sha256=str(catalog_metadata["catalog_sha256"]),
    )
    if not args.print_command and checked != 1:
        identity_errors.append(
            f"expected exactly one runtime identity trace, found {checked}"
        )
    failure_class, failure_reasons = classify_structured_failure(
        output_dir,
        proxy_stats,
        identity_errors,
        completed.returncode,
    )
    retry_class, retry_reasons = classify_retry_recommendation(
        output_dir,
        proxy_stats,
    )
    manifest = {
        "schema_version": 2,
        "started_at": started_at,
        "finished_at": utc_now(),
        "model": args.model,
        "strategy": getattr(args, "strategy", "baseline"),
        "upstream_model": getattr(args, "upstream_model", None) or args.model,
        "provider": "deepseek",
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "request_isolation": (
            "deepseek_responses_user" if args.user_id else "none"
        ),
        "user_id": args.user_id,
        "reasoning_effort": args.reasoning_effort,
        "codex_model_metadata_policy": {
            "source": "official_deepseek_model_catalog",
            **catalog_metadata,
            "max_output_tokens": (
                proxy_stats.get("limits", {}).get("max_output_tokens")
                if isinstance(proxy_stats, dict)
                else None
            ),
        },
        "runner_returncode": completed.returncode,
        "runtime_identity_traces_checked": checked,
        "runtime_identity_errors": identity_errors,
        "failure_class": failure_class,
        "failure_reasons": failure_reasons,
        "retry_class": retry_class,
        "retry_reasons": retry_reasons,
        "api_key_persisted": False,
        "responses_proxy_stats": proxy_stats,
    }
    if not args.print_command or output_dir.exists():
        atomic_write_json(output_dir / MANIFEST_NAME, manifest)
    if identity_errors:
        for error in identity_errors:
            print(f"RUNTIME_IDENTITY_ERROR: {error}", file=sys.stderr)
        return 2
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
