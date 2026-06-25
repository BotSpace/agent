"""Optional OpenTelemetry instrumentation — only loaded if installed."""

from __future__ import annotations

from ..config import get_settings
from .metrics import (
    LLM_INPUT_TOKENS,
    LLM_OUTPUT_TOKENS,
    LLM_REQUESTS,
    LLM_REQUEST_DURATION,
    record_llm_usage,
)


__all__ = [
    "LLM_INPUT_TOKENS",
    "LLM_OUTPUT_TOKENS",
    "LLM_REQUESTS",
    "LLM_REQUEST_DURATION",
    "configure_otel",
    "record_llm_usage",
]


def configure_otel(app=None) -> bool:  # noqa: ANN001
    s = get_settings()
    if not s.otel_endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": "agent"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=s.otel_endpoint)))
    trace.set_tracer_provider(provider)

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except ImportError:
            pass
    return True
