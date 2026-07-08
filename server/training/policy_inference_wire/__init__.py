"""Policy inference wire contract for workers and model ranks."""

from server.training.policy_inference_wire.device import (
    DevicePolicyRequestBuffer,
    allocate_device_policy_request_buffer,
    unpack_policy_request_batch_into,
)
from server.training.policy_inference_wire.request import (
    WIRE_MAX_PAIR_PLAN_COUNT,
    WIRE_MAX_TRACE_COUNT,
    build_policy_request_wire,
    decode_policy_request_metadata,
    max_policy_request_wire_bytes,
)
from server.training.policy_inference_wire.response import (
    build_completed_policy_response_wire,
    build_policy_response_wire_batch,
    build_rejected_policy_response_wire,
    decode_policy_response,
    decode_policy_response_wire,
)
from server.training.policy_inference_wire.types import (
    CompletedPolicyResponse,
    DevicePolicyRequestBatch,
    PolicyRequestMetadata,
    PolicyRequestRoute,
    PolicyRequestWire,
    PolicyRequestWireBatch,
    PolicyResponse,
    PolicyResponseWire,
    RejectedPolicyResponse,
)

__all__ = (
    "CompletedPolicyResponse",
    "DevicePolicyRequestBatch",
    "DevicePolicyRequestBuffer",
    "PolicyRequestMetadata",
    "PolicyRequestRoute",
    "PolicyRequestWire",
    "PolicyRequestWireBatch",
    "PolicyResponse",
    "PolicyResponseWire",
    "RejectedPolicyResponse",
    "WIRE_MAX_PAIR_PLAN_COUNT",
    "WIRE_MAX_TRACE_COUNT",
    "build_completed_policy_response_wire",
    "allocate_device_policy_request_buffer",
    "build_policy_request_wire",
    "build_policy_response_wire_batch",
    "build_rejected_policy_response_wire",
    "decode_policy_response",
    "decode_policy_response_wire",
    "decode_policy_request_metadata",
    "max_policy_request_wire_bytes",
    "unpack_policy_request_batch_into",
)
