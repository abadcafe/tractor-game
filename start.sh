export TRACTOR_BOT_PLAYER="ai"
export TRACTOR_AI_BASE_URL="https://token-plan-cn.xiaomimimo.com/v1"
export TRACTOR_AI_API_KEY="你的 key"
export TRACTOR_AI_MODEL="mimo-v2.5"
export TRACTOR_AI_TIMEOUT_SECONDS
export TRACTOR_AI_MAX_RETRIES
export TRACTOR_AI_RETRY_DELAY_SECONDS
export TRACTOR_AI_DECISION_RETRIES
export TRACTOR_AI_MAX_OUTPUT_TOKENS
export TRACTOR_AI_LOG_PAYLOADS
export TRACTOR_AI_LOG_TOOL_USE
python -m uvicorn server.server:app --ws websockets-sansio --host 127.0.0.1 --port 8000
