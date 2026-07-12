export TRACTOR_BOT_PLAYER="ai"
export TRACTOR_AI_BASE_URL="https://token-plan-cn.xiaomimimo.com/v1"
export TRACTOR_AI_API_KEY="tp-c1o80q6d1sfs06cxd6pi8zhv0b2af7m0efkivrqyg871uf0d"
export TRACTOR_AI_MODEL="mimo-v2.5"
export TRACTOR_AI_HTTP_TIMEOUT_SECONDS=10
export TRACTOR_AI_HTTP_MAX_RETRIES=10
#export TRACTOR_AI_HTTP_RETRY_DELAY_SECONDS
#export TRACTOR_AI_DECISION_MAX_RETRIES
#export TRACTOR_AI_MAX_OUTPUT_TOKENS
exec python -m uvicorn server.web.app:app --ws websockets-sansio --host 127.0.0.1 --port 8000
