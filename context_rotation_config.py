"""Default configuration for chat/context rotation used by the executor.

Deployments may provide an override file with the same name to customize
rotation behaviour. Defaults disable auto-rotation to avoid surprises.
"""
ENABLE_AUTO_CHAT_ROTATION = True
COMPACTION_MODE = "heuristic"  # or 'model'
COMPACTION_MODEL = "gpt-5-mini"
TOKEN_LIMIT = 16000
ROTATION_THRESHOLD_PCT = 85
ENABLE_EXTENDED_THINKING = False
ENABLE_ADAPTIVE_THINKING = False
