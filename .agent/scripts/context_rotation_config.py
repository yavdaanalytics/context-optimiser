"""
context_rotation_config.py — Feature Flags + Thresholds

Controls AutoBeast context limit detection and automatic chat rotation.
"""
ENABLE_AUTO_CHAT_ROTATION = True
ROTATION_MODE = 'hybrid'
TOKEN_LIMIT = 200000
ROTATION_THRESHOLD_PCT = 85
ENABLE_COMPACTION = True
COMPACTION_MODE = 'model'
COMPACTION_MODEL = 'gpt-5-mini'
ENABLE_ADAPTIVE_THINKING = True
ENABLE_EXTENDED_THINKING = True
EXTENDED_THINKING_MAX_TOKENS = 10000
COMPACTION_STRATEGIES = ['tool_clearing', 'message_pruning']
KEEP_LAST_N_TURNS = 10
AUTO_ROTATE_AFTER_COMPACTION = True
PRESERVE_TASK_OUTPUTS = True
CREATE_CONTINUITY_LINK = True
NOTIFY_USER_ON_ROTATION = True
INCLUDE_CHAIN_HISTORY = True
LOG_ROTATION_EVENTS = True
ROTATION_LOG_PATH = '.agent/data/state/rotation_log.jsonl'

# === MERGED FROM C:\Users\AmitMohanty\.vscode\brain\.agent\tmp\moved_duplicates\20260328T133840Z\context_rotation_config.py ===
# """
# context_rotation_config.py — Feature Flags + Thresholds
# 
# Controls AutoBeast context limit detection and automatic chat rotation.
# """
# 
# # Feature flag: Enable/disable automatic chat rotation on context limit
# ENABLE_AUTO_CHAT_ROTATION = True
# 
# # Mode: 'automatic' (create new chat silently) | 'hybrid' (create + notify) | 'manual' (notify, wait for approval)
# ROTATION_MODE = "hybrid"
# 
# # Token budget
# TOKEN_LIMIT = 200_000  # AutoBeast context window (confirmed)
# ROTATION_THRESHOLD_PCT = 85  # Rotate at 85% capacity (~170k tokens)
# 
# # Compaction settings (now model-based with gpt-5-mini adaptive thinking)
# ENABLE_COMPACTION = True
# COMPACTION_MODE = "model"  # 'model' (gpt-5-mini adaptive thinking) | 'heuristic' (fallback)
# COMPACTION_MODEL = "gpt-5-mini"  # Free tier model with adaptive thinking
# ENABLE_ADAPTIVE_THINKING = True  # Start with adaptive thinking (fast, free)
# ENABLE_EXTENDED_THINKING = True  # Fallback to extended thinking if adaptive insufficient
# EXTENDED_THINKING_MAX_TOKENS = 10_000  # Cost control for extended thinking
# 
# # Legacy heuristic strategies (used only if COMPACTION_MODE='heuristic' or model unavailable)
# COMPACTION_STRATEGIES = ["tool_clearing", "message_pruning"]  # Order of preference
# KEEP_LAST_N_TURNS = 10  # For message_pruning strategy
# 
# # Rotation behavior
# AUTO_ROTATE_AFTER_COMPACTION = True  # Rotate only after compaction attempt fails to reduce below threshold
# PRESERVE_TASK_OUTPUTS = True  # Save old session outputs to state/archived/
# CREATE_CONTINUITY_LINK = True  # Update session.json with next_session pointer
# 
# # Notification
# NOTIFY_USER_ON_ROTATION = True
# INCLUDE_CHAIN_HISTORY = True  # Include list of all linked sessions in notification
# 
# # Logging
# LOG_ROTATION_EVENTS = True
# ROTATION_LOG_PATH = ".agent/data/state/rotation_log.jsonl"


# === MERGED FROM C:\Users\AmitMohanty\.vscode\brain\.agent\tmp\moved_duplicates\20260328T133840Z\context_rotation_config.py.1 ===
# """
# context_rotation_config.py — Feature Flags + Thresholds
# 
# Controls AutoBeast context limit detection and automatic chat rotation.
# """
# 
# # Feature flag: Enable/disable automatic chat rotation on context limit
# ENABLE_AUTO_CHAT_ROTATION = True
# 
# # Mode: 'automatic' (create new chat silently) | 'hybrid' (create + notify) | 'manual' (notify, wait for approval)
# ROTATION_MODE = "hybrid"
# 
# # Token budget
# TOKEN_LIMIT = 200_000  # AutoBeast context window (confirmed)
# ROTATION_THRESHOLD_PCT = 85  # Rotate at 85% capacity (~170k tokens)
# 
# # Compaction settings (now model-based with gpt-5-mini adaptive thinking)
# ENABLE_COMPACTION = True
# COMPACTION_MODE = "model"  # 'model' (gpt-5-mini adaptive thinking) | 'heuristic' (fallback)
# COMPACTION_MODEL = "gpt-5-mini"  # Free tier model with adaptive thinking
# ENABLE_ADAPTIVE_THINKING = True  # Start with adaptive thinking (fast, free)
# ENABLE_EXTENDED_THINKING = True  # Fallback to extended thinking if adaptive insufficient
# EXTENDED_THINKING_MAX_TOKENS = 10_000  # Cost control for extended thinking
# 
# # Legacy heuristic strategies (used only if COMPACTION_MODE='heuristic' or model unavailable)
# COMPACTION_STRATEGIES = ["tool_clearing", "message_pruning"]  # Order of preference
# KEEP_LAST_N_TURNS = 10  # For message_pruning strategy
# 
# # Rotation behavior
# AUTO_ROTATE_AFTER_COMPACTION = True  # Rotate only after compaction attempt fails to reduce below threshold
# PRESERVE_TASK_OUTPUTS = True  # Save old session outputs to state/archived/
# CREATE_CONTINUITY_LINK = True  # Update session.json with next_session pointer
# 
# # Notification
# NOTIFY_USER_ON_ROTATION = True
# INCLUDE_CHAIN_HISTORY = True  # Include list of all linked sessions in notification
# 
# # Logging
# LOG_ROTATION_EVENTS = True
# ROTATION_LOG_PATH = ".agent/data/state/rotation_log.jsonl"
