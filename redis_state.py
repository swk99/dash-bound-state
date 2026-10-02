"""Atomic bounded-window update; Lua preserves the existing float32 list layout."""
LUA_PUSH_TRIM = """
local w = tonumber(ARGV[2])
if not w or w < 1 or w ~= math.floor(w) then
    return redis.error_reply('window must be a positive integer')
end
redis.call('LPUSH', KEYS[1], ARGV[1])
redis.call('LTRIM', KEYS[1], 0, w - 1)
return redis.call('LLEN', KEYS[1])
"""

class RedisWindowWriter:
    def __init__(self, client):
        # redis-py Script caches SHA and recovers from NOSCRIPT.
        self.script = client.register_script(LUA_PUSH_TRIM)

    def push(self, key, payload, window):
        if isinstance(window, bool) or int(window) != window or window < 1:
            raise ValueError("window must be a positive integer")
        return int(self.script(keys=[key], args=[payload, int(window)]))
