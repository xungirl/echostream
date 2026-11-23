
-- KEYS: dedupe, hist, cnt, rec-cache   ARGV: ttl, item, wr, ts, action, cats(csv), hist_max, pos_threshold
if not redis.call('SET', KEYS[1], '1', 'NX', 'EX', tonumber(ARGV[1])) then
  return 0
end
redis.call('LPUSH', KEYS[2], ARGV[2] .. ':' .. ARGV[3] .. ':' .. ARGV[4])
redis.call('LTRIM', KEYS[2], 0, tonumber(ARGV[7]) - 1)
redis.call('HINCRBY', KEYS[3], 'n', 1)
redis.call('HINCRBY', KEYS[3], ARGV[5], 1)
redis.call('HINCRBYFLOAT', KEYS[3], 'sum_wr', ARGV[3])
redis.call('HSET', KEYS[3], 'last_ts', ARGV[4])
redis.call('DEL', KEYS[4])   -- cached recommendations are now stale
local positive = tonumber(ARGV[3]) > tonumber(ARGV[8])
if positive then redis.call('HINCRBY', KEYS[3], 'pos', 1) end
for c in string.gmatch(ARGV[6], '[^,]+') do
  if positive then redis.call('HINCRBY', KEYS[3], 'pos_cat:' .. c, 1) end
  if ARGV[5] == 'skip' then redis.call('HINCRBY', KEYS[3], 'skip_cat:' .. c, 1) end
end
return 1
