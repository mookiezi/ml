select timestamp, guild_name, channel_name, channel_id, content
from messages
where channel_id = '109806361373655040'
order by timestamp asc
limit 10