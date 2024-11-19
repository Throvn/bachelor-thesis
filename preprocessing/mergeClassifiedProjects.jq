# Drop duplicates by the unique "slug" field
def drop_duplicates:
  unique_by(.slug)[];

# Filter entries where priceUSD has at least 64 entries
def filter_priceUSD_length:
  select(.priceUSD | type == "array" and length >= 64);

# Main operation
drop_duplicates
| filter_priceUSD_length
