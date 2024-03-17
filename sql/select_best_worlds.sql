WITH worlds as (
  SELECT
    id,
    name,
    favorites + SQRT(visits) as value,
    thumbnail_image_url,
  FROM crawled.worlds
  WHERE thumbnail_image_url IS NOT NULL
)
SELECT
  id,
  name,
  thumbnail_image_url,
  value,
FROM worlds
QUALIFY ROW_NUMBER() OVER (PARTITION BY id ORDER BY value DESC) = 1
ORDER BY value DESC
LIMIT 5000
