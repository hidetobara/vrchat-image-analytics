WITH worlds as (
  SELECT
    id,
    name,
    author_name,
    favorites,
    visits,
    thumbnail_image_url,
    release_status,
  FROM crawled.worlds
  WHERE thumbnail_image_url IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY id ORDER BY visits DESC) = 1
)
SELECT
  id,
  author_name,
  name,
  favorites,
  thumbnail_image_url,
FROM worlds
WHERE release_status != "hidden"
ORDER BY favorites + SQRT(visits) DESC
LIMIT 15000
