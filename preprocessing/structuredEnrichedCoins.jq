map({
  slug,
  santiment: {
    priceUSD: (.santiment.priceUSD1 + .santiment.priceUSD2),
    twitterFollowers: (.santiment.twitterFollowers1 + .santiment.twitterFollowers2),
    devActivity: (.santiment.devActivity1 + .santiment.devActivity2)
  }
})
