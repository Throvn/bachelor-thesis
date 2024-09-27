import fs from "fs";
// import { ProxyAgent } from "undici";
// const dispatcher = new ProxyAgent('http://116.163.1.85:9999')

/**
 * @type {[{id: string, symbol: string, name: string}]}
 */
const allCoingeckoCoins = JSON.parse(fs.readFileSync("../allCoingeckoCoins.json"));
const allSantimentCoins = JSON.parse(fs.readFileSync("./allSantimentCoins.json"));

const queryableSlugs = []
for (const geckoCoin of allCoingeckoCoins) {
    if (allSantimentCoins.find((val) => val.slug === geckoCoin.id)) {
        queryableSlugs.push(geckoCoin.id)
    }
}
fs.writeFileSync("geckoSantiSlugs.json", JSON.stringify(queryableSlugs));

const getBodies = slug => ({
    priceUSD1: {
        query: `{
        getMetric(metric: "price_usd") {
            timeseriesData(
            selector: { slug: "${slug}" }
            from: "2009-01-02T00:00:00Z"
            to: "utc_now-4382d"
            interval: "1d"
            transform: {
                type: "none"
            }) {
                datetime
                value
            }
        }
    }`
    },
    priceUSD2: {
        query: `{
            getMetric(metric: "price_usd") {
                timeseriesData(
                selector: {
                    slug: "${slug}"
                }
                from: "utc_now-4381d"
                to: "2024-09-25T00:00:00Z"
                interval: "1d"
                transform: {
                    type:"none"
                }) {
                    datetime
                    value
                }
            }
        }`
    },
    twitterFollowers1: {
        query: `{
        getMetric(metric: "twitter_followers") {
            timeseriesData(
            selector: {
                slug: "${slug}"
            }
            from: "2009-01-02T00:00:00Z"
            to: "utc_now-4382d"
            interval: "1d"
            transform: {type: "consecutive_differences"}) {
                datetime
                value
            }
        }
    }`},
    twitterFollowers2: {
        query: `{
        getMetric(metric: "twitter_followers") {
            timeseriesData(
            selector: {
                slug: "${slug}"
            }
            from: "utc_now-4381d"
            to: "2024-09-25T00:00:00Z"
            interval: "1d"
            transform: {type: "consecutive_differences"}) {
                datetime
                value
            }
        }
    }`},
    devActivity1: {
        query: `{
        getMetric(metric: "dev_activity") {
            timeseriesData(
            slug: "${slug}"
            from: "2009-01-02T00:00:00Z"
            to: "utc_now-4382d"
            interval: "1d"
            transform: { type: "none" }) {
                datetime
                value
            }
        }
    }`},
    devActivity2: {
        query: `{
        getMetric(metric: "dev_activity") {
            timeseriesData(
            slug: "${slug}"
            from: "utc_now-4381d"
            to: "2024-09-25T00:00:00Z"
            interval: "1d"
            transform: { type: "none" }) {
                datetime
                value
            }
        }
    }`},
});

/**
 * Don't forget to update the slug globally before.
 */
const requestGenerator = async (body) => {
    // return fetch("https://api.santiment.net/graphiql?", {
    //     "credentials": "include",
    //     "headers": {
    //         "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:130.0) Gecko/20100101 Firefox/130.0",
    //         "Accept": "application/json",
    //         "Accept-Language": "en-GB,en;q=0.5",
    //         "Content-Type": "application/json",
    //         "Sec-Fetch-Dest": "empty",
    //         "Sec-Fetch-Mode": "cors",
    //         "Sec-Fetch-Site": "same-origin",
    //         "Priority": "u=0",
    //         "Pragma": "no-cache",
    //         "Cache-Control": "no-cache"
    //     },
    //     "referrer": "https://api.santiment.net/graphiql?query=%7B%0A%20%20networkGrowth(from%3A%20%222019-05-09T11%3A25%3A04.894Z%22%2C%20interval%3A%20%221d%22%2C%20slug%3A%20%22ethereum%22%2C%20to%3A%20%222019-06-23T11%3A25%3A04.894Z%22)%20%7B%0A%20%20%20%20newAddresses%0A%20%20%20%20datetime%0A%20%20%7D%0A%7D%0A&variables=%7B%7D",
    //     "body": body,
    //     "method": "POST",
    //     "mode": "cors"
    // });
    return fetch("https://api.santiment.net/graphql?", {
        // dispatcher,
        "headers": {
            "accept": "application/json",
            "accept-language": "en-GB,en-US;q=0.9,en;q=0.8,de;q=0.7,zh-TW;q=0.6,zh;q=0.5",
            "content-type": "application/json",
            "priority": "u=1, i",
            "sec-ch-ua": "\"Google Chrome\";v=\"129\", \"Not=A?Brand\";v=\"8\", \"Chromium\";v=\"129\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"macOS\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "cookie": "intercom-id-cyjjko9u=737c5180-369b-4da7-94c2-d2a93ecc4557; intercom-session-cyjjko9u=; intercom-device-id-cyjjko9u=b9417442-ac8f-4b69-81b7-5c37e8d8228d; AMP_MKTG_4acc1be088=JTdCJTdE; AMP_4acc1be088=JTdCJTIyZGV2aWNlSWQlMjIlM0ElMjI1MGRlY2IxYS1jNjA0LTRkYjctODg3NS1kOGQ1YjJlNTZmNjIlMjIlMkMlMjJzZXNzaW9uSWQlMjIlM0ExNzI3MjQ2ODI1NzY4JTJDJTIyb3B0T3V0JTIyJTNBZmFsc2UlMkMlMjJsYXN0RXZlbnRUaW1lJTIyJTNBMTcyNzI0NjgyODA2OCUyQyUyMmxhc3RFdmVudElkJTIyJTNBOCUyQyUyMnBhZ2VDb3VudGVyJTIyJTNBMSU3RA==; INGRESSCOOKIE=1727246832.745.10450.769973|a79b3b8e865ecfcc3bde4baf5fc0a8fa; mp_1e2fab759c4dcb54aec7d258dc77a278_mixpanel=%7B%22distinct_id%22%3A%20%22%24device%3A19227ef1774903-097f3436fa0b34-16525637-16a7f0-19227ef1774903%22%2C%22%24device_id%22%3A%20%2219227ef1774903-097f3436fa0b34-16525637-16a7f0-19227ef1774903%22%2C%22%24initial_referrer%22%3A%20%22https%3A%2F%2Fsantiment.net%2F%22%2C%22%24initial_referring_domain%22%3A%20%22santiment.net%22%7D; __stripe_mid=144a5f75-9df1-44e0-9081-70bd10b29caee76c41; __stripe_sid=0bfa5514-6e5f-4aa5-91a5-d8ad29b66deb4e93c0",
            "Referer": "https://api.santiment.net/graphiql?query=%7B%0A%20%20getMetric(metric%3A%20%22daily_active_addresses%22)%7B%0A%20%20%20%20timeseriesData(%0A%20%20%20%20%20%20selector%3A%20%7Bslug%3A%20%22bitcoin%22%7D%0A%20%20%20%20%20%20from%3A%20%222024-01-01T00%3A00%3A00Z%22%0A%20%20%20%20%20%20to%3A%20%222024-01-31T23%3A59%3A59Z%22%0A%20%20%20%20%20%20interval%3A%20%221d%22)%7B%0A%20%20%20%20%20%20%20%20datetime%0A%20%20%20%20%20%20%20%20value%0A%20%20%20%20%7D%0A%20%20%7D%0A%7D%0A",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        },
        "body": body,
        "method": "POST"
    });
    // return fetch("https://api.santiment.net/graphiql?", {
    //     method: "POST",
    //     // mode: "cors",
    //     // credentials: "include",
    //     headers: {
    //         "Accept": "application/json",
    //         "Accept-Language": "en-GB,en;q=0.5",
    //         "Content-Type": "application/json",
    //         "Priority": "u=0",
    //         "Pragma": "no-cache",
    //         "Cache-Control": "no-cache",
    //         // "Authorization": "Apikey rx33zbosn6wozm5o_3u3jmnr4zrb7lrwr",
    //         // "Authorization": "Apikey mxshtn3jcprw6elj_z6za263lukn75sn5",
    //         "Authorization": "Apikey abf5lf3pjxbjj4g7_kojxdgepmjdhbya7",
    //     },
    //     body,
    //     // referrer: "https://api.santiment.net/graphiql?query=%7B%0A%20%20price_usd%3A%20getMetric(metric%3A%20%22price_usd%22)%20%7B%0A%20%20%20%20timeseriesData(%0A%20%20%20%20%20%20selector%3A%20%7B%0A%20%20%20%20%20%20%20%20slug%3A%20%22bitcoin%22%0A%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20from%3A%20%222009-01-02T00%3A00%3A00Z%22%0A%20%20%20%20%20%20%20%20to%3A%20%22utc_now-4282d%22%0A%20%20%20%20%20%20%20%20interval%3A%20%221d%22%0A%20%20%20%20%20%20%20%20transform%3A%20%7B%0A%20%20%20%20%20%20%20%20%20%20type%3A%22none%22%0A%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%7D)%20%7B%0A%20%20%20%20%20%20%20%20datetime%0A%20%20%20%20%20%20%20%20value%0A%20%20%20%20%20%20%7D%0A%20%20%7D%0A%7D",
    // })
};

if (!fs.existsSync("noCoinsFoundSantiment.txt")) {
    fs.writeFileSync("noCoinsFoundSantiment.txt", "")
}
const noCoinsFoundSantiment = fs.readFileSync("noCoinsFoundSantiment.txt").toString("ascii").split("\n")
if (!fs.existsSync("coinsScrapedSantiment.txt")) {
    fs.writeFileSync("coinsScrapedSantiment.txt", "")
}
const coinsScraped = fs.readFileSync("coinsScrapedSantiment.txt").toString("ascii").split("\n")

console.info("Queryable Slugs:", queryableSlugs.length);
for (const slug of queryableSlugs) {
    console.log("Slug:", slug);

    if (noCoinsFoundSantiment.find((val) => val === slug)) {
        console.warn("Slug '" + slug + "' found in 'noCoinsFoundSantiment.txt' file... Skipping.")
        continue
    }
    if (coinsScraped.find(val => val === slug)) {
        console.warn("Slug '" + slug + "' already scraped... Skipping.")
        continue
    }

    const bodies = getBodies(slug);
    const requestsForCoin = await Promise.all(Object.values(bodies).map(body => requestGenerator(JSON.stringify(body))));
    const newCoin = {
        slug,
        santiment: {}
    }
    for (let index = 0; index < requestsForCoin.length; index++) {
        const res = requestsForCoin[index];
        if (!res.ok) {
            if (res.headers.get("x-ratelimit-remaining") == 0) {
                console.warn("Ratelimit reached, waiting for:", res.headers.get("x-ratelimit-reset"), "seconds")
                await new Promise((res, rej) => {
                    setTimeout(() => {
                        res();
                    }, 1000 * Number(res.headers.get("x-ratelimit-reset") || 10));
                });
            }
            console.error(res);
            throw new Error("Response not ok (see above)");
        }
        const json = await res.json();
        if (json["errors"]) {
            if (json["errors"][0].message.includes(" for project with slug ")) {
                console.warn("Project '" + slug + "' not found in santiment... Skipping.");
                fs.appendFileSync("noCoinsFoundSantiment.txt", slug + "\n")
                continue;
            } else {
                console.error("Slug:", slug);
                console.error("JSON:", JSON.stringify(json, undefined, 2));
                console.error("Body:", Object.values(bodies)[index])
                console.error(JSON.stringify(json["errors"], undefined, 2));
                throw new Error("GraphQL Error (see above)");
            }
        }
        const resName = Object.keys(bodies)[index];
        console.log(json)
        const resValue = json.data.getMetric.timeseriesData;
        console.info("adding", resName, "to", resValue);
        newCoin.santiment[resName] = resValue;
    }

    fs.appendFileSync("coinsScrapedSantiment.txt", slug + "\n");
    fs.appendFileSync("enrichedCoins.json", "," + JSON.stringify(newCoin));
}

/**
 * {
  price_usd: getMetric(metric: "price_usd") {
    timeseriesData(
      selector: {
        slug: "bitcoin"
      }
        from: "utc_now-4382d"
            to: "utc_now"
        interval: "1d"
        transform: {
          type:"none"
          
        }) {
        datetime
        value
      }
  }
  twitter_followers: getMetric(metric: "twitter_followers") {
    timeseriesData(
      selector: {
        slug: "bitcoin"
      }
      from: "utc_now-4382d"
      to: "utc_now"
      interval: "1d"
      transform: {type: "consecutive_differences"}) {
        datetime
        value
    }
  }
  devActivity: getMetric(metric: "dev_activity") {
    timeseriesData(
      slug: "santiment"
      from: "utc_now-4382d"
      to: "utc_now"
      interval: "7d"
      transform: { type: "none" }
    ) {
      datetime
      value
    }
  }
}
 */