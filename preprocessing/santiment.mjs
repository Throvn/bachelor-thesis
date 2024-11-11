import fs from "fs";

import allSantimentCoins from "./allSantimentCoins.json" with {type: "json"};

const queryableSlugs = allSantimentCoins.map((val) => val.slug);
const MAX_DAYS = 4381;

// Due to a limitation of the Santiment API, MAX_DAYS is the maximum amount of time one can get the values back from.
// Therefore we need to ask for the data twice.
const PARSE_DATE_RAW = new Date("2024-09-25T00:00:00Z");
const INTERMEDIATE_DATE = new Date(PARSE_DATE_RAW.getTime() - (MAX_DAYS * 24 * 60 * 60 * 1000)).toISOString();
const PARSE_DATE = PARSE_DATE_RAW.toISOString();

const getBodies = slug => ({
    priceUSD1: {
        query: `{
        getMetric(metric: "price_usd") {
            timeseriesData(
            selector: { slug: "${slug}" }
            from: "2009-01-02T00:00:00Z"
            to: "${INTERMEDIATE_DATE}"
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
                from: "${INTERMEDIATE_DATE}"
                to: "${PARSE_DATE}"
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
            to: "${INTERMEDIATE_DATE}"
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
            from: "${INTERMEDIATE_DATE}"
            to: "${PARSE_DATE}"
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
            to: "${INTERMEDIATE_DATE}"
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
            from: "${INTERMEDIATE_DATE}"
            to: "${PARSE_DATE}"
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

const FNAME_NO_COINS = "noCoinsFound.txt";
if (!fs.existsSync(FNAME_NO_COINS)) {
    fs.writeFileSync(FNAME_NO_COINS, "");
}
const noCoinsFoundSantiment = fs.readFileSync(FNAME_NO_COINS, { encoding: "ascii" }).split("\n");

const FNAME_COINS_SCRAPED = "coinsScraped.txt";
if (!fs.existsSync(FNAME_COINS_SCRAPED)) {
    fs.writeFileSync(FNAME_COINS_SCRAPED, "");
}
const coinsScraped = fs.readFileSync(FNAME_COINS_SCRAPED, { encoding: "ascii" }).split("\n");

console.info("Length of Queryable Slugs:", queryableSlugs.length);
for (const slug of queryableSlugs) {
    console.log("Slug:", slug);

    if (noCoinsFoundSantiment.find((val) => val === slug)) {
        console.warn(`Slug '${slug}' found in '${FNAME_NO_COINS}' file... Skipping.`);
        continue;
    }
    if (coinsScraped.find(val => val === slug)) {
        console.warn(`Slug '${slug}' already scraped... Skipping.`);
        continue;
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
            console.error(res);
            throw new Error("Response not ok (see above)");
        }
        const json = await res.json();
        if (json["errors"]) {
            if (json["errors"][0].message.includes(" for project with slug ")) {
                console.warn("Project '" + slug + "' not found in santiment... Skipping.");
                fs.appendFileSync(FNAME_NO_COINS, slug + "\n");
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

    fs.appendFileSync(FNAME_COINS_SCRAPED, slug + "\n");
    fs.appendFileSync("newlyAddedSantimentProjects.json", "," + JSON.stringify(newCoin) + "\n");
}

console.log("DONE!!!\n Don't forget to manually close the json file.");

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