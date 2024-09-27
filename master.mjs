import { plugin } from "./libs/cursor.js";
import "https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js";


const slug = new URLSearchParams(location.search).get("id")
if (!slug) {
    throw new Error("You need ./coin?id=___ in your path.");
}

const MONTHS = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December'
];

/**
 * 
 * @param {[{datetime: string, value: number}]} prices 
 */
function drawPrices(prices, timeRange) {
    const ctx = document.getElementById("tl-canvas");
    console.log("Price time range: ", timeRange);
    const chart = new Chart(
        ctx,
        {
            type: 'line',
            data: {
                labels: prices.map(row => {
                    const date = new Date(row.datetime);
                    const fmtDate = `${MONTHS[date.getMonth()]}, ${date.getFullYear()}`
                    return date.getTime();
                }),
                datasets: [
                    {
                        label: 'Price',
                        data: prices.map(row => row.value)
                    }
                ],
            },
            options: {
                elements: {
                    point: {
                        pointStyle: false,
                    },
                },
                scales: {
                    x: {
                        type: 'time',
                        min: timeRange[0],
                        max: timeRange[1],
                    }
                },
                onClick: (e) => {
                    console.info(e);
                    const canvasPosition = Chart.helpers.getRelativePosition(e, chart);
                    const points = chart.getElementsAtEventForMode(e, 'nearest', { intersect: false }, true);

                    if (points.length) {
                        const firstPoint = points[0];
                        const time = prices[firstPoint.index].datetime;
                        console.log("Timestamp: ", time, new Date(time).toDateString());

                        window.addPrediction(time);
                    }
                }
            },
            plugins: [plugin],
        }
    );
}

/**
 * 
 * @param {[{datetime: string, value: number}]} activity 
 */
function drawDevelopment(activity, timeRange) {
    if (activity.length === 0) {
        document.getElementById("tr-error")
            .innerText = "Note: No dev activity during time. Could be closed source."
    }

    const ctx = document.getElementById("tr-canvas");
    const chart = new Chart(
        ctx,
        {
            type: 'line',
            data: {
                labels: activity.map(row => {
                    const date = new Date(row.datetime);
                    const fmtDate = `${MONTHS[date.getMonth()]}, ${date.getFullYear()}`
                    return date.getTime();
                }),
                datasets: [
                    {
                        label: 'Commits per day',
                        data: activity.map(row => row.value),
                    }
                ]
            },
            options: {
                scales: {
                    x: {
                        type: 'time',
                        min: timeRange[0],
                        max: timeRange[1],
                    }
                },
                elements: {
                    point: {
                        pointStyle: false,
                    },
                },
            },
            plugins: [plugin],
        }
    );
}

/**
 * 
 * @param {[{datetime: string, value: number}]} followers 
 */
function drawFollowing(followers, timeRange) {
    if (!followers || followers.length === 0) {
        document.getElementById("bl-error")
            .innerText = "Note: No followers were tracked."
    }

    const ctx = document.getElementById("bl-canvas");
    let cumSum = 0;
    console.log("Followers Timerange: ", timeRange);
    const chart = new Chart(
        ctx,
        {
            type: 'line',
            data: {
                labels: followers.map(row => {
                    const date = new Date(row.datetime);
                    const fmtDate = `${MONTHS[date.getMonth()]}, ${date.getFullYear()}`
                    return date.getTime();
                }),
                datasets: [
                    {
                        label: 'Change in Twitter followers',
                        data: followers.map(row => {
                            cumSum += row.value;
                            return cumSum;
                        }),
                    }
                ]
            },
            options: {
                elements: {
                    point: {
                        pointStyle: false,
                    },
                },
                scales: {
                    x: {
                        type: 'time',
                        min: timeRange[0],
                        max: timeRange[1],
                    }
                },
            },
            plugins: [plugin],
        }
    );
}

/**
 * 
 * @param {{priceUSD: [{datetime: string, value: number}]|null, devActivity: any|null, twitterFollowers: any|null}} data 
 * @returns {[number]}
 */
function getMaxTimespan(data) {
    const min = [], max = [];

    // Cut off all leading 0s.
    let beginning = true;
    data.devActivity = data.devActivity.reduce((prevVal, row) => {
        if (!beginning) {
            prevVal.push(row);
        } else if (beginning && row.value > 0) {
            beginning = false;
            prevVal.push(row);
        }
        return prevVal;
    }, []);

    if (data.priceUSD && data.priceUSD.length) {
        min.push(new Date(data.priceUSD[0].datetime).getTime());
        max.push(new Date(data.priceUSD.at(-1).datetime).getTime());
    }
    if (data.devActivity && data.devActivity.length) {
        min.push(new Date(data.devActivity[0].datetime).getTime());
        max.push(new Date(data.devActivity.at(-1).datetime).getTime());
    }
    if (data.twitterFollowers && data.twitterFollowers.length) {
        min.push(new Date(data.twitterFollowers[0].datetime).getTime());
        max.push(new Date(data.twitterFollowers.at(-1).datetime).getTime());
    }

    const minMillis = Math.min(...min);
    const maxMillis = Math.max(...max);

    // const dateRange = [];
    // for (let time = minMillis; time <= maxMillis; time + 1000 * 60 * 60 * 24) {
    //     dateRange.push(new Date(time).toISOString());
    // }

    return [minMillis, maxMillis];
}

fetch("/api/" + slug)
    .then(res => res.json())
    .then((json) => {
        const maxTimespan = getMaxTimespan(json.santiment);
        drawPrices(json.santiment.priceUSD, maxTimespan);
        drawDevelopment(json.santiment.devActivity, maxTimespan);
        drawFollowing(json.santiment.twitterFollowers, maxTimespan);
    })
    .catch(console.error)