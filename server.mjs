import * as fs from "fs";
import path from "path";
import { cwd } from "process";
import express from "express";
import bodyParser from "body-parser";
const app = express();

app.use(express.static("./"))
app.use(bodyParser.json({}));

/**
 * @type {[string]}
 */
import allDAOs from "./preprocessing/normalizedAllSantimentProjects.json" with {type: "json"};
function searchBySlug(arr, key) {
    const middleIndex = Math.floor(arr.length / 2)
    const middleValue = arr[middleIndex]
    // console.log(middleValue.slug, key)
    if (middleValue.slug === key) return middleValue
    if (arr.length <= 1) return false;
    if (middleValue.slug < key) {
        return searchBySlug(arr.slice(middleIndex), key)
    } else if (middleValue.slug > key) {
        return searchBySlug(arr.slice(0, middleIndex), key)
    }
    return false
}

app.get("/api/:slug", (req, res) => {
    console.log(req.params.slug);
    const data = searchBySlug(allDAOs, req.params.slug);
    return res.send(data);
});

app.post("/api/:slug", (req, res) => {
    if (!Object.keys(req.body).includes("timestamp")) {
        res.status(400).send({
            msg: "timestamp field missing",
            error: "provide the timestamp at which the dao turned inactive. or 0/1 for a DAO being in-/active through the entire timespan."
        });
        return;
    }

    const data = searchBySlug(allDAOs, req.params.slug);
    data.santiment["isActive"] = [];
    const prediction = new Date(req.body.timestamp).getTime()
    for (const price of data.santiment.priceUSD) {
        const time = new Date(price["datetime"]).getTime()
        if (prediction > 1) {
            // Before the time event coin was active.
            // After the time, coin was inactive.
            // Works, because date format goes from biggest to smallest unit and prefixes single digits with 0s.
            data.santiment.isActive.push(Number(time < prediction));
        } else {
            data.santiment.isActive.push(prediction);
        }
        // trainingData.push({
        //     id,
        //     prediction,
        //     prices,
        // });
    }
    console.log("isActive Length: ", data.santiment.isActive.length);
    console.log("priceUSD Length: ", data.santiment.priceUSD.length);

    try {
        fs.appendFileSync("classifiedProjects3.json", ",\n" + JSON.stringify({
            slug: req.params.slug,
            ...data.santiment
        }));
        console.log("Classified '" + req.params.slug + "'");
        res.status(200).send();
    } catch (e) {
        res.status(500).send();
    }
})

app.get("/", (req, res) => {
    res.sendFile(path.join(cwd(), "index.html"))
});

app.listen(3000, () => {
    console.log("Listening on port *:3000")
});