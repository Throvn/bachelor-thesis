import fs from "fs";
let skip = 30000;

const queryBuilder = (skip) => `
query Spaces {
    spaces(
        first: 1000,
        skip: ${skip},
        orderBy: "created",
        orderDirection: asc
    ) {
        id
        name
        about
        network
        symbol
        strategies {
            name
            params
        }
        admins
        members
        filters {
            minScore
            onlyMembers
        }
        plugins
    }
}`;

fs.writeFileSync("30000.json", "[")
let isFirstBatch = true

const main = async () => {
    const request = await fetch("https://hub.snapshot.org/graphql", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
        },
        body: JSON.stringify({
            query: queryBuilder(skip),
        }),
    })
    if (!request.ok) {
        console.warn("[Skip: " + skip + "] Request failed: " + await request.text())
        setTimeout(async () => {
            await main()
        }, 1000 * 60)
        return;
    }
    const data = await request.json();
    const spaces = data.data.spaces;
    console.log("First space: ", spaces[0])
    const jsonString = JSON.stringify(spaces, undefined, 2)
    fs.appendFileSync("30000.json",
        (isFirstBatch ? "" : ",") + jsonString.substring(1, jsonString.length - 2))


    isFirstBatch = false
    if (skip < 90000) {
        skip += 1000;
        await main();
    } else {
        // Close file.
        fs.appendFileSync("30000.json", "]")
    }
}

main()
    .then(() => console.log("Finished."))