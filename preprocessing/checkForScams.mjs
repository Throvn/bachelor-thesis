import * as fs from "fs";
// Scamdb taken from: https://github.com/scamsniffer/scam-database/blob/main/blacklist/address.json
const scamAddresses = JSON.parse(fs.readFileSync("./scamdb.json"));
const defDaos = JSON.parse(fs.readFileSync("./definitelyDaos.json"));

let numOfScams = 0;

for (const dao of defDaos) {
    // console.log(dao.name);
    const strategies = dao.snapshot?.strategies;
    if (!strategies) { continue; }
    for (const strategy of strategies) {
        const address = strategy.params?.address
        if (!address) {
            continue;
        }
        for (const scam of scamAddresses) {
            if (scam === address) {
                console.warn(`SCAM FOUND: ${dao.name} (${dao.id})`)
                numOfScams += 1;
                break;
            }
        }
    }
}
console.log("Scams found:", numOfScams);
