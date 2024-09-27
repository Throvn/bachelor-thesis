import fs from "fs";

/** @type{[{id: string, symbol: string, name: string}]} */
const allCoins = JSON.parse(fs.readFileSync("../allCoingeckoCoins.json"))

function searchSymbolSnapshot(arr, key) {
    const middleIndex = Math.floor(arr.length / 2)
    const middleValue = arr[middleIndex]
    if (middleValue.symbol === key) return middleValue
    if (arr.length <= 1) return false;
    if (middleValue.symbol < key) {
        return searchSymbolSnapshot(arr.slice(middleIndex), key)
    } else if (middleValue.symbol > key) {
        return searchSymbolSnapshot(arr.slice(0, middleIndex), key)
    }
    return false
}

// Spaces sorted by symbol field
const allSpaces = JSON.parse(fs.readFileSync("./sortedSpaces.json"))
const definitelyDaos = []
let totalDaos = 0
for (const coin of allCoins) {
    // id === slug
    coin.id

    const result = searchSymbolSnapshot(allSpaces, coin.symbol)
    if (result) {
        coin.snapshot = result
        definitelyDaos.push(coin)
        totalDaos++
    }
}
console.log("Total DAOs:", totalDaos)
fs.writeFileSync("definitelyDaos.json", JSON.stringify(definitelyDaos))
