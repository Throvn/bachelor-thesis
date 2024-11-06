import fs from "fs";

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

/**
 * 
 * @param {string} path Either: `./allSantimentCoins.json` or `../allCoingeckoCoins.json`
 */
function addToDao(path) {
    /** @type{[{id: string, symbol: string, name: string}]} */
    const allCoins = JSON.parse(fs.readFileSync(path))

    for (const coin of allCoins) {
        // id === slug
        coin.id = coin.id || coin.slug
        coin.symbol = coin.symbol || coin.ticker

        const result = searchSymbolSnapshot(allSpaces, coin.symbol)
        if (result) {
            coin.snapshot = result
            definitelyDaos.push(coin.id)
            totalDaos++
        }
    }
}

addToDao("./allSantimentCoins.json")
addToDao("../allCoingeckoCoins.json")

console.log(definitelyDaos)
const set = new Set(definitelyDaos)

const geckoSantiSlugs = JSON.parse(fs.readFileSync("./geckoSantiSlugs.json"))
const matches = geckoSantiSlugs.filter(value => {
    return set.has(value);
})

console.log("Total DAOs:", totalDaos, set.size)
console.log("Size of current DAOs:", matches.length)
fs.writeFileSync("definitelyDaoSlugs.json", JSON.stringify([...set]))
