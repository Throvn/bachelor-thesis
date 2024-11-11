function find(arr1 = [], arr2 = [], path1 = [], path2 = [], foundCallback, notFoundCallback = () => { }) {
    const getValueByPath = (obj, path) => path.reduce((acc, key) => acc && acc[key], obj);

    for (let item1 of arr1) {
        let found = false;
        for (let item2 of arr2) {
            const value1 = getValueByPath(item1, path1);
            const value2 = getValueByPath(item2, path2);

            if (value1 === value2) {
                foundCallback(item1);
                found = true;
                break;
            }
        }
        if (!found) {
            notFoundCallback(item1);
        }
    }
}


/** Step 1:
 * Get all available Coingecko Projects.
 */
import allGeckoProjects from "../allCoingeckoCoins.json" with { type: "json" };
console.log("1. Length of all CoinGecko Projects:", allGeckoProjects.length);
console.assert(allGeckoProjects, "NO COINGECKO PROJECTS FOUND")

/** Step 2:
 * Get all available Santiment Projects.
 */
import allSantimentProjectNames from "./allSantimentCoins.json" with {type: "json"};
console.log("2. Length of all Santiment Projects:", allSantimentProjectNames.length);
console.assert(allSantimentProjectNames, "NO SANTIMENT PROJECT NAMES FOUND")

/**
 * Step 3:
 * Since we can only use Santiment (because they have all of the required Data), we actually don't need any Gecko projects.
 * But just out of curiousity, how many of CoinGecko Projects are covered in Santiment?
 */
let projectsInBoth = 0;
find(allGeckoProjects, allSantimentProjectNames, ["id"], ["slug"], () => projectsInBoth++);
console.log("3. Length of Projects available in both:", projectsInBoth);

/**
 * Step 4:
 * Continue the normal process.
 * Fetch all necessary metadata from Santiment for all Santiment projects.
 * This includes also projects which are NOT DAOs.
 */
console.log("4. Did you source all Projects through 'santiment.mjs' and turn it into valid json?");
import allSantimentProjects from "./allSantimentProjects.json" with {type: "json"};
const projectNameSet = new Set();
console.log("\tChecking for duplicates...")
for (const project of allSantimentProjects) {
    if (projectNameSet.has(project.slug)) {
        throw new Error(`Duplicate slug '${project.slug}' found in './allSantimentProjects.json'`);
    }
    projectNameSet.add(project.slug);
}
console.log("\t\x1b[32mCHECK Successful.\x1b[0m")
console.log("\tThere are a total of:", allSantimentProjects.length, "projects available. Probably not all of them are DAOs though.");
console.log("   \tThis file can also be checked for duplicates using:");
console.log(`   \t\tcat allSantimentProjects.json|jq -sc ".[] | group_by(.)[] | select(length > 1) | {key: first|tostring, value: length}"`);

/**
 * Step 5:
 * Take the scraped projects and reform them to easier understand JSON.
 * Aka. merge pricesUSD1 and pricesUSD2, devActivity1 and devActivity2, ...
 * Use jq for that.
 */
console.log("5. Did you merge the time series arrays together using this command?");
console.log(`   \tcat allSantimentProjects.json|jq -f structuredEnrichedCoins.jq -rc > normalizedAllSantimentProjects.json`);
import normalizedAllSantimentProjects from "./normalizedAllSantimentProjects.json" with {type: "json"};
for (const sample of normalizedAllSantimentProjects) {
    if (Object.keys(sample.santiment).join("").match(/1|2/g)) {
        console.log(Object.keys(sample.santiment))
        throw new Error("Santiment projects are in wrong format. '.santiment.keys' include numbers.")
    }
}
console.log("\t\x1b[32mCHECK Successful.\x1b[0m")


/**
 * Step 6:
 * Get all Snapshot Spaces.
 * Also sort them (using jq) to have faster comparison access when comparing the ticker later on.
 * SORT THEM BY SYMBOL, as this is the matching criterion. 
 */
console.log("6. Did you source all Snapshot spaces through 'snapshotScaraper.mjs' and turn it into valid json?");
import allSnapshotSpaces from "./sortedSpaces.json" with {type: "json"};
console.log("   Length of all Snapshot Spaces:", allSnapshotSpaces.length)
if (!allSnapshotSpaces.length) {
    throw new Error("NO SNAPSHOT SPACES FOUND")
}


import { compareUTF8 } from "./utf8";
let prev = allSnapshotSpaces[0].symbol;
for (let i = 1; i < allSnapshotSpaces.length; i++) {
    const curr = allSnapshotSpaces[i].symbol;

    if (compareUTF8(prev, curr) > 0) {
        console.error(prev, ">=", curr)
        throw new Error("Snapshot spaces are not sorted by 'symbol' property.");
    }
    prev = curr;
}
console.log("\t\x1b[32mCHECK Successful.\x1b[0m");

/**
 * Step 7:
 * Remove all projects from 'normalizedAllSantimentProjects.json' which don't have a matching Snapshot Ticker symbol.
 * Ticker Symbols for the Santiment projects can be found in 'normalizedAllSantimentCoins.json'.
 * We use python as this is better for handling larger files by default.
 */
