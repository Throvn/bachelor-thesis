import fs from "fs";
if (!Bun.version) {
    throw new Error("Run script with the Bun runtime...\nOtherwise the files cannot be read.")
}

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

console.log("\tChecking if './normalizedAllSantimentProjects.json' is sorted by 'slug'.")
import { compareUTF8 } from "./utf8.js";
let prev = normalizedAllSantimentProjects[0].slug;
for (let i = 1; i < normalizedAllSantimentProjects.length; i++) {
    const curr = normalizedAllSantimentProjects[i].slug;

    if (compareUTF8(prev, curr) > 0) {
        console.error(prev, ">=", curr)
        throw new Error("Santiment projects are not sorted by 'slug' property.");
    }
    prev = curr;
}
console.log("\t\x1b[32mCHECK Successful.\x1b[0m");


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


prev = allSnapshotSpaces[0].symbol;
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
const santimentTicker = new Set()

console.log("7. Matching slugs to get TICKER symbol...")
// First find the ticker of all 'normalizedAllSantimentProjects'.
find(allSantimentProjectNames, normalizedAllSantimentProjects, ["slug"], ["slug"], (val) => {
    santimentTicker.add({
        slug: val.slug,
        ticker: val.ticker
    });
});
// Then use the projects ticker to compare if they have a Snapshot space.
const definitelyDaoSlugs = new Set();
find(santimentTicker, allSnapshotSpaces, ["ticker"], ["symbol"], (space) => {
    definitelyDaoSlugs.add(space.slug);
});

console.info("Out of", santimentTicker.size, "Santiment projects,", definitelyDaoSlugs.size, "are DAOs.");

/**
 * Step 8:
 * Strip the current dataset of all non-DAOs.
 */
console.info("8. Strip '../classifiedProjects.json' of all non-definite DAOs");
console.info("   '../classifiedProjects.json' contains all classifications done over the shared dataset between Coingecko and Santiment regardless of the project being a DAO or not.")

import allClassifiedProjects from "../classifiedProjects.json" with {type: "json"};
import { ChildProcess, exec, execSync } from "child_process";

const definitelyDaos = [];
find(allClassifiedProjects, definitelyDaoSlugs, ["slug"], [], (classifiedDao) => {
    definitelyDaos.push(classifiedDao);
});

console.log("\tAlready", definitelyDaos.length, "DAOs were classified.");
const numberOfRemainingClassifications = definitelyDaoSlugs.size - definitelyDaos.length;
console.log("\tTherefore, only", numberOfRemainingClassifications, "DAOs which were added by Santiment to be classified.")


/**
 * Step 9:
 * Write currently already classified DAOs to 'classifications.json'.
 */
console.log("9. Write already classified DAOs (not projects!) to 'classifications.json'.");
if (fs.existsSync("classifications.json")) {
    console.info("\tFile 'classifications.json' already exists. Skipping...")
} else {
    await Bun.write("classifications.json", JSON.stringify(definitelyDaos));
    console.info("\tWritten 'classifications.json' with already classified DAOs.")
    console.info("\t", definitelyDaos.length, "DAOs.")
}

if (fs.existsSync("remainingClassificationSlugs.json")) {
    console.info("\tFile 'remainingClassificationSlugs.json' already exists. Skipping...")
} else {
    // Take out the already classified ones, because we don't need to reclassifiy them.
    const remainingDaoSlugs = [];
    find(definitelyDaoSlugs, definitelyDaos, [], ["slug"], (found) => { }, (notFoundSlug) => {
        remainingDaoSlugs.push(notFoundSlug);
    });
    await Bun.write("remainingClassificationSlugs.json", JSON.stringify(remainingDaoSlugs));
    console.info("\tWritten 'remainingClassificationSlugs.json' for all of the remaining slugs.")
    console.info("\t", remainingDaoSlugs.length, "DAO slugs.")

    if (remainingDaoSlugs.length !== numberOfRemainingClassifications) {
        throw new Error("Remaining DAO slugs should be equal to number of Remaining Classifications!")
    }
}

/**
 * Step 10:
 */
console.info("------------")
console.info("10. Now you should have classified the remaining DAOs using 'bun run ../server.mjs'.")
console.info("\tThe new classifications were written to '../classifiedProjects2.json'.")
console.info("------------")

/**
 * Step 11: Merge only real classified DAOs together.
 * To make sure that we don't lose any data, we put the additional classifications (which come from the 'whole' Santiment dataset) in an extra file called '../classifiedProjects2.json'.
 */
console.log("11. Merging the already classified DAOs (GeckoSanti Dataset (./classifications.json)) together with the additional ones (../classifiedProjects2.json).")
console.log("\tWrite all of the classified DAOs in one file names 'allClassificationsRaw.json'")
console.log("\t\tjq -s 'flatten | group_by(.slug) | map(reduce .[] as $x ({}; . * $x))' ./classifications.json ../classifiedProjects2.json -cMa > allClassificationsRaw.json")
console.log("\tRemove all projects which have less than 64 priceUSD entries, as this is the minimum required to give a prediction.")
console.log("\t\tjq 'unique_by(.slug) | map(select(.priceUSD | length >= 64))' ./allClassificationsRaw.json > ./allClassifications.json")
console.log("\tOutput of all classified projects with an adequate length and without duplicates: './allClassifications.json'")
console.log("Length of unsanitized dataset:", Number(execSync(`~/./jq "length" ./allClassificationsRaw.json`).toString()))
console.log('Length of complete dataset:', Number(execSync(`~/./jq "length" ./allClassifications.json`).toString()))

/**
 * Step 12:
 * Check the distribution of active/inactive DAOs in the dataset.
 */
console.log(execSync("python ../model/balance.py").toString())