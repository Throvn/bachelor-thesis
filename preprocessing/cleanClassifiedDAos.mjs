import * as fs from "fs";
import allClassified from "../classifiedDAOs.json" with {type: "json"};
import enrDao from "./smallerStructuredEnrichedCoins.json" with {type: "json"};
import defDao from "./definitelyDaoSlugs.json" with {type: "json"};
console.log("All Prjs: ", allClassified.length)
console.log("Enr Prjs:", enrDao.length)
console.log("Def DAOs: ", defDao.length)
const newDaos = [];
for (const classified of allClassified) {
    for (const daoSlug of defDao) {
        if (classified.slug === daoSlug) {
            newDaos.push(classified);
            console.log("Found: ", daoSlug)
            break;
        }
    }
}

fs.writeFileSync("cleanedClassifiedDaos.json", JSON.stringify(newDaos))
console.log("# of real DAOs:", newDaos.length)