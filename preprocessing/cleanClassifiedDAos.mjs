import * as fs from "fs";
import allClassified from "../classifiedDAOs.json" with {type: "json"};
import enrDao from "./smallerStructuredEnrichedCoins.json" with {type: "json"};
import defDao from "./definitelyDaos.json" with {type: "json"};
import slugs from "./geckoSantiSlugs.json" with {type: "json"};
console.log("All Prjs: ", allClassified.length)
console.log("Enr Prjs:", enrDao.length)
console.log("Def DAOs: ", defDao.length)
const newDaos = [];
for (const slug of slugs) {
    let lastFound = "";
    for (const dao of defDao) {
        if (slug.trim() === dao.id.trim()) {
            newDaos.push(dao);
            console.log("Found: ", slug)
            lastFound = dao.id;
            break;
        }
    }
    if (lastFound !== slug) {
        console.log("Not found: " + slug);
    }
}
fs.writeFileSync("cleanedClassifiedDaos.json", JSON.stringify(newDaos))
console.log("# of real DAOs:", newDaos.length)