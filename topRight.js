import { plugin } from "./libs/cursor.js";
import { Octokit, App } from "https://esm.sh/octokit";
const octokit = new Octokit({});

const params = new URLSearchParams(location.search);
const id = params.get('id');

const ctx = document.getElementById("tr-canvas");

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

fetch(`https://api.coingecko.com/api/v3/coins/${id}`)
    .then(res => {
        if (res.status !== 200) {
            throw new Error("[Coingecko] Could not get coin with id");
        }

        return res.json()
    })
    .then(json => json.links.repos_url.github[0])
    .then(url => {
        // get org name from github link
        try {
            const org = url.split("/")[3];
            if (!window.state) window.state = {};
            window.state.org = org;
            return octokit.request('GET /orgs/{org}/repos', {
                org,
                headers: {
                    'X-GitHub-Api-Version': '2022-11-28'
                }
            })
        } catch (e) {
            document.getElementById("tr-error")
                .innerText = e + "\n\n Probably no github repo found.";
            throw new Error('No github repo found...');
        }
    })
    .then(org => {
        console.info(org);
        if (org.status !== 200) {
            throw new Error("[Github] Could not get org");
        }

        // Get the public repo with the most stars
        let bestRepo = { stargazers_count: -1 };
        for (let repo of org.data) {
            if (bestRepo.stargazers_count < repo.stargazers_count) {
                bestRepo = repo;
            }
        }
        console.log(bestRepo);
        return bestRepo;
    })
    .then(repo => {
        return octokit.rest.repos.getCommitActivityStats({
            owner: window.state.org,
            repo: repo.name,
        });
    })
    .then(commitData => {
