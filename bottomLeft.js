const ctx = document.getElementById("bl-canvas");
const data = [];

// const res = new DOMParser(await(await fetch("https://socialblade.com/twitter/user/elonmusk")).text());

console.log("Hello")
fetch("preprocessing/structuredEnrichedCoins.json")
    .then(res => res.json())
    .then(json => {
        console.log(json[0], json.length)
    })
    .catch(console.error)

for (let i = 0; i < 1000; i++) {
    const point = { x: Math.random(), y: Math.random() };
    data.push(point);
}
const chart = new Chart(
    ctx,
    {
        type: 'scatter',
        data: {
            labels: data.map(row => row.x),
            datasets: [
                {
                    label: 'Acquisitions by year',
                    data: data.map(row => row.y)
                }
            ]
        }
    }
);

