export const plugin = {
    id: 'corsair',
    defaults: {
        width: 1,
        color: '#FF4949',
        dash: [3, 3],
    },
    afterInit: (chart, args, opts) => {
        chart.corsair = {
            x: 0,
            y: 0,
        }
    },
    afterEvent: (chart, args) => {
        const { inChartArea } = args
        const { type, x, y } = args.event

        chart.corsair = { x, y, draw: inChartArea }
        chart.draw()
    },
    beforeDatasetsDraw: (chart, args, opts) => {
        const { ctx } = chart
        const { top, bottom, left, right } = chart.chartArea
        const { x, y, draw } = chart.corsair
        if (!draw) return

        ctx.save()

        ctx.beginPath()
        ctx.lineWidth = opts.width
        ctx.strokeStyle = opts.color
        ctx.setLineDash(opts.dash)
        ctx.moveTo(x, bottom)
        ctx.lineTo(x, top)
        // ctx.moveTo(left, y)
        // ctx.lineTo(right, y)
        ctx.stroke()

        ctx.restore()
    },
}