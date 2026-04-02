// ─── Constants ────────────────────────────────────────────────────────
const TICKERS = ['MSFT','AAPL','NVDA','AMD','GOOG','META','TSM','TSLA','PLTR','APP','MCD','COST'];
const PLOTLY_CFG = {responsive: true, displayModeBar: false};
const PLOTLY_LAYOUT_BASE = {
    margin: {l: 50, r: 20, t: 36, b: 40},
    font: {family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif', size: 11},
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
    xaxis: {gridcolor: '#ecf0f1'},
    yaxis: {gridcolor: '#ecf0f1'},
};

// RdYlGn_r colorscale for river bands
function bandColor(i, total) {
    const colors = [
        '#1a9850','#66bd63','#a6d96a','#d9ef8b','#fee08b',
        '#fdae61','#f46d43','#d73027','#a50026'
    ];
    const idx = Math.min(Math.floor(i / total * colors.length), colors.length - 1);
    return colors[idx];
}

// ─── Router ──────────────────────────────────────────────────────────
function initRouter() {
    window.addEventListener('hashchange', route);
    route();
}

function route() {
    const hash = location.hash || '#/';
    updateNavActive(hash);

    if (hash.startsWith('#/stock/')) {
        const ticker = hash.replace('#/stock/', '').toUpperCase();
        showStock(ticker);
    } else {
        showDashboard();
    }
}

function updateNavActive(hash) {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('href') === hash);
    });
}

// ─── Nav Setup ───────────────────────────────────────────────────────
function initNav() {
    const list = document.getElementById('ticker-list');
    TICKERS.forEach(t => {
        const a = document.createElement('a');
        a.href = `#/stock/${t}`;
        a.className = 'nav-btn';
        a.textContent = t;
        list.appendChild(a);
    });
}

// ─── Loading ─────────────────────────────────────────────────────────
function showLoading(msg) {
    document.getElementById('loading-text').textContent = msg || 'Loading...';
    document.getElementById('loading').style.display = 'flex';
}
function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

// ─── API Helper ──────────────────────────────────────────────────────
async function api(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
}

// ─── Dashboard ───────────────────────────────────────────────────────
async function showDashboard() {
    const app = document.getElementById('app');
    app.innerHTML = `
        <div class="fed-status" id="fed-status">
            <div>
                <div class="fed-label">Federal Funds Rate</div>
                <div class="fed-rate" id="fed-rate">--</div>
            </div>
            <div class="fed-target" id="fed-target">Target: -- ~ --</div>
        </div>
        <div class="dashboard-grid">
            <div class="card"><div id="chart-yield" class="chart-container"></div></div>
            <div class="card"><div id="chart-rates" class="chart-container"></div></div>
            <div class="card"><div id="chart-indices" class="chart-container"></div></div>
            <div class="card"><div id="chart-gold" class="chart-container"></div></div>
        </div>
    `;

    showLoading('Fetching dashboard data...');

    // Fetch all in parallel
    const [ratesData, indicesData, goldData] = await Promise.all([
        api('/api/dashboard/rates').catch(e => null),
        api('/api/dashboard/indices').catch(e => null),
        api('/api/dashboard/gold').catch(e => null),
    ]);

    hideLoading();

    if (ratesData) renderRates(ratesData);
    if (indicesData) renderIndices(indicesData);
    if (goldData) renderGold(goldData);
}

function renderRates(data) {
    // Fed status
    const ff = data.fed_funds;
    if (ff) {
        // Find latest effective rate
        let eff = null, lower = null, upper = null;
        for (let i = ff.dates.length - 1; i >= 0; i--) {
            if (eff === null && ff.effective && ff.effective[i] !== null) eff = ff.effective[i];
            if (lower === null && ff.target_lower && ff.target_lower[i] !== null) lower = ff.target_lower[i];
            if (upper === null && ff.target_upper && ff.target_upper[i] !== null) upper = ff.target_upper[i];
            if (eff !== null && lower !== null && upper !== null) break;
        }
        if (eff !== null) document.getElementById('fed-rate').textContent = eff.toFixed(2) + '%';
        else if (lower !== null && upper !== null) document.getElementById('fed-rate').textContent = ((lower+upper)/2).toFixed(2) + '%';
        if (lower !== null && upper !== null) document.getElementById('fed-target').textContent = `Target: ${lower.toFixed(2)}% ~ ${upper.toFixed(2)}%`;
    }

    // Yield Curve
    const yc = data.yield_curve;
    if (yc) {
        Plotly.newPlot('chart-yield', [{
            x: yc.maturities,
            y: yc.yields,
            mode: 'lines+markers',
            line: {color: '#2980b9', width: 2},
            marker: {size: 6},
            fill: 'tozeroy',
            fillcolor: 'rgba(41,128,185,0.12)',
        }], {
            ...PLOTLY_LAYOUT_BASE,
            title: {text: 'US Treasury Yield Curve', font: {size: 13}},
            yaxis: {...PLOTLY_LAYOUT_BASE.yaxis, title: 'Yield (%)'},
        }, PLOTLY_CFG);
    }

    // Rate Expectations
    const traces = [];
    if (ff) {
        // Filter from 2020
        const startIdx = ff.dates.findIndex(d => d >= '2020-01-01');
        const today = new Date().toISOString().slice(0, 10);
        const todayIdx = ff.dates.findIndex(d => d > today);
        const endIdx = todayIdx === -1 ? ff.dates.length : todayIdx;
        const histDates = ff.dates.slice(startIdx, endIdx);

        // Target range band
        if (ff.target_lower && ff.target_upper) {
            const lo = ff.target_lower.slice(startIdx, endIdx);
            const hi = ff.target_upper.slice(startIdx, endIdx);
            traces.push({
                x: histDates, y: hi, mode: 'lines', line: {width: 0, shape: 'hv'},
                showlegend: false, hoverinfo: 'skip',
            });
            traces.push({
                x: histDates, y: lo, mode: 'lines', line: {width: 0, shape: 'hv'},
                fill: 'tonexty', fillcolor: 'rgba(52,152,219,0.2)',
                name: 'Target range', hoverinfo: 'skip',
            });
        }

        // Effective rate
        if (ff.effective) {
            const effDates = [], effVals = [];
            for (let i = startIdx; i < endIdx; i++) {
                if (ff.effective[i] !== null) {
                    effDates.push(ff.dates[i]);
                    effVals.push(ff.effective[i]);
                }
            }
            traces.push({
                x: effDates, y: effVals, mode: 'lines',
                line: {color: '#2c3e50', width: 1.5}, name: 'Effective rate',
            });
        }
    }

    // Futures
    if (data.fed_futures) {
        // Connect from latest effective rate
        let lastEff = null;
        if (ff && ff.effective) {
            for (let i = ff.effective.length - 1; i >= 0; i--) {
                if (ff.effective[i] !== null) { lastEff = ff.effective[i]; break; }
            }
        }
        const futDates = [...data.fed_futures.dates];
        const futRates = [...data.fed_futures.implied_rate];
        if (lastEff !== null) {
            futDates.unshift(new Date().toISOString().slice(0, 10));
            futRates.unshift(lastEff);
        }
        traces.push({
            x: futDates, y: futRates, mode: 'lines+markers',
            line: {color: '#e74c3c', width: 2, dash: 'dash'},
            marker: {size: 4}, name: 'Market-implied path',
        });
    }

    // FOMC dots
    if (data.fomc_dots && data.fomc_dots.dates.length > 0) {
        const dots = data.fomc_dots;
        const errDown = dots.range_low.length ? dots.median.map((m, i) => m - dots.range_low[i]) : [];
        const errUp = dots.range_high.length ? dots.range_high.map((h, i) => h - dots.median[i]) : [];
        const dotTrace = {
            x: dots.dates, y: dots.median, mode: 'markers',
            marker: {color: '#27ae60', size: 8, symbol: 'diamond'},
            name: 'FOMC dots',
        };
        if (errDown.length || errUp.length) {
            dotTrace.error_y = {
                type: 'data', symmetric: false,
                array: errUp.length ? errUp : dots.median.map(() => 0),
                arrayminus: errDown.length ? errDown : dots.median.map(() => 0),
                color: '#27ae60', thickness: 1.5, width: 4,
            };
        }
        traces.push(dotTrace);
    }

    if (traces.length) {
        // Determine x-axis range: extend past last FOMC dot or futures date
        let xEnd = new Date().toISOString().slice(0,10);
        if (data.fomc_dots && data.fomc_dots.dates.length) {
            const lastDot = data.fomc_dots.dates[data.fomc_dots.dates.length - 1];
            if (lastDot > xEnd) xEnd = lastDot;
        }
        if (data.fed_futures && data.fed_futures.dates.length) {
            const lastFut = data.fed_futures.dates[data.fed_futures.dates.length - 1];
            if (lastFut > xEnd) xEnd = lastFut;
        }
        // Add 3 months padding
        const endDate = new Date(xEnd);
        endDate.setMonth(endDate.getMonth() + 3);
        xEnd = endDate.toISOString().slice(0, 10);

        Plotly.newPlot('chart-rates', traces, {
            ...PLOTLY_LAYOUT_BASE,
            title: {text: 'Fed Funds Rate & Market Expectations', font: {size: 13}},
            xaxis: {
                ...PLOTLY_LAYOUT_BASE.xaxis,
                range: ['2020-01-01', xEnd],
                dtick: 'M6',
                tickformat: '%Y-%m',
                tickangle: -30,
            },
            yaxis: {...PLOTLY_LAYOUT_BASE.yaxis, rangemode: 'tozero', title: 'Rate (%)'},
            legend: {font: {size: 9}, x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.7)'},
            shapes: [{
                type: 'line', x0: new Date().toISOString().slice(0,10),
                x1: new Date().toISOString().slice(0,10),
                y0: 0, y1: 1, yref: 'paper',
                line: {color: 'gray', width: 1, dash: 'dash'},
            }],
        }, PLOTLY_CFG);
    }
}

function renderIndices(data) {
    const traces = [];
    const colors = ['#2c3e50', '#e74c3c', '#27ae60', '#8e44ad'];
    let i = 0;
    for (const [sym, info] of Object.entries(data)) {
        if (!info.values.length) continue;
        const base = info.values[0];
        const normalized = info.values.map(v => ((v / base) - 1) * 100);
        const lastPct = normalized[normalized.length - 1].toFixed(1);
        traces.push({
            x: info.dates, y: normalized, mode: 'lines',
            line: {color: colors[i % colors.length], width: 1.5},
            name: `${info.name} (${lastPct > 0 ? '+' : ''}${lastPct}%)`,
        });
        i++;
    }
    // Zero line
    const shapes = [{
        type: 'line', y0: 0, y1: 0, x0: 0, x1: 1,
        xref: 'paper', line: {color: '#bbb', width: 0.5},
    }];
    Plotly.newPlot('chart-indices', traces, {
        ...PLOTLY_LAYOUT_BASE,
        title: {text: 'US Indices (6M % Change)', font: {size: 13}},
        yaxis: {...PLOTLY_LAYOUT_BASE.yaxis, title: 'Change (%)'},
        legend: {font: {size: 9}, x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.7)'},
        shapes: shapes,
    }, PLOTLY_CFG);
}

function renderGold(data) {
    if (!data.dates || !data.dates.length) return;
    Plotly.newPlot('chart-gold', [{
        x: data.dates, y: data.values, mode: 'lines',
        line: {color: '#DAA520', width: 1.5},
        fill: 'tozeroy', fillcolor: 'rgba(218,165,32,0.08)',
    }], {
        ...PLOTLY_LAYOUT_BASE,
        title: {text: 'Gold (GC=F) 6-Month', font: {size: 13}},
        yaxis: {...PLOTLY_LAYOUT_BASE.yaxis, title: 'Price ($)'},
    }, PLOTLY_CFG);
}


// ─── Stock Detail ────────────────────────────────────────────────────

async function showStock(ticker) {
    const app = document.getElementById('app');
    app.innerHTML = `
        <h1 class="stock-header">${ticker} - Stock Analysis</h1>
        <div class="tab-bar" id="tab-bar">
            <button class="tab-btn active" data-tab="financials">Financial Statements</button>
            <button class="tab-btn" data-tab="pe-river">Forward P/E River</button>
            <button class="tab-btn" data-tab="dcf-river">DCF Valuation</button>
            <button class="tab-btn" data-tab="eps">EPS Analysis</button>
        </div>
        <div id="tab-financials" class="tab-content active"></div>
        <div id="tab-pe-river" class="tab-content"></div>
        <div id="tab-dcf-river" class="tab-content"></div>
        <div id="tab-eps" class="tab-content"></div>
    `;

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
        });
    });

    // Show loading in each tab
    ['financials','pe-river','dcf-river','eps'].forEach(t => {
        document.getElementById('tab-' + t).innerHTML =
            '<div style="text-align:center;padding:60px;color:#95a5a6;"><div class="spinner" style="margin:0 auto 12px;"></div>Loading...</div>';
    });

    // Single combined API call — fetches everything once
    try {
        const data = await api(`/api/stock/${ticker}/all`);

        if (data.financials) renderFinancials('tab-financials', data.financials);
        else document.getElementById('tab-financials').innerHTML = '<p style="padding:20px;color:#95a5a6;">No financial data</p>';

        if (data.eps) renderEPS('tab-eps', data.eps, ticker);
        else document.getElementById('tab-eps').innerHTML = '<p style="padding:20px;color:#95a5a6;">No EPS data</p>';

        if (data.pe_river && !data.pe_river.error) renderPERiver('tab-pe-river', data.pe_river, ticker);
        else document.getElementById('tab-pe-river').innerHTML = `<p style="padding:20px;color:#95a5a6;">${data.pe_river?.error || 'No data'}</p>`;

        if (data.dcf_river && !data.dcf_river.error) renderDCFRiver('tab-dcf-river', data.dcf_river, ticker);
        else document.getElementById('tab-dcf-river').innerHTML = `<p style="padding:20px;color:#95a5a6;">${data.dcf_river?.error || 'No data'}</p>`;
    } catch(e) {
        ['financials','pe-river','dcf-river','eps'].forEach(t => {
            document.getElementById('tab-' + t).innerHTML =
                `<p style="color:#e74c3c;padding:20px;">Failed to load: ${e.message}</p>`;
        });
    }
}


// ─── Financial Statements ────────────────────────────────────────────
function renderFinancials(containerId, data) {
    const container = document.getElementById(containerId);

    container.innerHTML = `
        <div class="fin-freq-tabs">
            <button class="fin-freq-btn active" data-freq="annual">Annual</button>
            <button class="fin-freq-btn" data-freq="quarterly">Quarterly</button>
        </div>
        <div id="fin-annual" class="fin-table-wrap"></div>
        <div id="fin-quarterly" class="fin-table-wrap" style="display:none;"></div>
    `;

    // Toggle buttons
    container.querySelectorAll('.fin-freq-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            container.querySelectorAll('.fin-freq-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('fin-annual').style.display = btn.dataset.freq === 'annual' ? '' : 'none';
            document.getElementById('fin-quarterly').style.display = btn.dataset.freq === 'quarterly' ? '' : 'none';
        });
    });

    if (data.annual) buildFinTable('fin-annual', data.annual);
    if (data.quarterly) buildFinTable('fin-quarterly', data.quarterly);
}

function buildFinTable(containerId, tableData) {
    if (!tableData || !tableData.rows || !tableData.rows.length) return;
    const dates = tableData.dates;

    let html = '<table><thead><tr><th>Item</th>';
    dates.forEach(d => { html += `<th>${d}</th>`; });
    html += '</tr></thead><tbody>';

    tableData.rows.forEach(row => {
        if (row.is_header) {
            html += `<tr class="section-header"><td colspan="${dates.length + 1}">${row.label.replace(/---/g, '').trim()}</td></tr>`;
        } else {
            html += `<tr><td>${row.label}</td>`;
            dates.forEach(d => {
                const full_d = Object.keys(row.values).find(k => k.startsWith(d));
                const val = full_d ? row.values[full_d] : '--';
                html += `<td>${val}</td>`;
            });
            html += '</tr>';
        }
    });

    html += '</tbody></table>';
    document.getElementById(containerId).innerHTML = html;
}


// ─── PE River ────────────────────────────────────────────────────────
function renderPERiver(containerId, data, ticker) {
    const container = document.getElementById(containerId);
    container.innerHTML = '<div id="chart-pe" class="chart-container" style="height:500px;"></div>';

    const traces = [];
    const n = data.bands.length;

    // Bands (bottom to top = highest PE first)
    for (let i = n - 1; i >= 0; i--) {
        const band = data.bands[i];
        // Upper boundary
        traces.push({
            x: data.dates, y: band.lower, mode: 'lines',
            line: {width: 0}, showlegend: false, hoverinfo: 'skip',
        });
        traces.push({
            x: data.dates, y: band.upper, mode: 'lines',
            line: {width: 0}, fill: 'tonexty',
            fillcolor: hexToRgba(bandColor(i, n), 0.5),
            name: band.label,
        });
    }

    // Price
    traces.push({
        x: data.dates, y: data.price, mode: 'lines',
        line: {color: 'black', width: 2}, name: 'Price',
        connectgaps: false,
    });

    Plotly.newPlot('chart-pe', traces, {
        ...PLOTLY_LAYOUT_BASE,
        title: {text: `${ticker} Forward P/E River`, font: {size: 14}},
        yaxis: {...PLOTLY_LAYOUT_BASE.yaxis, title: 'Price ($)'},
        legend: {font: {size: 8}, x: 1.02, y: 0.5, xanchor: 'left'},
        margin: {l: 50, r: 160, t: 40, b: 40},
        shapes: [{
            type: 'line', x0: data.today, x1: data.today,
            y0: 0, y1: 1, yref: 'paper',
            line: {color: 'gray', width: 1, dash: 'dash'},
        }],
    }, PLOTLY_CFG);
}


// ─── DCF River ───────────────────────────────────────────────────────
function renderDCFRiver(containerId, data, ticker) {
    const container = document.getElementById(containerId);
    container.innerHTML = '<div id="chart-dcf" class="chart-container" style="height:550px;"></div>';

    const traces = [];
    const n = data.bands.length;

    // Bands
    for (let i = n - 1; i >= 0; i--) {
        const band = data.bands[i];
        traces.push({
            x: data.dates, y: band.lower, mode: 'lines',
            line: {width: 0}, showlegend: false, hoverinfo: 'skip',
        });
        traces.push({
            x: data.dates, y: band.upper, mode: 'lines',
            line: {width: 0}, fill: 'tonexty',
            fillcolor: hexToRgba(bandColor(i, n), 0.5),
            name: band.label,
        });
    }

    // IV line
    if (data.iv_line) {
        traces.push({
            x: data.dates, y: data.iv_line, mode: 'lines',
            line: {color: 'purple', width: 2, dash: 'dash'},
            name: `IV fair value (ERP=${data.iv_erp_now}%)`,
        });
    }

    // Price
    traces.push({
        x: data.dates, y: data.price, mode: 'lines',
        line: {color: 'black', width: 2}, name: 'Price',
        connectgaps: false,
    });

    // EPS on secondary Y-axis
    if (data.actual_eps && data.actual_eps.dates.length) {
        traces.push({
            x: data.actual_eps.dates, y: data.actual_eps.values,
            mode: 'lines+markers', yaxis: 'y2',
            line: {color: '#8B4513', width: 2},
            marker: {size: 3, symbol: 'square'},
            name: 'Actual EPS (trailing 4Q)',
        });
    }
    if (data.estimate_eps && data.estimate_eps.dates.length) {
        traces.push({
            x: data.estimate_eps.dates, y: data.estimate_eps.values,
            mode: 'lines+markers', yaxis: 'y2',
            line: {color: '#4169E1', width: 2, dash: 'dash'},
            marker: {size: 3, symbol: 'diamond'},
            name: 'Estimate EPS (fwd 4Q)',
        });
    }

    // Title
    let titleParts = [`${ticker} DCF Fair Value`];
    if (data.implied_erp !== null) titleParts.push(`Price ERP: ${data.implied_erp}%`);
    if (data.iv_erp_now !== null) titleParts.push(`IV ERP: ${(data.iv_erp_now * 100).toFixed(1)}%`);

    Plotly.newPlot('chart-dcf', traces, {
        ...PLOTLY_LAYOUT_BASE,
        title: {text: titleParts.join(' | '), font: {size: 13}},
        yaxis: {...PLOTLY_LAYOUT_BASE.yaxis, title: 'Price ($)', range: [0, data.y_top]},
        yaxis2: {
            title: 'EPS ($)', overlaying: 'y', side: 'right',
            titlefont: {color: '#8B4513'}, tickfont: {color: '#8B4513'},
            gridcolor: 'rgba(0,0,0,0)',
        },
        legend: {font: {size: 8}, x: 1.12, y: 0.5, xanchor: 'left'},
        margin: {l: 50, r: 180, t: 40, b: 40},
        shapes: [{
            type: 'line', x0: data.today, x1: data.today,
            y0: 0, y1: 1, yref: 'paper',
            line: {color: 'gray', width: 1, dash: 'dash'},
        }],
    }, PLOTLY_CFG);
}


// ─── EPS Bar Chart ───────────────────────────────────────────────────
function renderEPS(containerId, data, ticker) {
    const container = document.getElementById(containerId);
    container.innerHTML = '<div id="chart-eps" class="chart-container" style="height:450px;"></div>';

    // Split past (has actual) vs future
    const pastIdx = [], futureIdx = [];
    for (let i = 0; i < data.actual.length; i++) {
        if (data.actual[i] !== null) pastIdx.push(i);
        else if (data.estimate[i] !== null) futureIdx.push(i);
    }
    // Show last 12 past + up to 8 future
    const showPast = pastIdx.slice(-12);
    const showFuture = futureIdx.slice(0, 8);
    const showIdx = [...showPast, ...showFuture];

    const quarters = showIdx.map(i => data.quarters[i]);
    const estimates = showIdx.map(i => data.estimate[i]);
    const actuals = showIdx.map(i => data.actual[i]);

    // Beat/miss colors for actual bars
    const actualColors = actuals.map((a, i) => {
        if (a === null) return 'rgba(0,0,0,0)';
        return a >= estimates[i] ? '#2ecc71' : '#e74c3c';
    });
    const actualVals = actuals.map(a => a === null ? 0 : a);

    const traces = [
        {
            x: quarters, y: estimates, type: 'bar', name: 'Estimate',
            marker: {color: '#95a5a6', opacity: 0.7},
        },
        {
            x: quarters, y: actualVals, type: 'bar', name: 'Actual',
            marker: {color: actualColors, opacity: 0.9},
        },
    ];

    const shapes = [];
    if (showPast.length > 0 && showFuture.length > 0) {
        // Divider between past and future
        const divIdx = showPast.length - 0.5;
        shapes.push({
            type: 'line', x0: divIdx, x1: divIdx,
            y0: 0, y1: 1, yref: 'paper',
            line: {color: 'gray', width: 1, dash: 'dash'},
        });
    }

    Plotly.newPlot('chart-eps', traces, {
        ...PLOTLY_LAYOUT_BASE,
        title: {text: `${ticker} Quarterly EPS: Actual vs Estimate`, font: {size: 14}},
        barmode: 'group',
        yaxis: {...PLOTLY_LAYOUT_BASE.yaxis, title: 'EPS ($)'},
        xaxis: {...PLOTLY_LAYOUT_BASE.xaxis, tickangle: -45, tickfont: {size: 10}},
        legend: {font: {size: 10}},
        shapes: shapes,
        annotations: showFuture.length > 0 ? [{
            x: quarters[showPast.length],
            y: 1, yref: 'paper', yanchor: 'top',
            text: 'Future &rarr;', showarrow: false,
            font: {size: 10, color: 'gray'},
        }] : [],
    }, PLOTLY_CFG);
}


// ─── Utility ─────────────────────────────────────────────────────────
function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}


// ─── Init ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initNav();
    initRouter();
});
