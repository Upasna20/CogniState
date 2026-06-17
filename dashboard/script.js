const API = "http://localhost:5050";

async function startDaemon() {
    await fetch(`${API}/start`, {
        method: "POST"
    });
}

async function stopDaemon() {
    await fetch(`${API}/stop`, {
        method: "POST"
    });
}

function updateStatus(data) {

    const statusEl = document.getElementById("status");

    statusEl.className = "status";

    const map = {
        normal: "🟢 NORMAL",
        stress: "🔴 STRESS",
        calibrating: "🟠 CALIBRATING",
        idle: "⚪ IDLE"
    };

    statusEl.textContent =
        map[data.status] || data.status;

    if (data.status === "normal")
        statusEl.classList.add("normal");

    if (data.status === "stress")
        statusEl.classList.add("stress");

    if (data.status === "calibrating")
        statusEl.classList.add("calibrating");

    document.getElementById("statusText")
        .textContent = data.status_text;

    document.getElementById("calibration")
        .textContent = `${data.cal_count}/4`;

    document.getElementById("windows")
        .textContent = data.window_count;

    document.getElementById("normalCount")
        .textContent = data.normal_count;

    document.getElementById("stressCount")
        .textContent = data.stress_count;

    document.getElementById("lastResult")
        .textContent = data.last_result || "-";
}

function updateFeatures(features) {

    const tbody =
        document.querySelector("#featuresTable tbody");

    tbody.innerHTML = "";

    for (const [key, value] of Object.entries(features)) {

        const row = document.createElement("tr");

        row.innerHTML = `
            <td>${key}</td>
            <td>${value}</td>
        `;

        tbody.appendChild(row);
    }
}

function updateHistory(history) {

    const container =
        document.getElementById("history");

    container.innerHTML = "";

    history
        .slice()
        .reverse()
        .forEach(item => {

            const div =
                document.createElement("div");

            div.className =
                "history-item";

            div.innerHTML = `
                Window ${item.window}
                |
                ${item.result}
                |
                ${item.time}
            `;

            container.appendChild(div);
        });
}

function updateLogs(logs) {

    const container =
        document.getElementById("logs");

    container.innerHTML = "";

    logs.forEach(log => {

        const div =
            document.createElement("div");

        div.className =
            "log-entry";

        div.textContent =
            `[${log.time}] ${log.msg}`;

        container.appendChild(div);
    });
}

async function fetchStatus() {

    try {

        const response =
            await fetch(`${API}/status`);

        const data =
            await response.json();

        updateStatus(data);
        updateFeatures(data.last_features);
        updateHistory(data.history);
        updateLogs(data.log);

    } catch (err) {

        document.getElementById("status")
            .textContent =
            "Backend Offline";
    }
}

fetchStatus();

setInterval(fetchStatus, 2000);