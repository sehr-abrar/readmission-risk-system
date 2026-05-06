const API = "http://127.0.0.1:5000";

async function handleSubmit(event) {
    event.preventDefault();

    const age       = document.getElementById("age").value;
    const gender    = document.getElementById("gender").value;
    const insulin   = document.getElementById("insulin").value;
    const time      = parseFloat(document.getElementById("time").value);
    const meds      = parseFloat(document.getElementById("meds").value);
    const inpatient = parseFloat(document.getElementById("inpatient").value);
    const emergency = parseFloat(document.getElementById("emergency").value);

    if (!age || !gender || !insulin || isNaN(time) || isNaN(meds) || isNaN(inpatient) || isNaN(emergency)) {
        alert("Please fill in all fields.");
        return;
    }

    const btn = document.getElementById("submit-btn");
    btn.disabled = true;
    btn.textContent = "Predicting...";

    try {
        const response = await fetch(`${API}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                age, gender, insulin,
                time_in_hospital: time,
                num_medications: meds,
                number_inpatient: inpatient,
                number_emergency: emergency,
            }),
        });

        const data = await response.json();

        if (data.error) {
            alert("Prediction error: " + data.error);
            return;
        }

        showResult(data.probability);
        loadFeatureImportance();
    } catch (err) {
        alert("Could not reach the backend. Make sure Flask is running on port 5000.\n\n" + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Predict Readmission Risk";
    }
}

function showResult(probability) {
    const pct = Math.round(probability * 100);

    let level, labelText, note;
    if (pct < 20) {
        level = "low";
        labelText = "Low Risk";
        note = "Model estimates a low probability of readmission within 30 days.";
    } else if (pct < 40) {
        level = "medium";
        labelText = "Moderate Risk";
        note = "Model estimates a moderate probability of readmission within 30 days. Follow-up monitoring is advisable.";
    } else {
        level = "high";
        labelText = "High Risk";
        note = "Model estimates a high probability of readmission within 30 days. Close follow-up is strongly recommended.";
    }

    document.getElementById("risk-label").textContent = labelText;
    document.getElementById("risk-label").className = `risk-label ${level}`;
    document.getElementById("risk-score").textContent = `Risk score: ${pct}%`;

    const bar = document.getElementById("risk-bar");
    bar.className = `risk-bar ${level}`;
    bar.style.width = `${pct}%`;

    document.getElementById("result-note").textContent = note;
    document.getElementById("result-card").style.display = "block";
    document.getElementById("result-card").scrollIntoView({ behavior: "smooth" });
}

async function loadFeatureImportance() {
    try {
        const response = await fetch(`${API}/feature-importance`);
        const data = await response.json();
        renderImportance(data.slice(0, 14));
        document.getElementById("importance-card").style.display = "block";
    } catch {
        // silently skip if backend doesn't have this route yet
    }
}

function renderImportance(features) {
    const maxAbs = Math.max(...features.map(f => Math.abs(f.coefficient)));
    const chart = document.getElementById("importance-chart");
    chart.innerHTML = "";

    features.forEach(({ feature, coefficient }) => {
        const pct = (Math.abs(coefficient) / maxAbs) * 50;
        const isPositive = coefficient > 0;

        const row = document.createElement("div");
        row.className = "importance-row";

        const name = document.createElement("div");
        name.className = "importance-name";
        name.textContent = formatFeatureName(feature);
        name.title = feature;

        const track = document.createElement("div");
        track.className = "importance-track";

        const centerLine = document.createElement("div");
        centerLine.className = "importance-center-line";

        const fill = document.createElement("div");
        fill.className = `importance-fill ${isPositive ? "positive" : "negative"}`;
        fill.style.width = `${pct}%`;

        track.appendChild(centerLine);
        track.appendChild(fill);

        const val = document.createElement("div");
        val.className = "importance-value";
        val.textContent = (coefficient > 0 ? "+" : "") + coefficient.toFixed(3);

        row.appendChild(name);
        row.appendChild(track);
        row.appendChild(val);
        chart.appendChild(row);
    });

    const legend = document.createElement("p");
    legend.style.cssText = "font-size:0.78rem;color:#718096;margin-top:14px;text-align:center";
    legend.textContent = "Red bars = increases risk  |  Green bars = decreases risk";
    chart.appendChild(legend);
}

function formatFeatureName(raw) {
    const map = {
        time_in_hospital: "Days in hospital",
        num_medications: "Number of medications",
        number_inpatient: "Prior inpatient visits",
        number_emergency: "Prior emergency visits",
    };
    if (map[raw]) return map[raw];
    return raw
        .replace(/^(age|gender|insulin)_/, (_, prefix) => prefix + ": ")
        .replace(/_/g, " ");
}
