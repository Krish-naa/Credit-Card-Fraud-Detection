// ---------- Build the V1-V28 input fields dynamically ----------
const vGrid = document.getElementById("v-grid");
for (let i = 1; i <= 28; i++) {
    const label = document.createElement("label");
    label.textContent = "V" + i;
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.name = "V" + i;
    input.value = "0";
    input.required = true;
    label.appendChild(input);
    vGrid.appendChild(label);
}

// ---------- Tab switching ----------
document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById("panel-" + tab.dataset.tab).classList.add("active");
    });
});

// ---------- Sample data (a real legitimate transaction) ----------
const SAMPLE = {
    Time: 0, V1: -1.3598071336738, V2: -0.0727811733098497, V3: 2.53634673796914,
    V4: 1.37815522427443, V5: -0.338320769942518, V6: 0.462387777762292,
    V7: 0.239598554061257, V8: 0.0986979012610507, V9: 0.363786969611213,
    V10: 0.0907941719789316, V11: -0.551599533260813, V12: -0.617800855762348,
    V13: -0.991389847235408, V14: -0.311169353699879, V15: 1.46817697209427,
    V16: -0.470400525259478, V17: 0.207971241929242, V18: 0.0257905801985591,
    V19: 0.403992960255733, V20: 0.251412098239705, V21: -0.018306777944153,
    V22: 0.277837575558899, V23: -0.110473910188767, V24: 0.0669280749146731,
    V25: 0.128539358273528, V26: -0.189114843888824, V27: 0.133558376740387,
    V28: -0.0210530534538215, Amount: 149.62
};

const form = document.getElementById("predict-form");

document.getElementById("sample-btn").addEventListener("click", () => {
    for (const [key, value] of Object.entries(SAMPLE)) {
        if (form.elements[key]) form.elements[key].value = value;
    }
});

document.getElementById("reset-btn").addEventListener("click", () => {
    for (const el of form.elements) {
        if (el.name) el.value = el.name === "Amount" ? "149.62" : "0";
    }
    document.getElementById("result").classList.remove("show", "fraud", "legit");
});

// ---------- Animated gauge ----------
const RADIUS = 60;
const CIRC = 2 * Math.PI * RADIUS;
const gaugeFill = document.getElementById("gauge-fill");
gaugeFill.style.strokeDasharray = CIRC;
gaugeFill.style.strokeDashoffset = CIRC;

function animateGauge(prob, isFraud) {
    const color = isFraud ? "#f43f5e" : "#22c55e";
    gaugeFill.style.stroke = color;
    // reset then animate
    gaugeFill.style.strokeDashoffset = CIRC;
    const target = CIRC * (1 - prob);
    requestAnimationFrame(() => {
        gaugeFill.style.strokeDashoffset = target;
    });

    // count-up number
    const numEl = document.getElementById("gauge-num");
    const end = Math.round(prob * 100);
    let cur = 0;
    const step = Math.max(1, Math.round(end / 30));
    clearInterval(numEl._timer);
    numEl._timer = setInterval(() => {
        cur += step;
        if (cur >= end) { cur = end; clearInterval(numEl._timer); }
        numEl.textContent = cur + "%";
    }, 20);
}

// ---------- Predict ----------
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = {};
    for (const el of form.elements) {
        if (el.name) data[el.name] = parseFloat(el.value);
    }

    const resultBox = document.getElementById("result");
    const loading = document.getElementById("loading");
    resultBox.classList.remove("show", "fraud", "legit");
    loading.style.display = "block";

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        loading.style.display = "none";

        if (!res.ok) {
            const err = await res.json();
            resultBox.classList.add("show", "fraud");
            document.getElementById("verdict-tag").innerHTML = "&#9888; Error";
            document.getElementById("verdict-note").textContent = err.detail || ("Request failed (" + res.status + ")");
            document.getElementById("gauge-num").textContent = "--";
            return;
        }

        const out = await res.json();
        const isFraud = out.prediction === 1;
        resultBox.classList.add("show", isFraud ? "fraud" : "legit");

        animateGauge(out.probability, isFraud);

        document.getElementById("verdict-tag").innerHTML = isFraud
            ? "&#128680; Fraudulent"
            : "&#9989; Legitimate";
        document.getElementById("verdict-note").textContent = isFraud
            ? "This transaction shows patterns consistent with fraud. It should be flagged for review."
            : "This transaction appears normal and is unlikely to be fraudulent.";
    } catch (err) {
        loading.style.display = "none";
        resultBox.classList.add("show", "fraud");
        document.getElementById("verdict-tag").innerHTML = "&#9888; Error";
        document.getElementById("verdict-note").textContent = "Request failed: " + err.message;
    }
});

// ---------- Dropzone (batch upload) ----------
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fname = document.getElementById("fname");

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) fname.textContent = "Selected: " + fileInput.files[0].name;
});

["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        fname.textContent = "Selected: " + e.dataTransfer.files[0].name;
    }
});

document.getElementById("csv-form").addEventListener("submit", () => {
    const btn = document.getElementById("upload-btn");
    btn.innerHTML = "Processing...";
    btn.disabled = true;
});
