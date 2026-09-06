// Build the V1-V28 input fields dynamically
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

// A real legitimate sample row from the dataset (for quick testing)
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

document.getElementById("sample-btn").addEventListener("click", () => {
    const form = document.getElementById("predict-form");
    for (const [key, value] of Object.entries(SAMPLE)) {
        if (form.elements[key]) form.elements[key].value = value;
    }
});

document.getElementById("predict-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const data = {};
    for (const el of form.elements) {
        if (el.name) data[el.name] = parseFloat(el.value);
    }

    const resultBox = document.getElementById("result");
    resultBox.className = "result";
    resultBox.textContent = "Predicting...";

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        if (!res.ok) {
            const err = await res.json();
            resultBox.className = "result fraud";
            resultBox.textContent = "Error: " + (err.detail || res.status);
            return;
        }
        const out = await res.json();
        resultBox.className = "result " + (out.prediction === 1 ? "fraud" : "legit");
        const pct = (out.probability * 100).toFixed(2);
        resultBox.innerHTML = `<strong>${out.label}</strong><span class="prob">Fraud probability: ${pct}%</span>`;
    } catch (err) {
        resultBox.className = "result fraud";
        resultBox.textContent = "Request failed: " + err.message;
    }
});
