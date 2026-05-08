let video;
let scanButton;
let captureButton;
let trainButton;
let labelSelect;
let datasetStatus;
let resultLabel;
let resultDescription;

function queryElements() {
    video = document.getElementById("video");
    scanButton = document.getElementById("scanButton");
    captureButton = document.getElementById("captureButton");
    trainButton = document.getElementById("trainButton");
    labelSelect = document.getElementById("labelSelect");
    datasetStatus = document.getElementById("datasetStatus");
    resultLabel = document.getElementById("resultLabel");
    resultDescription = document.getElementById("resultDescription");
}

async function initCamera() {
    if (!video) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false,
        });
        video.srcObject = stream;
    } catch (error) {
        resultLabel.textContent = "ERROR";
        resultDescription.textContent = "Tidak dapat mengakses kamera. Izinkan akses webcam di browser.";
        scanButton.disabled = true;
        captureButton.disabled = true;
        trainButton.disabled = true;
    }
}

function buildDescription(label) {
    if (label === "organik") {
        return "Sampah organik dapat terurai secara alami. Contoh: sisa makanan, daun, dan sisa tanaman.";
    }
    if (label === "anorganik") {
        return "Sampah anorganik tidak mudah terurai. Contoh: plastik, botol, dan kemasan logam.";
    }
    if (label === "B3") {
        return "Sampah B3 adalah limbah berbahaya yang harus dikelola khusus karena dapat merusak lingkungan dan kesehatan.";
    }
    return "Hasil klasifikasi tidak tersedia.";
}

function getImageData() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.9);
}

async function fetchDatasetStats() {
    try {
        const response = await fetch("/stats");
        const data = await response.json();
        if (response.ok) {
            datasetStatus.textContent = `Dataset: Organik=${data.organik}, Anorganik=${data.anorganik}, B3=${data.B3}`;
        } else {
            datasetStatus.textContent = data.error || "Gagal memuat status dataset.";
        }
    } catch (error) {
        datasetStatus.textContent = "Gagal memuat status dataset.";
    }
}

async function scan() {
    if (!video.videoWidth || !video.videoHeight) {
        resultLabel.textContent = "TUNGGU";
        resultDescription.textContent = "Kamera belum siap. Coba lagi sebentar.";
        return;
    }

    scanButton.disabled = true;
    scanButton.textContent = "MENSKANN...";

    const imageData = getImageData();

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageData }),
        });
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Terjadi kesalahan server.");
        }

        resultLabel.textContent = result.label.toUpperCase();
        resultDescription.textContent = buildDescription(result.label);
    } catch (error) {
        resultLabel.textContent = "ERROR";
        resultDescription.textContent = error.message;
    }

    scanButton.disabled = false;
    scanButton.textContent = "SCAN";
}

async function collectSample() {
    if (!video.videoWidth || !video.videoHeight) {
        datasetStatus.textContent = "Kamera belum siap. Coba lagi sebentar.";
        return;
    }

    captureButton.disabled = true;
    captureButton.textContent = "MENYIMPAN...";

    const imageData = getImageData();
    if (!labelSelect) {
        throw new Error("Elemen label tidak tersedia.");
    }
    const label = labelSelect.value;

    try {
        const response = await fetch("/collect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageData, label }),
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Gagal menyimpan sampel.");
        }
        datasetStatus.textContent = `Dataset: Organik=${result.counts.organik}, Anorganik=${result.counts.anorganik}, B3=${result.counts.B3}`;
    } catch (error) {
        datasetStatus.textContent = error.message;
    }

    captureButton.disabled = false;
    captureButton.textContent = "Simpan Sampel";
}

async function trainModel() {
    const epochs = parseInt(prompt("Jumlah epoch untuk pelatihan:", "10"), 10);
    if (!epochs || epochs <= 0) {
        return;
    }

    trainButton.disabled = true;
    trainButton.textContent = "LATIHING...";
    resultLabel.textContent = "-";
    resultDescription.textContent = "Model sedang dilatih. Tunggu sampai proses selesai...";

    try {
        const response = await fetch("/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ epochs }),
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Gagal melatih model.");
        }
        resultLabel.textContent = "SELESAI";
        resultDescription.textContent = `Latih selesai. Akurasi ${Math.round(result.result.accuracy * 100)}%, validasi ${Math.round(result.result.val_accuracy * 100)}%.`;
        await fetchDatasetStats();
    } catch (error) {
        resultLabel.textContent = "ERROR";
        resultDescription.textContent = error.message;
    }

    trainButton.disabled = false;
    trainButton.textContent = "Latih Model";
}

window.addEventListener("DOMContentLoaded", () => {
    queryElements();
    initCamera();
    fetchDatasetStats();

    if (scanButton) {
        scanButton.addEventListener("click", scan);
    }
    if (captureButton) {
        captureButton.addEventListener("click", collectSample);
    }
    if (trainButton) {
        trainButton.addEventListener("click", trainModel);
    }
});
