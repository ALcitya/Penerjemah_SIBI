// ==========================================================================
// SIBI Sign Language Translator - Interaction Logic
// ==========================================================================

// 1. Tab Switching Functionality
function switchTab(tabName) {
    const cameraSection = document.getElementById("camera-section");
    const fileSection = document.getElementById("file-section");
    const cameraBtn = document.getElementById("tab-camera-btn");
    const fileBtn = document.getElementById("tab-file-btn");

    if (tabName === 'camera') {
        if (cameraSection) cameraSection.classList.add("active");
        if (fileSection) fileSection.classList.remove("active");
        if (cameraBtn) cameraBtn.classList.add("active");
        if (fileBtn) fileBtn.classList.remove("active");
    } else {
        if (fileSection) fileSection.classList.add("active");
        if (cameraSection) cameraSection.classList.remove("active");
        if (fileBtn) fileBtn.classList.add("active");
        if (cameraBtn) cameraBtn.classList.remove("active");
        
        // Stop active camera recording when switching tabs
        stopRekam();
    }
}

// 2. File Input Display Listener
document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("videoUpload");
    const fileNameDisplay = document.getElementById("fileNameDisplay");

    if (fileInput && fileNameDisplay) {
        fileInput.addEventListener("change", function() {
            if (this.files && this.files.length > 0) {
                fileNameDisplay.textContent = "📁 " + this.files[0].name;
                fileNameDisplay.classList.add("has-file");
            } else {
                fileNameDisplay.textContent = "Belum ada berkas dipilih";
                fileNameDisplay.classList.remove("has-file");
            }
        });
    }
});

// 3. Result Display Helper
function updateResultUI(text, isLoading = false) {
    const hasil = document.getElementById("hasil-prediksi");
    const placeholder = document.getElementById("result-placeholder");
    
    if (hasil) {
        hasil.textContent = text;
        if (text && text.trim().length > 0) {
            hasil.style.display = "block";
            if (placeholder) placeholder.style.display = "none";
        } else {
            hasil.style.display = "none";
            if (placeholder) placeholder.style.display = "flex";
        }

        if (isLoading) {
            hasil.classList.add("loading-pulse");
        } else {
            hasil.classList.remove("loading-pulse");
        }
    }
}

// 4. Copy Result to Clipboard
function copyResultText() {
    const hasil = document.getElementById("hasil-prediksi");
    if (hasil && hasil.textContent && hasil.textContent.trim().length > 0) {
        navigator.clipboard.writeText(hasil.textContent).then(() => {
            const btn = document.getElementById("copyResultBtn");
            if (btn) {
                const origText = btn.innerHTML;
                btn.innerHTML = '<i class="fa-solid fa-check" style="color: #10b981;"></i> Tersalin!';
                setTimeout(() => {
                    btn.innerHTML = origText;
                }, 2000);
            }
        }).catch(err => {
            console.error("Gagal menyalin teks:", err);
        });
    }
}

// 5. Video Upload Form Submission
const uploadForm = document.getElementById("UploadForm");
if (uploadForm) {
    uploadForm.addEventListener("submit", async function(event){
        event.preventDefault();

        updateResultUI("Sedang memprediksi ...", true);
        const formData = new FormData(this);
        
        try {
            const response = await fetch(this.action, {
                method: this.method,
                body: formData
            });
            console.log("Status server:", response.status);
            const data = await response.json();
            updateResultUI(data.prediction, false);
        } catch (err) {
            console.error("Error upload:", err);
            updateResultUI("Gagal terhubung ke server atau terjadi kesalahan.", false);
        }
    });
}

// 6. Live Camera Recording Logic
let mediaRecorder = null;
let recordedchunks = [];
let stream = null;

const preview = document.getElementById("preview");
const viewfinderPlaceholder = document.getElementById("viewfinder-placeholder");
const cameraStatusText = document.getElementById("camera-status-text");
const cameraStatusBadge = document.getElementById("camera-status");

function setRecordingUI(isRecording) {
    const mulaiBtn = document.getElementById("mulai");
    const stopBtn = document.getElementById("stop");
    
    if (mulaiBtn) mulaiBtn.style.display = isRecording ? "none" : "inline-flex";
    if (stopBtn) stopBtn.style.display = isRecording ? "inline-flex" : "none";

    if (cameraStatusBadge && cameraStatusText) {
        if (isRecording) {
            cameraStatusBadge.classList.add("recording");
            cameraStatusText.textContent = "Sedang Merekam...";
        } else {
            cameraStatusBadge.classList.remove("recording");
            cameraStatusText.textContent = "Kamera Siap";
        }
    }
}

async function mulaiRekam() {
    try {
        setRecordingUI(true);
        recordedchunks = [];
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        
        if (preview) {
            preview.srcObject = stream;
            preview.style.display = "block";
        }
        if (viewfinderPlaceholder) {
            viewfinderPlaceholder.style.display = "none";
        }

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                recordedchunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const blob = new Blob(recordedchunks, { type: "video/webm" });
            const formData = new FormData();
            formData.append("video", blob, "rekaman.webm");
            
            updateResultUI("Sedang memprediksi ...", true);

            try {
                const response = await fetch("/record_predict", {
                    method: "POST",
                    body: formData
                });
                console.log("Status server:", response.status);
                const data = await response.json();
                console.log("Data hasil:", data);
                updateResultUI(data.prediction, false);
            } catch (err) {
                console.error("Error prediksi rekaman:", err);
                updateResultUI("Terjadi kesalahan saat memproses video rekaman.", false);
            }
        };

        mediaRecorder.start();
    } catch (err) {
        console.error("Gagal mengakses kamera:", err);
        setRecordingUI(false);
        alert("Tidak dapat mengakses kamera. Pastikan izin kamera telah diberikan.");
    }
}

function stopRekam() {
    setRecordingUI(false);

    if (mediaRecorder && mediaRecorder.state !== "inactive"){
        mediaRecorder.stop();
    }
    if (stream){
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    if (preview) {
        preview.srcObject = null;
        preview.style.display = "none";
    }
    if (viewfinderPlaceholder) {
        viewfinderPlaceholder.style.display = "flex";
    }
}

const mulaiBtn = document.getElementById("mulai");
const stopBtn = document.getElementById("stop");
if (mulaiBtn) mulaiBtn.onclick = mulaiRekam;
if (stopBtn) stopBtn.onclick = stopRekam;
