document.addEventListener("DOMContentLoaded", () => {
    const mp = document.getElementById("mpStatus");
    const model = document.getElementById("modelStatus");
    const camera = document.getElementById("cameraStatus");

    if (mp || model || camera) {
        fetch("/status")
        .then(r => r.json())
        .then(data => {
            if (mp) mp.textContent = data.mediapipe ? "● MediaPipe Ready" : "● MediaPipe Error";
            if (model) model.textContent = data.model_loaded ? "● Model Ready" : "● Model Not Trained";
            if (camera) camera.textContent = data.camera ? "● Camera Ready" : "● Camera Error";
        })
        .catch(() => {
            if (mp) mp.textContent = "● Status Error";
        });
    }
});