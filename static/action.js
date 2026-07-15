// todo - action form
document.getElementById("UploadForm").addEventListener("submit", async function(event){
    event.preventDefault();

    const hasil = document.getElementById("hasil-prediksi");
    hasil.textContent = "Sedang memprediksi ...";
    const formData = new FormData(this);
    const response = await fetch(this.action, {
        method:this.method,
        body:formData
    });
    console.log("Status:", response.status);
    const data = await response.json();
    hasil.textContent = data.prediction;
});
// todo - perekaman langsung
let mediaRecorder
let recordedchunks=[]
let stream=null

const preview =document.getElementById("preview")
async function mulaiRekam(){
    preview.style.display="block"
    document.getElementById("mulai").style.display="none"

    recordedchunks=[]
    stream= await navigator.mediaDevices.getUserMedia({video:true})
    preview.srcObject=stream
    mediaRecorder=new MediaRecorder(stream)
    mediaRecorder.ondataavailable=event=>{
        if(event.data.size>0){
            recordedchunks.push(event.data)}
    }
    mediaRecorder.onstop=async()=>{
        const blob = new Blob(recordedchunks,{type:"video/webm"})
        const formData = new FormData()
        formData.append("video",blob,"rekaman.webm")
        document.getElementById("hasil-prediksi").textContent="Sedang memprediksi ..."
        const response = await fetch("/record_predict",{
            method:"POST",
            body:formData})
        console.log("Status:", response.status)
        const data = await response.json()
        console.log("Data:", data)
        const hasil = document.getElementById("hasil-prediksi")
        hasil.textContent = data.prediction
        }
    mediaRecorder.start()
}
function stopRekam(){
    document.getElementById("stop").style.display="block"
    preview.style.display="none"

    if(mediaRecorder && mediaRecorder.state !=="inactive"){
        mediaRecorder.stop()
    }
    if(stream){
        stream.getTracks().forEach(track=>track.stop())
        preview.srcObject=null
        preview.style.display="none"
    }
}
document.getElementById("mulai").onclick=mulaiRekam
document.getElementById("stop").onclick=stopRekam

