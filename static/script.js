// function uploadPDF() {
//     let fileInput = document.getElementById("pdfUpload");
//     let formData = new FormData();
//     formData.append("file", fileInput.files[0]);
//
//     fetch("/upload", {
//         method: "POST",
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => alert(data.message))
//     .catch(error => console.error("Error:", error));
// }
//
// function askQuestion() {
//     let question = document.getElementById("questionInput").value;
//
//     fetch("/ask", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ question: question })
//     })
//     .then(response => response.json())
//     .then(data => {
//         document.getElementById("answerOutput").innerText = data.answer;
//     })
//     .catch(error => console.error("Error:", error));
// }








// function uploadPDF() {
//     let fileInput = document.getElementById("pdfUpload");
//     let formData = new FormData();
//     formData.append("file", fileInput.files[0]);

//     fetch("/upload", {
//         method: "POST",
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         alert(data.message);
//     })
//     .catch(error => console.error("Error:", error));
// }

// function askQuestion() {
//     let question = document.getElementById("questionInput").value;
//     let answerOutput = document.getElementById("answerOutput");
//     let loader = document.getElementById("loader");

//     if (question.trim() === "") {
//         alert("Please enter a question.");
//         return;
//     }

//     // Show Loader
//     loader.style.display = "block";
//     answerOutput.innerText = "";

//     fetch("/ask", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ question: question })
//     })
//     .then(response => response.json())
//     .then(data => {
//         // Hide Loader
//         loader.style.display = "none";
//         answerOutput.innerText = data.answer;
//     })
//     .catch(error => {
//         console.error("Error:", error);
//         loader.style.display = "none";
//         answerOutput.innerText = "Error fetching answer.";
//     });
// }









// function uploadPDF() {
//     let fileInput = document.getElementById("pdfUpload");
//     let progressBar = document.getElementById("uploadProgress");
//     let progressText = document.getElementById("progressText");
//     let formData = new FormData();

//     if (fileInput.files.length === 0) {
//         alert("Please select a file to upload.");
//         return;
//     }

//     formData.append("file", fileInput.files[0]);

//     let xhr = new XMLHttpRequest();
//     xhr.open("POST", "/upload", true);

//     // Update progress bar during upload
//     xhr.upload.onprogress = function(event) {
//         if (event.lengthComputable) {
//             let percentComplete = Math.round((event.loaded / event.total) * 100);
//             progressBar.style.width = percentComplete + "%";
//             progressText.innerText = percentComplete + "% Uploaded";
//         }
//     };

//     xhr.onload = function() {
//         if (xhr.status === 200) {
//             progressText.innerText = "✅ File Uploaded Successfully!";
//             progressBar.style.width = "100%";
//         } else {
//             progressText.innerText = "❌ Upload Failed!";
//             progressBar.style.width = "0%";
//         }
//     };

//     xhr.onerror = function() {
//         progressText.innerText = "❌ Network Error!";
//         progressBar.style.width = "0%";
//     };

//     // Show progress bar and start upload
//     document.getElementById("uploadStatus").style.display = "block";
//     xhr.send(formData);
// }

// function askQuestion() {
//     let question = document.getElementById("questionInput").value;
//     let answerOutput = document.getElementById("answerOutput");
//     let loader = document.getElementById("loader");

//     if (question.trim() === "") {
//         alert("Please enter a question.");
//         return;
//     }

//     // Show Loader
//     loader.style.display = "block";
//     answerOutput.innerText = "";

//     fetch("/ask", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ question: question })
//     })
//     .then(response => response.json())
//     .then(data => {
//         // Hide Loader
//         loader.style.display = "none";
//         answerOutput.innerText = data.answer;
//     })
//     .catch(error => {
//         console.error("Error:", error);
//         loader.style.display = "none";
//         answerOutput.innerText = "Error fetching answer.";
//     });
// }








