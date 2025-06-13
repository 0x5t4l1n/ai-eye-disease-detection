const API_BASE_URL = 'http://localhost:5000';

function predictSingle() {
    const fileInput = document.getElementById('singleImage');
    const returnProbabilities = document.getElementById('singleProbabilities').checked;
    const resultDiv = document.getElementById('singleResult');
    
    if (!fileInput.files[0]) {
        resultDiv.innerHTML = '<p class="error">Please select an image.</p>';
        return;
    }
    
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    formData.append('return_probabilities', returnProbabilities);
    
    resultDiv.innerHTML = 'Processing...';
    
    fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            let html = `<p><strong>Filename:</strong> ${data.filename}</p>
                        <p><strong>CDR:</strong> ${data.data.cdr}</p>
                        <p><strong>Severity:</strong> ${data.data.severity}</p>
                        <p><strong>Description:</strong> ${data.data.severity_description}</p>
                        <p><strong>Confidence:</strong> ${(data.data.confidence * 100).toFixed(2)}%</p>
                        <p><strong>Risk Level:</strong> ${data.data.risk_level}</p>`;
            if (data.data.probabilities) {
                html += '<p><strong>Probabilities:</strong></p><ul>';
                for (const [key, value] of Object.entries(data.data.probabilities)) {
                    html += `<li>${key}: ${(value * 100).toFixed(2)}%</li>`;
                }
                html += '</ul>';
            }
            resultDiv.innerHTML = html;
        } else {
            resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
        }
    })
    .catch(error => {
        resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    });
}

function predictBatch() {
    const fileInput = document.getElementById('batchImages');
    const returnProbabilities = document.getElementById('batchProbabilities').checked;
    const resultDiv = document.getElementById('batchResult');
    
    if (!fileInput.files.length) {
        resultDiv.innerHTML = '<p class="error">Please select at least one image.</p>';
        return;
    }
    
    const formData = new FormData();
    Array.from(fileInput.files).forEach((file, index) => {
        formData.append(`image${index}`, file);
    });
    formData.append('return_probabilities', returnProbabilities);
    
    resultDiv.innerHTML = 'Processing...';
    
    fetch(`${API_BASE_URL}/api/predict-batch`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            let html = `<p><strong>Total Images:</strong> ${data.total_images}</p>
                        <p><strong>Successful:</strong> ${data.successful_predictions}</p>
                        <p><strong>Failed:</strong> ${data.failed_predictions}</p>`;
            data.data.forEach(result => {
                html += `<div>
                    <p><strong>Image ${result.image_index}:</strong> ${result.filename}</p>`;
                if (result.success) {
                    html += `<p>CDR: ${result.cdr}</p>
                             <p>Severity: ${result.severity}</p>
                             <p>Description: ${result.severity_description}</p>
                             <p>Confidence: ${(result.confidence * 100).toFixed(2)}%</p>
                             <p>Risk Level: ${result.risk_level}</p>`;
                    if (result.probabilities) {
                        html += '<p>Probabilities:</p><ul>';
                        for (const [key, value] of Object.entries(result.probabilities)) {
                            html += `<li>${key}: ${(value * 100).toFixed(2)}%</li>`;
                        }
                        html += '</ul>';
                    }
                } else {
                    html += `<p class="error">Error: ${result.error}</p>`;
                }
                html += '</div>';
            });
            resultDiv.innerHTML = html;
        } else {
            resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
        }
    })
    .catch(error => {
        resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    });
}

function getModelInfo() {
    const resultDiv = document.getElementById('modelInfo');
    resultDiv.innerHTML = 'Loading...';
    
    fetch(`${API_BASE_URL}/api/model-info`)
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            resultDiv.innerHTML = `<p><strong>Model Type:</strong> ${data.data.model_type}</p>
                                  <p><strong>Architecture:</strong> ${data.data.architecture}</p>
                                  <p><strong>Device:</strong> ${data.data.device}</p>
                                  <p><strong>Supported Formats:</strong> ${data.data.supported_formats.join(', ')}</p>
                                  <p><strong>Severity Classes:</strong> ${data.data.severity_classes.join(', ')}</p>`;
        } else {
            resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
        }
    })
    .catch(error => {
        resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    });
}