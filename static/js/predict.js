// Prediction Page JavaScript

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const tabName = this.dataset.tab;
        
        // Update buttons
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(tabName + '-tab').classList.add('active');
    });
});

// Toggle feature visibility
const toggleBtn = document.getElementById('toggle-features');
if (toggleBtn) {
    toggleBtn.addEventListener('click', function() {
        const moreFeatures = document.getElementById('more-features');
        if (moreFeatures.style.display === 'none') {
            moreFeatures.style.display = 'block';
            this.innerHTML = '<i class="fas fa-chevron-up"></i> Hide Extra Features';
        } else {
            moreFeatures.style.display = 'none';
            this.innerHTML = '<i class="fas fa-chevron-down"></i> Show All Features';
        }
    });
}

// Manual prediction form
const predictForm = document.getElementById('predict-form');
if (predictForm) {
    predictForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showResult(data.result);
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during prediction');
        });
    });
}

function showResult(result) {
    const resultCard = document.getElementById('result-manual');
    resultCard.style.display = 'block';
    
    document.getElementById('result-prediction').textContent = result.prediction;
    document.getElementById('result-confidence').textContent = 
        `Confidence: ${result.confidence.toFixed(1)}%`;
    
    document.getElementById('prob-exoplanet').textContent = 
        result.probability_exoplanet.toFixed(1) + '%';
    document.getElementById('prob-non-exoplanet').textContent = 
        result.probability_non_exoplanet.toFixed(1) + '%';
    
    document.getElementById('progress-exoplanet').style.width = 
        result.probability_exoplanet + '%';
    document.getElementById('progress-non-exoplanet').style.width = 
        result.probability_non_exoplanet + '%';
    
    // Update icon
    const icon = document.getElementById('result-icon');
    if (result.prediction === 'Exoplanet') {
        icon.innerHTML = '<i class="fas fa-globe" style="color: #52C41A;"></i>';
    } else {
        icon.innerHTML = '<i class="fas fa-times-circle" style="color: #FF4D4F;"></i>';
    }
    
    // Scroll to result
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// File upload
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const fileName = document.getElementById('file-name');

if (fileInput && uploadArea) {
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#4A90E2';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.3)';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.3)';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            fileName.textContent = files[0].name;
        }
    });
    
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            fileName.textContent = this.files[0].name;
        }
    });
}

// Batch upload form
const uploadForm = document.getElementById('upload-form');
if (uploadForm) {
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        
        document.getElementById('upload-btn').innerHTML = 
            '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        document.getElementById('upload-btn').disabled = true;
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('upload-btn').innerHTML = 
                '<i class="fas fa-paper-plane"></i> Analyze Batch';
            document.getElementById('upload-btn').disabled = false;
            
            if (data.success && data.batch) {
                showBatchResults(data);
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('upload-btn').innerHTML = 
                '<i class="fas fa-paper-plane"></i> Analyze Batch';
            document.getElementById('upload-btn').disabled = false;
            alert('An error occurred during batch analysis');
        });
    });
}

function showBatchResults(data) {
    const resultCard = document.getElementById('result-batch');
    resultCard.style.display = 'block';
    
    document.getElementById('batch-total').textContent = data.total;
    document.getElementById('batch-exoplanets').textContent = data.exoplanets;
    document.getElementById('batch-non-exoplanets').textContent = 
        data.total - data.exoplanets;
    
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = '';
    
    data.results.forEach(result => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${result.index + 1}</td>
            <td>${result.prediction}</td>
            <td>${result.confidence.toFixed(1)}%</td>
        `;
        tbody.appendChild(row);
    });
    
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

