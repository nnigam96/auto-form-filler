// Auto Form Filler - Frontend Logic

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const passportInput = document.getElementById('passport');
    const g28Input = document.getElementById('g28');
    const extractBtn = document.getElementById('extract-btn');
    const fillBtn = document.getElementById('fill-btn');
    const resetBtn = document.getElementById('reset-btn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');

    // File input handlers
    passportInput.addEventListener('change', (e) => {
        updateFileDisplay('passport', e.target.files[0]);
    });

    g28Input.addEventListener('change', (e) => {
        updateFileDisplay('g28', e.target.files[0]);
    });

    // Extract only button
    extractBtn.addEventListener('click', async () => {
        if (!validateFiles()) return;
        await submitForm('/extract', false);
    });

    // Extract & Fill button
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!validateFiles()) return;
        await submitForm('/fill-form', true);
    });

    // Reset button
    resetBtn.addEventListener('click', () => {
        resetForm();
    });

    function updateFileDisplay(type, file) {
        const box = document.getElementById(`${type}-box`);
        const nameEl = document.getElementById(`${type}-name`);

        if (file) {
            box.classList.add('has-file');
            nameEl.textContent = file.name;
        } else {
            box.classList.remove('has-file');
            nameEl.textContent = '';
        }
    }

    function validateFiles() {
        if (!passportInput.files[0]) {
            showError('Please select a passport file');
            return false;
        }
        if (!g28Input.files[0]) {
            showError('Please select a G-28 file');
            return false;
        }
        return true;
    }

    async function submitForm(endpoint, includeScreenshot) {
        // Show loading
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        error.classList.add('hidden');
        
        // Update loading message
        const loadingText = loading.querySelector('p');
        const originalText = loadingText.textContent;
        loadingText.textContent = includeScreenshot 
            ? 'Extracting data and filling form... This may take 30-60 seconds.'
            : 'Extracting data from documents...';

        // Disable buttons
        extractBtn.disabled = true;
        fillBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('passport', passportInput.files[0]);
            formData.append('g28', g28Input.files[0]);
            
            // Always use headless mode for form filling (faster, no browser window)
            if (includeScreenshot) {
                formData.append('headless', 'true');
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle detailed error messages
                let errorMsg = data.detail || 'Request failed';
                if (Array.isArray(errorMsg)) {
                    errorMsg = errorMsg.join(', ');
                }
                throw new Error(errorMsg);
            }

            displayResults(data, includeScreenshot);

        } catch (err) {
            showError(err.message);
        } finally {
            loadingText.textContent = originalText;
            loading.classList.add('hidden');
            extractBtn.disabled = false;
            fillBtn.disabled = false;
        }
    }

    function displayResults(data, includeScreenshot) {
        results.classList.remove('hidden');

        // Passport data
        const passportData = data.passport_data || data.extraction?.passport_data;
        const passportEl = document.getElementById('passport-data');
        if (passportData) {
            passportEl.textContent = JSON.stringify(passportData, null, 2);
            document.getElementById('passport-results').classList.remove('hidden');
        } else {
            passportEl.textContent = 'No passport data extracted';
        }

        // G-28 data
        const g28Data = data.g28_data || data.extraction?.g28_data;
        const g28El = document.getElementById('g28-data');
        if (g28Data) {
            g28El.textContent = JSON.stringify(g28Data, null, 2);
            document.getElementById('g28-results').classList.remove('hidden');
        } else {
            g28El.textContent = 'No G-28 data extracted';
        }

        // Screenshot
        const screenshotContainer = document.getElementById('screenshot-container');
        const noScreenshot = document.getElementById('no-screenshot');
        if (includeScreenshot && data.screenshot_url) {
            screenshotContainer.classList.remove('hidden');
            noScreenshot.classList.add('hidden');
            document.getElementById('screenshot').src = data.screenshot_url;
        } else {
            screenshotContainer.classList.add('hidden');
            noScreenshot.classList.remove('hidden');
        }

        // Fill stats
        const statsContent = document.getElementById('stats-content');
        if (data.fill_result) {
            const filled = data.fill_result.filled_fields?.length || 0;
            const failed = data.fill_result.failed_fields?.length || 0;
            const total = filled + failed;
            const rate = total > 0 ? Math.round((filled / total) * 100) : 0;

            statsContent.innerHTML = `
                <div class="stat-box success">
                    <div class="stat-value">${filled}</div>
                    <div class="stat-label">Fields Filled</div>
                </div>
                <div class="stat-box ${failed > 0 ? 'warning' : ''}">
                    <div class="stat-value">${failed}</div>
                    <div class="stat-label">Fields Failed</div>
                </div>
                <div class="stat-box ${rate >= 80 ? 'success' : rate >= 50 ? 'warning' : 'error'}">
                    <div class="stat-value">${rate}%</div>
                    <div class="stat-label">Fill Rate</div>
                </div>
                ${failed > 0 ? `
                <div class="failed-fields">
                    <h4>Failed Fields:</h4>
                    <ul>
                        ${data.fill_result.failed_fields.map(f => `<li>${f}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}
            `;
        } else {
            const extractionSuccess = data.success || (data.extraction && data.extraction.success);
            statsContent.innerHTML = `
                <div class="stat-box ${extractionSuccess ? 'success' : 'error'}">
                    <div class="stat-value">${extractionSuccess ? '✓' : '✗'}</div>
                    <div class="stat-label">Extraction ${extractionSuccess ? 'Success' : 'Failed'}</div>
                </div>
                ${data.errors && data.errors.length > 0 ? `
                <div class="error-details">
                    <h4>Errors:</h4>
                    <ul>
                        ${data.errors.map(e => `<li>${e}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}
            `;
        }
    }

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            
            // Update buttons
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update content
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`tab-${tabName}`).classList.add('active');
        });
    });

    function showError(message) {
        error.classList.remove('hidden');
        document.getElementById('error-message').textContent = message;
        results.classList.add('hidden');
    }

    function resetForm() {
        // Clear file inputs
        passportInput.value = '';
        g28Input.value = '';
        
        // Reset file display
        updateFileDisplay('passport', null);
        updateFileDisplay('g28', null);
        
        // Hide results and errors
        results.classList.add('hidden');
        error.classList.add('hidden');
        loading.classList.add('hidden');
        
        // Reset tabs to first tab
        document.querySelectorAll('.tab-btn').forEach((btn, idx) => {
            btn.classList.toggle('active', idx === 0);
        });
        document.querySelectorAll('.tab-content').forEach((content, idx) => {
            content.classList.toggle('active', idx === 0);
        });
        
        // Clear result content
        document.getElementById('passport-data').textContent = '';
        document.getElementById('g28-data').textContent = '';
        document.getElementById('stats-content').innerHTML = '';
        document.getElementById('screenshot').src = '';
        
        // Enable buttons
        extractBtn.disabled = false;
        fillBtn.disabled = false;
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});
