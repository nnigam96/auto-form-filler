// Auto Form Filler - Frontend Logic with HITL Support

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const passportInput = document.getElementById('passport');
    const g28Input = document.getElementById('g28');
    const extractBtn = document.getElementById('extract-btn');
    const fillBtn = document.getElementById('fill-btn');
    const resetBtn = document.getElementById('reset-btn');
    const headlessToggle = document.getElementById('headless-toggle');
    const llmToggle = document.getElementById('llm-toggle');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');

    // HITL Modal elements
    const hitlModal = document.getElementById('hitl-modal');
    const hitlConfirmBtn = document.getElementById('hitl-confirm');
    const hitlCancelBtn = document.getElementById('hitl-cancel');
    const comparisonFields = document.getElementById('comparison-fields');
    const fraudWarning = document.getElementById('fraud-warning');
    const fraudWarningText = document.getElementById('fraud-warning-text');

    // Store current extraction state for HITL flow
    let currentFileId = null;
    let currentSelections = {};

    // File input handlers
    passportInput.addEventListener('change', (e) => {
        updateFileDisplay('passport', e.target.files[0]);
    });

    g28Input.addEventListener('change', (e) => {
        updateFileDisplay('g28', e.target.files[0]);
    });

    // Extract only button - always use V5 for passport with HITL
    extractBtn.addEventListener('click', async () => {
        if (!passportInput.files[0]) {
            showError('Please select a passport file');
            return;
        }
        await extractWithHITL();
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

    // HITL Modal handlers
    hitlConfirmBtn.addEventListener('click', () => {
        confirmHITLSelections();
    });

    hitlCancelBtn.addEventListener('click', () => {
        closeHITLModal();
    });

    // Close modal on overlay click
    hitlModal.addEventListener('click', (e) => {
        if (e.target === hitlModal) {
            closeHITLModal();
        }
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

    // =============================================================================
    // Extraction with HITL Support
    // =============================================================================

    // Store extraction results for combining after HITL
    let pendingPassportData = null;
    let pendingG28File = null;

    async function extractWithHITL() {
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        error.classList.add('hidden');

        const useLlm = llmToggle?.checked ?? true;
        const loadingText = loading.querySelector('p');
        loadingText.textContent = 'Extracting passport data (V5)...';

        extractBtn.disabled = true;
        fillBtn.disabled = true;
        pendingG28File = g28Input.files[0] || null;

        try {
            // Step 1: Extract passport with V5
            const formData = new FormData();
            formData.append('file', passportInput.files[0]);

            const url = `/extract/passport?use_llm=${useLlm}`;
            console.log('[HITL] Extracting passport:', url);

            const response = await fetch(url, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            console.log('[HITL] Passport response:', data);

            if (!response.ok) {
                throw new Error(data.detail || 'Passport extraction failed');
            }

            currentFileId = data.file_id;
            pendingPassportData = data;

            // Step 2: Check if HITL is needed for passport
            if (data.needs_human_review) {
                console.log('[HITL] Needs review, showing modal');
                // Modal will call continueAfterHITL() when confirmed
                showHITLModal(data);
            } else {
                console.log('[HITL] No review needed, continuing...');
                await continueAfterHITL(data.final_data);
            }

        } catch (err) {
            console.error('[HITL] Error:', err);
            showError(err.message);
            loading.classList.add('hidden');
            extractBtn.disabled = false;
            fillBtn.disabled = false;
        }
    }

    async function continueAfterHITL(passportData) {
        const loadingText = loading.querySelector('p');

        try {
            let g28Data = null;

            // Step 3: Extract G-28 if uploaded
            if (pendingG28File) {
                loading.classList.remove('hidden');
                loadingText.textContent = 'Extracting G-28 data...';

                const formData = new FormData();
                formData.append('file', pendingG28File);

                console.log('[HITL] Extracting G-28...');
                const response = await fetch('/upload/g28', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();
                console.log('[HITL] G-28 response:', data);

                if (response.ok && data.success) {
                    g28Data = data.data;
                }
            }

            // Step 4: Display combined results
            displayCombinedResults(passportData, g28Data, pendingPassportData);

        } catch (err) {
            console.error('[HITL] G-28 Error:', err);
            // Still show passport results even if G-28 fails
            displayCombinedResults(passportData, null, pendingPassportData);
        } finally {
            loading.classList.add('hidden');
            extractBtn.disabled = false;
            fillBtn.disabled = false;
            pendingPassportData = null;
            pendingG28File = null;
        }
    }

    function displayCombinedResults(passportData, g28Data, rawPassportResponse) {
        results.classList.remove('hidden');

        // Passport data
        const passportEl = document.getElementById('passport-data');
        if (passportData) {
            const displayData = {
                ...passportData,
                _meta: {
                    confidence: rawPassportResponse?.overall_confidence,
                    mrz_checksum_valid: rawPassportResponse?.mrz_checksum_valid,
                    version: rawPassportResponse?.version,
                }
            };
            passportEl.textContent = JSON.stringify(displayData, null, 2);
        } else {
            passportEl.textContent = 'No passport data extracted';
        }
        document.getElementById('passport-results').classList.remove('hidden');

        // G-28 data
        const g28El = document.getElementById('g28-data');
        if (g28Data) {
            g28El.textContent = JSON.stringify(g28Data, null, 2);
        } else {
            g28El.textContent = pendingG28File ? 'G-28 extraction failed' : 'G-28 not uploaded';
        }
        document.getElementById('g28-results').classList.remove('hidden');

        // Stats
        const statsContent = document.getElementById('stats-content');
        const confidence = Math.round((rawPassportResponse?.overall_confidence || 0) * 100);
        const checksumStatus = rawPassportResponse?.mrz_checksum_valid ? '✓ Valid' : '✗ Invalid';
        const fraudFlags = rawPassportResponse?.fraud_flags || [];

        statsContent.innerHTML = `
            <div class="stat-box ${confidence >= 80 ? 'success' : confidence >= 50 ? 'warning' : 'error'}">
                <div class="stat-value">${confidence}%</div>
                <div class="stat-label">Confidence</div>
            </div>
            <div class="stat-box ${rawPassportResponse?.mrz_checksum_valid ? 'success' : 'warning'}">
                <div class="stat-value">${checksumStatus}</div>
                <div class="stat-label">MRZ Checksum</div>
            </div>
            <div class="stat-box ${fraudFlags.length ? 'error' : 'success'}">
                <div class="stat-value">${fraudFlags.length}</div>
                <div class="stat-label">Fraud Flags</div>
            </div>
            ${fraudFlags.length ? `
            <div class="failed-fields" style="grid-column: 1 / -1;">
                <h4>Fraud Flags:</h4>
                <ul>
                    ${fraudFlags.map(f => `<li>${f}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
        `;
    }

    // =============================================================================
    // HITL Modal Functions
    // =============================================================================

    function showHITLModal(data) {
        currentSelections = {};
        comparisonFields.innerHTML = '';

        // Show fraud warnings if any
        if (data.fraud_flags && data.fraud_flags.length > 0) {
            fraudWarning.classList.remove('hidden');
            fraudWarningText.textContent = data.fraud_flags.join('; ');
        } else {
            fraudWarning.classList.add('hidden');
        }

        // Build comparison fields
        const fields = data.fields || {};
        const fieldOrder = [
            'surname', 'given_names', 'passport_number', 'nationality',
            'date_of_birth', 'sex', 'expiry_date', 'country',
            'place_of_birth', 'issue_date'
        ];

        for (const fieldName of fieldOrder) {
            const field = fields[fieldName];
            if (!field) continue;

            const fieldEl = createComparisonField(fieldName, field);
            comparisonFields.appendChild(fieldEl);
        }

        hitlModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }

    function createComparisonField(fieldName, field) {
        const container = document.createElement('div');
        container.className = 'comparison-field';

        if (!field.needs_review) {
            container.classList.add('aligned');
        }

        const displayName = fieldName.replace(/_/g, ' ');

        container.innerHTML = `
            <div class="field-header">${displayName}</div>
            <div class="field-options">
                <div class="field-option" data-field="${fieldName}" data-source="mrz">
                    <div class="option-source mrz">MRZ (Machine Readable)</div>
                    <div class="option-value ${!field.mrz_value ? 'empty' : ''}">${field.mrz_value || '(not found)'}</div>
                    <div class="option-note">Checksummed data from barcode</div>
                </div>
                <div class="field-option" data-field="${fieldName}" data-source="visual">
                    <div class="option-source visual">Visual (Printed Text)</div>
                    <div class="option-value ${!field.visual_value ? 'empty' : ''}">${field.visual_value || '(not found)'}</div>
                    <div class="option-note">What appears on the document</div>
                </div>
            </div>
        `;

        // Add click handlers for selection
        const options = container.querySelectorAll('.field-option');
        options.forEach(opt => {
            opt.addEventListener('click', () => {
                // Remove selected from siblings
                options.forEach(o => o.classList.remove('selected'));
                // Add selected to clicked
                opt.classList.add('selected');
                // Store selection
                const source = opt.dataset.source;
                const value = source === 'mrz' ? field.mrz_value : field.visual_value;
                currentSelections[fieldName] = value;
            });
        });

        // Pre-select based on final_value or source
        if (field.final_value !== null) {
            const source = field.source === 'visual' ? 'visual' : 'mrz';
            const preselect = container.querySelector(`[data-source="${source}"]`);
            if (preselect) {
                preselect.classList.add('selected');
                currentSelections[fieldName] = field.final_value;
            }
        } else if (field.needs_review) {
            // For conflicts, pre-select MRZ as it's checksummed
            const mrzOption = container.querySelector('[data-source="mrz"]');
            if (mrzOption && field.mrz_value) {
                mrzOption.classList.add('selected');
                currentSelections[fieldName] = field.mrz_value;
            }
        }

        return container;
    }

    function closeHITLModal() {
        hitlModal.classList.add('hidden');
        document.body.style.overflow = '';
        currentSelections = {};
    }

    async function confirmHITLSelections() {
        // Validate all conflicting fields have selections
        const conflictFields = comparisonFields.querySelectorAll('.comparison-field:not(.aligned)');
        let allSelected = true;

        conflictFields.forEach(field => {
            const fieldName = field.querySelector('.field-option').dataset.field;
            if (currentSelections[fieldName] === undefined) {
                allSelected = false;
                field.style.borderColor = '#dc2626';
            } else {
                field.style.borderColor = '';
            }
        });

        if (!allSelected) {
            alert('Please select a value for all conflicting fields');
            return;
        }

        // Build final passport data from selections
        // Merge original final_data with user corrections
        const finalPassportData = { ...(pendingPassportData?.final_data || {}) };
        for (const [field, value] of Object.entries(currentSelections)) {
            finalPassportData[field] = value;
        }

        console.log('[HITL] User confirmed selections:', currentSelections);
        console.log('[HITL] Final passport data:', finalPassportData);

        closeHITLModal();

        // Continue with G-28 extraction if needed
        await continueAfterHITL(finalPassportData);
    }

    // =============================================================================
    // Full Pipeline Flow (Extract & Fill)
    // =============================================================================

    async function submitForm(endpoint, includeScreenshot) {
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        error.classList.add('hidden');

        const loadingText = loading.querySelector('p');
        const originalText = loadingText.textContent;
        loadingText.textContent = includeScreenshot
            ? 'Extracting data and filling form... This may take 30-60 seconds.'
            : 'Extracting data from documents...';

        extractBtn.disabled = true;
        fillBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('passport', passportInput.files[0]);
            formData.append('g28', g28Input.files[0]);

            if (includeScreenshot) {
                formData.append('headless', headlessToggle.checked ? 'true' : 'false');
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
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

            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

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
        passportInput.value = '';
        g28Input.value = '';

        updateFileDisplay('passport', null);
        updateFileDisplay('g28', null);

        results.classList.add('hidden');
        error.classList.add('hidden');
        loading.classList.add('hidden');

        document.querySelectorAll('.tab-btn').forEach((btn, idx) => {
            btn.classList.toggle('active', idx === 0);
        });
        document.querySelectorAll('.tab-content').forEach((content, idx) => {
            content.classList.toggle('active', idx === 0);
        });

        document.getElementById('passport-data').textContent = '';
        document.getElementById('g28-data').textContent = '';
        document.getElementById('stats-content').innerHTML = '';
        document.getElementById('screenshot').src = '';

        extractBtn.disabled = false;
        fillBtn.disabled = false;

        currentFileId = null;
        currentSelections = {};
        pendingPassportData = null;
        pendingG28File = null;

        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Keyboard handler for modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !hitlModal.classList.contains('hidden')) {
            closeHITLModal();
        }
    });
});
