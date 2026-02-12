// static/js/app.js
/**
 * ========================================================
 * GEMINI DOCUMENTATION GENERATOR - FRONTEND APPLICATION
 * ========================================================
 * 
 * Two AI Agents:
 * 1. Docstring Agent - Generates Google-style docstrings for Python code
 * 2. README Agent - Generates comprehensive README.md from projects (ZIP/Path)
 * 
 * Version: 1.0.0
 * 
 * ========================================================
 */

// ==================== CONSTANTS & CONFIGURATION ====================

const CONFIG = {
    API: {
        DOCSTRING: '/api/generate',
        README_PATH: '/api/generate-readme',
        README_ZIP: '/api/upload-zip',
        HEALTH: '/health',
        VERSION: '/api/version'
    },
    LIMITS: {
        ZIP_MAX_SIZE: 50 * 1024 * 1024, // 50MB
        FILE_MAX_SIZE: 100 * 1024,       // 100KB
        MAX_FILES: 500,
        ALLOWED_EXTENSIONS: ['.zip']
    },
    UI: {
        STATUS_DURATION: 5000,
        ANIMATION_DURATION: 300
    }
};

// ==================== STATE MANAGEMENT ====================

const AppState = {
    docstring: {
        current: '',
        elementName: '',
        elementType: '',
        isGenerating: false
    },
    readme: {
        current: '',
        projectName: '',
        projectPath: '',
        stats: null,
        isProcessing: false,
        uploadProgress: 0
    },
    ui: {
        currentMode: 'docstring'
    },
    icons: {
        function: '⚡',
        class: '🏛️',
        method: '🔧',
        unknown: '📄',
        directory: '📁',
        file: '📄',
        python: '🐍',
        zip: '📦'
    }
};

// ==================== UTILITY FUNCTIONS ====================

const Utils = {
    path: {
        basename(path) {
            if (!path) return '';
            return path.split(/[\\/]/).pop();
        }
    },

    string: {
        escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },

        formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },

        pluralize(count, singular, plural = null) {
            if (count === 1) return `${count} ${singular}`;
            return `${count} ${plural || singular + 's'}`;
        }
    },

    validate: {
        isZipFile(filename) {
            if (!filename) return false;
            return filename.toLowerCase().endsWith('.zip');
        },

        isWithinSizeLimit(size, maxSize = CONFIG.LIMITS.ZIP_MAX_SIZE) {
            return size <= maxSize;
        },

        isValidPath(path) {
            if (!path || typeof path !== 'string') return false;
            return !/[<>:"|?*]/.test(path);
        }
    },

    debounce(func, wait = 300) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// ==================== DOM MANAGER ====================

const DOM = {
    get(id) {
        return document.getElementById(id);
    },

    getElements: {
        modeBtns: () => document.querySelectorAll('.mode-btn'),
        docstringMode: () => document.getElementById('docstringMode'),
        readmeMode: () => document.getElementById('readmeMode'),
        codeInput: () => document.getElementById('codeInput'),
        outputContent: () => document.getElementById('outputContent'),
        generateBtn: () => document.getElementById('generateBtn'),
        copyBtn: () => document.getElementById('copyBtn'),
        elementBadge: () => document.getElementById('elementBadge'),
        inputStats: () => document.getElementById('inputStats'),
        fileInput: () => document.getElementById('fileInput'),
        dropZone: () => document.getElementById('dropZone'),
        folderPathInput: () => document.getElementById('folderPathInput'),
        uploadProgress: () => document.getElementById('uploadProgress'),
        progressFill: () => document.querySelector('.progress-fill'),
        progressStatus: () => document.querySelector('.progress-status'),
        analysisResults: () => document.getElementById('analysisResults'),
        projectStats: () => document.getElementById('projectStats'),
        readmeContent: () => document.getElementById('readmeContent'),
        copyReadmeBtn: () => document.getElementById('copyReadmeBtn'),
        downloadReadmeBtn: () => document.getElementById('downloadReadmeBtn'),
        statusMessage: () => document.getElementById('statusMessage')
    }
};

// ==================== API SERVICE ====================

const ApiService = {
    async generateDocstring(code) {
        try {
            const response = await fetch(CONFIG.API.DOCSTRING, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `HTTP error ${response.status}`);
            return data;
        } catch (error) {
            console.error('Docstring API error:', error);
            throw error;
        }
    },

    async generateReadmeFromPath(projectPath) {
        try {
            const response = await fetch(CONFIG.API.README_PATH, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ project_path: projectPath })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `HTTP error ${response.status}`);
            return data;
        } catch (error) {
            console.error('README path API error:', error);
            throw error;
        }
    },

    async generateReadmeFromZip(file, onProgress = null) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            formData.append('zip_file', file);

            const xhr = new XMLHttpRequest();

            if (onProgress) {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percent = Math.round((e.loaded / e.total) * 100);
                        onProgress(percent);
                    }
                });
            }

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        resolve(JSON.parse(xhr.responseText));
                    } catch {
                        reject(new Error('Invalid JSON response'));
                    }
                } else {
                    try {
                        const error = JSON.parse(xhr.responseText);
                        reject(new Error(error.error || `HTTP error ${xhr.status}`));
                    } catch {
                        reject(new Error(`HTTP error ${xhr.status}`));
                    }
                }
            });

            xhr.addEventListener('error', () => reject(new Error('Network error')));
            xhr.addEventListener('abort', () => reject(new Error('Upload aborted')));

            xhr.open('POST', CONFIG.API.README_ZIP);
            xhr.send(formData);
        });
    },

    async healthCheck() {
        try {
            const response = await fetch(CONFIG.API.HEALTH);
            return await response.json();
        } catch {
            return { status: 'unhealthy' };
        }
    }
};

// ==================== UI SERVICE ====================

const UIService = {
    showStatus(message, type = 'info', duration = CONFIG.UI.STATUS_DURATION) {
        const el = DOM.getElements.statusMessage();
        if (!el) return;

        if (this.statusTimeout) clearTimeout(this.statusTimeout);

        el.textContent = message;
        el.className = `status ${type}`;
        el.style.display = 'block';

        if (type === 'success' || type === 'info') {
            this.statusTimeout = setTimeout(() => {
                el.style.display = 'none';
            }, duration);
        }
    },

    hideStatus() {
        const el = DOM.getElements.statusMessage();
        if (el) el.style.display = 'none';
    },

    updateProgress(percent, status = null) {
        const fill = DOM.getElements.progressFill();
        const statusEl = DOM.getElements.progressStatus();

        if (fill) fill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
        if (statusEl && status) statusEl.textContent = status;

        AppState.readme.uploadProgress = percent;
    },

    setLoading(isLoading, elementId = 'generateBtn') {
        const btn = DOM.get(elementId);
        if (!btn) return;

        if (elementId === 'generateBtn') {
            AppState.docstring.isGenerating = isLoading;
            btn.innerHTML = isLoading 
                ? '<span class="loading"></span> Generating...' 
                : '<span class="btn-icon">✨</span> Generate Docstring';
        } else if (elementId === 'analyzePathBtn') {
            AppState.readme.isProcessing = isLoading;
            btn.innerHTML = isLoading 
                ? '<span class="loading"></span> Analyzing...' 
                : '<span class="btn-icon">🔍</span> Analyze';
        }

        btn.disabled = isLoading;
    },

    escapeHtml(text) {
        return Utils.string.escapeHtml(text);
    },

    formatDocstring(docstring) {
        if (!docstring) return '';
        return docstring
            .replace(/^"""|"""$/g, '')
            .replace(/\\n/g, '\n')
            .trim();
    }
};

// ==================== DOCSTRING MODULE ====================

const DocstringModule = {
    init() {
        this.bindEvents();
        this.updateInputStats();
        this.loadSample();
    },

    bindEvents() {
        const input = DOM.getElements.codeInput();
        if (!input) return;

        input.addEventListener('input', this.debouncedUpdateStats.bind(this));
        input.addEventListener('keydown', this.handleKeyDown.bind(this));
        input.addEventListener('input', this.autoResize.bind(this));
    },

    debouncedUpdateStats: Utils.debounce(function() {
        this.updateInputStats();
    }, 100),

    handleKeyDown(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            this.generate();
        }

        if (e.key === 'Tab') {
            e.preventDefault();
            const start = e.target.selectionStart;
            const end = e.target.selectionEnd;
            e.target.value = e.target.value.substring(0, start) + '    ' + e.target.value.substring(end);
            e.target.selectionStart = e.target.selectionEnd = start + 4;
        }
    },

    autoResize(e) {
        const el = e.target;
        el.style.height = 'auto';
        el.style.height = el.scrollHeight + 'px';
    },

    updateInputStats() {
        const input = DOM.getElements.codeInput();
        const statsEl = DOM.getElements.inputStats();

        // FIX: Check if statsEl exists before setting innerHTML
        if (!input || !statsEl) return;

        const value = input.value;
        const lines = value.split('\n').length;
        const chars = value.length;

        statsEl.innerHTML = `${Utils.string.pluralize(lines, 'line')} • ${Utils.string.formatBytes(chars)}`;
    },

    loadSample() {
        const input = DOM.getElements.codeInput();
        if (!input || input.value.trim()) return;

        const sample = `def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)`;

        input.value = sample;
        this.autoResize({ target: input });
        this.updateInputStats();
    },

    async generate() {
        const input = DOM.getElements.codeInput();
        if (!input) return;

        const code = input.value.trim();

        if (!code) {
            UIService.showStatus('Please enter Python code', 'error');
            this.showPlaceholder();
            return;
        }

        try {
            UIService.setLoading(true, 'generateBtn');
            UIService.hideStatus();
            this.resetOutput();

            const data = await ApiService.generateDocstring(code);

            if (data.success) {
                this.displayDocstring(data);
                UIService.showStatus('✅ Docstring generated successfully!', 'success');
            } else {
                throw new Error(data.error || 'Generation failed');
            }
        } catch (error) {
            this.showError(error.message);
            UIService.showStatus(`Error: ${error.message}`, 'error');
        } finally {
            UIService.setLoading(false, 'generateBtn');
        }
    },

    displayDocstring(data) {
        const outputEl = DOM.getElements.outputContent();
        const badgeEl = DOM.getElements.elementBadge();
        const copyBtn = DOM.getElements.copyBtn();

        if (!outputEl) return;

        AppState.docstring.current = data.docstring;
        AppState.docstring.elementName = data.element_name;
        AppState.docstring.elementType = data.element_type;

        const formatted = UIService.formatDocstring(data.docstring);
        outputEl.innerHTML = `<pre class="docstring-output">${UIService.escapeHtml(formatted)}</pre>`;

        if (badgeEl && data.element_type && data.element_name) {
            const icon = AppState.icons[data.element_type] || '📄';
            badgeEl.innerHTML = `${icon} ${data.element_type}: ${UIService.escapeHtml(data.element_name)}`;
            badgeEl.style.display = 'inline-flex';
        }

        if (copyBtn) {
            copyBtn.style.display = 'inline-flex';
        }
    },

    showError(message) {
        const outputEl = DOM.getElements.outputContent();
        if (!outputEl) return;

        outputEl.innerHTML = `
            <div class="placeholder">
                <div class="placeholder-icon">❌</div>
                <div class="placeholder-text" style="color: var(--error-500);">${UIService.escapeHtml(message)}</div>
                <div class="placeholder-hint">Check your code and try again</div>
            </div>
        `;
    },

    showPlaceholder() {
        const outputEl = DOM.getElements.outputContent();
        if (!outputEl) return;

        outputEl.innerHTML = `
            <div class="placeholder">
                <div class="placeholder-icon">✨</div>
                <div class="placeholder-text">Generated docstring will appear here</div>
                <div class="placeholder-hint">Ctrl+Enter to generate</div>
            </div>
        `;
    },

    resetOutput() {
        const badgeEl = DOM.getElements.elementBadge();
        const copyBtn = DOM.getElements.copyBtn();

        if (badgeEl) badgeEl.style.display = 'none';
        if (copyBtn) copyBtn.style.display = 'none';

        AppState.docstring.current = '';
    },

    clearInput() {
        const input = DOM.getElements.codeInput();
        if (input) {
            input.value = '';
            this.updateInputStats();
            this.autoResize({ target: input });
        }
        this.resetOutput();
    },

    clearAll() {
        this.clearInput();
        this.showPlaceholder();
        UIService.hideStatus();
    },

    async copyDocstring() {
        if (!AppState.docstring.current) {
            UIService.showStatus('No docstring to copy', 'warning');
            return;
        }

        try {
            await navigator.clipboard.writeText(AppState.docstring.current);

            const btn = DOM.getElements.copyBtn();
            if (btn) {
                const original = btn.innerHTML;
                btn.innerHTML = '✅ Copied!';
                setTimeout(() => { btn.innerHTML = original; }, 2000);
            }

            UIService.showStatus('📋 Docstring copied!', 'success');
        } catch (err) {
            UIService.showStatus('Failed to copy', 'error');
        }
    }
};

// ==================== README MODULE ====================

const ReadmeModule = {
    init() {
        this.bindEvents();
        this.initDragDrop();
    },

    bindEvents() {
        const fileInput = DOM.getElements.fileInput();
        const dropZone = DOM.getElements.dropZone();
        const pathInput = DOM.getElements.folderPathInput();

        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleZipFile(e.target.files[0]);
                }
            });
        }

        if (dropZone) {
            dropZone.addEventListener('click', () => fileInput?.click());
        }

        if (pathInput) {
            pathInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.generateFromPath();
                }
            });
        }
    },

    initDragDrop() {
        const dropZone = DOM.getElements.dropZone();
        if (!dropZone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('drag-over');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleZipFile(files[0]);
            }
        });
    },

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    },

    async handleZipFile(file) {
        if (!Utils.validate.isZipFile(file.name)) {
            UIService.showStatus('Please upload a ZIP file', 'error');
            return;
        }

        if (!Utils.validate.isWithinSizeLimit(file.size)) {
            UIService.showStatus(`ZIP exceeds ${Utils.string.formatBytes(CONFIG.LIMITS.ZIP_MAX_SIZE)}`, 'error');
            return;
        }

        AppState.readme.uploadedFile = file;

        const progressEl = DOM.getElements.uploadProgress();
        if (progressEl) progressEl.style.display = 'block';

        UIService.showStatus(`📦 Uploading ${file.name}...`, 'info');

        try {
            const data = await ApiService.generateReadmeFromZip(file, (percent) => {
                UIService.updateProgress(percent, `Uploading: ${percent}%`);
            });

            UIService.updateProgress(100, 'Complete!');

            setTimeout(() => {
                const progressEl = DOM.getElements.uploadProgress();
                if (progressEl) progressEl.style.display = 'none';
                UIService.updateProgress(0);
            }, 1000);

            if (data.success) {
                this.displayAnalysis(data);
                UIService.showStatus(`✅ Analyzed ${data.file_count || 0} files`, 'success');
            } else {
                throw new Error(data.error || 'Failed to process ZIP');
            }
        } catch (error) {
            const progressEl = DOM.getElements.uploadProgress();
            if (progressEl) progressEl.style.display = 'none';
            UIService.showStatus(`Error: ${error.message}`, 'error');
        }
    },

    async generateFromPath() {
        const pathInput = DOM.getElements.folderPathInput();
        if (!pathInput) return;

        const projectPath = pathInput.value.trim();

        if (!projectPath) {
            UIService.showStatus('Please enter a project path', 'error');
            return;
        }

        if (!Utils.validate.isValidPath(projectPath)) {
            UIService.showStatus('Invalid path format', 'error');
            return;
        }

        try {
            UIService.setLoading(true, 'analyzePathBtn');
            UIService.showStatus('🔍 Analyzing project...', 'info');

            const data = await ApiService.generateReadmeFromPath(projectPath);

            if (data.success) {
                this.displayAnalysis(data);
                UIService.showStatus('✅ README generated!', 'success');
            } else {
                throw new Error(data.error || 'Failed to generate README');
            }
        } catch (error) {
            UIService.showStatus(`Error: ${error.message}`, 'error');
        } finally {
            UIService.setLoading(false, 'analyzePathBtn');
        }
    },

    displayAnalysis(data) {
        AppState.readme.current = data.readme;
        AppState.readme.projectName = data.project_name;
        AppState.readme.stats = data.summary;

        const resultsEl = DOM.getElements.analysisResults();
        if (resultsEl) {
            resultsEl.style.display = 'block';
            resultsEl.scrollIntoView({ behavior: 'smooth' });
        }

        this.displayStats(data.summary);
        this.displayReadme(data.readme);
    },

    displayStats(stats) {
        const statsEl = DOM.getElements.projectStats();
        if (!statsEl || !stats) return;

        const formatNumber = (num) => num?.toLocaleString() || '0';

        statsEl.innerHTML = `
            <div class="stat-card">
                <div class="stat-icon">📁</div>
                <div class="stat-value">${formatNumber(stats.total_dirs)}</div>
                <div class="stat-label">Directories</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📄</div>
                <div class="stat-value">${formatNumber(stats.total_files)}</div>
                <div class="stat-label">Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⚡</div>
                <div class="stat-value">${formatNumber(stats.functions?.length)}</div>
                <div class="stat-label">Functions</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🏛️</div>
                <div class="stat-value">${formatNumber(stats.classes?.length)}</div>
                <div class="stat-label">Classes</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📦</div>
                <div class="stat-value">${formatNumber(stats.dependencies?.length)}</div>
                <div class="stat-label">Dependencies</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📏</div>
                <div class="stat-value">${formatNumber(stats.total_lines)}</div>
                <div class="stat-label">Lines</div>
            </div>
        `;
    },

    displayReadme(readme) {
        const contentEl = DOM.getElements.readmeContent();
        if (!contentEl) return;

        contentEl.innerHTML = `<pre class="readme-output">${UIService.escapeHtml(readme)}</pre>`;
    },

    async copyReadme() {
        if (!AppState.readme.current) {
            UIService.showStatus('No README to copy', 'warning');
            return;
        }

        try {
            await navigator.clipboard.writeText(AppState.readme.current);

            const btn = DOM.getElements.copyReadmeBtn();
            if (btn) {
                const original = btn.innerHTML;
                btn.innerHTML = '✅ Copied!';
                setTimeout(() => { btn.innerHTML = original; }, 2000);
            }

            UIService.showStatus('📋 README copied!', 'success');
        } catch (err) {
            UIService.showStatus('Failed to copy', 'error');
        }
    },

    downloadReadme() {
        if (!AppState.readme.current) {
            UIService.showStatus('No README to download', 'warning');
            return;
        }

        const blob = new Blob([AppState.readme.current], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = AppState.readme.projectName ? `README_${AppState.readme.projectName}.md` : 'README.md';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        UIService.showStatus('⬇️ README downloaded!', 'success');
    }
};

// ==================== MODE TOGGLE MODULE ====================

const ModeToggle = {
    init() {
        const btns = DOM.getElements.modeBtns();
        if (!btns || !btns.length) return;

        btns.forEach(btn => {
            btn.addEventListener('click', () => this.switchMode(btn.dataset.mode));
        });

        this.switchMode('docstring');
    },

    switchMode(mode) {
        const btns = DOM.getElements.modeBtns();
        const docstringPanel = DOM.getElements.docstringMode();
        const readmePanel = DOM.getElements.readmeMode();

        if (!btns || !docstringPanel || !readmePanel) return;

        btns.forEach(btn => {
            if (btn.dataset.mode === mode) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        if (mode === 'docstring') {
            docstringPanel.classList.add('active');
            readmePanel.classList.remove('active');
            AppState.ui.currentMode = 'docstring';

            // FIX: Update input stats when switching to docstring mode
            setTimeout(() => {
                DocstringModule.updateInputStats();
            }, 50);
        } else {
            docstringPanel.classList.remove('active');
            readmePanel.classList.add('active');
            AppState.ui.currentMode = 'readme';
        }
    }
};

// ==================== APP INITIALIZATION ====================

const App = {
    async init() {
        console.log('🚀 Initializing Documentation Generator...');

        ModeToggle.init();
        DocstringModule.init();
        ReadmeModule.init();

        await this.checkHealth();
        this.exposePublicMethods();

        console.log('✅ Application initialized');
    },

    async checkHealth() {
        try {
            await ApiService.healthCheck();
        } catch (error) {
            console.warn('⚠️ API health check failed');
        }
    },

    exposePublicMethods() {
        window.generateDocstring = () => DocstringModule.generate();
        window.copyDocstring = () => DocstringModule.copyDocstring();
        window.clearInput = () => DocstringModule.clearInput();
        window.clearAll = () => DocstringModule.clearAll();
        window.loadExample = () => DocstringModule.loadSample();

        window.generateReadmeFromPath = () => ReadmeModule.generateFromPath();
        window.copyReadme = () => ReadmeModule.copyReadme();
        window.downloadReadme = () => ReadmeModule.downloadReadme();
    }
};

// ==================== START APPLICATION ====================

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => App.init());
} else {
    App.init();
}
