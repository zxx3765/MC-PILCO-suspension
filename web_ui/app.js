// MC-PILCO Web Dashboard Frontend Controller

document.addEventListener("DOMContentLoaded", () => {
    const UI_VERSION = "2026.06.09.1";

    // DOM Elements - Sweep Config Form
    const scriptSelect = document.getElementById("script-select");
    const seedInput = document.getElementById("seed-input");
    const deviceSelect = document.getElementById("device-select");
    const trialsInput = document.getElementById("trials-input");
    const sweepSelect = document.getElementById("sweep-select");
    const customJsonContainer = document.getElementById("custom-json-container");
    const customConfigsInput = document.getElementById("custom-configs-input");
    const startBtn = document.getElementById("start-btn");
    const stopBtn = document.getElementById("stop-btn");
    
    // DOM Elements - Single Run Config Form & Accordions
    const singleStartBtn = document.getElementById("single-start-btn");
    const configSelect = document.getElementById("config-template-select");
    const configNameInput = document.getElementById("config-template-name");
    const saveTemplateBtn = document.getElementById("save-template-btn");
    const refreshTemplateBtn = document.getElementById("refresh-template-btn");
    
    // DOM Elements - Header & Global status
    const overallStatusBadge = document.getElementById("overall-status-badge");
    const overallStatusText = document.getElementById("overall-status-text");
    const elapsedTimeDisplay = document.getElementById("elapsed-time-display");
    const dashboardVersionBadge = document.getElementById("dashboard-version-badge");
    if (dashboardVersionBadge) {
        dashboardVersionBadge.textContent = `UI v${UI_VERSION} / API loading / PID ?`;
    }
    
    // DOM Elements - Progress Monitor
    const progressBar = document.getElementById("progress-bar-element");
    const progressPercentText = document.getElementById("progress-percent-text");
    const progressRatioText = document.getElementById("progress-ratio-text");
    const estRemainingText = document.getElementById("estimated-remaining-text");
    
    // DOM Elements - Current Config Details
    const currentLabelVal = document.getElementById("current-label-val");
    const currentStatusVal = document.getElementById("current-status-val");
    const currentLrVal = document.getElementById("current-lr-val");
    const currentEpochsVal = document.getElementById("current-epochs-val");
    const currentStepsVal = document.getElementById("current-steps-val");
    const currentStepVal = document.getElementById("current-step-val");
    const currentCostVal = document.getElementById("current-cost-val");
    const currentTrialVal = document.getElementById("current-trial-val");
    
    // DOM Elements - GP Errors & Terminal
    const gpErrorsList = document.getElementById("gp-errors-list");
    const consoleOutputBox = document.getElementById("console-output-box");
    const clearConsoleBtn = document.getElementById("clear-console-btn");
    const currentRunLeaderboardTbody = document.getElementById("current-run-leaderboard-tbody");
    const savedLeaderboardTbody = document.getElementById("saved-leaderboard-tbody");
    
    // DOM Elements - Analysis tab
    const loadPlotsBtn = document.getElementById("load-plots-btn");
    const analysisRunSelect = document.getElementById("analysis-run-select");
    const analysisSeedInput = document.getElementById("analysis-seed-input");
    const analysisRootInput = document.getElementById("analysis-root-input");
    const analysisRunsStatus = document.getElementById("analysis-runs-status");
    
    // DOM Elements - Leaderboard management & uploads
    const leaderboardSelect = document.getElementById("leaderboard-select");
    const currentRunSaveTargetSelect = document.getElementById("current-run-save-target-select");
    const newLeaderboardNameInput = document.getElementById("new-leaderboard-name");
    const saveLeaderboardBtn = document.getElementById("save-leaderboard-btn");
    const deleteLeaderboardBtn = document.getElementById("delete-leaderboard-btn");
    const uploadCurrentBtn = document.getElementById("upload-current-btn");
    const uploadPastBtn = document.getElementById("upload-past-btn");
    
    // State variables
    let currentLeaderboard = "default";
    let lastRunInfo = {
        runName: "",
        seed: 1,
        resultRoot: "./results_tmp/quarter_car_gym"
    };
    
    // Charts variables
    let paretoChart = null;
    let knownLogLines = new Set();
    let isMonitoring = false;
    let pollIntervalId = null;
    
    // API URL Base (同源部署，空值即为当前 Host)
    const API_BASE = "";

    // Helper to handle fetch responses and handle JSON/HTML errors gracefully
    async function handleResponse(res) {
        if (!res.ok) {
            const contentType = res.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                const errData = await res.json();
                throw new Error(errData.message || "请求失败");
            } else {
                const text = await res.text();
                throw new Error(`HTTP ${res.status}: ${text || '服务器错误'}`);
            }
        }
        return res.json();
    }

    // ==========================================================================
    // Tabs Navigation & Collapsible Accordions
    // ==========================================================================
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            
            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(`tab-${tabId}`).classList.add("active");
            
            // Toggle side panel visibility based on active tab
            if (tabId === "leaderboard-view") {
                document.querySelector(".progress-panel").style.display = "none";
                document.querySelector(".console-panel").style.display = "none";
                document.querySelector(".app-grid").style.gridTemplateColumns = "1fr";
            } else {
                document.querySelector(".progress-panel").style.display = "";
                document.querySelector(".console-panel").style.display = "";
                document.querySelector(".app-grid").style.gridTemplateColumns = "";
            }
            
            if (tabId === "analysis-view") {
                syncAnalysisInputsToLastRun();
                loadAnalysisRuns(lastRunInfo.runName);
            } else if (tabId === "leaderboard-view") {
                loadLeaderboardList().then(() => {
                    loadHistory();
                });
                if (paretoChart) {
                    setTimeout(() => {
                        paretoChart.resize();
                        paretoChart.update();
                    }, 50);
                }
            } else if (tabId === "sweep-run") {
                loadLeaderboardList();
            }
        });
    });

    const accordionTitles = document.querySelectorAll(".accordion-title");
    accordionTitles.forEach(title => {
        title.addEventListener("click", () => {
            title.parentElement.classList.toggle("open");
        });
    });

    // Open first accordion section by default
    const firstAccordion = document.querySelector(".accordion-item");
    if (firstAccordion) {
        firstAccordion.classList.add("open");
    }

    // Toggle Custom JSON Textarea
    sweepSelect.addEventListener("change", () => {
        if (sweepSelect.value === "custom") {
            customJsonContainer.style.display = "flex";
        } else {
            customJsonContainer.style.display = "none";
        }
    });

    clearConsoleBtn.addEventListener("click", () => {
        consoleOutputBox.innerHTML = '<div class="console-line system-msg">[SYSTEM] 终端日志历史已被清空...</div>';
        knownLogLines.clear();
    });

    // ==========================================================================
    // Config Form Management
    // ==========================================================================
    function getFormConfig() {
        const train = {};
        document.querySelectorAll("[data-train-field]").forEach(input => {
            if (input.type === "checkbox") {
                train[input.getAttribute("data-train-field")] = input.checked ? "True" : "False";
            } else {
                train[input.getAttribute("data-train-field")] = input.value;
            }
        });
        
        const seed = train["seed"] || "1";
        const result_root = train["result_root"] || "./results_tmp/quarter_car_gym";
        const run_name = train["run_name"] || "baseline";
        
        return {
            "conda_env": "mc-pilco",
            "overwrite_existing": document.getElementById("single-overwrite-check").checked,
            "plot_after_train": true,
            "train": train,
            "plot": {
                "seed": seed,
                "result_root": result_root,
                "run_name": run_name,
                "log_dir": "",
                "legacy dir_path": `${result_root}_seed`
            }
        };
    }

    async function loadConfigList() {
        try {
            const res = await fetch(`${API_BASE}/api/configs`);
            const data = await res.json();
            if (data.success && data.configs) {
                configSelect.innerHTML = '<option value="">-- 新建配置 --</option>';
                data.configs.forEach(name => {
                    const opt = document.createElement("option");
                    opt.value = name;
                    opt.textContent = name;
                    configSelect.appendChild(opt);
                });
            }
        } catch (err) {
            console.error("加载配置列表失败:", err);
        }
    }

    configSelect.addEventListener("change", async () => {
        const name = configSelect.value;
        if (!name) {
            configNameInput.value = "";
            return;
        }
        configNameInput.value = name;
        try {
            const res = await fetch(`${API_BASE}/api/configs?name=${encodeURIComponent(name)}`);
            const data = await res.json();
            if (data.success && data.config) {
                const config = data.config;
                if (config.train) {
                    document.querySelectorAll("[data-train-field]").forEach(input => {
                        const fieldName = input.getAttribute("data-train-field");
                        if (fieldName in config.train) {
                            const val = config.train[fieldName];
                            if (input.tagName === "SELECT") {
                                input.value = val;
                            } else if (input.type === "checkbox") {
                                input.checked = (String(val).toLowerCase() === "true" || val === 1 || val === true);
                            } else {
                                input.value = val;
                            }
                        }
                    });
                }
                if (config.overwrite_existing !== undefined) {
                    document.getElementById("single-overwrite-check").checked = config.overwrite_existing;
                }
            }
        } catch (err) {
            console.error("读取配置详情出错:", err);
        }
    });

    saveTemplateBtn.addEventListener("click", async () => {
        const name = configNameInput.value.trim();
        if (!name) {
            alert("请输入配置模版名称。");
            return;
        }
        const configPayload = getFormConfig();
        try {
            const res = await fetch(`${API_BASE}/api/configs`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name: name,
                    config: configPayload
                })
            });
            const data = await res.json();
            if (data.success) {
                alert(`配置模版 "${data.name}" 保存成功！`);
                await loadConfigList();
                configSelect.value = data.name;
            } else {
                alert(`保存失败: ${data.message}`);
            }
        } catch (err) {
            alert(`请求失败: ${err}`);
        }
    });

    refreshTemplateBtn.addEventListener("click", loadConfigList);

    // ==========================================================================
    // Initialize Dashboard UI & Charts
    // ==========================================================================
    function initChart() {
        const ctx = document.getElementById("pareto-chart-canvas").getContext("2d");
        
        paretoChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: '参数组合 (Comfort vs Safety)',
                    data: [],
                    backgroundColor: '#26a69a',
                    borderColor: '#26a69a',
                    pointRadius: 6,
                    pointHoverRadius: 9,
                    showLine: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const item = context.raw;
                                return [
                                    `配置: ${item.label}`,
                                    `簧上加速度 (Comfort): ${item.x.toFixed(4)} m/s²`,
                                    `轮胎动变形 (Safety): ${item.y.toFixed(4)} m`,
                                    `最终Cost: ${item.cost.toFixed(4)}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: '簧上加速度 RMS (Comfort) [m/s²] - 越小越好',
                            color: '#94a3b8',
                            font: { size: 11 }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: '轮胎动变形 RMS (Safety) [m] - 越小越好',
                            color: '#94a3b8',
                            font: { size: 11 }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    }
                }
            }
        });
    }

    // ==========================================================================
    // API Operations & Task Launchers
    // ==========================================================================
    
    function getActiveResultRoot() {
        const singleRootInput = document.querySelector("[data-train-field='result_root']");
        if (singleRootInput && singleRootInput.value.trim()) {
            return singleRootInput.value.trim();
        }
        if (analysisRootInput && analysisRootInput.value.trim()) {
            return analysisRootInput.value.trim();
        }
        return "./results_tmp/quarter_car_gym";
    }

    function getEvalCostSortValue(row) {
        const cost = Number(row && row.eval_cost);
        return Number.isFinite(cost) ? cost : Number.POSITIVE_INFINITY;
    }

    function sortByEvalCost(results) {
        return [...(results || [])].sort((a, b) => getEvalCostSortValue(a) - getEvalCostSortValue(b));
    }

    function escapeHtml(value) {
        return String(value ?? "").replace(/[&<>"']/g, (char) => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            "\"": "&quot;",
            "'": "&#39;"
        }[char]));
    }

    // Load History & Render Leaderboard + Pareto Chart
    async function loadHistory() {
        const root = getActiveResultRoot();
        try {
            const res = await fetch(`${API_BASE}/api/history?result_root=${encodeURIComponent(root)}&leaderboard=${encodeURIComponent(currentLeaderboard)}`);
            const data = await res.json();
            if (data.success) {
                const rankedResults = sortByEvalCost(data.results);
                renderSavedLeaderboard(rankedResults);
                updateChartData(rankedResults);
            }
        } catch (err) {
            console.error("加载历史数据失败:", err);
        }
    }

    function renderCurrentRunLeaderboard(results) {
        if (!results || results.length === 0) {
            currentRunLeaderboardTbody.innerHTML = `
                <tr>
                    <td colspan="8" class="table-empty">暂无评估完的数据</td>
                </tr>`;
            return;
        }
        
        currentRunLeaderboardTbody.innerHTML = sortByEvalCost(results).map((r, idx) => {
            const label = escapeHtml(r.label);
            return `
                <tr>
                    <td>${idx + 1}</td>
                    <td title="${label}">${label}</td>
                    <td>${r.lr}</td>
                    <td>${r.model_epochs}</td>
                    <td>${r.opt_steps}</td>
                    <td>${r.rms_acc ? Number(r.rms_acc).toFixed(4) : "-"}</td>
                    <td>${r.rms_tire ? Number(r.rms_tire).toFixed(4) : "-"}</td>
                    <td><strong>${r.eval_cost ? Number(r.eval_cost).toFixed(4) : "-"}</strong></td>
                </tr>`;
        }).join("");
    }

    function renderSavedLeaderboard(results) {
        if (!results || results.length === 0) {
            savedLeaderboardTbody.innerHTML = `
                <tr>
                    <td colspan="9" class="table-empty">暂无评估完的数据</td>
                </tr>`;
            return;
        }
        
        savedLeaderboardTbody.innerHTML = results.map((r, idx) => {
            const label = escapeHtml(r.label);
            const entryId = escapeHtml(r.entry_id || "");
            const rowIndex = r._row_index === undefined ? "" : escapeHtml(r._row_index);
            return `
                <tr>
                    <td>${idx + 1}</td>
                    <td title="${label}">${label}</td>
                    <td>${r.lr}</td>
                    <td>${r.model_epochs}</td>
                    <td>${r.opt_steps}</td>
                    <td>${r.rms_acc ? Number(r.rms_acc).toFixed(4) : "-"}</td>
                    <td>${r.rms_tire ? Number(r.rms_tire).toFixed(4) : "-"}</td>
                    <td><strong>${r.eval_cost ? Number(r.eval_cost).toFixed(4) : "-"}</strong></td>
                    <td>
                        <button class="delete-entry-btn btn-icon" data-label="${label}" data-entry-id="${entryId}" data-row-index="${rowIndex}" title="从排行榜删除该项" style="color: var(--danger); padding: 4px 8px; font-size: 11px;">
                            <i class="fa-solid fa-trash"></i>
                        </button>
                    </td>
                </tr>`;
        }).join("");
    }

    async function loadLeaderboardList() {
        const root = getActiveResultRoot();
        try {
            const res = await fetch(`${API_BASE}/api/leaderboards?result_root=${encodeURIComponent(root)}`);
            const data = await res.json();
            if (data.success && data.leaderboards) {
                // Populate Tab 4 select
                if (leaderboardSelect) {
                    const prevSelection = leaderboardSelect.value || currentLeaderboard;
                    leaderboardSelect.innerHTML = "";
                    data.leaderboards.forEach(lb => {
                        const opt = document.createElement("option");
                        opt.value = lb;
                        opt.textContent = lb === "default" ? "默认 (default)" : lb;
                        leaderboardSelect.appendChild(opt);
                    });
                    
                    if (data.leaderboards.includes(prevSelection)) {
                        leaderboardSelect.value = prevSelection;
                        currentLeaderboard = prevSelection;
                    } else {
                        leaderboardSelect.value = "default";
                        currentLeaderboard = "default";
                    }
                }
                
                // Populate Tab 2 select
                if (currentRunSaveTargetSelect) {
                    const prevSaveSelection = currentRunSaveTargetSelect.value || "default";
                    currentRunSaveTargetSelect.innerHTML = "";
                    data.leaderboards.forEach(lb => {
                        const opt = document.createElement("option");
                        opt.value = lb;
                        opt.textContent = lb === "default" ? "默认 (default)" : lb;
                        currentRunSaveTargetSelect.appendChild(opt);
                    });
                    
                    if (data.leaderboards.includes(prevSaveSelection)) {
                        currentRunSaveTargetSelect.value = prevSaveSelection;
                    } else {
                        currentRunSaveTargetSelect.value = "default";
                    }
                }
            }
        } catch (err) {
            console.error("加载排行榜列表失败:", err);
        }
    }

    async function saveLeaderboard() {
        const newName = newLeaderboardNameInput.value.trim();
        if (!newName) {
            alert("请输入新排行榜的名称。");
            return;
        }
        
        const root = getActiveResultRoot();
        try {
            const res = await fetch(`${API_BASE}/api/leaderboard/save`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    result_root: root,
                    current_name: currentLeaderboard,
                    new_name: newName
                })
            });
            const data = await handleResponse(res);
            if (data.success) {
                alert(data.message || "另存为成功！");
                newLeaderboardNameInput.value = "";
                currentLeaderboard = data.name;
                await loadLeaderboardList();
                await loadHistory();
            } else {
                alert(`保存失败: ${data.message}`);
            }
        } catch (err) {
            alert(`请求失败: ${err.message || err}`);
        }
    }

    async function deleteLeaderboard() {
        if (currentLeaderboard === "default") {
            alert("不能删除默认排行榜！");
            return;
        }
        if (!confirm(`您确定要永久删除排行榜 "${currentLeaderboard}" 吗？此操作无法恢复！`)) {
            return;
        }
        
        const root = getActiveResultRoot();
        try {
            const res = await fetch(`${API_BASE}/api/leaderboard/delete`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    result_root: root,
                    name: currentLeaderboard
                })
            });
            const data = await handleResponse(res);
            if (data.success) {
                alert(data.message || "删除成功。");
                currentLeaderboard = "default";
                await loadLeaderboardList();
                await loadHistory();
            } else {
                alert(`删除失败: ${data.message}`);
            }
        } catch (err) {
            alert(`请求失败: ${err.message || err}`);
        }
    }

    async function deleteLeaderboardEntry(label, entryId = "", rowIndex = "") {
        if (!confirm(`您确定要从排行榜中删除配置 "${label}" 吗？`)) {
            return;
        }
        
        const root = getActiveResultRoot();
        const payload = {
            result_root: root,
            leaderboard: currentLeaderboard,
            label: label
        };
        if (entryId) {
            payload.entry_id = entryId;
        }
        if (rowIndex !== "") {
            payload.row_index = rowIndex;
        }

        try {
            const res = await fetch(`${API_BASE}/api/leaderboard/delete_entry`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await handleResponse(res);
            if (data.success) {
                await loadHistory();
            } else {
                alert(`删除失败: ${data.message}`);
            }
        } catch (err) {
            alert(`请求失败: ${err.message || err}`);
        }
    }

    async function uploadRunToLeaderboard(runName, seed, resultRoot) {
        try {
            const res = await fetch(`${API_BASE}/api/leaderboard/upload`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    run_name: runName,
                    seed: parseInt(seed) || 1,
                    result_root: resultRoot,
                    leaderboard: currentLeaderboard
                })
            });
            const data = await handleResponse(res);
            if (data.success) {
                alert(data.message || "上传成功！");
                await loadHistory();
            } else {
                alert(`上传失败: ${data.message}`);
            }
        } catch (err) {
            alert(`请求失败: ${err.message || err}`);
        }
    }

    async function uploadCurrentRunToLeaderboard() {
        const root = getActiveResultRoot();
        const targetLb = currentRunSaveTargetSelect ? currentRunSaveTargetSelect.value : "default";
        
        try {
            const res = await fetch(`${API_BASE}/api/leaderboard/upload_current`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    result_root: root,
                    leaderboard: targetLb
                })
            });
            const data = await handleResponse(res);
            if (data.success) {
                alert(data.message || "当前排行结果已成功添加到持久排行榜！");
                
                if (leaderboardSelect) {
                    leaderboardSelect.value = targetLb;
                    currentLeaderboard = targetLb;
                }
                
                uploadCurrentBtn.disabled = true;
                
                await loadHistory();
                
                const lbTabBtn = document.querySelector('.tab-btn[data-tab="leaderboard-view"]');
                if (lbTabBtn) {
                    lbTabBtn.click();
                }
            } else {
                alert(`添加失败: ${data.message}`);
            }
        } catch (err) {
            alert(`请求失败: ${err.message || err}`);
        }
    }

    function syncAnalysisInputsToLastRun() {
        if (lastRunInfo.seed !== undefined && analysisSeedInput) {
            analysisSeedInput.value = lastRunInfo.seed;
        }
        if (lastRunInfo.resultRoot && analysisRootInput) {
            analysisRootInput.value = lastRunInfo.resultRoot;
        }
    }

    function updateChartData(results) {
        if (!paretoChart) return;
        
        // 将数据映射到散点图
        const chartPoints = results
            .filter(r => r.rms_acc && r.rms_tire)
            .map(r => ({
                x: Number(r.rms_acc),
                y: Number(r.rms_tire),
                label: r.label,
                cost: Number(r.eval_cost)
            }));
            
        paretoChart.data.datasets[0].data = chartPoints;
        paretoChart.update();
    }

    // Launch Single Run Subprocess
    async function startSingleRun() {
        const configPayload = getFormConfig();
        const payload = {
            train_mode: document.getElementById("single-train-mode-select").value,
            overwrite_existing: document.getElementById("single-overwrite-check").checked,
            train: configPayload.train
        };
        
        singleStartBtn.disabled = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        
        consoleOutputBox.innerHTML = '<div class="console-line system-msg">[SYSTEM] 单次训练任务已下发，正在启动计算子进程...</div>';
        knownLogLines.clear();
        
        try {
            const res = await fetch(`${API_BASE}/api/start_single`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                appendConsoleLine(`[SYSTEM] 训练进程启动成功。正在执行...`, "system-msg");
                startPolling();
            } else {
                alert(`启动失败: ${data.message}`);
                singleStartBtn.disabled = false;
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        } catch (err) {
            alert(`网络连接失败: ${err}`);
            singleStartBtn.disabled = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }
    }

    // Launch Sweep Subprocess
    async function startSweep() {
        // 自定义校验
        let customConfigs = "";
        if (sweepSelect.value === "custom") {
            customConfigs = customConfigsInput.value.trim();
            if (!customConfigs) {
                alert("在自定义扫参模式下，必须提供 JSON 格式的参数配置。");
                return;
            }
            try {
                JSON.parse(customConfigs);
            } catch (err) {
                alert(`JSON 格式错误，请检查输入:\n${err.message}`);
                return;
            }
        }
        
        const config = {
            script: scriptSelect.value,
            seed: parseInt(seedInput.value) || 1,
            device: deviceSelect.value,
            sweep_mode: sweepSelect.value,
            num_trials: parseInt(trialsInput.value) || 2,
            custom_configs: customConfigs
        };

        startBtn.disabled = true;
        singleStartBtn.disabled = true;
        
        consoleOutputBox.innerHTML = '<div class="console-line system-msg">[SYSTEM] 扫参任务成功下发，正在启动后台计算引擎...</div>';
        knownLogLines.clear();
        
        try {
            const res = await fetch(`${API_BASE}/api/start`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(config)
            });
            const data = await res.json();
            if (data.success) {
                appendConsoleLine(`[SYSTEM] 扫参引擎启动成功。正在计算...`, "system-msg");
                stopBtn.disabled = false;
                startPolling();
            } else {
                alert(`启动失败: ${data.message}`);
                startBtn.disabled = false;
                singleStartBtn.disabled = false;
            }
        } catch (err) {
            alert("与后端服务器通信失败。");
            startBtn.disabled = false;
            singleStartBtn.disabled = false;
        }
    }

    // Terminate Subprocess
    async function stopSweep() {
        if (!confirm("确定要强行终止当前的计算任务吗？这会杀死正在训练的 Python 子进程。")) {
            return;
        }
        
        try {
            const res = await fetch(`${API_BASE}/api/stop`, { method: "POST" });
            const data = await res.json();
            if (data.success) {
                appendConsoleLine(`[SYSTEM] 计算任务已被人工中止。`, "err-msg");
                stopBtn.disabled = true;
            } else {
                alert(`中止任务失败: ${data.message}`);
            }
        } catch (err) {
            alert("与后端通信失败。");
        }
    }

    // ==========================================================================
    // Polling Loop & State Sync
    // ==========================================================================
    function startPolling() {
        if (isMonitoring) return;
        isMonitoring = true;
        
        pollIntervalId = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE}/api/status`);
                const state = await res.json();
                
                syncUIState(state);
                
                if (!state.running) {
                    stopPolling();
                    syncAnalysisInputsToLastRun();
                    loadHistory();
                    loadLeaderboardList();
                    loadAnalysisRuns(lastRunInfo.runName);
                }
            } catch (err) {
                console.error("轮询状态出错:", err);
            }
        }, 1000);
    }

    function stopPolling() {
        if (!isMonitoring) return;
        isMonitoring = false;
        clearInterval(pollIntervalId);
        
        // 恢复按钮状态
        startBtn.disabled = false;
        singleStartBtn.disabled = false;
        stopBtn.disabled = true;
    }

    function syncUIState(state) {
        // 1. 头部运行状态
        if (state.running) {
            overallStatusBadge.className = "status-indicator running";
            overallStatusText.textContent = `正在运行 - ${state.status}`;
        } else {
            overallStatusBadge.className = "status-indicator idle";
            overallStatusText.textContent = `系统空闲 (${state.status})`;
        }
        elapsedTimeDisplay.textContent = `已耗时: ${state.elapsed.toFixed(1)}s`;
        if (dashboardVersionBadge) {
            const apiVersion = state.dashboard_version || "old";
            const pidText = state.server_pid ? `PID ${state.server_pid}` : "PID ?";
            dashboardVersionBadge.textContent = `UI v${UI_VERSION} / API v${apiVersion} / ${pidText}`;
            dashboardVersionBadge.title = `UI version: ${UI_VERSION}\nAPI version: ${apiVersion}\nStarted: ${state.server_started_at || "unknown"}\nWorkspace: ${state.workspace || "unknown"}`;
        }
        
        // 2. 进度条
        const percent = state.total_configs > 0 ? (state.completed.length / state.total_configs) * 100 : 0;
        progressBar.style.width = `${percent}%`;
        progressPercentText.textContent = `${percent.toFixed(0)}%`;
        progressRatioText.textContent = `完成组数: ${state.completed.length} / ${state.total_configs}`;
        
        // 估算剩余时间
        if (state.completed.length > 0 && state.running) {
            const avgTime = state.elapsed / state.completed.length;
            const remainingTime = avgTime * (state.total_configs - state.completed.length);
            estRemainingText.textContent = `预计剩余: ${remainingTime.toFixed(1)}s`;
        } else if (state.running) {
            estRemainingText.textContent = `预计剩余: 估算中...`;
        } else {
            estRemainingText.textContent = `预计剩余: -`;
        }
        
        // 3. 当前超参详情
        currentLabelVal.textContent = state.label || "-";
        currentStatusVal.textContent = state.status || "-";
        currentLrVal.textContent = state.lr || "-";
        currentEpochsVal.textContent = state.model_epochs || "-";
        currentStepsVal.textContent = state.opt_steps || "-";
        currentTrialVal.textContent = state.trial_str || "-";
        
        if (state.status === "控制策略更新" || state.status === "物理评估") {
            currentStepVal.textContent = `Step ${state.opt_step}`;
            currentCostVal.textContent = state.opt_cost ? Number(state.opt_cost).toFixed(4) : "-";
        } else {
            currentStepVal.textContent = "-";
            currentCostVal.textContent = "-";
        }
        
        // 4. GP MSE 拟合误差
        if (state.gp_errors && Object.keys(state.gp_errors).length > 0) {
            gpErrorsList.innerHTML = Object.entries(state.gp_errors).map(([gp, err]) => {
                return `
                    <div class="gp-error-card">
                        <span class="gp-label">${gp}</span>
                        <span class="gp-val">${err}</span>
                    </div>`;
            }).join("");
        } else {
            gpErrorsList.innerHTML = `<div class="gp-empty">等待 GP 拟合数据...</div>`;
        }
        
        // 5. 终端流式日志
        if (state.console_feed && state.console_feed.length > 0) {
            state.console_feed.forEach(line => {
                if (!knownLogLines.has(line)) {
                    knownLogLines.add(line);
                    let isErr = line.includes("[ERROR]") || line.includes("Exception") || line.includes("ModuleNotFoundError");
                    appendConsoleLine(line, isErr ? "err-msg" : "");
                }
            });
        }
        
        // 6. 渲染当前运行结果的排行榜
        renderCurrentRunLeaderboard(state.completed || []);

        // 7. 更新当前/上一次运行的元数据
        if (state.label && state.label !== "未启动" && state.label !== "-") {
            lastRunInfo.runName = state.label;
            if (state.seed !== undefined) lastRunInfo.seed = state.seed;
            if (state.result_root !== undefined) lastRunInfo.resultRoot = state.result_root;
            
            const hasData = state.completed && state.completed.length > 0;
            if (!state.running && (state.status === "已结束" || state.status === "已完成" || state.status === "物理评估" || state.status === "已中止") && hasData) {
                uploadCurrentBtn.disabled = false;
            } else {
                uploadCurrentBtn.disabled = true;
            }
        } else {
            uploadCurrentBtn.disabled = true;
        }
    }

    function appendConsoleLine(text, className = "") {
        const lineEl = document.createElement("div");
        lineEl.className = `console-line ${className}`;
        lineEl.textContent = text;
        consoleOutputBox.appendChild(lineEl);
        
        // 滚动到底部
        consoleOutputBox.scrollTop = consoleOutputBox.scrollHeight;
    }

    // ==========================================================================
    // Experiment Results Analysis
    // ==========================================================================
    async function loadAnalysisRuns(preferredRun = "") {
        const seed = analysisSeedInput.value;
        const root = analysisRootInput.value;
        const requestedRun = typeof preferredRun === "string" ? preferredRun : "";
        const currentSelected = requestedRun || analysisRunSelect.value;

        analysisRunSelect.innerHTML = '<option value="">-- 正在加载实验 --</option>';
        analysisRunSelect.disabled = true;
        if (analysisRunsStatus) {
            analysisRunsStatus.textContent = `正在查询 seed=${seed}, root=${root}`;
        }
        
        try {
            const runsUrl = `${API_BASE}/api/runs?seed=${encodeURIComponent(seed)}&result_root=${encodeURIComponent(root)}`;
            const res = await fetch(runsUrl);
            const data = await handleResponse(res);
            if (data.success && data.runs) {
                if (data.resolved_seed && data.resolved_seed !== seed) {
                    analysisSeedInput.value = data.resolved_seed;
                }
                analysisRunSelect.innerHTML = data.runs.length > 0
                    ? '<option value="">-- 请选择实验 --</option>'
                    : '<option value="">-- 未找到实验 --</option>';
                data.runs.forEach(run => {
                    const opt = document.createElement("option");
                    opt.value = run;
                    opt.textContent = run;
                    analysisRunSelect.appendChild(opt);
                });
                if (data.runs.includes(currentSelected)) {
                    analysisRunSelect.value = currentSelected;
                }
                if (analysisRunsStatus) {
                    const count = data.count ?? data.runs.length;
                    const fallbackText = data.fallback ? `，已自动切换到 seed=${data.resolved_seed}` : "";
                    const dirText = data.seed_dir
                        ? `，目录: ${data.seed_dir}`
                        : `，后端未返回目录，请重启 web_dashboard_server.py。请求: ${runsUrl || `/api/runs seed=${seed}, root=${root}`}`;
                    analysisRunsStatus.textContent = count > 0
                        ? `已找到 ${count} 个实验${fallbackText}${dirText}`
                        : `未找到实验${dirText}`;
                }
                console.info(`加载实验列表: ${data.count ?? data.runs.length} 个`, data.seed_dir || "");
            }
        } catch (err) {
            console.error("加载实验列表出错:", err);
            analysisRunSelect.innerHTML = '<option value="">-- 加载失败 --</option>';
            if (analysisRunsStatus) {
                analysisRunsStatus.textContent = `加载失败: ${err.message || err}`;
            }
        } finally {
            analysisRunSelect.disabled = false;
        }
    }

    analysisSeedInput.addEventListener("change", () => loadAnalysisRuns());
    analysisRootInput.addEventListener("change", () => loadAnalysisRuns());

    loadPlotsBtn.addEventListener("click", async () => {
        const runName = analysisRunSelect.value;
        const seed = analysisSeedInput.value;
        const root = analysisRootInput.value;
        
        if (!runName) {
            alert("请先选择一个实验！");
            return;
        }
        
        try {
            const res = await fetch(`${API_BASE}/api/plots?run_name=${encodeURIComponent(runName)}&seed=${encodeURIComponent(seed)}&result_root=${encodeURIComponent(root)}`);
            const data = await res.json();
            
            const gallery = document.getElementById("plots-gallery");
            const tabsContainer = document.getElementById("gallery-tabs");
            const displayImg = document.getElementById("gallery-display-image");
            const caption = document.getElementById("gallery-display-caption");
            
            if (data.success && data.plots && data.plots.length > 0) {
                gallery.style.display = "block";
                tabsContainer.innerHTML = "";
                
                data.plots.forEach((file, index) => {
                    const tabBtn = document.createElement("button");
                    tabBtn.className = "gallery-tab-btn" + (index === 0 ? " active" : "");
                    let label = file;
                    if (file === "rms_trend.png") label = "RMS 趋势 (RMS Trend)";
                    else if (file === "learning_plot.png") label = "策略学习曲线 (Learning)";
                    else if (file.includes("rollout")) label = "轨迹预测 (Rollout Path)";
                    else if (file.includes("road")) label = "路面输入激励 (Road Profile)";
                    else if (file.includes("state")) label = "状态轨迹分布 (States)";
                    
                    tabBtn.textContent = label;
                    tabBtn.addEventListener("click", () => {
                        document.querySelectorAll(".gallery-tab-btn").forEach(b => b.classList.remove("active"));
                        tabBtn.classList.add("active");
                        displayImg.src = `${API_BASE}/api/plot_file?run_name=${encodeURIComponent(runName)}&seed=${encodeURIComponent(seed)}&result_root=${encodeURIComponent(root)}&file=${encodeURIComponent(file)}`;
                        caption.textContent = label;
                    });
                    tabsContainer.appendChild(tabBtn);
                });
                
                // Show first plot
                const firstFile = data.plots[0];
                displayImg.src = `${API_BASE}/api/plot_file?run_name=${encodeURIComponent(runName)}&seed=${encodeURIComponent(seed)}&result_root=${encodeURIComponent(root)}&file=${encodeURIComponent(firstFile)}`;
                let firstLabel = firstFile;
                if (firstFile === "rms_trend.png") firstLabel = "RMS 趋势 (RMS Trend)";
                else if (firstFile === "learning_plot.png") firstLabel = "策略学习曲线 (Learning)";
                caption.textContent = firstLabel;
            } else {
                gallery.style.display = "none";
                alert("该实验目录下未找到生成的可视化 PNG 图片。请确认实验已经运行完成且 log_plot 脚本已生成图表。");
            }
        } catch (err) {
            alert("加载图片列表失败: " + err);
        }
    });

    // ==========================================================================
    // Bind Event Listeners
    // ==========================================================================
    startBtn.addEventListener("click", startSweep);
    stopBtn.addEventListener("click", stopSweep);
    singleStartBtn.addEventListener("click", startSingleRun);

    // Leaderboard event listeners
    leaderboardSelect.addEventListener("change", () => {
        currentLeaderboard = leaderboardSelect.value;
        loadHistory();
    });

    saveLeaderboardBtn.addEventListener("click", saveLeaderboard);
    deleteLeaderboardBtn.addEventListener("click", deleteLeaderboard);

    // Upload events
    uploadCurrentBtn.addEventListener("click", () => {
        uploadCurrentRunToLeaderboard();
    });

    uploadPastBtn.addEventListener("click", () => {
        const runName = analysisRunSelect.value;
        const seed = analysisSeedInput.value;
        const root = analysisRootInput.value;
        if (!runName) {
            alert("请先在列表中选择一个历史实验！");
            return;
        }
        uploadRunToLeaderboard(runName, seed, root);
    });

    // Delete single entry (event delegation)
    savedLeaderboardTbody.addEventListener("click", (e) => {
        const btn = e.target.closest(".delete-entry-btn");
        if (btn) {
            const label = btn.getAttribute("data-label") || "";
            const entryId = btn.getAttribute("data-entry-id") || "";
            const rowIndex = btn.getAttribute("data-row-index") || "";
            if (label || entryId || rowIndex) {
                deleteLeaderboardEntry(label, entryId, rowIndex);
            }
        }
    });

    // ==========================================================================
    // Startup & Initialization
    // ==========================================================================
    initChart();
    loadHistory();
    loadLeaderboardList();
    loadConfigList();
    loadAnalysisRuns();
    
    // 启动时检查服务器是否已经在运行（如页面刷新了）
    fetch(`${API_BASE}/api/status`)
        .then(res => res.json())
        .then(state => {
            syncUIState(state);
            if (state.running) {
                startBtn.disabled = true;
                singleStartBtn.disabled = true;
                stopBtn.disabled = false;
                startPolling();
            }
        })
        .catch(err => {
            console.error("Failed to load initial status:", err);
            if (dashboardVersionBadge) {
                dashboardVersionBadge.textContent = `UI v${UI_VERSION} / API unavailable / PID ?`;
                dashboardVersionBadge.title = "Initial /api/status request failed. Make sure web_dashboard_server.py is running.";
            }
        });
});
