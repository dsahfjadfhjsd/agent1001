/**
 * Agent1001 Real-time Web Demo JavaScript
 * Handles real-time communication with backend and UI interactions
 */

class Agent1001Demo {
    constructor() {
        this.socket = io();
        this.isRunning = false;
        this.currentRound = 0;
        this.totalRounds = 0;
        this.charts = {};
        this.notifications = [];
        this.postsData = new Map(); // Store post data for detail view
        
        this.initializeEventListeners();
        this.initializeSocketListeners();
        this.initializeUI();
        this.startTimeUpdater();
    }

    // Initialize event listeners
    initializeEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // Parameter panel collapse
        document.getElementById('collapse-params').addEventListener('click', () => {
            this.toggleParameterPanel();
        });

        // Control buttons
        document.getElementById('start-simulation').addEventListener('click', () => {
            this.startSimulation();
        });

        document.getElementById('stop-simulation').addEventListener('click', () => {
            this.stopSimulation();
        });

        document.getElementById('reset-params').addEventListener('click', () => {
            this.resetParameters();
        });

        document.getElementById('clear-logs').addEventListener('click', () => {
            this.clearLogs();
        });

        // Auto-scroll toggle
        document.getElementById('auto-scroll-posts').addEventListener('click', (e) => {
            e.target.classList.toggle('active');
        });

        // Modal close events
        document.getElementById('close-post-detail').addEventListener('click', () => {
            this.closePostDetailModal();
        });

        // Close modal when clicking overlay
        document.getElementById('post-detail-modal').addEventListener('click', (e) => {
            if (e.target.id === 'post-detail-modal') {
                this.closePostDetailModal();
            }
        });

        // ESC key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closePostDetailModal();
            }
        });
    }

    // Initialize socket event listeners
    initializeSocketListeners() {
        this.socket.on('connect', () => {
            this.updateConnectionStatus('connected');
            this.addLog('info', 'Connected to server');
        });

        this.socket.on('disconnect', () => {
            this.updateConnectionStatus('disconnected');
            this.addLog('error', 'Disconnected from server');
        });

        this.socket.on('status_update', (data) => {
            this.handleStatusUpdate(data);
        });

        this.socket.on('initialization_result', (data) => {
            this.handleInitializationResult(data);
        });

        this.socket.on('simulation_start', (data) => {
            this.handleSimulationStart(data);
        });

        this.socket.on('round_start', (data) => {
            this.handleRoundStart(data);
        });

        this.socket.on('realtime_update', (data) => {
            this.handleRealtimeUpdate(data);
        });

        this.socket.on('round_complete', (data) => {
            this.handleRoundComplete(data);
        });

        this.socket.on('evaluation_start', (data) => {
            this.handleEvaluationStart(data);
        });

        this.socket.on('evaluation_complete', (data) => {
            this.handleEvaluationComplete(data);
        });

        this.socket.on('simulation_complete', (data) => {
            this.handleSimulationComplete(data);
        });

        this.socket.on('simulation_error', (data) => {
            this.handleSimulationError(data);
        });
    }

    // Update per-round simulation progress (posts processed within current round)
    updateSimulationProgress(data) {
        try {
            const processed = data?.processed_posts ?? 0;
            const total = data?.total_posts ?? 0;
            const roundIdx = data?.round_index ?? this.currentRound;
            if (roundIdx) this.currentRound = roundIdx;

            // Base round display
            const span = document.getElementById('current-round-display');
            const baseText = `轮次: ${this.currentRound}/${this.totalRounds}`;
            if (total > 0) {
                span.textContent = `${baseText} · 进度: ${processed}/${total}`;
            } else {
                span.textContent = baseText;
            }
        } catch (_) {
            // no-op
        }
    }

    // Initialize UI components
    initializeUI() {
        this.hideLoading();
        this.updateConnectionStatus('connecting');
        
        // Initialize charts
        this.initializeCharts();
        
        // Set default parameter values
        this.resetParameters();
        
        // Add initial logs
        this.addLog('info', 'Agent1001 Demo initialized');
    }

    // Time updater
    startTimeUpdater() {
        setInterval(() => {
            const now = new Date();
            document.getElementById('current-time').textContent = 
                now.toLocaleTimeString('zh-CN', { hour12: false });
        }, 1000);
    }

    // Tab switching
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `${tabName}-tab`);
        });

        // Resize charts if analytics tab is selected
        if (tabName === 'analytics') {
            setTimeout(() => this.resizeCharts(), 100);
        }
    }

    // Parameter panel toggle
    toggleParameterPanel() {
        const panel = document.querySelector('.parameter-panel');
        const btn = document.getElementById('collapse-params');
        
        panel.classList.toggle('collapsed');
        
        const icon = btn.querySelector('i');
        if (panel.classList.contains('collapsed')) {
            icon.className = 'fas fa-chevron-right';
        } else {
            icon.className = 'fas fa-chevron-left';
        }
    }

    // Connection status update
    updateConnectionStatus(status) {
        const indicator = document.getElementById('connection-status');
        const text = document.getElementById('connection-text');
        
        indicator.className = `fas fa-circle status-indicator ${status}`;
        
        switch (status) {
            case 'connected':
                text.textContent = '已连接';
                break;
            case 'connecting':
                text.textContent = '连接中...';
                break;
            case 'disconnected':
                text.textContent = '连接断开';
                break;
        }
    }

    // Get current parameters
    getParameters() {
        return {
            rounds_before_eval: parseInt(document.getElementById('rounds-before-eval').value),
            rounds_after_eval: parseInt(document.getElementById('rounds-after-eval').value),
            posts_per_round: parseInt(document.getElementById('posts-per-round').value),
            users_per_post: parseInt(document.getElementById('users-per-post').value),
            rounds_per_post: parseInt(document.getElementById('rounds-per-post').value),
            total_users: parseInt(document.getElementById('total-users').value),
            max_concurrent_users: parseInt(document.getElementById('max-concurrent').value),
            thinking_model: document.getElementById('thinking-model').value,
            enable_prompt_export: document.getElementById('enable-prompt-export').checked,
            posts_source: document.getElementById('posts-source').value,
            csv_path: document.getElementById('csv-path').value,
            max_posts: parseInt(document.getElementById('max-posts').value),
            action_probability: parseFloat(document.getElementById('action-probability').value),
            comment_probability: parseFloat(document.getElementById('comment-probability').value),
            batch_id: (document.getElementById('batch-id').value || '').trim()
        };
    }

    // Reset parameters to defaults
    resetParameters() {
        document.getElementById('rounds-before-eval').value = 2;
        document.getElementById('rounds-after-eval').value = 1;
        document.getElementById('posts-per-round').value = 3;
        document.getElementById('users-per-post').value = 6;
        document.getElementById('rounds-per-post').value = 1;
        document.getElementById('total-users').value = 24;
        document.getElementById('max-concurrent').value = 10;
        document.getElementById('thinking-model').value = 'qwen-max';
        document.getElementById('enable-prompt-export').checked = false;
        document.getElementById('posts-source').value = 'csv';
        document.getElementById('csv-path').value = 'Data/integrated_data/XMSU7D_integrated_articles.csv';
        document.getElementById('max-posts').value = 12;
        document.getElementById('action-probability').value = 0.7;
        document.getElementById('comment-probability').value = 0.5;
        document.getElementById('batch-id').value = '';

        this.showNotification('success', '参数已重置为默认值');
    }

    // Start simulation
    startSimulation() {
        if (this.isRunning) {
            this.showNotification('warning', '仿真已在运行中');
            return;
        }

        const parameters = this.getParameters();
        
        // Update UI immediately for better user feedback
        document.getElementById('start-simulation').disabled = true;
        document.getElementById('stop-simulation').disabled = false;
        this.updateSimulationStatus('启动中', 'running');
        
        this.addLog('info', 'Starting simulation with parameters:', parameters);
        this.showNotification('info', '正在启动仿真...');
        
        // Emit start simulation event
        this.socket.emit('start_simulation', { parameters });
    }

    // Stop simulation
    stopSimulation() {
        this.socket.emit('stop_simulation');
        
        // Update UI
        document.getElementById('start-simulation').disabled = false;
        document.getElementById('stop-simulation').disabled = true;
        
        this.updateSimulationStatus('停止', 'stopped');
        this.addLog('warning', 'Simulation stopped by user');
        this.showNotification('warning', '仿真已停止');
    }

    // Handle status updates
    handleStatusUpdate(data) {
        this.addLog(data.type, data.message);
        
        if (data.type === 'init_success') {
            this.showNotification('success', '系统初始化成功');
        } else if (data.type === 'init_error') {
            this.showNotification('error', '系统初始化失败');
        } else if (data.type === 'connected') {
            this.showNotification('info', '已连接到服务器');
        }
    }
    
    // Handle initialization result
    handleInitializationResult(data) {
        if (data.success) {
            this.addLog('success', data.message || 'Demo initialization successful');
            this.showNotification('success', '系统准备就绪');
        } else {
            this.addLog('error', data.message || 'Demo initialization failed');
            this.showNotification('error', '系统初始化失败');
        }
    }

    // Handle simulation start
    handleSimulationStart(data) {
        this.isRunning = true;
        this.totalRounds = data.total_rounds;
        this.currentRound = 0;
        
        this.updateSimulationStatus('运行中', 'running');
        this.updateRoundDisplay(0, data.total_rounds);
        
        this.clearDisplayAreas();
        
        this.addLog('success', `Simulation started - Total rounds: ${data.total_rounds}`);
        this.showNotification('success', '仿真开始运行');
    }

    // Handle round start
    handleRoundStart(data) {
        this.currentRound = data.round;
        this.updateRoundDisplay(data.round, this.totalRounds);
        
        const statusText = data.type === 'initial' ? '初始轮次' : '优化轮次';
        this.updateSimulationStatus(`${statusText} - 轮次 ${data.round}`, 'running');
        
        this.addLog('info', `Round ${data.round} (${data.type}) started`);
    }

    // Handle real-time updates
    handleRealtimeUpdate(data) {
        switch (data.type) {
            case 'post_distributed':
                // Pass current round info to post item
                const roundInfo = {
                    round: this.currentRound,
                    type: data.data.round_type || 'initial',
                    scenario_name: data.data.scenario_name,
                    scenario_id: data.data.scenario_id
                };
                this.addPostItem(data.data, roundInfo);
                break;
            case 'user_simulation_start':
                this.addUserActivity(data.data.user_id, 'thinking', '正在思考...');
                break;
            case 'user_action':
                this.handleUserAction(data.data);
                break;
            case 'simulation_progress':
                this.updateSimulationProgress(data.data);
                break;
        }
    }

    // Handle round completion
    handleRoundComplete(data) {
        this.updateStats(data.stats);
        this.addLog('success', `Round ${data.round} (${data.type}) completed`);
        
        // Update charts
        this.updateCharts(data);
    }

    // Handle evaluation start
    handleEvaluationStart(data) {
        this.updateSimulationStatus('评估中', 'evaluating');
        this.addLog('info', 'Evaluation phase started');
        this.showNotification('info', '开始评估阶段...');
    }

    // Handle evaluation completion
    handleEvaluationComplete(data) {
        this.displayEvaluationResults(data.evaluation);
        this.addLog('success', 'Evaluation completed');
        this.showNotification('success', '评估完成');
    }

    // Handle simulation completion
    handleSimulationComplete(data) {
        this.isRunning = false;
        this.updateSimulationStatus('完成', 'completed');
        
        // Reset buttons
        document.getElementById('start-simulation').disabled = false;
        document.getElementById('stop-simulation').disabled = true;
        
        this.updateStats(data.final_stats);
        this.addLog('success', 'Simulation completed successfully');
        this.showNotification('success', '仿真完成！');
    }

    // Handle simulation error
    handleSimulationError(data) {
        this.isRunning = false;
        this.updateSimulationStatus('错误', 'error');
        
        // Reset buttons
        document.getElementById('start-simulation').disabled = false;
        document.getElementById('stop-simulation').disabled = true;
        
        this.addLog('error', `Simulation error: ${data.error}`);
        this.showNotification('error', `仿真错误: ${data.error}`);
    }

    // Update simulation status
    updateSimulationStatus(text, type) {
        const badge = document.getElementById('simulation-status');
        const span = badge.querySelector('span');
        const icon = badge.querySelector('i');
        
        span.textContent = text;
        badge.className = `status-badge ${type}`;
        
        // Update icon based on type
        switch (type) {
            case 'running':
                icon.className = 'fas fa-play';
                break;
            case 'evaluating':
                icon.className = 'fas fa-search';
                break;
            case 'completed':
                icon.className = 'fas fa-check';
                break;
            case 'error':
                icon.className = 'fas fa-exclamation-triangle';
                break;
            case 'stopped':
                icon.className = 'fas fa-stop';
                break;
            default:
                icon.className = 'fas fa-pause';
        }
    }

    // Update round display
    updateRoundDisplay(current, total) {
        document.getElementById('current-round-display').textContent = `轮次: ${current}/${total}`;
    }

    // Update statistics
    updateStats(stats) {
        document.getElementById('total-posts').textContent = stats.total_posts || 0;
        document.getElementById('total-users').textContent = stats.total_users || 0;
        document.getElementById('total-actions').textContent = stats.total_actions || 0;
    }

    // Clear display areas
    clearDisplayAreas() {
        this.clearContainer('posts-container');
        this.clearContainer('users-container');
        this.clearContainer('evaluation-container');
    }

    // Clear container and show empty state
    clearContainer(containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-inbox"></i>
                <p>等待数据...</p>
            </div>
        `;
    }

    // Add post item
    addPostItem(postData, roundInfo = {}) {
        const container = document.getElementById('posts-container');
        
        // Remove empty state if present
        if (container.querySelector('.empty-state')) {
            container.innerHTML = '';
        }
        
        // Store post data for detail view
        const postId = postData.post_id;
        this.postsData.set(postId, {
            ...postData,
            roundInfo,
            comments: [],
            likes: 0,
            shares: 0,
            timestamp: new Date().toISOString()
        });
        
        // Determine round badge class and text
        const roundNumber = roundInfo.round || this.currentRound;
        const roundType = roundInfo.type || 'initial';
        const badgeClass = roundType === 'optimized' ? 'post-eval' : 'pre-eval';
        
        // Check if this is a re-distribution of an existing post
        const existingPost = container.querySelector(`[data-post-id="${postId}"]`);
        let badgeText;
        if (existingPost) {
            // This is a second distribution
            badgeText = roundType === 'optimized' ? `第二次分发` : `第二次分发`;
        } else {
            // This is first distribution
            badgeText = roundType === 'optimized' ? `优化轮次 ${roundNumber}` : `初次分发`;
        }
        
        const postElement = document.createElement('div');
        postElement.className = 'post-item';
        postElement.setAttribute('data-post-id', postId);
        postElement.innerHTML = `
            <div class="post-round-badge ${badgeClass}">${badgeText}</div>
            <div class="post-header">
                <div>
                    <div class="post-title">${postData.title || `帖子 ${postId}`}</div>
                    <div class="post-meta">ID: ${postId} | 时间: ${new Date().toLocaleTimeString()}</div>
                </div>
            </div>
            <div class="post-content">${postData.content || '内容加载中...'}</div>
            <div class="post-interactions">
                <div class="interaction-stat">
                    <i class="fas fa-heart"></i>
                    <span class="likes-count">0</span>
                </div>
                <div class="interaction-stat">
                    <i class="fas fa-comment"></i>
                    <span class="comments-count">0</span>
                </div>
                <div class="interaction-stat">
                    <i class="fas fa-share"></i>
                    <span class="shares-count">0</span>
                </div>
            </div>
        `;
        
        // Add click event for post detail
        postElement.addEventListener('click', () => {
            this.showPostDetail(postId);
        });
        
        container.appendChild(postElement);
        
        // Auto-scroll if enabled
        if (document.getElementById('auto-scroll-posts').classList.contains('active')) {
            container.scrollTop = container.scrollHeight;
        }
    }

    // Add user activity
    addUserActivity(userId, status, activity) {
        const container = document.getElementById('users-container');
        
        // Remove empty state if present
        if (container.querySelector('.empty-state')) {
            container.innerHTML = '';
        }
        
        let userElement = container.querySelector(`[data-user-id="${userId}"]`);
        
        if (!userElement) {
            userElement = document.createElement('div');
            userElement.className = 'user-item';
            userElement.setAttribute('data-user-id', userId);
            userElement.innerHTML = `
                <div class="user-avatar">${userId.substring(0, 2).toUpperCase()}</div>
                <div class="user-info">
                    <div class="user-name">用户 ${userId}</div>
                    <div class="user-activity">${activity}</div>
                </div>
                <div class="user-status ${status}"></div>
            `;
            container.appendChild(userElement);
        } else {
            // Update existing user
            userElement.querySelector('.user-activity').textContent = activity;
            userElement.querySelector('.user-status').className = `user-status ${status}`;
            userElement.className = `user-item ${status === 'thinking' ? 'active' : ''}`;
        }
        
        // Update active users count
        const activeUsers = container.querySelectorAll('.user-item.active').length;
        document.getElementById('active-users-count').textContent = activeUsers;
    }

    // Handle user action
    handleUserAction(actionData) {
        // Update user status
        this.addUserActivity(actionData.user_id, 'active', `执行了${actionData.action_type}`);
        
        // Update post interactions
        this.updatePostInteractions(actionData.post_id, actionData.action_type, actionData.content);
        
        // Add to stats
        const currentActions = parseInt(document.getElementById('total-actions').textContent) + 1;
        document.getElementById('total-actions').textContent = currentActions;
    }

    // Update post interactions
    updatePostInteractions(postId, actionType, content = '') {
        // Update stored post data
        if (this.postsData.has(postId)) {
            const postData = this.postsData.get(postId);
            
            switch (actionType) {
                case 'like':
                    postData.likes += 1;
                    break;
                case 'comment':
                    postData.comments.push({
                        author: `用户${Math.random().toString(36).substr(2, 4)}`,
                        content: content || '这是一条模拟用户评论',
                        timestamp: new Date().toISOString()
                    });
                    break;
                case 'share':
                    postData.shares += 1;
                    break;
            }
            
            this.postsData.set(postId, postData);
        }
        
        // Update UI
        const posts = document.querySelectorAll('.post-item');
        posts.forEach(post => {
            const postIdElement = post.querySelector('.post-meta');
            if (postIdElement && postIdElement.textContent.includes(postId)) {
                let selector, currentCount;
                
                switch (actionType) {
                    case 'like':
                        selector = '.likes-count';
                        break;
                    case 'comment':
                        selector = '.comments-count';
                        break;
                    case 'share':
                        selector = '.shares-count';
                        break;
                }
                
                if (selector) {
                    const countElement = post.querySelector(selector);
                    currentCount = parseInt(countElement.textContent) + 1;
                    countElement.textContent = currentCount;
                    
                    // Add visual feedback
                    const statElement = countElement.closest('.interaction-stat');
                    statElement.classList.add('active');
                    setTimeout(() => statElement.classList.remove('active'), 2000);
                }
            }
        });
    }

    // Show post detail modal
    showPostDetail(postId) {
        const postData = this.postsData.get(postId);
        if (!postData) {
            console.warn('Post data not found for ID:', postId);
            return;
        }

        // Update modal content
        document.getElementById('modal-post-title').textContent = postData.title || `帖子 ${postId}`;
        document.getElementById('modal-post-id').textContent = postId;
        
        // Enhanced round info with scenario
        const roundType = postData.roundInfo?.type === 'optimized' ? '优化' : '初始';
        const scenarioInfo = postData.roundInfo?.scenario_name ? ` | ${postData.roundInfo.scenario_name}` : '';
        document.getElementById('modal-post-round').textContent = 
            `${roundType}轮次 ${postData.roundInfo?.round || this.currentRound}${scenarioInfo}`;
        
        document.getElementById('modal-post-time').textContent = 
            new Date(postData.timestamp).toLocaleString();
        document.getElementById('modal-post-content').textContent = postData.content || '内容加载中...';
        
        // Update stats
        document.getElementById('modal-likes-count').textContent = postData.likes || 0;
        document.getElementById('modal-comments-count').textContent = postData.comments?.length || 0;
        document.getElementById('modal-shares-count').textContent = postData.shares || 0;
        
        // Update comments
        this.updateModalComments(postData.comments || []);
        
        // Show modal
        document.getElementById('post-detail-modal').classList.add('show');
        document.body.style.overflow = 'hidden'; // Prevent background scrolling
    }

    // Close post detail modal
    closePostDetailModal() {
        document.getElementById('post-detail-modal').classList.remove('show');
        document.body.style.overflow = ''; // Restore scrolling
    }

    // Update modal comments
    updateModalComments(comments) {
        const container = document.getElementById('modal-comments-container');
        
        if (!comments || comments.length === 0) {
            container.innerHTML = `
                <div class="empty-comments">
                    <i class="fas fa-comment-slash"></i>
                    <p>暂无评论</p>
                </div>
            `;
            return;
        }

        container.innerHTML = comments.map(comment => `
            <div class="comment-item">
                <div class="comment-header">
                    <div class="comment-author">${comment.author}</div>
                    <div class="comment-time">${new Date(comment.timestamp).toLocaleString()}</div>
                </div>
                <div class="comment-content">${comment.content}</div>
            </div>
        `).join('');
    }

    // Display evaluation results
    displayEvaluationResults(evaluation) {
        const container = document.getElementById('evaluation-container');
        
        container.innerHTML = `
            <div class="evaluation-summary">
                <h4>评估摘要</h4>
                <div class="eval-metrics">
                    <div class="eval-metric">
                        <span class="metric-label">总体相似度</span>
                        <span class="metric-value">${(evaluation.similarity_score || 0.85).toFixed(2)}</span>
                    </div>
                    <div class="eval-metric">
                        <span class="metric-label">用户参与度</span>
                        <span class="metric-value">${(evaluation.engagement_score || 0.78).toFixed(2)}</span>
                    </div>
                    <div class="eval-metric">
                        <span class="metric-label">分发效果</span>
                        <span class="metric-value">${(evaluation.distribution_score || 0.82).toFixed(2)}</span>
                    </div>
                </div>
                <div class="eval-suggestions">
                    <h5>优化建议</h5>
                    <ul>
                        ${(evaluation.suggestions || ['提高内容质量', '优化分发时机', '增强用户互动']).map(s => `<li>${s}</li>`).join('')}
                    </ul>
                </div>
            </div>
        `;
    }

    // Initialize charts
    initializeCharts() {
        // Rounds chart
        const roundsCtx = document.getElementById('rounds-chart');
        if (roundsCtx) {
            this.charts.rounds = new Chart(roundsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '每轮互动数',
                        data: [],
                        borderColor: '#64ffda',
                        backgroundColor: 'rgba(100, 255, 218, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: '轮次互动统计',
                            color: '#ffffff'
                        },
                        legend: {
                            labels: {
                                color: '#ffffff'
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#a0a9c0' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            ticks: { color: '#a0a9c0' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
        }

        // Interactions chart
        const interactionsCtx = document.getElementById('interactions-chart');
        if (interactionsCtx) {
            this.charts.interactions = new Chart(interactionsCtx, {
                type: 'doughnut',
                data: {
                    labels: ['点赞', '评论', '分享'],
                    datasets: [{
                        data: [0, 0, 0],
                        backgroundColor: [
                            '#667eea',
                            '#f093fb',
                            '#4facfe'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: '互动类型分布',
                            color: '#ffffff'
                        },
                        legend: {
                            labels: {
                                color: '#ffffff'
                            }
                        }
                    }
                }
            });
        }
    }

    // Update charts
    updateCharts(data) {
        if (this.charts.rounds) {
            this.charts.rounds.data.labels.push(`轮次 ${data.round}`);
            this.charts.rounds.data.datasets[0].data.push(data.stats.total_actions || 0);
            this.charts.rounds.update();
        }
    }

    // Resize charts
    resizeCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.resize();
        });
    }

    // Add log entry
    addLog(type, message, data = null) {
        const container = document.getElementById('logs-content');
        
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        logEntry.innerHTML = `
            <div class="log-timestamp">[${timestamp}]</div>
            <div class="log-message">${message}</div>
            ${data ? `<pre>${JSON.stringify(data, null, 2)}</pre>` : ''}
        `;
        
        container.appendChild(logEntry);
        container.scrollTop = container.scrollHeight;
    }

    // Clear logs
    clearLogs() {
        document.getElementById('logs-content').innerHTML = '';
        this.addLog('info', 'Logs cleared');
    }

    // Show notification
    showNotification(type, message) {
        const container = document.getElementById('notifications');
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; color: inherit; cursor: pointer;">×</button>
            </div>
        `;
        
        container.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    // Show/hide loading
    showLoading(message = '加载中...') {
        const overlay = document.getElementById('loading-overlay');
        const messageEl = document.getElementById('loading-message');
        
        messageEl.textContent = message;
        overlay.classList.remove('hidden');
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        overlay.classList.add('hidden');
    }
}

// Initialize demo when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.agent1001Demo = new Agent1001Demo();
});
