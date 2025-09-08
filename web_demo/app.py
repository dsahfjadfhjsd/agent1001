#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent1001 Real-time Web Demo
Modern interface for social media simulation with real-time updates
"""

import asyncio
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Any

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from demo_2 import MultiScenarioSmartDemo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # avoid double logging via root

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'agent1001_demo_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Logging: redirect stdout/stderr to logger and broadcast to UI ---
class TeeStream(io.TextIOBase):
    """Tee stdout to both logger (for UI) and original console (for raw demo_2 prints)."""
    def __init__(self, logger, level=logging.INFO, console_stream=None):
        self.logger = logger
        self.level = level
        self.console_stream = console_stream or sys.__stdout__
        self._buffer = ''

    def write(self, buf):
        if not buf:
            return 0
        try:
            # Write raw to console (preserve demo_2 style emojis and formatting)
            self.console_stream.write(buf)
            self.console_stream.flush()
        except Exception:
            pass

        # Also forward line-wise to logger for UI streaming
        try:
            buf_norm = buf.replace('\r', '\n')
            for line in buf_norm.splitlines():
                line = line.strip()
                if line:
                    self.logger.log(self.level, line)
        except Exception:
            pass
        return len(buf)

    def flush(self):
        try:
            self.console_stream.flush()
        except Exception:
            pass

class SocketIOLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Map logging levels to UI types
            level = record.levelno
            if level >= logging.ERROR:
                ui_type = 'error'
            elif level >= logging.WARNING:
                ui_type = 'warning'
            elif level >= logging.INFO:
                ui_type = 'info'
            else:
                ui_type = 'info'
            socketio.emit('status_update', {
                'type': ui_type,
                'message': msg,
                'timestamp': datetime.now().isoformat()
            })
        except Exception:
            pass

# Attach handlers once
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
if not any(isinstance(h, SocketIOLogHandler) for h in logger.handlers):
    sio_handler = SocketIOLogHandler()
    sio_handler.setLevel(logging.INFO)
    sio_handler.setFormatter(formatter)
    logger.addHandler(sio_handler)

# Redirect prints to both console (raw) and UI logger
sys.stdout = TeeStream(logger, logging.INFO, console_stream=sys.__stdout__)
# Keep stderr as real stderr to avoid recursion; Flask/werkzeug logs still go to stderr
sys.stderr = sys.__stderr__

class RealTimeDemo:
    """Real-time web demo wrapper that orchestrates demo_2.MultiScenarioSmartDemo with Socket.IO"""

    def __init__(self):
        self.demo: Optional[MultiScenarioSmartDemo] = None
        self.is_running = False
        self.current_round = 0
        self.simulation_stats = {
            'total_rounds': 0,
            'total_posts': 0,
            'total_users': 0,
            'total_actions': 0,
            'evaluation_results': {}
        }
        
    def initialize_demo(self, config: Dict[str, Any]) -> bool:
        """Initialize the demo with given configuration"""
        try:
            logger.info("Starting demo initialization (lazy mode, no engines will be created)...")

            # In the web app, initialization should be LIGHTWEIGHT.
            # Do NOT construct MultiScenarioSmartDemo here because DISTAgent's constructor
            # allocates asyncio primitives requiring a running event loop. The SocketIO
            # connect thread has no loop, which previously caused: 
            # "There is no current event loop in thread ...".
            # We'll defer all heavy initialization to run_simulation() which runs inside
            # a dedicated asyncio event loop in a background thread.

            # Store configuration only
            self.config = config or {}

            socketio.emit('status_update', {
                'type': 'init_success',
                'message': 'Demo initialized (lazy). Ready to start simulation.',
                'timestamp': datetime.now().isoformat()
            })
            logger.info("🎉 Demo initialization completed (lazy)")
            return True
            
        except Exception as e:
            error_msg = f"Initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            socketio.emit('status_update', {
                'type': 'init_error', 
                'message': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def emit_realtime_update(self, update_type: str, data: Dict[str, Any]):
        """Route real-time updates from demo_2 to the appropriate Socket.IO channel"""
        try:
            if update_type in {"round_start", "round_complete", "evaluation_start", "evaluation_complete"}:
                # Keep current_round in sync if provided
                if isinstance(data, dict) and 'round' in data:
                    try:
                        self.current_round = int(data['round'])
                    except Exception:
                        pass
                # Enrich evaluation payload for front-end simplified display
                if update_type == "evaluation_complete" and isinstance(data, dict) and 'evaluation' in data:
                    eval_data = data.get('evaluation', {}) or {}
                    try:
                        # Derive simple metrics expected by UI (with safe fallbacks)
                        sim_vs_real = eval_data.get('simulation_vs_real', {}) or {}
                        distribution_impact = eval_data.get('distribution_impact', {}) or {}
                        suggestions = distribution_impact.get('optimization_suggestions', []) or []
                        simple = dict(eval_data)
                        simple['similarity_score'] = sim_vs_real.get('similarity_score', 0.0)
                        # Heuristic placeholders if not provided
                        simple['engagement_score'] = float(simple.get('comment_similarity', {}).get('overall_stats', {}).get('average_similarity', 0.7))
                        simple['distribution_score'] = float(distribution_impact.get('distribution_features', {}).get('post_selection_strategy', {}).get('content_diversity', 0.8))
                        simple['suggestions'] = suggestions
                        data = {'evaluation': simple}
                    except Exception:
                        pass
                socketio.emit(update_type, data)
                return

            # Default: pack into a unified realtime_update channel
            socketio.emit('realtime_update', {
                'type': update_type,
                'data': data,
                'round': self.current_round,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Realtime emit failed ({update_type}): {e}")
    
    async def run_simulation(self, parameters: Dict[str, Any]):
        """Run the full demo_2 pipeline with real-time updates"""
        try:
            self.is_running = True
            self.current_round = 0
            # Reset stats for a fresh run
            self.simulation_stats = {
                'total_rounds': 0,
                'total_posts': 0,
                'total_users': 0,
                'total_actions': 0,
                'evaluation_results': {}
            }

            # Extract parameters
            num_rounds_before_eval = int(parameters.get('rounds_before_eval', 1))
            num_rounds_after_eval = int(parameters.get('rounds_after_eval', 1))
            posts_per_round = int(parameters.get('posts_per_round', 2))
            users_per_post = int(parameters.get('users_per_post', 6))
            rounds_per_post = int(parameters.get('rounds_per_post', 1))
            max_concurrent_users = int(parameters.get('max_concurrent_users', 10))
            thinking_model = parameters.get('thinking_model', 'qwen-max')
            enable_prompt_export = bool(parameters.get('enable_prompt_export', False))
            posts_source = parameters.get('posts_source', 'csv')
            csv_path = parameters.get('csv_path', 'Data/integrated_data/XMSU7D_integrated_articles.csv')
            max_posts = int(parameters.get('max_posts', 12))
            action_probability = float(parameters.get('action_probability', 0.7))
            comment_probability = float(parameters.get('comment_probability', 0.5))
            total_users = int(parameters.get('total_users', 24))
            batch_id = parameters.get('batch_id') or None

            # Create a fresh demo instance for this run to fully align with demo_2.py flow
            self.demo = MultiScenarioSmartDemo(batch_id=batch_id)
            self.demo.realtime_callback = self.emit_realtime_update

            # Load posts according to source
            if posts_source == 'csv':
                try:
                    if os.path.exists(csv_path):
                        posts = self.demo.load_posts_from_csv(csv_path, max_posts=max_posts)
                    else:
                        raise FileNotFoundError(f"CSV not found: {csv_path}")
                except Exception as e:
                    logger.warning(f"CSV load failed ({e}), falling back to sample posts")
                    posts = self.demo._load_sample_posts()
            else:
                posts = self.demo._load_sample_posts()

            self.demo.posts_data = posts
            self.demo.initialize_users(total_users=total_users)
            self.simulation_stats['total_users'] = len(self.demo.users_data or [])

            # Apply knobs to demo
            self.demo.posts_per_scenario = posts_per_round
            self.demo.users_per_post = users_per_post
            self.demo.rounds_per_post = rounds_per_post

            # Pre-compute scenarios to know total planned rounds
            try:
                self.demo.build_dynamic_scenarios(self.demo.posts_data)
            except Exception:
                pass
            planned_total_rounds = len(self.demo.scenarios or []) * (num_rounds_before_eval + num_rounds_after_eval)

            # Emit simulation start
            socketio.emit('simulation_start', {
                'parameters': parameters,
                'total_rounds': planned_total_rounds,
                'timestamp': datetime.now().isoformat()
            })

            # Initialize systems
            await self.demo.initialize()

            # Adjust simulation engine config based on UI
            try:
                if self.demo.sim_engine and self.demo.sim_engine.config:
                    self.demo.sim_engine.config.max_concurrent_requests = max_concurrent_users
                    self.demo.sim_engine.config.model_name = thinking_model
                    self.demo.sim_engine.config.export_prompts = enable_prompt_export
                    self.demo.sim_engine.config.action_probability = action_probability
                    self.demo.sim_engine.config.comment_probability = comment_probability
                    # Keep export directory consistent with batch id
                    self.demo.sim_engine.config.prompt_export_dir = f"Output/prompt_exports/{self.demo.batch_id}"
                    # Also refresh simulator internals that depend on config
                    if getattr(self.demo.sim_engine, 'simulator', None):
                        # Update concurrency semaphore to reflect new max
                        try:
                            self.demo.sim_engine.simulator.semaphore = asyncio.Semaphore(max_concurrent_users)
                        except Exception:
                            pass
            except Exception:
                pass

            # Run the pipeline (round_start/complete & evaluation events are emitted from demo_2)
            await self.demo.run(pre_eval_cycles=num_rounds_before_eval, post_eval_cycles=num_rounds_after_eval)

            # Final summary/stats
            rounds = self.demo.simulation_results.get('rounds', [])
            final_stats = {
                'total_rounds': len(rounds),
                'total_posts': sum(len(r.get('posts', [])) for r in rounds),
                'total_users': len(self.demo.users_data or []),
                'total_actions': sum(int(r.get('total_actions', 0)) for r in rounds)
            }
            socketio.emit('simulation_complete', {
                'final_stats': final_stats,
                'summary': self.demo.simulation_results,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Simulation error: {str(e)}")
            socketio.emit('simulation_error', {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        finally:
            self.is_running = False

# Global demo instance
demo_instance = RealTimeDemo()

@app.route('/')
def index():
    """Main demo page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current demo status"""
    return jsonify({
        'is_running': demo_instance.is_running,
        'current_round': demo_instance.current_round,
        'stats': demo_instance.simulation_stats
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status_update', {
        'type': 'connected',
        'message': 'Connected to Agent1001 Demo',
        'timestamp': datetime.now().isoformat()
    })
    
    # Auto-initialize demo
    success = demo_instance.initialize_demo({})
    if success:
        emit('initialization_result', {
            'success': True,
            'message': 'Demo ready to start',
            'timestamp': datetime.now().isoformat()
        })
    else:
        emit('initialization_result', {
            'success': False,
            'message': 'Demo initialization failed',
            'timestamp': datetime.now().isoformat()
        })

@socketio.on('initialize_demo')
def handle_initialize(data):
    """Initialize demo with configuration"""
    config = data.get('config', {})
    success = demo_instance.initialize_demo(config)
    
    emit('initialization_result', {
        'success': success,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('start_simulation')
def handle_start_simulation(data):
    """Start simulation with parameters"""
    if demo_instance.is_running:
        emit('error', {'message': 'Simulation already running'})
        return
    
    parameters = data.get('parameters', {})
    
    # Run simulation in background thread
    def run_async():
        # Ensure Windows uses Selector policy (consistent with demo_2.py)
        if sys.platform.startswith("win"):
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception:
                pass
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(demo_instance.run_simulation(parameters))
        loop.close()
    
    thread = Thread(target=run_async)
    thread.daemon = True
    thread.start()

@socketio.on('stop_simulation')
def handle_stop_simulation():
    """Stop running simulation"""
    demo_instance.is_running = False
    emit('simulation_stopped', {
        'message': 'Simulation stopped by user',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Create static and template directories if they don't exist
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True) 
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting Agent1001 Real-time Demo Server...")
    print("📡 Server will be available at: http://localhost:5000")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
