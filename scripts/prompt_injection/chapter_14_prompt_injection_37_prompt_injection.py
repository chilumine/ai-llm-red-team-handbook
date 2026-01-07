#!/usr/bin/env python3
"""
2. Anomaly Detection Dashboard

Source: Chapter_14_Prompt_Injection
Category: prompt_injection
"""

from flask import Flask, render_template, jsonify
import threading

import argparse
import sys

# real_time_dashboard.py

app = Flask(__name__)

class RealTimeMonitor:
    def __init__(self):
        self.active_sessions = {}
        self.recent_alerts = []

    def monitor_stream(self):
        """Monitor LLM interactions in real-time"""
        while True:
            event = self.get_next_event()

            if event.type == 'new_query':
                self.check_for_injection(event)

            elif event.type == 'unusual_response':
                self.flag_anomaly(event)

    def check_for_injection(self, event):
        score = self.calculate_injection_likelihood(event.user_input)

        if score > 0.8:
            self.recent_alerts.append({
                'severity': 'HIGH',
                'user_id': event.user_id,
                'message': 'Likely injection attempt',
                'input': event.user_input[:100]
            })

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/alerts')
def get_alerts():
    return jsonify(monitor.recent_alerts)

monitor = RealTimeMonitor()

# Start monitoring in background
threading.Thread(target=monitor.monitor_stream, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=True)
