import time
from collections import deque
import numpy as np

class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.stats = {
            'detection': deque(maxlen=window_size),
            'extraction': deque(maxlen=window_size),
            'classification': deque(maxlen=window_size),
            'total_frame': deque(maxlen=window_size)
        }
        self.start_times = {}

    def start(self, stage):
        self.start_times[stage] = time.time()

    def stop(self, stage):
        if stage in self.start_times:
            duration = (time.time() - self.start_times[stage]) * 1000 # in ms
            self.stats[stage].append(duration)
            return duration
        return 0

    def get_report(self):
        report = {}
        for stage, times in self.stats.items():
            if times:
                report[stage] = {
                    'avg': np.mean(times),
                    'p95': np.percentile(times, 95) if len(times) >= 20 else np.mean(times)
                }
        
        # Calculate FPS
        if self.stats['total_frame']:
            avg_frame_time = np.mean(self.stats['total_frame'])
            report['fps'] = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
            
        return report

    def log_report(self):
        report = self.get_report()
        print("\n--- Performance Report ---")
        if 'fps' in report:
            print(f"FPS: {report['fps']:.2f}")
        for stage, metrics in report.items():
            if stage == 'fps': continue
            print(f"{stage.capitalize()}: Avg {metrics['avg']:.2f}ms | P95 {metrics['p95']:.2f}ms")
