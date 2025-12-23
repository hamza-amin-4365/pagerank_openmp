import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

class PageRankAnalyzer:
    def __init__(self, csv_file):
        """Initialize analyzer with benchmark results CSV"""
        self.df = pd.read_csv(csv_file)
        self.title = Path(csv_file).stem
        
    def plot_speedup(self, filename=None):
        """Generate speedup vs threads graph"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        threads = self.df['Threads']
        speedup = self.df['Speedup']
        
        # Actual speedup
        ax.plot(threads, speedup, 'o-', linewidth=2.5, markersize=8, 
                label='Actual Speedup', color='#2E86AB')
        
        # Ideal speedup (linear)
        ideal_speedup = threads
        ax.plot(threads, ideal_speedup, '--', linewidth=2, 
                label='Ideal Linear Speedup', color='#A23B72', alpha=0.7)
        
        ax.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
        ax.set_title(f'PageRank Speedup Analysis\n{self.title}', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        ax.set_xticks(threads)
        
        # Add value labels on points
        for x, y in zip(threads, speedup):
            ax.annotate(f'{y:.2f}x', xy=(x, y), xytext=(0, 10),
                       textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        else:
            plt.savefig(f'{self.title}_speedup.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_efficiency(self, filename=None):
        """Generate efficiency vs threads graph"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        threads = self.df['Threads']
        efficiency = self.df['Efficiency']
        
        # Actual efficiency
        ax.plot(threads, efficiency, 'o-', linewidth=2.5, markersize=8,
                label='Actual Efficiency', color='#F18F01')
        
        # Ideal efficiency (100%)
        ideal = [100] * len(threads)
        ax.axhline(y=100, color='green', linestyle='--', linewidth=2, 
                   label='Ideal (100%)', alpha=0.7)
        
        # Good threshold (80%)
        ax.axhline(y=80, color='orange', linestyle=':', linewidth=1.5, 
                   alpha=0.5, label='Good Threshold (80%)')
        
        ax.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
        ax.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Parallel Efficiency Analysis\n{self.title}',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        ax.set_xticks(threads)
        ax.set_ylim([0, 110])
        
        # Add value labels
        for x, y in zip(threads, efficiency):
            ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 10),
                       textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        else:
            plt.savefig(f'{self.title}_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_scalability(self, filename=None):
        """Generate strong scaling graph (time vs threads)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        threads = self.df['Threads']
        parallel_time = self.df['ParallelTime']
        
        # Actual execution time
        ax.plot(threads, parallel_time, 'o-', linewidth=2.5, markersize=8,
                label='Parallel Execution Time', color='#06A77D')
        
        ax.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'Strong Scaling: Execution Time\n{self.title}',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        ax.set_xticks(threads)
        
        # Add value labels
        for x, y in zip(threads, parallel_time):
            ax.annotate(f'{y:.2f}s', xy=(x, y), xytext=(0, 10),
                       textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        else:
            plt.savefig(f'{self.title}_scalability.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_combined(self, filename=None):
        """Generate combined analysis with 4 subplots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        threads = self.df['Threads']
        speedup = self.df['Speedup']
        efficiency = self.df['Efficiency']
        parallel_time = self.df['ParallelTime']
        seq_time = self.df['SequentialTime'][0]  # Should be same for all rows
        
        # Subplot 1: Speedup
        axes[0, 0].plot(threads, speedup, 'o-', linewidth=2.5, markersize=8,
                       color='#2E86AB', label='Actual')
        axes[0, 0].plot(threads, threads, '--', linewidth=2, color='#A23B72',
                       alpha=0.7, label='Ideal')
        axes[0, 0].set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Speedup vs Threads', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].set_xticks(threads)
        
        # Subplot 2: Efficiency
        axes[0, 1].plot(threads, efficiency, 'o-', linewidth=2.5, markersize=8,
                       color='#F18F01', label='Actual')
        axes[0, 1].axhline(y=100, color='green', linestyle='--', linewidth=2,
                          alpha=0.7, label='Ideal')
        axes[0, 1].axhline(y=80, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
        axes[0, 1].set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Parallel Efficiency', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].set_xticks(threads)
        axes[0, 1].set_ylim([0, 110])
        
        # Subplot 3: Execution Time
        axes[1, 0].plot(threads, parallel_time, 'o-', linewidth=2.5, markersize=8,
                       color='#06A77D')
        axes[1, 0].axhline(y=seq_time, color='red', linestyle='--', linewidth=2,
                          alpha=0.7, label='Sequential Time')
        axes[1, 0].set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Strong Scaling: Execution Time', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].set_xticks(threads)
        
        # Subplot 4: Speedup per Thread
        speedup_per_thread = speedup / threads
        axes[1, 1].plot(threads, speedup_per_thread, 'o-', linewidth=2.5, markersize=8,
                       color='#D62246')
        axes[1, 1].axhline(y=1.0, color='green', linestyle='--', linewidth=2,
                          alpha=0.7, label='Ideal (1.0)')
        axes[1, 1].set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Speedup / Thread', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Speedup Efficiency per Thread', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].set_xticks(threads)
        
        plt.suptitle(f'PageRank Parallel Analysis - {self.title}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        else:
            plt.savefig(f'{self.title}_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """Print detailed summary statistics"""
        print("\n" + "="*80)
        print("PAGERANK BENCHMARK SUMMARY")
        print("="*80 + "\n")
        
        print(self.df.to_string(index=False))
        
        print("\n" + "-"*80)
        print("STATISTICS:")
        print("-"*80)
        
        # Peak speedup
        max_speedup_idx = self.df['Speedup'].idxmax()
        max_speedup = self.df.loc[max_speedup_idx, 'Speedup']
        max_speedup_threads = self.df.loc[max_speedup_idx, 'Threads']
        print(f"Peak Speedup: {max_speedup:.2f}x at {max_speedup_threads} threads")
        
        # Best efficiency
        max_eff_idx = self.df['Efficiency'].idxmax()
        max_eff = self.df.loc[max_eff_idx, 'Efficiency']
        max_eff_threads = self.df.loc[max_eff_idx, 'Threads']
        print(f"Best Efficiency: {max_eff:.2f}% at {max_eff_threads} threads")
        
        # Scaling quality
        threads_range = self.df['Threads'].max() - self.df['Threads'].min()
        speedup_range = self.df['Speedup'].max() - self.df['Speedup'].min()
        scaling_rate = speedup_range / threads_range if threads_range > 0 else 0
        print(f"Average Scaling Rate: {scaling_rate:.4f} speedup per thread")
        
        # Overhead analysis
        last_row = self.df.iloc[-1]
        overhead = (1 - (last_row['Speedup'] / last_row['Threads'])) * 100
        print(f"Total Parallelization Overhead: {overhead:.2f}%")
        
        print("\n" + "="*80 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_pagerank.py <csv_file> [output_dir]")
        print("\nExample:")
        print("  python3 analyze_pagerank.py pagerank_results.csv")
        print("  python3 analyze_pagerank.py pagerank_results_100000_nodes.csv ./plots")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    if not Path(csv_file).exists():
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Analyze
    analyzer = PageRankAnalyzer(csv_file)
    analyzer.print_summary()
    
    # Generate plots
    base_name = Path(csv_file).stem
    
    print("Generating visualizations...")
    analyzer.plot_speedup(f"{output_dir}/{base_name}_speedup.png")
    analyzer.plot_efficiency(f"{output_dir}/{base_name}_efficiency.png")
    analyzer.plot_scalability(f"{output_dir}/{base_name}_scalability.png")
    analyzer.plot_combined(f"{output_dir}/{base_name}_combined.png")
    
    print("\nAll visualizations generated successfully!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
