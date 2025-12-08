import numpy as np
import matplotlib.pyplot as plt

def generate_async_matrices(N):
    """
    Generate A and B matrices for N-agent asynchronous consensus.
    Assumes complete graph (everyone sees everyone).
    
    Args:
        N: Number of agents
        
    Returns:
        A: Current state contribution matrix (upper triangular)
        B: Next state contribution matrix (lower triangular)
        degree: Node degree (N-1 for complete graph)
    """
    degree = N - 1
    
    # A matrix: agents see "future" neighbors (j > i) at current time n
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = -degree
        for j in range(i+1, N):
            A[i, j] = 1
    
    # B matrix: agents see "past" neighbors (j < i) at next time n+1
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            B[i, j] = 1
    
    return A, B, degree


def compute_transformation_matrix(A, B, alpha):
    """
    Compute M = (I - αB)^(-1) * (I + αA)
    """
    N = A.shape[0]
    I = np.eye(N)
    
    try:
        M = np.linalg.inv(I - alpha * B) @ (I + alpha * A)
        return M
    except np.linalg.LinAlgError:
        print(f"WARNING: (I - αB) is singular at α={alpha}")
        return None


def analyze_async_consensus(N, alpha, initial_states=None, n_iterations=200):
    """
    Analyze asynchronous consensus for N agents.
    
    Args:
        N: Number of agents
        alpha: Consensus parameter
        initial_states: Initial state values (if None, random)
        n_iterations: Number of iterations to simulate
        
    Returns:
        results dictionary with eigenvalues, drift, history, etc.
    """
    # Generate matrices
    A, B, degree = generate_async_matrices(N)
    
    # Initial states
    if initial_states is None:
        np.random.seed(42)
        initial_states = np.random.uniform(20, 100, N)
    
    X0 = np.array(initial_states)
    true_average = np.mean(X0)
    
    # Compute transformation matrix
    M = compute_transformation_matrix(A, B, alpha)
    
    if M is None:
        return None
    
    # Eigenvalue analysis
    eigenvalues = np.linalg.eig(M)[0]
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]  # Sort descending
    max_eig_magnitude = np.max(np.abs(eigenvalues[1:]))  # Exclude eigenvalue at 1
    
    # Simulate consensus
    X = X0.copy()
    history = [X.copy()]
    
    for _ in range(n_iterations):
        X = M @ X
        history.append(X.copy())
        
        # Check for divergence
        if np.any(np.abs(X) > 1e10):
            break
    
    history = np.array(history)
    
    # Final values and drift
    final_values = history[-1]
    consensus_value = final_values[0]  # All agents should converge to same value
    drift = abs(consensus_value - true_average)
    drift_percent = (drift / true_average) * 100
    
    # Check convergence
    converged = np.allclose(final_values, consensus_value, atol=1e-3)
    diverged = np.any(np.abs(final_values) > 1e6)
    
    return {
        'N': N,
        'alpha': alpha,
        'A': A,
        'B': B,
        'M': M,
        'eigenvalues': eigenvalues,
        'max_eig_magnitude': max_eig_magnitude,
        'initial_states': X0,
        'true_average': true_average,
        'final_values': final_values,
        'consensus_value': consensus_value,
        'drift': drift,
        'drift_percent': drift_percent,
        'history': history,
        'converged': converged,
        'diverged': diverged
    }


def print_results(results):
    """Print analysis results in a nice format."""
    print(f"\n{'='*70}")
    print(f"N = {results['N']} Agents, α = {results['alpha']}")
    print(f"{'='*70}")
    
    print(f"\nInitial states: {results['initial_states']}")
    print(f"True average: {results['true_average']:.3f}")
    
    print(f"\nA matrix (current state contributions):")
    print(results['A'])
    
    print(f"\nB matrix (next state contributions):")
    print(results['B'])
    
    print(f"\nTransformation Matrix M:")
    print(results['M'])
    
    print(f"\nEigenvalues:")
    for i, eig in enumerate(results['eigenvalues']):
        if np.isreal(eig):
            print(f"  λ_{i+1} = {eig.real:.6f}")
        else:
            print(f"  λ_{i+1} = {eig.real:.6f} + {eig.imag:.6f}j  (magnitude: {np.abs(eig):.6f})")
    
    print(f"\nMax eigenvalue magnitude (excluding 1): {results['max_eig_magnitude']:.6f}")
    print(f"Stable? {results['max_eig_magnitude'] < 1.0}")
    
    print(f"\nFinal values: {results['final_values']}")
    print(f"Consensus value: {results['consensus_value']:.3f}")
    print(f"Drift from true average: {results['drift']:.3f} ({results['drift_percent']:.2f}%)")
    
    if results['diverged']:
        print(f"\nStatus: DIVERGED")
    elif results['converged']:
        print(f"\nStatus: CONVERGED")
    else:
        print(f"\nStatus: NOT CONVERGED")


def visualize_consensus(results, save_name=None):
    """Visualize consensus convergence."""
    N = results['N']
    alpha = results['alpha']
    history = results['history']
    true_avg = results['true_average']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: State trajectories
    ax1 = axes[0]
    colors = plt.cm.rainbow(np.linspace(0, 1, N))
    for i in range(N):
        ax1.plot(history[:, i], label=f'Agent {i}', color=colors[i], linewidth=2)
    ax1.axhline(true_avg, color='k', linestyle='--', linewidth=2, label=f'True Avg ({true_avg:.1f})')
    ax1.axhline(results['consensus_value'], color='r', linestyle=':', linewidth=2, 
                label=f'Consensus ({results["consensus_value"]:.1f})')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('State Value', fontsize=12)
    ax1.set_title(f'Asynchronous Consensus\nN={N}, α={alpha}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error convergence
    ax2 = axes[1]
    errors = np.abs(history - true_avg)
    for i in range(N):
        ax2.plot(errors[:, i], color=colors[i], linewidth=2, alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Error from True Average', fontsize=12)
    ax2.set_title(f'Error Convergence\nFinal Drift: {results["drift"]:.3f} ({results["drift_percent"]:.1f}%)', 
                  fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Eigenvalue spectrum
    ax3 = axes[2]
    eigenvalues = results['eigenvalues']
    real_parts = eigenvalues.real
    imag_parts = eigenvalues.imag
    magnitudes = np.abs(eigenvalues)
    
    # Plot eigenvalues in complex plane
    ax3.scatter(real_parts, imag_parts, s=100, c=magnitudes, cmap='viridis', 
                edgecolors='black', linewidths=2, zorder=3)
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax3.plot(np.cos(theta), np.sin(theta), 'r--', linewidth=2, alpha=0.5, label='Unit Circle')
    
    # Annotate eigenvalues
    for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
        ax3.annotate(f'λ{i+1}', (re, im), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax3.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    ax3.set_xlabel('Real Part', fontsize=12)
    ax3.set_ylabel('Imaginary Part', fontsize=12)
    ax3.set_title(f'Eigenvalue Spectrum\nMax |λ| (excl. 1): {results["max_eig_magnitude"]:.4f}', 
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        print(f"[OK] Plot saved to: {save_name}")
    
    return fig


def compare_different_N_values(alphas=[0.3], N_values=[3, 5, 7, 10]):
    """Compare drift for different numbers of agents."""
    print(f"\n{'='*70}")
    print(f"COMPARING DIFFERENT N VALUES")
    print(f"{'='*70}")
    
    results_table = []
    
    for N in N_values:
        for alpha in alphas:
            results = analyze_async_consensus(N, alpha)
            if results:
                results_table.append({
                    'N': N,
                    'alpha': alpha,
                    'drift': results['drift'],
                    'drift_percent': results['drift_percent'],
                    'max_eig': results['max_eig_magnitude'],
                    'stable': results['max_eig_magnitude'] < 1.0
                })
    
    # Print table
    print(f"\n{'N':<5} {'α':<6} {'Drift':<10} {'Drift %':<10} {'Max |λ|':<12} {'Stable?':<10}")
    print("-"*70)
    for r in results_table:
        stable_str = "✓" if r['stable'] else "✗"
        print(f"{r['N']:<5} {r['alpha']:<6.1f} {r['drift']:<10.3f} {r['drift_percent']:<10.2f} {r['max_eig']:<12.6f} {stable_str:<10}")
    
    return results_table


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    # Test 1: 5-agent system with α=0.3
    print("\n" + "="*70)
    print("TEST 1: 5-AGENT PENTAGON FORMATION (α=0.3)")
    print("="*70)
    
    results_5 = analyze_async_consensus(N=5, alpha=0.3)
    print_results(results_5)
    fig1 = visualize_consensus(results_5, save_name='async_5agents_alpha03.png')
    
    # Test 2: Compare 3-agent vs 5-agent
    print("\n" + "="*70)
    print("TEST 2: 3-AGENT vs 5-AGENT COMPARISON")
    print("="*70)
    
    results_3 = analyze_async_consensus(N=3, alpha=0.3, 
                                        initial_states=[100.0, 50.0, 20.0])
    
    print(f"\n3 Agents:")
    print(f"  True average: {results_3['true_average']:.3f}")
    print(f"  Consensus value: {results_3['consensus_value']:.3f}")
    print(f"  Drift: {results_3['drift']:.3f} ({results_3['drift_percent']:.2f}%)")
    
    print(f"\n5 Agents:")
    print(f"  True average: {results_5['true_average']:.3f}")
    print(f"  Consensus value: {results_5['consensus_value']:.3f}")
    print(f"  Drift: {results_5['drift']:.3f} ({results_5['drift_percent']:.2f}%)")
    
    print(f"\nConclusion: More agents → MORE drift due to accumulated phase lag")
    
    # Test 3: Different N values
    comparison_table = compare_different_N_values(
        alphas=[0.3],
        N_values=[3, 5, 7, 10, 15]
    )
    
    # Test 4: Critical alpha for 5 agents
    print("\n" + "="*70)
    print("TEST 3: FINDING CRITICAL α FOR 5 AGENTS")
    print("="*70)
    
    test_alphas = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
    
    print(f"\n{'α':<8} {'Drift':<12} {'Drift %':<12} {'Max |λ|':<12} {'Status':<15}")
    print("-"*70)
    
    for alpha in test_alphas:
        results = analyze_async_consensus(N=5, alpha=alpha)
        if results:
            status = "STABLE" if results['max_eig_magnitude'] < 1.0 else "UNSTABLE"
            if results['diverged']:
                status = "DIVERGED"
            print(f"{alpha:<8.2f} {results['drift']:<12.3f} {results['drift_percent']:<12.2f} "
                  f"{results['max_eig_magnitude']:<12.6f} {status:<15}")
    
    plt.show()