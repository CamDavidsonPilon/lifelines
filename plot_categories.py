import json
import os
import matplotlib.pyplot as plt

def generate_plot(json_path, output_path, title, is_before=True):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    categories = [item['categoria'].capitalize() for item in data]
    counts = [item['ocorrencias'] for item in data]
    percentages = [item['percentual'] for item in data]
    
    # Modern premium palette
    colors = ['#2B5C8F', '#D95F02', '#7570B3', '#E7298A'] # Steel blue, orange, purple, magenta
    if not is_before:
        colors = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A'] # Green-ish for after
        
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(categories, counts, color=colors, edgecolor='none', width=0.6)
    
    # Grid lines behind bars
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE', linestyle='-', linewidth=1)
    ax.xaxis.grid(False)
    
    # Remove top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Labels and title
    ax.set_title(title, fontsize=14, pad=20, weight='bold', color='#333333')
    ax.set_ylabel('Ocorrências', fontsize=12, labelpad=10, color='#333333')
    ax.tick_params(colors='#333333', labelsize=10)
    
    # Add counts and percentages on top of bars
    for bar, count, pct in zip(bars, counts, percentages):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(counts) * 0.02), 
                f'{count}\n({pct}%)', 
                ha='center', va='bottom', fontsize=9, weight='bold', color='#444444')
                
    # Extra space on top for labels
    ax.set_ylim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Re-generated plot: {output_path}")

def main():
    # Before pylint category plot
    generate_plot(
        'metrics-before-pylint/pylint_distribuicao_categorias_antes.json',
        'metrics-before-pylint/categorias_antes.png',
        'Distribuição das Categorias de Mensagens (Pylint) - Antes da Refatoração',
        is_before=True
    )
    
    # After pylint category plot
    generate_plot(
        'metrics-after-pylint/pylint_distribuicao_categorias_depois.json',
        'metrics-after-pylint/categorias_depois.png',
        'Distribuição das Categorias de Mensagens (Pylint) - Após a Refatoração',
        is_before=False
    )

if __name__ == '__main__':
    main()
