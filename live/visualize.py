"""
Trading vizualizatsiyasi — har savdodan keyin yangilanadi
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI yo'q, faqat fayl saqlash
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

LOG_PATH = "live/futures_trades.csv"
IMG_PATH = "live/trading_dashboard.png"

def plot_dashboard():
    path = Path(LOG_PATH)
    if not path.exists() or path.stat().st_size == 0:
        print("Hali savdo yo'q")
        return

    df = pd.read_csv(LOG_PATH, parse_dates=['time'])
    if len(df) == 0:
        return

    df['ret_pct'] = df['ret_pct'].astype(float)
    df['capital'] = df['capital'].astype(float)
    df['cumret']  = (df['capital'] / df['capital'].iloc[0] - 1) * 100
    df['win']     = df['ret_pct'] > 0

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0f1117')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    COLORS = {
        'green' : '#00C853',
        'red'   : '#FF3D00',
        'blue'  : '#7B68EE',
        'gold'  : '#FFD700',
        'gray'  : '#888888',
        'bg'    : '#1a1d27',
    }

    # ── Panel 1: Kapital o'sishi ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(COLORS['bg'])
    ax1.plot(df.index, df['capital'],
             color=COLORS['blue'], linewidth=2, label='Kapital')
    ax1.fill_between(df.index, df['capital'].iloc[0],
                     df['capital'], alpha=0.15, color=COLORS['blue'])
    ax1.axhline(df['capital'].iloc[0], color=COLORS['gray'],
                linestyle='--', linewidth=0.8, alpha=0.5)

    # Win/Loss nuqtalari
    wins   = df[df['win']]
    losses = df[~df['win']]
    ax1.scatter(wins.index,   wins['capital'],
                color=COLORS['green'], s=40, zorder=5, label='Win')
    ax1.scatter(losses.index, losses['capital'],
                color=COLORS['red'],   s=40, zorder=5, label='Loss')

    ax1.set_title('Kapital o\'sishi', color='white', fontsize=13)
    ax1.set_ylabel('USDT', color='white')
    ax1.tick_params(colors='white')
    ax1.spines[:].set_color('#333344')
    ax1.legend(facecolor=COLORS['bg'], labelcolor='white', fontsize=9)

    # ── Panel 2: Har savdo PnL ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(COLORS['bg'])
    colors_bar = [COLORS['green'] if w else COLORS['red'] for w in df['win']]
    ax2.bar(df.index, df['ret_pct'], color=colors_bar, alpha=0.8, width=0.6)
    ax2.axhline(0, color='white', linewidth=0.8)
    ax2.set_title('Har savdo PnL (%)', color='white', fontsize=11)
    ax2.set_ylabel('%', color='white')
    ax2.tick_params(colors='white')
    ax2.spines[:].set_color('#333344')

    # ── Panel 3: Kumulativ return ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(COLORS['bg'])
    color_line = COLORS['green'] if df['cumret'].iloc[-1] >= 0 else COLORS['red']
    ax3.plot(df.index, df['cumret'], color=color_line, linewidth=2)
    ax3.fill_between(df.index, 0, df['cumret'],
                     alpha=0.15, color=color_line)
    ax3.axhline(0, color='white', linewidth=0.8)
    ax3.set_title('Kumulativ Return (%)', color='white', fontsize=11)
    ax3.set_ylabel('%', color='white')
    ax3.tick_params(colors='white')
    ax3.spines[:].set_color('#333344')

    # ── Panel 4: Win/Loss taqsimoti ──
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(COLORS['bg'])
    wins_count   = df['win'].sum()
    losses_count = (~df['win']).sum()
    ax4.pie([wins_count, losses_count],
            labels=[f'Win ({wins_count})', f'Loss ({losses_count})'],
            colors=[COLORS['green'], COLORS['red']],
            autopct='%1.1f%%', startangle=90,
            textprops={'color': 'white'})
    ax4.set_title('Win/Loss taqsimoti', color='white', fontsize=11)

    # ── Panel 5: Statistika ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(COLORS['bg'])
    ax5.axis('off')

    total_ret  = df['cumret'].iloc[-1]
    win_rate   = df['win'].mean() * 100
    avg_win    = df[df['win']]['ret_pct'].mean() if wins_count > 0 else 0
    avg_loss   = df[~df['win']]['ret_pct'].mean() if losses_count > 0 else 0
    best_trade = df['ret_pct'].max()
    worst_trade= df['ret_pct'].min()
    capital_now= df['capital'].iloc[-1]

    stats = [
        ('Jami savdo',     f"{len(df)} ta"),
        ('Win rate',       f"{win_rate:.1f}%"),
        ('Avg win',        f"+{avg_win:.3f}%"),
        ('Avg loss',       f"{avg_loss:.3f}%"),
        ('Eng yaxshi',     f"+{best_trade:.3f}%"),
        ('Eng yomon',      f"{worst_trade:.3f}%"),
        ('Kapital',        f"${capital_now:.2f}"),
        ('Jami return',    f"{total_ret:+.2f}%"),
    ]

    y = 0.95
    for label, value in stats:
        color = COLORS['gold'] if 'return' in label.lower() or 'Kapital' in label else 'white'
        ax5.text(0.05, y, label + ':', color=COLORS['gray'],
                 fontsize=11, transform=ax5.transAxes)
        ax5.text(0.55, y, value, color=color,
                 fontsize=11, fontweight='bold', transform=ax5.transAxes)
        y -= 0.12

    ax5.set_title('Statistika', color='white', fontsize=11)

    # Sarlavha
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f'BTC/USDT Trading Dashboard — {now}',
                 color='white', fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(IMG_PATH, dpi=130, bbox_inches='tight',
                facecolor='#0f1117', edgecolor='none')
    plt.close()
    print(f"Dashboard saqlandi: {IMG_PATH}")


if __name__ == '__main__':
    plot_dashboard()
