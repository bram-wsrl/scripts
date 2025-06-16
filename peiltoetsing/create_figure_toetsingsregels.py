import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


fig = plt.figure(layout="constrained")
gs = GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, :])

# extents
xmin = 0
xmax = 1
ymin = 0
ymax = 1

# apply to axes
for i, ax in enumerate(fig.axes):
    if i < 4:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else: # ax5
        ax.set_yticks([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([5/24, 8/24, 18/24, 21/24])
        ax.set_xticklabels(['15-03', '01-05', '01-10', '15-11'])

# vastpeil
ax1.set_title("Vastpeil")
ax1.hlines(0.5, 0, 1, color="black", linewidth=2, linestyle="--")
ax1.hlines(0.2, 0, 1, color="orange", linewidth=2, linestyle="-")
ax1.hlines(0.8, 0, 1, color="orange", linewidth=2, linestyle="-")
ax1.text(xmin + 0.1, 0.5 + 0.05, 'GPGVAST', fontsize=6, color='black', weight='bold')
ax1.text(xmin + 0.1, 0.2 + 0.05, '- WP_OND_MAR + LOCATIECORRECTIE', fontsize=6, color='black')
ax1.text(xmin + 0.1, 0.8 + 0.05, '+ ZP_BOV_MAR + LOCATIECORRECTIE', fontsize=6, color='black')
# streefpeil
ax2.set_title("Streefpeil")
ax2.hlines(0.5, 0, 1, color="black", linewidth=2, linestyle="--")
ax2.hlines(0.2, 0, 1, color="orange", linewidth=2, linestyle="-")
ax2.hlines(0.8, 0, 1, color="orange", linewidth=2, linestyle="-")
ax2.text(xmin + 0.1, 0.5 + 0.05, 'GPGSTREEF', fontsize=6, color='black', weight='bold')
ax2.text(xmin + 0.1, 0.2 + 0.05, '- WP_OND_MAR + LOCATIECORRECTIE', fontsize=6, color='black')
ax2.text(xmin + 0.1, 0.8 + 0.05, '+ ZP_BOV_MAR + LOCATIECORRECTIE', fontsize=6, color='black')
# natuurlijkpeil
ax3.set_title("Natuurlijk peil")
ax3.hlines(0.4, 0, 1, color="black", linewidth=2, linestyle="--")
ax3.hlines(0.6, 0, 1, color="black", linewidth=2, linestyle="--")
ax3.hlines(0.2, 0, 1, color="orange", linewidth=2, linestyle="-")
ax3.hlines(0.8, 0, 1, color="orange", linewidth=2, linestyle="-")
ax3.text(xmin + 0.1, 0.4 + 0.05, 'GPGMIN', fontsize=6, color='black', weight='bold')
ax3.text(xmin + 0.1, 0.6 + 0.05, 'GPGMAX', fontsize=6, color='black', weight='bold')
ax3.text(xmin + 0.1, 0.2 + 0.05, '- WP_OND_MAR + LOCATIECORRECTIE', fontsize=6, color='black')
ax3.text(xmin + 0.1, 0.8 + 0.05, '+ ZP_BOV_MAR + LOCATIECORRECTIE', fontsize=6, color='black')
# flexpeil
ax4.set_title("Flexibel peil")
ax4.hlines(0.4, 0, 1, color="black", linewidth=2, linestyle="--")
ax4.hlines(0.6, 0, 1, color="black", linewidth=2, linestyle="--")
ax4.hlines(0.2, 0, 1, color="orange", linewidth=2, linestyle="-")
ax4.hlines(0.8, 0, 1, color="orange", linewidth=2, linestyle="-")
ax4.text(xmin + 0.1, 0.4 + 0.05, 'GPGMIN', fontsize=6, color='black', weight='bold')
ax4.text(xmin + 0.1, 0.6 + 0.05, 'GPGMAX', fontsize=6, color='black', weight='bold')
ax4.text(xmin + 0.1, 0.2 + 0.05, '- WP_OND_MAR + LOCATIECORRECTIE', fontsize=6, color='black')
ax4.text(xmin + 0.1, 0.8 + 0.05, '+ ZP_BOV_MAR + LOCATIECORRECTIE', fontsize=6, color='black')
#seizoenspeil
ax5.set_title("Seizoenspeil")
ax5.hlines(0.45, 0, 5/24, color="black", linewidth=2, linestyle="--")
ax5.hlines(0.45, 21/24, 1, color="black", linewidth=2, linestyle="--")
ax5.hlines(0.55, 8/24, 18/24, color="black", linewidth=2, linestyle="--")
ax5.plot([5/24, 8/24], [0.45, 0.55], color="black", linewidth=1, linestyle="dotted")
ax5.plot([18/24, 21/24], [0.55, 0.45], color="black", linewidth=1, linestyle="dotted")
ax5.hlines(0.25, 0, 8/24, color="orange", linewidth=2, linestyle="-")
ax5.hlines(0.25, 18/24, 1, color="orange", linewidth=2, linestyle="-")
ax5.hlines(0.35, 8/24, 18/24, color="orange", linewidth=2, linestyle="-")
ax5.hlines(0.65, 0, 5/24, color="orange", linewidth=2, linestyle="-")
ax5.hlines(0.65, 21/24, 1, color="orange", linewidth=2, linestyle="-")
ax5.hlines(0.75, 5/24, 21/24, color="orange", linewidth=2, linestyle="-")
ax5.text(xmin + 1/48, 0.45 + 0.05, 'GPGWNTPL', fontsize=6, color='black', weight='bold')
ax5.text(xmin + 17/48, 0.55 + 0.05, 'GPGZMRPL', fontsize=6, color='black', weight='bold')
ax5.text(xmin + 1/48, 0.25 + 0.05, '- WP_OND_MAR + LOCATIECORRECTIE', fontsize=6, color='black')
ax5.text(xmin + 1/48, 0.65 + 0.05, '+ WP_BOV_MAR\n + LOCATIECORRECTIE', fontsize=6, color='black')
ax5.text(xmin + 17/48, 0.35 + 0.05, '- ZP_OND_MAR + LOCATIECORRECTIE', fontsize=6, color='black')
ax5.text(xmin + 17/48, 0.75 + 0.05, '+ ZP_BOV_MAR + LOCATIECORRECTIE', fontsize=6, color='black')

fig.suptitle("Toetsingsregels")
fig.savefig("toetsingsregels.jpeg", dpi=150, bbox_inches='tight')
