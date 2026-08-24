"""
5 Publication-Quality Figures for LIFE Paper using Updated Results (results (11))
Nature Medicine / Lancet Digital Health aesthetic
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['figure.dpi'] = 150

BASE11 = 'extracted_data/results11/edge_opt_results'
BASE_PCAM = 'extracted_data/pcam/edge_opt_results'
OUT    = 'extracted_data/pcam/plots_v2'
os.makedirs(OUT, exist_ok=True)

SLATE   = '#2C3E50'
TEAL    = '#16A085'
AMBER   = '#F39C12'
CORAL   = '#E74C3C'
VIOLET  = '#8E44AD'
SILVER  = '#95A5A6'
LGREY   = '#ECF0F1'
DGREY   = '#7F8C8D'

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Horizontal Lollipop (Architecture Comparison)
# ══════════════════════════════════════════════════════════════════════════════
def parse_mean_std(s):
    nums = re.findall(r'[\d.]+', str(s))
    return float(nums[0]), float(nums[1]) if len(nums) > 1 else 0.0

df_cods  = pd.read_csv(f'{BASE_PCAM}/cods_paper_architecture_table.csv')
models, aucs, stds = [], [], []
for _, row in df_cods.iterrows():
    m, s = parse_mean_std(row['Test AUC (mean\xb1sd)'] if 'Test AUC (mean\xb1sd)' in row else row.iloc[2])
    models.append(row['Model'])
    aucs.append(m)
    stds.append(s)

order   = np.argsort(aucs)
models  = [models[i] for i in order]
aucs    = [aucs[i]   for i in order]
stds    = [stds[i]   for i in order]

LABELS = {
    'resnet18':            'ResNet-18',
    'mobilenetv3_small_100': 'MobileNetV3-Small',
    'efficientnet_b0':     'EfficientNet-B0',
    'edgenext_x_small':    'EdgeNeXt-X-Small',
    'mobilevitv2_050':     'MobileViTv2-050',
    'fastvit_t8':          'FastViT-T8',
    'repvit_m0_9':         'RepViT-M0.9',
    'mobilevit_s':         'MobileViT-S (Ours)',
}
fam_color = {
    'resnet18':            SILVER,
    'mobilenetv3_small_100': SILVER,
    'efficientnet_b0':     SILVER,
    'edgenext_x_small':    DGREY,
    'mobilevitv2_050':     DGREY,
    'fastvit_t8':          DGREY,
    'repvit_m0_9':         DGREY,
    'mobilevit_s':         TEAL,
}

fig, ax = plt.subplots(figsize=(7.5, 4.8))
y = np.arange(len(models))

for i, (mod, auc, std) in enumerate(zip(models, aucs, stds)):
    col = fam_color.get(mod, SILVER)
    highlight = mod == 'mobilevit_s'
    lw  = 2.5 if highlight else 1.2
    ms  = 10  if highlight else 7
    ax.hlines(i, 0.85, auc - std, colors=col, linewidth=lw, alpha=0.7)
    ax.errorbar(auc, i, xerr=std, fmt='none', ecolor=col, elinewidth=lw, capsize=3, capthick=lw)
    ax.scatter(auc, i, color=col, s=ms**2, zorder=5, edgecolors='white', linewidth=1.2)
    ax.text(auc + std + 0.002, i, f'{auc:.4f}', va='center', ha='left',
            fontsize=8.5, color=SLATE if highlight else DGREY,
            fontweight='bold' if highlight else 'normal')

ax.set_yticks(y)
ax.set_yticklabels([LABELS.get(m, m) for m in models], fontsize=9.5)
ax.tick_params(axis='y', which='both', length=0)
ax.set_xlabel('Test AUC  (mean ± SD, 3 seeds)', fontsize=10, labelpad=8)
ax.set_xlim(0.855, 0.982)
ax.axvline(aucs[-1], color=TEAL, linestyle='--', linewidth=1.2, alpha=0.35, zorder=0)

p1 = mpatches.Patch(color=SILVER, label='Conventional CNN baseline')
p2 = mpatches.Patch(color=DGREY,  label='Post-2021 mobile/edge')
p3 = mpatches.Patch(color=TEAL,   label='MobileViT-S (selected)')
ax.legend(handles=[p1, p2, p3], fontsize=8.5, frameon=False,
          loc='lower right', handlelength=1.2)

ax.set_title('Fig 1 · Architecture Comparison (MobileViT-S Backbone Selection)',
             fontsize=11, fontweight='bold', pad=10, color=SLATE)
ax.grid(axis='x', linestyle=':', alpha=0.4, color=DGREY)
ax.set_axisbelow(True)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT}/Fig1_Architecture_Lollipop.png', dpi=300, bbox_inches='tight')
plt.close()
print('Fig 1 done')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Slope / Spaghetti chart (Quantization Danger - Updated from Results 11)
# ══════════════════════════════════════════════════════════════════════════════
df_abl = pd.read_csv(f'{BASE11}/mixed_precision_ablation_table.csv')
fp32_auc = 0.9841
fp32_f1  = 0.9517

cands = {
    'STEM_CONV\n(dangerous)': {'auc': 0.8164, 'f1': 0.7204, 'color': CORAL,  'lw': 2.2, 'ls': '-'},
    'SAFE_CONV\n(selected)':  {'auc': 0.9829, 'f1': 0.9481, 'color': TEAL,   'lw': 2.5, 'ls': '-'},
    'SAFE_CONV\n(full calib)':{'auc': 0.9829, 'f1': 0.9481, 'color': TEAL,   'lw': 1.5, 'ls': '--'},
}

fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.4), sharey=False)
metrics   = [('auc', 'Test AUC', 0.75, 1.0), ('f1', 'Test F1', 0.65, 1.0)]
refs      = [fp32_auc, fp32_f1]
ref_lbls  = [f'FP32 ref\n{fp32_auc:.4f}', f'FP32 ref\n{fp32_f1:.4f}']

for ax, (key, ylabel, ylo, yhi), ref, rlbl in zip(axes, metrics, refs, ref_lbls):
    x_left, x_right = 0.0, 1.0
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(ylo, yhi)
    ax.axvline(x_left,  color=DGREY, linewidth=0.8, alpha=0.5)
    ax.axvline(x_right, color=DGREY, linewidth=0.8, alpha=0.5)
    ax.axhline(ref, color=SLATE, linestyle=':', linewidth=1.2, alpha=0.6)

    ax.scatter(x_left, ref, color=SLATE, s=80, zorder=5, edgecolors='white', linewidth=1)
    ax.text(x_left - 0.05, ref, f'{ref:.4f}', va='center', ha='right', fontsize=8, color=SLATE)

    for label, cfg in cands.items():
        val  = cfg[key]
        col  = cfg['color']
        lw   = cfg['lw']
        ls   = cfg['ls']
        ax.plot([x_left, x_right], [ref, val], color=col, linewidth=lw,
                linestyle=ls, alpha=0.85)
        ax.scatter(x_right, val, color=col, s=85, zorder=6, edgecolors='white', linewidth=1)
        ax.text(x_right + 0.05, val, f'{val:.4f}', va='center', ha='left',
                fontsize=8, color=col, fontweight='bold')

    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(['FP32\n(reference)', 'INT8\n(quantized)'], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.tick_params(axis='x', length=0)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_position(('outward', 4))
    ax.grid(axis='y', linestyle=':', alpha=0.35)

p_safe = mpatches.Patch(color=TEAL,  label='SAFE_CONV (selected policy)')
p_stem = mpatches.Patch(color=CORAL, label='STEM_CONV (dangerous — early stem collapse)')
fig.legend(handles=[p_safe, p_stem], fontsize=8.5, frameon=False,
           loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03))

fig.suptitle('Fig 2 · Quantization Policy: AUC and F1 Protection via SAFE_CONV',
             fontsize=11, fontweight='bold', color=SLATE, y=1.10)
plt.tight_layout()
plt.savefig(f'{OUT}/Fig2_Slope_Quantization.png', dpi=300, bbox_inches='tight')
plt.close()
print('Fig 2 done')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Grouped Dot + CI Forest Plot (Clinical Metrics from Results 11)
# ══════════════════════════════════════════════════════════════════════════════
df_forest = pd.read_csv(f'{BASE11}/forest_plot_ci_aptos.csv')

# Key variants:
# ONNX FP32 (test), ONNX FP16 (test), ONNX INT8 Conv-only (test), ONNX INT8 Conv+MatMul (test)
metrics_data = {
    'AUC': {
        'FP32':            (0.9841, 0.9759, 0.9908),
        'FP16':            (0.9841, 0.9758, 0.9907),
        'INT8 Conv-only':  (0.9829, 0.9744, 0.9902),
        'INT8 Conv+MatMul':(0.9832, 0.9748, 0.9903),
    },
    'F1 Score': {
        'FP32':            (0.9517, None, None),
        'FP16':            (0.9531, None, None),
        'INT8 Conv-only':  (0.9481, None, None),
        'INT8 Conv+MatMul':(0.9509, None, None),
    },
    'Sensitivity': {
        'FP32':            (0.9543, None, None),
        'FP16':            (0.9570, None, None),
        'INT8 Conv-only':  (0.9570, None, None),
        'INT8 Conv+MatMul':(0.9624, None, None),
    },
    'Specificity': {
        'FP32':            (0.9474, None, None),
        'FP16':            (0.9474, None, None),
        'INT8 Conv-only':  (0.9363, None, None),
        'INT8 Conv+MatMul':(0.9363, None, None),
    },
}

variant_styles = {
    'FP32':            (SLATE,  'o',  80,  'FP32 (reference)'),
    'FP16':            (AMBER,  's',  70,  'FP16'),
    'INT8 Conv-only':  (VIOLET, '^',  65,  'INT8 (Conv-only)'),
    'INT8 Conv+MatMul':(TEAL,   'D',  70,  'INT8 (Conv+MatMul)'),
}

fig, ax = plt.subplots(figsize=(7.5, 4.4))

metric_names = list(metrics_data.keys())
n_metrics    = len(metric_names)
n_variants   = len(variant_styles)
offsets      = np.linspace(-0.28, 0.28, n_variants)

for mi, mname in enumerate(metric_names):
    for vi, (vname, (col, mk, ms, vlbl)) in enumerate(variant_styles.items()):
        val, ci_lo, ci_hi = metrics_data[mname][vname]
        y_pos = mi + offsets[vi]
        ax.scatter(val, y_pos, color=col, marker=mk, s=ms, zorder=5,
                   edgecolors='white', linewidth=0.8)
        if ci_lo is not None:
            ax.hlines(y_pos, ci_lo, ci_hi, colors=col, linewidth=2.0, alpha=0.6)
            ax.vlines([ci_lo, ci_hi], y_pos-0.04, y_pos+0.04, colors=col,
                      linewidth=1.5, alpha=0.6)

for i in range(1, n_metrics):
    ax.axhline(i - 0.5, color=LGREY, linewidth=1.0, zorder=0)

ax.set_yticks(range(n_metrics))
ax.set_yticklabels(metric_names, fontsize=10)
ax.tick_params(axis='y', length=0)
ax.set_xlabel('Metric Score', fontsize=10, labelpad=8)
ax.set_xlim(0.910, 1.000)
ax.spines['left'].set_visible(False)
ax.grid(axis='x', linestyle=':', alpha=0.4, color=DGREY)
ax.set_axisbelow(True)

legend_handles = [
    mpatches.Patch(color=col, label=vlbl)
    for _, (col, mk, ms, vlbl) in variant_styles.items()
]
ax.legend(handles=legend_handles, fontsize=8.5, frameon=False,
          loc='lower left', handlelength=1.2)

ax.set_title('Fig 3 · Clinical Diagnostic Metrics Preserved Across Precision Variants',
             fontsize=11, fontweight='bold', pad=10, color=SLATE)
plt.tight_layout()
plt.savefig(f'{OUT}/Fig3_Forest_Clinical.png', dpi=300, bbox_inches='tight')
plt.close()
print('Fig 3 done')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Compression Journey (Updated from Gen4 Results 11)
# ══════════════════════════════════════════════════════════════════════════════
journey = [
    ('FP32\n(baseline)', 19.01, 0.9841, 0.9759, 0.9908, SLATE),
    ('FP16',             10.51, 0.9841, 0.9758, 0.9907, AMBER),
    ('INT8\n(Conv-only)',19.14, 0.9829, 0.9744, 0.9902, VIOLET),
    ('INT8\n(Conv+MatMul)',10.90,0.9832, 0.9748, 0.9903, TEAL),
]
labels_j = [r[0] for r in journey]
sizes_j  = [r[1] for r in journey]
aucs_j   = [r[2] for r in journey]
ci_lo_j  = [r[3] for r in journey]
ci_hi_j  = [r[4] for r in journey]
colors_j = [r[5] for r in journey]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 5.6),
                                      sharex=True, gridspec_kw={'height_ratios':[1,1.6]})
x = np.arange(len(journey))

bars = ax_top.bar(x, sizes_j, color=colors_j, width=0.55, alpha=0.80, edgecolor='white')
for bar, sz in zip(bars, sizes_j):
    ax_top.text(bar.get_x() + bar.get_width()/2, sz + 0.4,
                f'{sz:.2f} MB', ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color=bar.get_facecolor())
ax_top.set_ylabel('Model Size (MB)', fontsize=9.5)
ax_top.set_ylim(0, 24)
ax_top.set_yticks([0, 5, 10, 15, 19.01, 20])
ax_top.axhline(19.01, linestyle=':', color=SLATE, linewidth=1, alpha=0.5)
ax_top.text(3.45, 19.5, 'FP32 baseline', fontsize=7.5, color=SLATE, ha='right')
ax_top.set_title('Fig 4 · Compression Journey: 43% Footprint Reduction with High Diagnostic Fidelity',
                 fontsize=11, fontweight='bold', pad=10, color=SLATE)
ax_top.grid(axis='y', linestyle=':', alpha=0.35)
ax_top.spines['bottom'].set_visible(False)
ax_top.tick_params(axis='x', length=0)

ax_bot.plot(x, aucs_j, color=SLATE, linewidth=2.2, marker='o', markersize=9,
            markeredgecolor='white', markeredgewidth=1.5, zorder=5, label='Test AUC')

ax_bot.fill_between(x, ci_lo_j, ci_hi_j, alpha=0.15, color=SLATE, label='95% CI')
ax_bot.axhline(aucs_j[0], color=SLATE, linestyle='--', linewidth=1.0, alpha=0.45)

for xi, (auc, col) in enumerate(zip(aucs_j, colors_j)):
    ax_bot.scatter(xi, auc, color=col, s=90, zorder=6, edgecolors='white', linewidth=1.5)
    ax_bot.text(xi, auc - 0.0012, f'{auc:.4f}', ha='center', va='top',
                fontsize=8, color=col, fontweight='bold')

ax_bot.set_ylim(0.960, 0.998)
ax_bot.set_yticks([0.965, 0.975, 0.9841, 0.990])
ax_bot.set_ylabel('Test AUC', fontsize=9.5)
ax_bot.set_xticks(x)
ax_bot.set_xticklabels(labels_j, fontsize=9.5)
ax_bot.tick_params(axis='x', length=0)
ax_bot.grid(axis='y', linestyle=':', alpha=0.35)
ax_bot.legend(fontsize=8.5, frameon=False, loc='lower left')

for i in [1, 3]:
    reduction = (1 - sizes_j[i] / sizes_j[0]) * 100
    ax_top.annotate(f'−{reduction:.0f}%',
                    xy=(i, sizes_j[i] + 1.0), ha='center', fontsize=8, color=colors_j[i],
                    fontweight='bold')

plt.tight_layout(h_pad=0.5)
plt.savefig(f'{OUT}/Fig4_Compression_Journey.png', dpi=300, bbox_inches='tight')
plt.close()
print('Fig 4 done')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Seed Robustness with Val vs Test AUC (Updated from Results 11 Metadata)
# ══════════════════════════════════════════════════════════════════════════════
df_seeds = pd.read_csv(f'{BASE_PCAM}/mobilevit_seed_robustness.csv')

fig, ax = plt.subplots(figsize=(7.0, 3.8))

seeds    = df_seeds['Seed'].values
val_aucs = df_seeds['Best_Val_AUC'].values
tst_aucs = df_seeds['Test_AUC'].values
tst_lo   = df_seeds['Test_AUC_CI_low'].values
tst_hi   = df_seeds['Test_AUC_CI_high'].values

x = np.arange(len(seeds))
width = 0.22

ax.bar(x - width/2, val_aucs, width=width*0.9, color=AMBER, alpha=0.65,
       edgecolor='white', label='Best Val AUC')
ax.bar(x + width/2, tst_aucs, width=width*0.9, color=SLATE, alpha=0.80,
       edgecolor='white', label='Test AUC')

ax.errorbar(x + width/2, tst_aucs,
            yerr=[tst_aucs - tst_lo, tst_hi - tst_aucs],
            fmt='none', ecolor=TEAL, elinewidth=2.0, capsize=5, capthick=2.0,
            zorder=6, label='95% CI (test)')

for xi, (va, ta, lo, hi) in enumerate(zip(val_aucs, tst_aucs, tst_lo, tst_hi)):
    ax.text(xi - width/2, va + 0.0008, f'{va:.4f}', ha='center', va='bottom',
            fontsize=7.8, color=AMBER, fontweight='bold')
    ax.text(xi + width/2, hi + 0.001, f'{ta:.4f}', ha='center', va='bottom',
            fontsize=7.8, color=SLATE, fontweight='bold')

mean_tst = tst_aucs.mean()
ax.axhline(mean_tst, linestyle='--', color=TEAL, linewidth=1.5, alpha=0.7,
           label=f'Mean test AUC = {mean_tst:.4f}')

ax.set_xticks(x)
ax.set_xticklabels([f'Seed {s}' for s in seeds], fontsize=10)
ax.tick_params(axis='x', length=0)
ax.set_ylabel('AUC', fontsize=10)
ax.set_ylim(0.920, 0.985)
ax.set_yticks([0.93, 0.94, 0.95, 0.96, 0.97])
ax.spines['left'].set_position(('outward', 4))
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=8.5, frameon=False, loc='lower right', ncol=3)
ax.set_title('Fig 5 · MobileViT-S Seed Robustness: Validation and Test AUC with 95% CI',
             fontsize=11, fontweight='bold', pad=10, color=SLATE)

plt.tight_layout()
plt.savefig(f'{OUT}/Fig5_Seed_Robustness_Bar.png', dpi=300, bbox_inches='tight')
plt.close()
print('Fig 5 done')

print(f'\nAll 5 figures successfully updated in {OUT}/ using results (11) data!')
