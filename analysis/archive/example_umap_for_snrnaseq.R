# ==============================================================================
# make_umaps.R — UMAP panels for B02-04 snRNAseq (Gm1673 / 5xFAD AAV project)
# Follows ../../figure_aesthetics/FIGURE_AESTHETICS.md (theme_pub / Nature style)
# Data: umap_plot_data.csv (extracted from data/adata_annotated.h5ad by
#       extract_data.py — obs metadata + UMAP coords + raw/X log-norm expr
#       for Gm1673, 5xFAD_APP695, 5xFAD_PSEN1, Vipr1, Vipr2)
# ==============================================================================

library(dplyr)
library(ggplot2)
library(ggrepel)
library(scales)
library(RColorBrewer)

# Run from this script's directory (umap_figures/), where umap_plot_data.csv lives.
df <- readr::read_csv("umap_plot_data.csv", show_col_types = FALSE)

# Collapse to a simple binary transduction call.
df <- df %>%
  mutate(transduction_binary = if_else(transduction_status == "Untransduced",
                                        "Untransduced", "Transduced"))

out_dir <- "umap"
dir.create(out_dir, showWarnings = FALSE)

# ── Theme (scRNAseq figure theme, per FIGURE_AESTHETICS.md) ─────────────────
theme_pub <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      text             = element_text(family = "Helvetica"),
      panel.border     = element_rect(color = "black", fill = NA, linewidth = 0.6),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text        = element_blank(),
      axis.ticks       = element_blank(),
      axis.title       = element_text(size = 11, color = "black"),
      plot.title       = element_text(size = 13, hjust = 0.5, face = "plain"),
      legend.text      = element_text(size = 9),
      legend.title     = element_text(size = 10),
      plot.margin      = margin(6, 6, 6, 6)
    )
}

POINT_SIZE  <- 0.25
POINT_ALPHA <- 0.7
LABEL_SIZE  <- 3.0

# ── Cluster label positions: median UMAP coord per cell_type_detailed ───────
# (cell_type_detailed splits "Oligo" into its OPC / NFOL / MOL lineage substates)
# Categories below MIN_LABEL_N are dropped from labeling: a near-singleton
# category's "centroid" is just that 1-2 cells' coordinate, and ggrepel's
# collision-avoidance can drag its text label onto a nearby, unrelated,
# actually-populated cluster -- misleading even though technically correct.
MIN_LABEL_N <- 20
centroids <- df %>%
  group_by(cell_type_detailed) %>%
  summarise(UMAP_1 = median(UMAP_1), UMAP_2 = median(UMAP_2), n = n(), .groups = "drop") %>%
  filter(n >= MIN_LABEL_N)

add_labels <- function(p, size = LABEL_SIZE) {
  p + geom_text_repel(
    data = centroids, aes(x = UMAP_1, y = UMAP_2, label = cell_type_detailed),
    inherit.aes = FALSE, size = size, family = "Helvetica", color = "black",
    fontface = "plain", bg.color = "white", bg.r = 0.12,
    max.overlaps = Inf, seed = 1
  )
}

save_plot <- function(p, name, width = 6.5, height = 5.5) {
  ggsave(file.path(out_dir, paste0(name, ".pdf")), p, width = width, height = height, dpi = 600)
  ggsave(file.path(out_dir, paste0(name, ".png")), p, width = width, height = height, dpi = 300)
}

# Discrete-color legends default to point-sized swatches, which are hard to
# see at POINT_SIZE. Blow up just the legend key glyphs (not the plotted points).
big_legend <- function(size = 5) {
  guides(color = guide_legend(override.aes = list(size = size, alpha = 1)))
}

# ==============================================================================
# 1. Cell type (fine-grained, incl. OPC/NFOL/MOL oligodendrocyte lineage split)
# ==============================================================================
n_ct <- length(unique(df$cell_type_detailed))
ct_pal <- colorRampPalette(brewer.pal(12, "Paired"))(n_ct)
names(ct_pal) <- sort(unique(df$cell_type_detailed))

p1 <- ggplot(df, aes(UMAP_1, UMAP_2, color = cell_type_detailed)) +
  geom_point(size = POINT_SIZE, alpha = POINT_ALPHA, stroke = 0) +
  scale_color_manual(values = ct_pal, guide = "none") +
  labs(title = "Cell Type", x = "UMAP 1", y = "UMAP 2") +
  theme_pub() +
  coord_equal()
p1 <- add_labels(p1)
save_plot(p1, "01_cell_type")

# ==============================================================================
# 2. Transduction status: binary Transduced vs. Untransduced
# ==============================================================================
transd_pal <- c("Transduced" = "#c44e52", "Untransduced" = "grey80")

df_transd <- df %>% arrange(transduction_binary != "Untransduced")  # untransduced first (behind)

p2 <- ggplot(df_transd, aes(UMAP_1, UMAP_2, color = transduction_binary)) +
  geom_point(size = POINT_SIZE, alpha = POINT_ALPHA, stroke = 0) +
  scale_color_manual(values = transd_pal, name = "Transduction\nstatus") +
  labs(title = "Transduction Status", x = "UMAP 1", y = "UMAP 2") +
  theme_pub() +
  theme(legend.position = "right") +
  big_legend() +
  coord_equal()
p2 <- add_labels(p2)
save_plot(p2, "02_transduction_status")

# ── Shared continuous-expression helper ──────────────────────────────────────
# Zero-inflated gene expression: a plain 2-color grey->red gradient washes out
# (low, non-zero values look identical to true-zero grey). Use a punchy 4-stop
# ramp instead so any real signal pops clearly off the zero/background points,
# and boost point size/alpha for the plotted (non-zero) cells.
EXPR_PAL <- c("grey85", "#fed976", "#e31a1c", "#4d0000")

plot_expr <- function(gene_col, title) {
  d <- df %>% arrange(.data[[gene_col]])  # highest expressers plotted last (on top)
  is_zero <- d[[gene_col]] <= 0
  p <- ggplot() +
    geom_point(
      data = d[is_zero, ], aes(UMAP_1, UMAP_2), color = "grey88",
      size = POINT_SIZE, alpha = 0.5, stroke = 0
    ) +
    geom_point(
      data = d[!is_zero, ], aes(UMAP_1, UMAP_2, color = .data[[gene_col]]),
      size = POINT_SIZE * 1.8, alpha = 0.9, stroke = 0
    ) +
    scale_color_gradientn(colors = EXPR_PAL[-1], name = "Log-norm.\nexpression") +
    labs(title = title, x = "UMAP 1", y = "UMAP 2") +
    theme_pub() +
    theme(legend.position = "right") +
    coord_equal()
  add_labels(p)
}

# ==============================================================================
# 3. Gm1673 expression
# ==============================================================================
p3 <- plot_expr("Gm1673", "Gm1673 Expression")
save_plot(p3, "03_gm1673_expression")

# ==============================================================================
# 4. 5xFAD APP/PSEN1 transgene expression (mean of 5xFAD_APP695 + 5xFAD_PSEN1)
# ==============================================================================
p4 <- plot_expr("FAD_APP_PSEN1_score", "5xFAD APP/PSEN1 Transgene Expression")
save_plot(p4, "04_5xfad_app_psen1_expression")

# ==============================================================================
# 5. VPAC1 (Vipr1) expression
# ==============================================================================
p5 <- plot_expr("Vipr1", "VPAC1 (Vipr1) Expression")
save_plot(p5, "05_vpac1_expression")

# ==============================================================================
# 6. VPAC2 (Vipr2) expression
# ==============================================================================
p6 <- plot_expr("Vipr2", "VPAC2 (Vipr2) Expression")
save_plot(p6, "06_vpac2_expression")

# ==============================================================================
# 7. DAM: binary state + continuous score
# ==============================================================================
dam_pal <- c("DAM" = "#c44e52", "Homeostatic" = "#4A8689", "Non-microglia" = "grey85")
df_dam <- df %>%
  mutate(.order = case_when(
    DAM_binary == "Non-microglia" ~ 0,
    DAM_binary == "Homeostatic"   ~ 1,
    DAM_binary == "DAM"           ~ 2
  )) %>%
  arrange(.order)

p7a <- ggplot(df_dam, aes(UMAP_1, UMAP_2, color = DAM_binary)) +
  geom_point(size = POINT_SIZE, alpha = POINT_ALPHA, stroke = 0) +
  scale_color_manual(values = dam_pal, name = "Microglia\nstate") +
  labs(title = "DAM State (Binary)", x = "UMAP 1", y = "UMAP 2") +
  theme_pub() +
  theme(legend.position = "right") +
  big_legend() +
  coord_equal()
p7a <- add_labels(p7a)
save_plot(p7a, "07a_dam_binary")

df_score <- df %>% arrange(dam_score)
p7b <- ggplot(df_score, aes(UMAP_1, UMAP_2, color = dam_score)) +
  geom_point(size = POINT_SIZE, alpha = POINT_ALPHA, stroke = 0) +
  scale_color_gradient2(low = "#08306b", mid = "grey90", high = "#67000d",
                         midpoint = median(df$dam_score), name = "DAM\nscore") +
  labs(title = "DAM Score (Continuous)", x = "UMAP 1", y = "UMAP 2") +
  theme_pub() +
  theme(legend.position = "right") +
  coord_equal()
p7b <- add_labels(p7b)
save_plot(p7b, "07b_dam_score")

# ==============================================================================
# 8. Broad class: 6 general_cell_type categories (same grouping + palette as
# the bar/DEG/volcano "split-by-general" panels elsewhere in this pipeline)
# ==============================================================================
general_pal <- c(
  "Astrocytes"          = "#6B8E6B",  # sage green   (Astr)
  "Excitatory Neurons"  = "#C4735B",  # terracotta   (Exc Neur)
  "Microglia"           = "#4A8689",  # teal         (Immune)
  "Inhibitory Neurons"  = "#5F7EA3",  # slate blue   (Inh Neur)
  "Oligodendrocytes"    = "#BDA04E",  # gold-brown   (Oli)
  "Vascular"            = "#8E6FA5"   # mauve        (Vas)
)
df$general_cell_type <- factor(df$general_cell_type, levels = names(general_pal))

p8 <- ggplot(df, aes(UMAP_1, UMAP_2, color = general_cell_type)) +
  geom_point(size = POINT_SIZE, alpha = POINT_ALPHA, stroke = 0) +
  scale_color_manual(values = general_pal, name = "Class") +
  labs(title = "General Cell Type", x = "UMAP 1", y = "UMAP 2") +
  theme_pub() +
  theme(legend.position = "right") +
  big_legend() +
  coord_equal()
p8 <- add_labels(p8)
save_plot(p8, "08_broad_class")

# ==============================================================================
# 9. Vip expression (mirrors Gm1673 / 5xFAD / Vipr1 / Vipr2 expression panels)
# ==============================================================================
p9 <- plot_expr("Vip", "Vip Expression")
save_plot(p9, "09_vip_expression")

# ==============================================================================
# 10. Experimental condition: WT / Control_AAV / Gm1673_AAV
# (same condition palette as bar/bar10_condition_pct and dam_condition/)
# ==============================================================================
cond_pal <- c("WT" = "grey60", "Control_AAV" = "#4A8689", "Gm1673_AAV" = "#c44e52")
cond_levels <- c("WT", "Control_AAV", "Gm1673_AAV")
df$condition <- factor(df$condition, levels = cond_levels)

df_cond <- df %>% arrange(condition != "WT")  # WT first (behind), AAV groups on top

p10 <- ggplot(df_cond, aes(UMAP_1, UMAP_2, color = condition)) +
  geom_point(size = POINT_SIZE, alpha = POINT_ALPHA, stroke = 0) +
  scale_color_manual(values = cond_pal, name = "Condition") +
  labs(title = "Experimental Condition", x = "UMAP 1", y = "UMAP 2") +
  theme_pub() +
  theme(legend.position = "right") +
  big_legend() +
  coord_equal()
p10 <- add_labels(p10)
save_plot(p10, "10_condition")

cat("Done. Panels written to", normalizePath(out_dir), "\n")
