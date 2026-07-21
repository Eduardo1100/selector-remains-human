.PHONY: reproduce reproduce-phase-a reproduce-litbench-inventory reproduce-litbench-surface reproduce-litbench-lm reproduce-litbench-embedding reproduce-litbench-reward reproduce-litbench-summary reproduce-litbench-full clean tables figures diff paper-assets paper-check paper-tex paper site publication-bundle publication-check

reproduce: reproduce-phase-a

reproduce-phase-a:
	python pipeline/run_all.py

reproduce-litbench-inventory:
	python pipeline/litbench/00_inventory_litbench.py
	python pipeline/litbench/04_prompt_domain_inventory.py

reproduce-litbench-surface:
	python pipeline/litbench/01_surface_baselines.py

reproduce-litbench-lm:
	python pipeline/litbench/02_nll_baselines.py
	python pipeline/litbench/03_combined_surface_nll_model.py
	python pipeline/litbench/05_prompt_conditioned_v.py
	python pipeline/litbench/06_prompt_v_surface_controls.py
	python pipeline/litbench/07_prompt_v_domain_contrast.py
	python pipeline/litbench/08_prompt_v_random_split_control.py

reproduce-litbench-embedding:
	python pipeline/litbench/09_prompt_domain_kernel_controls.py
	python pipeline/litbench/12_prompt_domain_embedding_operator_controls.py
	python pipeline/litbench/13_train_domain_embedding_probe.py
	python pipeline/litbench/14_cross_prompt_test_domain_embedding_probe.py

reproduce-litbench-reward:
	python pipeline/litbench/10_subset_matched_reward_baseline.py

reproduce-litbench-summary:
	python pipeline/litbench/11_subset_matched_baseline_table.py

reproduce-litbench-full: reproduce-litbench-inventory reproduce-litbench-surface reproduce-litbench-lm reproduce-litbench-embedding reproduce-litbench-reward reproduce-litbench-summary

tables:
	python pipeline/90_make_tables.py

figures:
	python pipeline/91_make_figures.py

paper-assets:
	python scripts/make_primary_figure.py

paper-check:
	python scripts/make_primary_figure.py --check
	python scripts/build_paper.py --check --pdf

paper-tex:
	python scripts/build_paper.py --tex

paper: paper-assets
	python scripts/build_paper.py --pdf

site: paper-assets
	python scripts/build_site.py

publication-bundle: site
	python scripts/check_publication.py --bundle

publication-check: paper-assets site
	python scripts/build_site.py --check
	python scripts/check_publication.py --bundle

diff:
	python pipeline/99_diff_against_scaffold.py

clean:
	find results -type f \( -name "*.csv" -o -name "*.json" -o -name "*.md" -o -name "*.txt" \) -delete
	find paper/tables -type f -delete
	find paper/figures -type f -delete

.PHONY: bootstrap-absolute-full
bootstrap-absolute-full:
	python pipeline/61_bootstrap_absolute_effects.py --n-boot 5000 --seed 123
