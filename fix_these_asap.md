Code verification / bug-fix list for your agent

Fix before re-running anything (data integrity):

Dataset size reconciliation. pca_report.json splits sum to 1,229 samples (860/184/185); the paper states 
𝑛
n=614. Check whether feature extraction ran on an augmented/duplicated set, or a stale intermediate cache — re-verify the actual Mendeley split sizes at the point PCA is fit.
Regime B parameter counts disagree across two logs (29/55/89 in one table vs. 25/49/81 in another, for the same 
𝑞
q). Off-by-
𝑞
q pattern suggests a bias-term double-count in one logging path. Confirm the true count from the model definition, not the logs.
Per-client AUC in the QFL round-by-round CSV is identical across Tier1/Tier2/Tier3 in every round — almost certainly logging the global/broadcast AUC three times instead of true per-client local evaluation. Re-instrument to log each client's local validation AUC before aggregation.
Classical FedAvg control collapses to AUC 0.5 / F1 0.0. Confirm this isn't a training bug (e.g., frozen weights, wrong loss, label mismatch) before reporting it as a real finding — a total collapse to chance is unusual enough to warrant a sanity check, not just a citation.

Re-run / extend (needed for the claims now in the draft):
5. QFL on the sweep-optimal architecture. The federated pipeline used 
𝑞
q=8,
ℓ
ℓ=2 (pre-sweep default); re-run FedAvg + DP sweep on 
𝑞
q=4,
ℓ
ℓ=3 (Regime A) and 
𝑞
q=6,
ℓ
ℓ=2 (Regime B) to see if the federated gap holds at the better-performing configs.
6. **Extend the DP 
𝜎
σ sweep well past 0.20** — current results show no real degradation in-range, so the "practical operating point" claim isn't yet supported by data; push until AUC visibly drops.
7. **Run external (KAU) validation on the Regime B checkpoints and the 
𝑞
q=4,
ℓ
ℓ=3 Regime A checkpoint** — currently only 
𝑞
q=8,
ℓ
ℓ=2 was validated externally, so the paper can't yet say whether the best models generalize any better or worse.
8. Add the parameter-matched classical MLP control (a ~9–30 parameter dense head over the same PCA-4 features) — needed to isolate whether the quantum circuit itself adds value over "any small trainable head," not just small parameter count.
9. Compute PR-AUC/average precision on the primary Mendeley test set, not just on KAU — currently AP is only reported externally, so precision-recall behavior on the source population is unstated.

Double-check / clarify (small but reviewer-visible):
10. Temperature scaling sign convention: 
𝑇
𝑜
𝑝
𝑡
𝑖
𝑚
𝑎
𝑙
=
0.186
<
1
T
optimal
	​

=0.186<1, but the calibration code comment says "
𝑇
>
1
T>1 means overconfident." Check whether the formula is 
𝑝
𝑇
=
𝜎
(
ℓ
/
𝑇
)
p
T
	​

=σ(ℓ/T) or 
𝜎
(
ℓ
⋅
𝑇
)
σ(ℓ⋅T), and whether logits or something else are being scaled — the reported direction is currently inconsistent with the stated rationale.
11. Confirm the KAU-BCMD case count used for external validation is 1,416 (not the 2,378 image-level figure that appeared in an earlier log) — make sure the eval script is pulling from the corrected count.
12. PCA linear-probe AUCs: methods text should read 0.983/0.986/0.986 for 
𝑛
n=4/6/8 (not the earlier 0.9927/0.9909/0.9942) — just confirm which run is authoritative before locking the number in.