# Part 2: Analysis of Prediction Parameterization

## Q1: Which prediction type successfully scales to higher dimensions? At what dimension do the other types start to fail?

**x-prediction (with x-loss)** is the only parameterization that successfully scales to higher dimensions. Across all three datasets (swiss_roll, gaussians, circles), x-pred + x-loss preserves recognizable data structure at D=2, D=8, and D=32. For example, at D=32, the swiss_roll spiral remains clearly visible, and the gaussians still show 8 distinct, tight clusters.

**v-prediction** begins to fail at **D=32**. At D=2, both v-pred + v-loss and v-pred + x-loss produce reasonable results across all datasets. At D=8, results are acceptable for swiss_roll and gaussians but already degrade for circles. By D=32, both v-pred variants completely fail on all datasets — the generated samples collapse into uniform scatter with no discernible structure.

**x-pred + v-loss** is a special case: it fails catastrophically at **all dimensions, including D=2**. Generated samples collapse into narrow, low-dimensional structures (e.g., a J-shaped curve for swiss_roll, a thin line for gaussians) that bear no resemblance to the ground truth. At D=32, the generated coordinates shrink to the order of 1e-8, indicating complete training failure.

## Q2: Does the choice of loss space affect success or failure? What does this tell us?

Yes, the choice of loss space **significantly** affects the results, but its impact depends on the prediction type.

**Evidence:**
- For **x-prediction**: switching from x-loss to v-loss causes complete collapse at every dimension — despite the model learning the same quantity (x), the loss space change alone is sufficient to destroy training. This demonstrates that loss space choice can be critical.
- For **v-prediction**: switching between v-loss and x-loss produces similar results — both succeed at low D and fail at high D. Here, the loss space has minimal impact.

**Explanation:** Both prediction type and loss space matter, but for different reasons:

1. **Prediction type** determines whether the model can scale to high dimensions. This is fundamentally about what the model is learning (see Q3).

2. **Loss space** determines whether the training is numerically stable. The x-pred + v-loss combination fails because converting x-prediction to v-space requires dividing by t: v = (z_t − x̂) / t. This introduces an implicit 1/t² weighting in the MSE loss:

$$\mathcal{L} = \left\|\frac{z_t - \hat{x}}{t} - v_{\text{target}}\right\|^2 = \frac{\|x - \hat{x}\|^2}{t^2}$$

When t → 0, even small prediction errors are amplified by 1/t², causing gradient explosion and training instability.

In contrast, the reverse conversion (v-pred to x-space) is x̂ = z_t − t · v̂, which involves **multiplying** by t. This dampens errors as t → 0, making it numerically safe. This asymmetry explains why v-pred is insensitive to loss space choice while x-pred is highly sensitive.

## Q3: Why does x-prediction succeed at high D while v-prediction fails? (Consider the rank of the prediction target.)

The key lies in the **effective rank** of each prediction target relative to the ambient dimension D.

**x-prediction target is low-rank.** The data x lies on a low-dimensional manifold regardless of the ambient space. For swiss_roll (intrinsic dimension ≈ 1), circles (intrinsic dimension ≈ 1–2), and gaussians (discrete modes), the projection into D=8 or D=32 does not increase the intrinsic complexity — the signal remains concentrated in a low-dimensional subspace, and the remaining dimensions are near-zero. Therefore, the x-prediction target has low effective rank independent of D, and a fixed-capacity network (5 hidden layers × 256 units) can learn it effectively even at D=32.

**v-prediction target is full-rank.** The velocity target v = ε − x, where ε ∼ N(0, I_D), has full rank D because the Gaussian noise ε contributes independently to all D dimensions. As D increases from 2 to 32, the network must accurately predict 32 independent noise components simultaneously. With a fixed model capacity and training budget (25K steps), the network cannot capture this full-rank, high-dimensional target, leading to degraded generation quality.

**Signal-to-noise perspective.** In x-prediction, the meaningful signal (data structure) occupies a small subspace and is easy to extract. In v-prediction, the signal (−x) is drowned out by the noise (ε) across all D dimensions — as D grows, the signal-to-noise ratio of the prediction target decreases, making learning progressively harder. This observation is consistent with the analysis in Rethinking Autoencoder (RAE), which demonstrates that x-prediction naturally exploits the low intrinsic dimensionality of data in high-dimensional ambient spaces.
