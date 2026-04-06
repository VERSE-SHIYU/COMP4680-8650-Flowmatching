# Part 3: Can We Rescue v-Prediction?

## Experiment Setup

We investigate whether v-prediction's failure at D=32 (observed in Part 2) can be overcome by increasing model capacity and training duration. All experiments use the swiss_roll dataset at D=32 with v-pred + v-loss, compared against x-pred + x-loss as the baseline.

| Configuration | Width | Parameters | Training Steps | Total Compute (relative) |
|---|---|---|---|---|
| x-pred baseline | 256 | 312K | 25K | 1x |
| v-pred (default) | 256 | 312K | 25K | 1x |
| v-pred (more steps) | 256 | 312K | 100K | 4x |
| v-pred (wider) | 512 | 1.15M | 50K | 7.4x |
| v-pred (wider) | 1024 | 4.4M | 50K | 28x |
| v-pred (rescue) | 1024 | 4.4M | 100K | 56x |

We also run x-pred + x-loss at each configuration for comparison.

## Q1 (2 marks): Is v-prediction's failure fundamental or can it be overcome?

v-prediction's failure at high D is **not fundamental** — it can be overcome by increasing model capacity and training duration. Our experiments demonstrate a clear progression:

- **256w / 25K** (default): Complete failure. Generated samples are uniform scatter with no discernible structure.
- **256w / 100K** (4x steps): Partial improvement. A faint spiral shape begins to emerge, but the result is still far from acceptable.
- **512w / 50K** (wider model): Clear improvement. The double-spiral structure is visible, though with significant noise and scatter.
- **1024w / 50K** (wider still): Good quality. The spiral is clearly recognizable with moderate scatter.
- **1024w / 100K** (widest + most steps): Near-successful rescue. The double-spiral structure is clearly preserved, achieving comparable (though slightly inferior) quality to x-pred 256w/25K.

These findings **support and extend** our observations from Part 2. Part 2 showed that v-prediction fails because the prediction target v = ε − x has full rank D, requiring the network to learn D independent components. Part 3 confirms this is a capacity issue, not a fundamental limitation: a sufficiently large network with enough training can learn the full-rank target.

## Q2 (3 marks): What approaches did you try? Compare compute cost.

We tried two complementary approaches: **(1) increasing model width** (from 256 to 512 to 1024 hidden units) and **(2) increasing training steps** (from 25K to 100K).

**Key finding: Model width is the dominant factor.** Increasing steps alone (256w/25K → 256w/100K) produced only marginal improvement — the spiral barely emerged. In contrast, increasing width (256w → 1024w at 50K steps) produced dramatic improvement. The combination of both (1024w/100K) achieved the best result.

**Compute cost comparison to achieve comparable quality at D=32:**

| | x-pred + x-loss | v-pred + v-loss (rescue) |
|---|---|---|
| Width | 256 | 1024 |
| Parameters | 312K | 4.4M (**14x**) |
| Training steps | 25K | 100K (**4x**) |
| FLOPs per step (proportional to params) | 1x | ~14x |
| **Total compute** | **1x** | **~56x** |

v-prediction required approximately **56 times more total compute** to achieve comparable (but still slightly inferior) sample quality compared to x-prediction at D=32. This massive cost difference directly reflects the rank mismatch identified in Part 2: v-prediction must learn a full-rank D=32 target, while x-prediction only needs to learn the 2-dimensional data manifold.

## Q3 (3 marks): Compare how x-prediction and v-prediction respond to your changes.

x-prediction and v-prediction respond **very differently** to increased capacity and training.

**v-prediction: dramatic improvement.** Going from 256w/25K (complete failure) to 1024w/100K (clear spiral) represents a qualitative transformation. Each increase in width produced visible improvement, and additional training steps provided further refinement. v-prediction is highly sensitive to both model capacity and training budget.

**x-prediction: marginal or no improvement.** x-pred + x-loss already produced excellent results at the smallest configuration (256w/25K). Increasing to 512w or 1024w, or extending training to 100K steps, did not produce any visible quality improvement — the spiral was already clean and tight at the baseline.

**Explanation:** This asymmetry follows directly from the rank of each prediction target:

- **x-target** has low effective rank (≈2, the intrinsic dimensionality of swiss_roll), regardless of ambient D=32. A small 256-wide network already has sufficient capacity to represent this low-rank function. Additional capacity is simply redundant.
- **v-target** has full rank D=32, because ε contributes independently to all 32 dimensions. A 256-wide network cannot adequately represent this high-rank function, but a 1024-wide network can — explaining the dramatic improvement. The additional training steps (25K → 100K) give the larger network time to converge on this more complex target.

This confirms the core insight from RAE: x-prediction's compute requirements are determined by the data's intrinsic dimensionality, while v-prediction's requirements scale with the ambient dimension D.

## Q4 (7 marks): Why does v-prediction work in SD3/FLUX despite failing here?

In practice, v-prediction is used successfully in Stable Diffusion 3 (SD3) and FLUX. Three key differences between those systems and our toy experiments explain why v-prediction succeeds in that setting.

### 1. VAE Latent Space Reduces the Rank Gap (Primary Reason)

The most fundamental difference lies in the **relationship between intrinsic and ambient dimensionality**.

In our toy experiments, the data has intrinsic dimension 2 embedded in ambient dimension D=32. The ratio is just 6.25%, creating a massive rank gap: the x-prediction target occupies a 2-dimensional subspace while the v-prediction target spans all 32 dimensions. As shown in Part 2, this rank mismatch is the root cause of v-prediction's failure — the network wastes capacity predicting noise in 30 dimensions that carry no data signal.

SD3 and FLUX do not operate directly in pixel space. Instead, images are first encoded through a pretrained VAE into a compact latent space. The VAE's training objective (reconstruction + regularization) ensures that the latent representation is **informationally dense**: nearly every dimension of the latent code carries meaningful signal about the image. This means the data's effective dimensionality in latent space is close to the ambient latent dimension.

As analyzed in RAE (Section 3), x-prediction's advantage over v-prediction is proportional to the gap between intrinsic and ambient dimensionality. When this gap is small (as in a well-trained VAE's latent space), the rank difference between x-target and v-target diminishes significantly. The v-prediction target v = ε − x still has rank equal to the latent dimension, but now x also has high effective rank — so the relative overhead of predicting v versus x is much smaller than in our toy setting.

### 2. Massive Model Capacity

Our Part 3 experiments demonstrated that v-prediction at D=32 can be rescued by increasing model width from 256 to 1024 (14x more parameters, from 312K to 4.4M). SD3 and FLUX use DiT (Diffusion Transformer) architectures with **billions of parameters** — orders of magnitude beyond what is needed to handle the full-rank v-prediction target in their latent space. Even if v-prediction requires more capacity than x-prediction, the capacity budget of these production models is so large that the overhead is negligible in practice.

### 3. Massive Training Scale

We showed that increasing training from 25K to 100K steps (4x) contributed to rescuing v-prediction. SD3 and FLUX train for **millions of steps** on datasets of **billions of images**. This enormous training budget allows the model to thoroughly learn the full-rank velocity field, easily compensating for any additional learning difficulty compared to x-prediction.

### Synthesis

These three factors work together. The VAE latent space **reduces the root cause** (the rank gap), while the massive model capacity and training scale **provide abundant resources** to handle whatever gap remains. In our toy experiments, the rank gap was extreme (intrinsic/ambient = 6%) and the resources were minimal (312K parameters, 25K steps) — creating conditions under which v-prediction's inefficiency becomes fatal. In SD3/FLUX, the gap is small and the resources are enormous, so v-prediction works well in practice.

This is consistent with RAE's analysis: x-prediction's efficiency advantage is most pronounced when the intrinsic dimensionality of data is far below the ambient dimension. In well-designed latent diffusion systems, this condition is weakened by construction, making the choice between x-prediction and v-prediction less consequential.
