# Part 4: One-Step Generation — Analysis

## 4.1 Sampling Efficiency

We evaluate our best model from Part 2 (x-pred + x-loss, D=32) across Euler step counts of 1, 2, 5, 10, 20, 50, 100, and 200 on all three datasets.

**Observations:**
- **1 step**: Complete failure. Samples collapse to a small blob near the origin with no recognizable structure.
- **2 steps**: Still largely unstructured, though the spatial range begins to expand.
- **5 steps**: Faint structure starts to emerge (rough circular shape for swiss_roll), but samples are very noisy.
- **10 steps**: Recognizable structure appears, but with significant scatter and missing detail.
- **20 steps**: Good quality. The spiral/modes/rings are clearly visible, though slightly noisier than 50 steps.
- **50 steps** (baseline): Clean, well-defined structure matching the ground truth.
- **100–200 steps**: Marginal improvement over 50 steps; diminishing returns.

**Conclusion:** Standard flow matching with Euler ODE sampling requires approximately 20–50 steps for acceptable quality. Below 10 steps, quality degrades severely, and 1-step generation is impossible with standard Euler integration. This motivates MeanFlow for single-step generation.

## 4.2 MeanFlow Implementation

We implement MeanFlow with the following setup:
- **Prediction type**: x-prediction (best from Part 2), implemented via velocity output with x-loss conversion at h=0
- **Architecture**: MeanFlowMLP — same 6-layer MLP as Part 1, extended with a second sinusoidal embedding for horizon h (input dimension D+256)
- **Training**: **400K** steps total. First **40K** steps use pure flow matching (h=0 only) as warmup. After warmup, **50/50** split between flow matching (h=0) and MeanFlow consistency (h>0). **Learning rate** is fixed at **1e-3** during warmup, then **cosine decay** to **1e-5** for the remainder. **Gradient clipping** (max norm 1.0) is applied every step.
- **Horizon curriculum**: `h_max` increases smoothly from **0.05** to **1.0** over the MeanFlow phase (cosine schedule in step). For each MF step, **h** is sampled as **u·h_max** with **u ~ Beta(2,5)** (emphasizing smaller horizons), then clamped so **h ≤ t − ε**.
- **Consistency target**: The stop-gradient target is **V̄(z_{t−h}, t−h, 0)** at the **reached** state, not at (z_t, t). For stability, this target is evaluated with an **EMA** copy of the network (decay 0.999); the JVP side still uses the training (“student”) weights. **Sampling** for the reported figures uses **EMA weights** by default.
- **JVP**: `torch.func.jvp` computes ∂V̄/∂h in forward-mode autodiff.
- **Logs**: Per-1k-step CSV under `outputs/part4/logs/` with separate FM vs MF batch averages.
- **Datasets**: swiss_roll, gaussians, circles at D=32.

## Q1 (2 marks): Why did you choose x-prediction for MeanFlow?

We chose x-prediction because Part 2 conclusively demonstrated that it is the only parameterization that successfully scales to D=32. At this dimension, v-prediction failed completely with the default model capacity (256 hidden units, 25K steps), producing random scatter instead of structured samples.

Since MeanFlow operates at D=32 across all three datasets, using v-prediction as the base would require the same massive capacity increase demonstrated in Part 3 (14x parameters, 56x compute) just to make the base flow matching work — before even addressing the additional complexity of MeanFlow training. x-prediction's ability to leverage the low intrinsic dimensionality of the data makes it the natural and efficient choice.

## Q2 (4 marks): Core idea of MeanFlow

Standard flow matching learns the **instantaneous velocity** v(z, t) = dz/dt at each point in the ODE trajectory. Sampling requires numerically integrating this ODE from t=1 (noise) to t=0 (data) with many small Euler steps, because each step only provides a local linear approximation of the trajectory.

MeanFlow instead learns the **mean velocity** V̄(z, t, h), which represents the average velocity over a finite horizon h. Specifically, V̄(z_t, t, h) = (1/h) ∫₀ʰ v(z_{t-s}, t-s) ds — the average velocity needed to transport z_t from time t to time t-h along the ODE trajectory.

The key difference: when h equals the full remaining time (h = t, transporting from t all the way to t=0), a single step z₀ = z_t - h · V̄(z_t, t, h) lands exactly on the clean data. This enables **one-step generation** because V̄ encodes the entire integrated trajectory, not just the local tangent.

**Training** exploits a consistency equation derived from the integral definition:
V̄(z, t, h) + h · ∂V̄/∂h = v(z_{t-h}, t-h)

This links the mean velocity at horizon h to the instantaneous velocity (h=0 case), providing a self-supervised training signal. The JVP ∂V̄/∂h is computed efficiently via forward-mode autodiff (`torch.func.jvp`), avoiding the need to actually integrate the ODE during training.

## Q3 (3 marks): Why is the h=0 portion needed?

The h=0 training portion is essential because it provides the **anchor** for the entire MeanFlow framework.

At h=0, V̄(z, t, 0) = v(z, t), the instantaneous velocity. This is trained with standard flow matching loss against the known ground truth target (ε − x for v-prediction, or x for x-prediction). Without this, there is no ground truth signal anywhere in the system.

The MeanFlow consistency loss (h>0) is a **self-consistency** condition: it pushes V̄(z,t,h) + h·∂V̄/∂h toward the stop-gradient value of V̄(z,t,0). If V̄(z,t,0) is inaccurate (because h=0 was not trained), the consistency loss would propagate errors to all horizons, leading to training collapse.

In essence, h=0 provides the ground truth foundation, and the consistency condition propagates this knowledge to larger horizons (h>0), enabling multi-step and eventually one-step generation. The two components are complementary: h=0 alone gives standard flow matching (requiring many steps), and h>0 alone has no anchor. Together, they enable few-step and one-step generation.

## Q4 (3 marks): Training cost comparison

MeanFlow is harder to train than standard flow matching for two reasons:

**1. JVP computational overhead.** The MeanFlow consistency loss requires computing ∂V̄/∂h via `torch.func.jvp`. This performs a forward-mode autodiff pass through the network, which has roughly the same cost as one additional forward pass. So each MeanFlow training step costs approximately **2x** a standard forward pass (one for V̄(z,t,h) and one for ∂V̄/∂h, computed jointly by JVP), plus one additional forward pass for the stop-gradient target V̄(z,t,0), totaling **~3x** per MeanFlow step.

**2. Split training budget.** With a 50/50 FM ratio, only half of training steps use standard flow matching. The other half uses the MeanFlow consistency loss. This means the base model (h=0) receives only half the training budget compared to pure flow matching, potentially requiring more total steps to achieve the same base quality.

**Overall cost estimate:** Compared to standard flow matching with the same total step count, MeanFlow requires approximately **1.5–2x** more wall-clock time per step (averaging the cheap FM steps and expensive MF steps). We use **400K** total steps with **40K** warmup, compared to 25K steps for standard flow matching — roughly **16×** more gradient steps before counting MF/JVP/EMA overhead; in practice total wall-clock is on the order of **~20–30×** our Part 2 run, depending on hardware.

## Q5 (7 marks): MeanFlow vs ground truth comparison

We observe several notable differences and artifacts between MeanFlow-generated samples and the ground truth across all three datasets at D=32. **Refer to the current PNGs in `outputs/part4/`** for the exact appearance after the latest training run.

### Overall pattern: Scale collapse at 1-step, gradual recovery with more steps

**1-step generation** typically shows **scale collapse**: samples are compressed toward the origin relative to the ground-truth range, because a single large-horizon step must approximate the full transport with one mean velocity. Exact numeric ranges depend on the run; compare visually to the left-column GT panels.

**2-step generation** usually **partially recovers** spatial extent but often still lacks clean structure — samples may look like diffuse scatter at an intermediate scale.

**5-step generation** often **recovers a more appropriate global range** but can still show dataset-specific artifacts (detailed below).

### Dataset-specific observations

**swiss_roll (5-step):** Samples cover the correct 2D range but appear as uniform scatter with no discernible spiral structure. In contrast, standard flow matching with 5 Euler steps already shows emerging spiral patterns (see Section 4.1). This suggests MeanFlow's mean velocity at h>0 smooths out the trajectory details that would produce curved structure.

**gaussians (5-step):** This dataset shows the most striking artifact. Instead of 8 distinct Gaussian clusters, the MeanFlow 5-step samples form a **ring pattern** — the samples are distributed around a circle of approximately the correct radius but without localized modes. This is a clear case of **mode averaging**: the 8 Gaussian modes are arranged in a circle, and the mean velocity field averages across the directions pointing toward different modes. The resulting mean trajectory points toward the circle on which the modes lie, but cannot discriminate between individual modes.

**circles (5-step):** Similar to swiss_roll — correct range but no recognizable concentric ring structure. The samples form a diffuse blob.

### Why do these artifacts occur?

The **exact** consistency condition is V̄(z,t,h) + h·∂V̄/∂h = v(z_{t−h}, t−h), i.e. the instantaneous velocity at the **reached** point (z_{t−h}, t−h). In our implementation, the stop-gradient target is **V̄(z_{t−h}, t−h, 0)** evaluated at that reached state (using an EMA network for stability), which matches this target pointwise. Remaining errors therefore come from **finite capacity**, **training noise**, and the **inherent difficulty** of learning V̄(z,t,h) for large h: the mean field must still approximate a **nonlocal** mapping (integrated dynamics) with a single network forward pass.

For **large h** (especially near one-step generation), the trajectory from z_t to z_{t−h} can cross high-curvature or **multimodal** regions of the velocity field, so even a correct algebraic form leaves **approximation error** in practice.

For the **gaussians** dataset, multimodality is especially severe: the velocity field must route toward different modes. The **mean** velocity tends to **average** competing directions, which produces **ring-like** or **blended** structure instead of sharp clusters. Standard multi-step Euler flow matching avoids this by taking many local steps; few-step MeanFlow cannot fully replicate that sequential routing.

Additionally, our **model capacity** (256-unit MLP) may be insufficient to represent V̄(z, t, h) as a function of three inputs (z, t, h). The mean velocity for large h encodes an entire integrated trajectory — a significantly more complex function than the instantaneous velocity. A larger model or transformer architecture, as used in the original MeanFlow paper for image generation, would likely perform better.
