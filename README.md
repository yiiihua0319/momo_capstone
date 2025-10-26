# Green Packaging Optimization for E-commerce

---

## Project Background

As Taiwan’s e-commerce penetration continues to grow, retailers such as **momo** face increasing packaging complexity across diverse product categories.  
This leads to:
- Inconsistent packaging sizes;
- Low box utilization and material waste;
- Rising ESG and green logistics requirements.

This project aims to **recommend optimal boxes and bags for each order** using data analytics and machine learning, in order to:
- **Increase box utilization rate,**
- **Reduce packaging waste,**
- **Support sustainable logistics.**

---

## Dataset Overview

| Dataset | Description | Records |
|----------|--------------|----------|
| `order.csv` | Order-level and item-level data (size, weight, fragility, delivery type, box used, API adoption) | 89,900 |
| `package.csv` | All available boxes and bags with internal/external dimensions and channel applicability | 480 |

**Data leakage prevention:**  
A custom mechanism was implemented to detect and remove duplicate order-item combinations across train/test splits to ensure model generalization.

---

## Methodology Overview

Pipeline architecture:

```

Single/Multi-box Classification
→ Multi-box Grouping
→ Bag vs Box Classification
→ Bag Recommendation
→ Box Recommendation
→ Reinforcement Learning Simulation

````

---

### Single vs. Multi-box Classification

**Goal:** Identify whether an order requires multiple boxes.

- **Rule-based pre-filtering**
  - Single box if only one item.
  - Multi-box if volume > max box or weight > channel limit.
  - Otherwise → pass to ML model.

- **Machine Learning Model**
  - Model: `RandomForestClassifier`
  - Features: total volume, weight, number of SKUs, density, product category, fragility ratio.
  - Handling imbalance: SMOTE, oversampling, `class_weight={0:1,1:5}`.
  - Accuracy (Single-box): **99.9%**  
  - Limitation: low recall for multi-box (≈0%), improved with hybrid rule + model workflow.

---

### Multi-box Grouping Logic

**Goal:** Automatically assign items into multiple boxes to minimize cost and maximize utilization.

**Process:**
1. Load channel-specific constraints (volume, weight, size limits)
2. Estimate minimum number of boxes
3. Sort items by descending volume
4. Initialize empty box structures
5. Assign items based on a scoring function:

```python
Score = (ShippingCost * 100)
      + SameSKU_Bonus
      + VolumeBalancePenalty
      + WeightBalancePenalty
      + FragileMixPenalty
````

**Result:**
53.85% box matching accuracy (mostly differing by item arrangement, not box count).
**Next step:** simulate “largest box first” policy to better match human packing behavior.

---

### Bag vs. Box Classification

**Goal:** Predict which orders can be shipped using reusable bags.

* Bag orders only ~1.2% → highly imbalanced dataset.
* Model: **Balanced Random Forest**
* Feature categories:

  * Volume-related (mean, max, std)
  * Weight-related (mean, max, density)
  * Fragile item ratio
  * Product category & delivery type (one-hot)
* Evaluation:

  * At recall ≥ 0.9, best precision = **0.047**
  * Reduced candidate orders for bag evaluation from **100% → 24%**

---

### Bag Recommendation via 3D-bin-packing

**Tool:** Modified open-source [`3D-bin-packing`](https://github.com/jerry800416/3D-bin-packing) library.

* Filters invalid bags using rule-based size thresholds:

  * Volume margin
  * Dimension margin
  * Channel limitations
  * Scaled PO-series bag dimensions
* Parameters tuned to balance **overestimation** (too small bag) vs **underestimation** (too large bag).

**Results:**

| Model Type           | Overall Accuracy | Box Orders | Bag Orders (lenient) |
| -------------------- | ---------------- | ---------- | -------------------- |
| High-accuracy config | **82.0%**        | 88.2%      | 26.3%                |
| Balanced config      | **65.1%**        | 66.5%      | **52.6%**            |

---

### Box Recommendation Models

Three approaches were compared:

| Method            | Accuracy | Utilization | Notes                      |
| ----------------- | -------- | ----------- | -------------------------- |
| 3D-bin-packing    | 48.75%   | 59.43%      | Physically realistic       |
| Random Forest     | 53.21%   | 55.38%      | Stable and explainable     |
| Gradient Boosting | 52.30%   | **62.86%**  | Highest packing efficiency |

---

### Reinforcement Learning Simulation (PyBullet)

**Objective:** Learn optimal 3D placement and packing policies through physical simulation.

* **Environment:** OpenAI Gym + PyBullet physics engine
* **Agent:** Actor-Critic (PPO)
* **Actions:**

  * Position (x, y, z)
  * Rotation (pitch, roll, yaw)
  * Item selection
  * Buffer/fold decisions
* **Challenges:**

  * Reward credit assignment (delayed feedback)
  * Linking observation and action in high-dimensional state space
* **Improvement directions:**

  * Simplify environment (single-box training)
  * Add height-map CNN & sequential LSTM encoding for spatial awareness

---

## Key Results

| Metric                         | Before   | After           |
| ------------------------------ | -------- | --------------- |
| Average box utilization        | ~60%     | **71%**         |
| Avg. number of candidate boxes | 25       | **9.89 (↓60%)** |
| Bag prediction workload        | 100%     | **24%**         |
| Processing time per order      | <0.2 sec | —               |

---

## Tech Stack

* **Languages & Tools:** Python, pandas, scikit-learn, PyBullet, 3D-bin-packing
* **ML Models:** RandomForest, Balanced RF, GradientBoosting, PPO (RL)
* **Frameworks:** OpenAI Gym, RLlib
* **Version Control:** GitHub
* **Repository:** [github.com/jerry800416/3D-bin-packing](https://github.com/jerry800416/3D-bin-packing)

---

## Future Directions

1. **Reinforcement Learning Refinement**

   * Start from single-box convergence before generalization.
   * Implement reward decomposition to improve policy learning.

2. **Algorithmic Optimization**

   * Integrate Extreme Points algorithm into RL for global optimization.
   * Improve physical simulation efficiency via distributed training.

3. **Practical Applications**

   * Embed model outputs in WMS systems for real-time packing suggestions.
   * Introduce ESG metrics (carbon & material savings) into the cost function.

---

## Conclusion

This project integrates **rule-based logic, machine learning, and reinforcement learning** into a unified packaging recommendation framework.
Our system improves average box utilization by over 10%, significantly reduces packaging waste, and demonstrates the potential for **data-driven, sustainable logistics** in large-scale e-commerce operations.

