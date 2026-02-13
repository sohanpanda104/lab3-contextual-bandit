# Lab 3: Contextual Bandit-Based News Article Recommendation System

## Reinforcement Learning Fundamentals

**Student Name:** Sohan Panda
**Roll Number:** U20230162
**Date:** February 2026

---

## Executive Summary

This project implements a comprehensive Contextual Multi-Armed Bandit system for personalized news article recommendations. The system combines user classification with three distinct bandit algorithms (Epsilon-Greedy, UCB, and SoftMax) to maximize user engagement through intelligent exploration-exploitation strategies.

**Key Findings:**

- **Best Algorithm:** Upper Confidence Bound (UCB) with C=2.0
- **Best Average Reward:** 8.4585
- **User Classification Accuracy:** 90.0%
- **Time Horizon:** 10,000 steps

---

## 1. Introduction

### 1.1 Problem Statement

The assignment required building a recommendation system that:

- Classifies users into contextual categories (User1, User2, User3)
- Maps news articles to arms (4 categories)
- Implements three bandit algorithms to optimize article recommendations
- Maximizes cumulative rewards over 10,000 time steps

### 1.2 Contextual Bandit Framework

A Contextual Bandit extends the Multi-Armed Bandit problem by incorporating context information:

- **Context (State):** User category (3 types)
- **Arms (Actions):** News categories (4 types)
- **Reward:** Engagement signal from the sampler
- **Goal:** Learn policy π(arm | context) that maximizes expected cumulative reward

### 1.3 Arm Mapping

The system uses a flattened 12-arm representation:

```
Arms 0-3:   (Context: User1, Categories: Entertainment, Education, Tech, Crime)
Arms 4-7:   (Context: User2, Categories: Entertainment, Education, Tech, Crime)
Arms 8-11:  (Context: User3, Categories: Entertainment, Education, Tech, Crime)
```

---

## 2. Methodology

### 2.1 Data Preprocessing 

**Dataset Overview:**

- News Articles: 209,527 articles with 6 features (link, headline, category, description, authors, date)
- Train Users: 2,000 users with 33 features and labels (user_1, user_2, user_3)
- Test Users: 2,000 users with 32 features (no labels)

**Preprocessing Steps:**

1. Handled missing values using mean imputation for numeric columns
2. Identified categorical features: `region_code`, `browser_version`
3. Applied Label Encoding to categorical features with unknown value handling
4. Normalized feature distributions across train and test sets

**Data Splits:**

- Training: 1,600 users (80%)
- Validation: 400 users (20%)

### 2.2 User Classification Model 

**Classifier:** Random Forest Classifier

- Number of estimators: 100
- Max depth: 10
- Training samples: 1,600
- Validation samples: 400

**Performance Metrics:**

```
                precision    recall  f1-score   support
        user_1       0.89      0.86      0.87       142
        user_2       0.97      0.89      0.93       142
        user_3       0.84      0.97      0.90       116
    ────────────────────────────────────────────────
    accuracy                           0.90       400
   macro avg       0.90      0.90      0.90       400
weighted avg       0.90      0.90      0.90       400
```

**Key Insight:** The classifier achieves 90% accuracy, ensuring reliable context detection for the recommendation engine.

---

### 2.3 Contextual Bandit Algorithms 

#### 2.3.1 Epsilon-Greedy Strategy 

**Algorithm Description:**

- With probability ε: select a random arm (exploration)
- With probability (1-ε): select the arm with highest Q-value (exploitation)
- Q-values updated using incremental averaging

**Hyperparameter Tuning:**
Tested ε ∈ {0.01, 0.1, 0.3}

**Results:**

| ε   | Avg Reward | Std Dev | Interpretation                      |
| ---- | ---------- | ------- | ----------------------------------- |
| 0.01 | 8.0218     | 1.9591  | High exploitation, good convergence |
| 0.1  | 7.6760     | 3.1889  | Balanced exploration-exploitation   |
| 0.3  | 6.1314     | 4.7849  | High exploration, high variance     |

**Key Finding:** Lower ε values provide better average reward by focusing on exploitation once good arms are identified. However, ε=0.01 still allows sufficient exploration.

#### 2.3.2 Upper Confidence Bound (UCB) 

**Algorithm Description:**

- Selects arm that maximizes: $Q(a) + C \sqrt{\frac{\ln(t)}{N(a)}}$
- First term: estimated reward
- Second term: optimistic bonus for exploration
- Balances exploration automatically based on visit counts

**Hyperparameter Tuning:**
Tested C ∈ {0.5, 1.0, 2.0}

**Results:**

| C   | Avg Reward | Std Dev | Interpretation                        |
| --- | ---------- | ------- | ------------------------------------- |
| 0.5 | 4.1748     | 0.8231  | Too conservative, limited exploration |
| 1.0 | 4.1862     | 0.8279  | Insufficient exploration bonus        |
| 2.0 | 8.4585     | 1.3261  | Optimal balance, best overall         |

**Key Finding:** UCB with C=2.0 provides the highest average reward (8.4585). The aggressive exploration bonus enables faster convergence to optimal arms.

#### 2.3.3 SoftMax Strategy 

**Algorithm Description:**

- Selects arm based on softmax probability distribution
- Probability of arm a: $P(a) = \frac{e^{Q(a)/\tau}}{\sum_{i} e^{Q(i)/\tau}}$
- Temperature τ controls exploration smoothness

**Configuration:**

- Temperature τ = 1.0 (as specified)
- Provides probability-based, smooth exploration

**Results:**

| τ  | Avg Reward | Std Dev |
| --- | ---------- | ------- |
| 1.0 | 7.9382     | 1.7950  |

**Key Finding:** SoftMax provides competitive performance with smooth probability-based exploration, avoiding the abrupt switches of ε-greedy.

---

### 2.4 Recommendation Engine

**Pipeline Implementation:**

1. **Classify User:** Input user features → Random Forest classifier → User context (user_1, user_2, or user_3)
2. **Select Category:** Apply best bandit model (UCB C=2.0) → Extract Q-values → Select arm with max Q-value → Map to news category
3. **Recommend Article:** Query news_articles.csv → Filter by selected category → Random sample → Return article headline

**Example Recommendations:**

```
User 1: Context=user_2, Category=Tech
  → Article: "Watch The Top 9 YouTube Videos Of The Week..."

User 2: Context=user_1, Category=Tech  
  → Article: "This Is The Robot Dallas Police Used To Kill Shooting Suspect..."

User 3: Context=user_1, Category=Tech
  → Article: "Mailbox App Gets 800,000-Person-Long Waiting List..."
```

---

### 2.5 Evaluation & Reporting 

#### 2.5.1 Classification Accuracy

- Validation Accuracy: **90.0%**
- Correctly classifies 9 out of 10 users
- Enables reliable context detection for bandits

#### 2.5.2 RL Simulation Results

- Time Horizon: **T = 10,000 steps**
- Total Arms: **12** (3 contexts × 4 categories)
- Sampler: Initialized with roll number **162**

#### 2.5.3 Performance Comparison

**Summary Statistics:**

```
     Algorithm Hyperparameter  Avg Reward  Std Dev  Final Reward
Epsilon-Greedy         ε=0.01    8.0218   1.9591      9.9762
Epsilon-Greedy          ε=0.1    7.6760   3.1889      9.5790
Epsilon-Greedy          ε=0.3    6.1314   4.7849     10.2141
           UCB          C=0.5    4.1748   0.8231      3.8203
           UCB          C=1.0    4.1862   0.8279      4.3460
           UCB          C=2.0    8.4585   1.3261      8.5287
       SoftMax            τ=1    7.9382   1.7950      7.6242
```

**Winner:** UCB with C=2.0 achieves 8.4585 average reward

#### 2.5.4 Key Visualizations

**Plot 1: Epsilon-Greedy Performance**

- ε=0.01: Smooth convergence to ~8.0 reward
- ε=0.1: More variance, slightly lower average
- ε=0.3: High variance, poorest performance

**Plot 2: UCB Performance**

- C=0.5 & C=1.0: Converge to ~4.2 reward (suboptimal)
- C=2.0: Strong convergence to ~8.45 reward (best)

**Plot 3: Algorithm Comparison**

- Best EG (ε=0.01): 8.02 avg reward, smooth
- UCB (C=2.0): 8.46 avg reward, fastest convergence
- SoftMax (τ=1): 7.94 avg reward, balanced

**Plot 4: Hyperparameter Sensitivity**

- EG variation: 1.89 (moderate sensitivity)
- UCB variation: 4.28 (high sensitivity to C)

---

## 3. Analysis and Insights

### 3.1 Algorithm Comparison

**Epsilon-Greedy:**

- **Pros:** Simple, interpretable, consistent performance
- **Cons:** Fixed exploration rate, inefficient exploration early on
- **Best Use:** When simplicity is important, or for baseline comparison

**Upper Confidence Bound (UCB):**

- **Pros:** Adaptive exploration, theoretically justified, best empirical performance
- **Cons:** Requires tuning of C parameter
- **Best Use:** When optimizing reward is critical; sensitive to hyperparameter

**SoftMax:**

- **Pros:** Smooth exploration, probabilistic decisions
- **Cons:** Moderate performance, temperature affects exploration
- **Best Use:** When smooth probability distribution is preferred

### 3.2 Hyperparameter Sensitivity

**Epsilon-Greedy (ε):**

- ε=0.01: Best (8.02), high exploitation
- ε=0.1: Medium (7.68), balanced
- ε=0.3: Worst (6.13), too much exploration
- **Recommendation:** Lower values preferred; diminishing returns at very low ε

**UCB (C):**

- C=0.5: Poor (4.17), insufficient exploration
- C=1.0: Poor (4.18), insufficient exploration
- C=2.0: Excellent (8.46), optimal balance
- **Recommendation:** Higher C better for 12-arm problem; exploration bonus critical

### 3.3 Convergence Behavior

1. **Epsilon-Greedy:**

   - Rough convergence period: first 2,000 steps
   - Settles to average by step 5,000
   - Stable but with exploration noise
2. **UCB:**

   - Very rapid convergence (except C=0.5, 1.0)
   - C=2.0 converges within first 1,000 steps
   - Higher exploration bonus enables faster learning
3. **SoftMax:**

   - Smooth convergence without sharp changes
   - Takes ~3,000 steps to stabilize
   - Gradual improvement benefits from soft probabilities

### 3.4 Context-Specific Performance

The Q-values learned by each algorithm vary by user context:

- **User1:** Generally lower rewards across categories
- **User2:** Strong preference for specific categories
- **User3:** Mixed performance, highest variance

This suggests that different user contexts genuinely have different optimal strategies, validating the contextual bandit approach.

---

## 4. System Effectiveness

### 4.1 End-to-End Pipeline

The complete recommendation system demonstrates:

1. **Classification:** 90% accuracy in context detection
2. **Decision-Making:** 8.46 average reward per recommendation
3. **Recommendation:** Successfully samples articles from target categories

### 4.2 Scalability Considerations

**Current System:**

- 3 user contexts
- 4 news categories
- 12 total arms
- 10,000 training steps

**Scalability:**

- Linear scaling with number of arms
- Classification model handles new users
- Bandit can be retrained with new data

---

## 5. Conclusions

### 5.1 Key Findings

1. **UCB algorithm with C=2.0 outperforms other strategies** with 8.4585 average reward
2. **Epsilon-Greedy with low ε (0.01) provides competitive performance** at 8.0218
3. **User classification achieves 90% accuracy**, enabling reliable contextualization
4. **Hyperparameter selection is critical**, especially for UCB (C=2.0 vs C=1.0)

### 5.2 Technical Insights

- **Exploration vs Exploitation:** Higher exploration is beneficial in early stages; UCB's adaptive approach outperforms fixed rates
- **Contextual Learning:** Different user contexts exhibit different reward distributions, justifying CMAB
- **Convergence:** UCB converges faster; Epsilon-Greedy provides stability

---

## 6. Implementation Details

### 6.1 Technical Stack

- **Language:** Python 3.12
- **Libraries:**
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib` & `seaborn`
  - `rlcmab-sampler`

### 6.2 Code Structure

**Modules Implemented:**

1. `EpsilonGreedyBandit`: Epsilon-greedy strategy
2. `UCBBandit`: Upper confidence bound strategy
3. `SoftMaxBandit`: SoftMax (Boltzmann) strategy
4. `recommend_article()`: End-to-end recommendation
5. `get_arm_index()`: Context-category to arm mapping

### 6.3 Files Generated

- `master.ipynb`: Main notebook with all results
- `bandit_results.png`: Performance comparison plots
- `q_values_analysis.png`: Learned Q-value visualizations

---

## 7. Submission Materials

- **GitHub Branch:** sohan_U20230162
- **Notebook:** lab3_results_162.ipynb
- **README:** This comprehensive report
- **Plots:** Embedded in notebook

---

**Project Completed:** February 2026
