# Project Description

Building an end-to-end, multi-objective short-video recommender system using KuaiRand, with multi-source retrieval, learning-to-rank, constrained re-ranking, bias-aware evaluation, and online-style serving.

Addressing a feed-ranking problem:

- Sequential interaction histories
- Timestamps and user/item features
- Multiple feedback signals
- Watch-time and engagement outcomes
- Randomly exposed items that support exposure-bias analysis
- Enough complexity to discuss retrieval, ranking, multi-task learning, debiasing, and serving

KuaiRand comes from Kuaishou recommendation logs and includes randomly exposed videos alongside rich interaction and side information. This makes it especially useful for studying exposure bias rather than treating every unseen item as a true negative.

For initial development, use a smaller KuaiRand subset rather than immediately processing the complete dataset. The full KuaiRand-27K release is substantial, so your architecture should support sampling and configurable data scales


## Data Sources
* [Kuairand Website: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos](https://kuairand.com/)
* [Arxiv](https://arxiv.org/abs/2208.08696): KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos
* [Kuairand Github Page (Dataset download)](https://github.com/chongminggao/KuaiRand)

## Project Scoping
```json
                       ┌──────────────────────┐
                       │ Interaction logging  │
                       │ views/watch/like/etc.│
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Point-in-time feature│
                       │ generation           │
                       └──────────┬───────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 │                │                │
                 ▼                ▼                ▼
          Two-tower ANN     Item-to-item     Popular/trending
            retrieval         retrieval        retrieval
                 │                │                │
                 └────────────────┼────────────────┘
                                  ▼
                       Candidate union: 500–1,000
                                  │
                                  ▼
                       Multi-task ranking model
                       ├── P(long watch)
                       ├── Expected watch time
                       ├── P(like)
                       └── P(follow/share)
                                  │
                                  ▼
                          Top 50 candidates
                                  │
                                  ▼
                          Slate re-ranking
                       ├── Diversity
                       ├── Freshness
                       ├── Creator caps
                       ├── Exploration
                       └── Safety filters
                                  │
                                  ▼
                      Final top 10–20 videos
```

This matches the general multi-stage structure used in large-scale recommender-system references: retrieval narrows a large catalog to candidates, ranking applies more expensive models, and final filtering or ordering constructs the delivered slate.

#### Phase 1: Define the prediction (Optimization) problem
##### Recommendation unit
For each user at time t:
```python
Input:
- Historical interactions before t
- User features available before t
- Item features available before t
- Context available at t

Output:
- Ranked list of K videos
```

##### Primary objective
Use long-watch probability or normalized watch-time as the primary relevance label. For example:
$$\text{watch ratio}=
\min\left(\frac{\text{play time}}{\text{video duration}}, 1\right)$$

##### Secondary objectives
Train or evaluate:
- Like
- Follow
- Comment
- Share
- Negative feedback, when available
- Expected watch time
- Completion rate

Your final ranking utility is defined as: $$U(u,i)=
w_1\cdot P(\text{long watch})+w_2\cdot P(\text{watch ratio})+w_3\cdot P(\text{like})
+w_4\cdot P(\text{follow})-w_5 \cdot P(\text{negative feedback})$$
The weights should be configuration parameters, not hard-coded business truth.

#### Phase 2: perform data auditing and temporal splitting
Audit: Create a data report covering:
- Number of users and items
- Interaction volume
- Date range
- User-history lengths, Item interaction frequencies, Label prevalence, Missingness, Duplicate interactions
- Watch-time distributions
- Cold users and cold items
- Random-exposure versus standard-exposure records

**Split chronologically**
Never use a random interaction split for the principal evaluation. Instead we use
```python
Train: interactions before T1
Validation: T1 through T2
Test: T2 through T3
```

#### Phase 3: establish simple baselines
Before neural retrieval, implement three baselines.
##### 3.1 Global popularity
Rank videos by recent engagement: $score(i)=\alpha\cdot \text{views 7d} + \beta \cdot\text{long watch rate 7d}+\gamma \cdot \text{like rate 7d}$
##### 3.2 Popularity by user segment
Create segments based on features such as:
- Activity level
- Preferred video categories
- Historical watch-time distribution
- Device or demographic features, where appropriate
##### 3.3 Item-to-item co-occurrence
For each pair of videos watched by the same user within a session or time window, calculate co-occurrence.
- Use Cosine similarity, Lift, Pointwise mutual information, Time-decayed co-occurrence

These baselines give you:
- A fallback for cold users
- Additional retrieval sources
- A sanity check for the deep model

#### Phase 4: create point-in-time-correct features
Here we separate the features into four groups:
- User features: interaction_count_1d, interaction_count_7d, interaction_count_30d, average_watch_ratio_7d, like_rate_30d, preferred_category, category_entropy, days_since_last_session, average_session_length, historical_negative_feedback_rate
- Item features: video_duration, video_category, creator_id, item_age_hours, views_1d, views_7d, long_watch_rate_7d, like_rate_7d, engagement_velocity
- User-item cross features: user_category_affinity, user_creator_affinity, similarity_to_recent_history, two_tower_similarity, prior_creator_interaction_count, duration_difference_from_user_preference, item_novelty_for_user
- Contextual features: hour_of_day, day_of_week, time_since_previous_interaction, session_position, device_context, recent_session_category

#### Phase 5: build multi-source candidate retrieval
Production recommendation systems would usually benefit from heterogeneous candidate sources because each one captures a different relevance mechanism.

Build separate user and item encoders:
$$u=f_\theta(user, history, context),
\\ \mathbf{v}=g_{\phi}(item,\ metadata), \\  s(u,i)=u^Tv$$














