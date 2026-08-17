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


## Data Sources:
* [Kuairand Website: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos](https://kuairand.com/)
* [Arxiv](https://arxiv.org/abs/2208.08696): KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos
* [Github Page](https://github.com/chongminggao/KuaiRand)

## Project 