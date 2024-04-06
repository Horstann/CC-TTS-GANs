# Generative Adversarial Networks for Time Series Simulations under Continuous Conditions
---
## Introduction
Computer-generated time series simulations have been heavily used in banks and hedge funds for risk management, pricing, volatility trading, hedging and more. Traditionally, these simulations use Monte-Carlo algorithms, which usually fail to capture fat tails or unsymmetric distributions.
<br>
To address this shortcoming, we propose the use of generative deep learning frameworks, in particular, Generative Adversarial Networks (GANs) given their strong capability in simulating synthetic data (images). However, there are 2 main challenges:
- Recent research on time-series GANs rarely focus on conditional generation, which could render such models useless in an industry setting where the incorporation of context is crucial.
- Conditional GANs have traditionally only used categorical conditions, while continuous conditions are much less straightforward to sample and train on. However, many financial indicators/metrics are often continuous in nature.

To this end, we propose the CC-TTS-GANs (Continuous Conditional Transformer-Based Time-Series GANs). CC-TTS-GANs possess the ability to incorporate some prior condition (such as implied volatility or other market indicators) which are assumed to be continuous in nature, to inform their own simulations. We then explore some of their applications in finance.

## Usage
To start the model training process, simply run `main.py`. <br>
All models are auto-saved in `logs/`. <br>
You can visualise model results in `Visualosation.ipynb`.

## Credits
This research wouldn't have been possible without the papers and repositories below.
[1] Li, X., Metsis, V., Wang, H., & Ngu, A. H. H. (2022, June 26). TTS-GAN: a transformer-based Time-Series Generative adversarial network. arXiv.org.https://arxiv.org/abs/2202.02691
[2] Ding, X., Wang, Y., Xu, Z., Welch, W. J., & Wang, Z. J. (2023, October 22). CCGAN: Continuous conditional generative adversarial networks for image generation. OpenReview. https://openreview.net/forum?id=PrzjugOsDeE