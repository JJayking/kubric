<div align="center" style="font-family: charter;">
<h1><i>Explainable Action Form Assessment</i>:</br> by Exploiting Multimodal Chain-of-Thoughts Reasoning</h1>

<br />
<a href="https://arxiv.org/abs/2512.15153" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2512.15153-red?logo=arxiv" height="20" />
</a>
<a href="https://github.com/MICLAB-BUPT/EFA" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/github-%23121011.svg?style=flat&logo=github&logoColor=white" height="20" />
</a>
<a href="https://www.kaggle.com/datasets/dd34dc6f49a960a31e03af896f85be526a72f8c9a684defd715c75d62bedbdc2" target="_blank">
    <img alt="Kaggle Dataset: CoT-AFA" src="https://img.shields.io/badge/Kaggle-CoT--AFA-20BEFF?logo=kaggle&logoColor=white" height="20" />
</a>
<div>
    Mengshi Qi<sup></sup>, </span>
    Yeteng Wu<sup></sup>, </span>
    Wulian Yun<sup></sup>, </span>
    Xianlin Zhang<sup></sup>, </span>
    Huadong Ma<sup></sup> </span>
</div>
<div>
    <sup>1</sup>State Key Laboratory of Networking and Switching Technology, Beijing University of Posts and Telecommunications, China </span>
</div>
 <!-- Assume this is Fig.1 from page1 -->
<p align="justify"><i>In real-world scenarios like fitness training and martial arts, evaluating if human actions conform to standard forms is essential for safety and effectiveness. Traditional video understanding focuses on what and where actions occur, but our work introduces the Action Form Assessment (AFA) task to assess how well actions are performed against objective standards. We present the CoT-AFA dataset, featuring diverse workout videos with Chain-of-Thought explanations that provide step-by-step reasoning, error analysis, and corrective solutions, enabling explainable feedback for skill improvement.</i></p>
</div>

## Release
- `2025-12-25` :rocket: Released the CoT-AFA dataset and EFA source code on GitHub.
- `2025-12-20` :hearts: Our paper is available on arXiv!
- `[PLACEHOLDER_FOR_FUTURE_UPDATES]` : Add future news here.

## Contents
- [Release](#release)
- [Contents](#contents)
- [CoT-AFA](#cot-afa)
- [Results](#results)
- [Citation](#citation)

## CoT-AFA
**Overview:** We introduce CoT-AFA, a diverse dataset for the Human Action Form Assessment (AFA) task. It includes 3,392 videos (364,812 frames) of fitness and martial arts actions, with annotations for action categories, standardization (standard/non-standard), multiple viewpoints, and Chain-of-Thought text explanations. The dataset supports tasks like action classification, quality assessment, and explainable feedback generation.
<img src="docs/resources/benchmark-stats.png" width="100%"/> <!-- Assume this combines Fig.2 and Fig.3 from page3 -->

CoT-AFA features a three-level lexicon (workout mode, type, category) and multi-view annotations for comprehensive analysis.

| Dataset | Workout modes | Workout types | Action categories | Standard Videos | Non-standard Videos | Total Videos | Total Frames | CoT Text Explanations |
|---------|---------------|---------------|-------------------|-----------------|---------------------|--------------|--------------|-----------------------|
| CoT-AFA | 2             | 28            | 141               | 2,242           | 1,150               | 3,392        | 364,812      | 3,392                 |

## Results
Our Explainable Fitness Assessor (EFA) framework achieves significant improvements:
- Explanation generation: +16.0% in CIDEr
- Action classification: +2.7% in accuracy
- Quality assessment: +2.1% in accuracy

<img src="docs/resources/architecture.png" width="100%"/> <!-- Assume Fig.5 from page6 -->

These results highlight the effectiveness of multimodal fusion and Chain-of-Thought reasoning in AFA.

## Citation
If you find our paper, dataset, or code useful, please cite:
```
@article{qi2025explainable,
  title={Explainable Action Form Assessment by Exploiting Multimodal Chain-of-Thoughts Reasoning},
  author={Qi, Mengshi and Wu, Yeteng and Yun, Wulian and Zhang, Xianlin and Ma, Huadong},
  journal={arXiv preprint arXiv:2512.15153},
  year={2025}
}
```
