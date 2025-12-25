<div align="center" style="font-family: charter;">
<h1><img src="https://via.placeholder.com/50x50/000000/FFFFFF?text=AFA" width="4%"/> Explainable Action Form Assessment by Exploiting Multimodal Chain-of-Thoughts Reasoning</h1>
<img src="docs/resources/teaser_fig1.png" width="50%"/>
<!-- Note: You should save Figure 1 from the PDF as teaser_fig1.png -->
<br />
<a href="https://arxiv.org/abs/2512.15153" target="_blank">
<img alt="arXiv" src="https://img.shields.io/badge/arXiv-2512.15153-red?logo=arxiv" height="20" />
</a>
<a href="[INSERT_PROJECT_PAGE_LINK]" target="_blank">
<img alt="Website" src="https://img.shields.io/badge/🌎_Website-EFA-blue.svg" height="20" />
</a>
<a href="https://github.com/MICLAB-BUPT/EFA" target="_blank">
<img alt="GitHub" src="https://img.shields.io/badge/github-%23121011.svg?style=flat&logo=github&logoColor=white" height="20" />
</a>
<a href="[INSERT_HUGGINGFACE_LINK]" target="_blank">
<img alt="HF Dataset: CoT-AFA" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Dataset-CoT--AFA-ffc107?color=ffc107&logoColor=white" height="20" />
</a>
<div>
<a href="[INSERT_LINK_IF_AVAILABLE]" target="_blank">Mengshi Qi</a><sup>1</sup>,</span>
<a href="[INSERT_LINK_IF_AVAILABLE]" target="_blank">Yeteng Wu</a><sup>1</sup>, </span>
<a href="[INSERT_LINK_IF_AVAILABLE]" target="_blank">Wulian Yun</a><sup>1</sup>,</span>
<a href="[INSERT_LINK_IF_AVAILABLE]" target="_blank">Xianlin Zhang</a><sup>1</sup>,</span>
<a href="[INSERT_LINK_IF_AVAILABLE]" target="_blank">Huadong Ma</a><sup>1</sup></span>
</div>
<div>
<sup>1</sup>State Key Laboratory of Networking and Switching Technology, Beijing University of Posts and Telecommunications&emsp;
</div>
<img src="docs/resources/architecture_fig5.png" width="100%"/>
<!-- Note: You should save Figure 5 from the PDF as architecture_fig5.png -->
<p align="justify"><i>In real-world scenarios—from fitness coaching to rehabilitation—simply knowing "what" action is being performed isn't enough; we need to know "how well" it adheres to objective standards. While current Action Quality Assessment (AQA) relies on subjective scoring, we introduce a new task: <b>Human Action Form Assessment (AFA)</b>. AFA focuses on whether an action meets standard forms and provides actionable feedback. To tackle this, we propose the <b>CoT-AFA</b> dataset and the <b>Explainable Fitness Assessor (EFA)</b> framework. Unlike traditional methods that offer isolated tips, our approach utilizes Multimodal Chain-of-Thought (CoT) reasoning to diagnose errors, explain the biomechanical consequences (e.g., "this causes strain on the shoulders"), and propose concrete solutions, bridging the gap between AI perception and expert-level coaching.</i></p>
</div>
Release
2025-12-17 :hearts: The paper Explainable Action Form Assessment by Exploiting Multimodal Chain-of-Thoughts Reasoning is released on arXiv.
2025-12-17 :rocket: The CoT-AFA dataset and source code are available at MICLAB-BUPT/EFA.
CoT-AFA Benchmark
Overview: We construct CoT-AFA, a diverse video dataset designed for the Human Action Form Assessment task. Unlike existing AQA datasets that lack explainability, CoT-AFA contains rich multi-level annotations including lexicon, standard/non-standard labels, duration, viewpoints, and crucial Chain-of-Thought text explanations.
<img src="docs/resources/dataset_stats_fig2.png" width="100%"/>
<!-- Note: You should save Figure 2 from the PDF as dataset_stats_fig2.png -->
CoT-AFA features:
Scale: 3,392 videos with over 364k frames.
Diversity: Covers 2 workout modes (Manual, Apparatus), 28 workout types, and 141 action categories.
Explainability: Includes 3,392 Chain-of-Thought text explanations generated via a rigorous pipeline involving LLMs (Gemini 2.0), VLMs (VideoChat), and human expert verification.
Objective Standards: Distinct separation between "Standard Form" and "Non-Standard Form" based on professional fitness guidelines.
Results
Quantitative Performance: We evaluate our EFA framework against state-of-the-art baselines on three tasks: Explanation Generation, Action Classification, and Quality Assessment. EFA achieves significant improvements, particularly in generating explainable feedback (+16.0% in CIDEr).
Method	BLEU	METEOR	CIDEr	ROUGE-L	Class Acc	Quality Acc
C3DAVG	39.4	18.8	16.1	30.3	-	-
SWINBERT	43.2	23.5	26.0	35.8	-	-
LAVENDER	47.1	22.8	28.7	36.0	-	-
RICA²	44.4	21.0	27.3	33.8	0.819	0.814
EFA (Ours)	49.2	23.8	33.3	37.7	0.959 (Top5)	0.818
Qualitative Visualization:
Below are examples of EFA's text generation. The model successfully identifies specific errors (e.g., "hunching shoulders") and provides logical reasoning for correction, matching the Ground-Truth expert annotations.
<img src="docs/resources/results_fig7.png" width="100%"/>
<!-- Note: You should save Figure 7 from the PDF as results_fig7.png -->
