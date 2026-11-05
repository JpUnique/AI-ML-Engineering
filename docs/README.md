# AI/ML Learning Journey

A hands-on repository documenting my learning journey in Artificial
Intelligence and Machine Learning, from foundational concepts to
practical implementations and, eventually, AI engineering and agentic
systems.

This repository is organised by learning stage and lesson. Each
practical project includes its source code, experiment outputs,
visualisations, and technical documentation so that the work is
reproducible and easy to review.

## Learning objectives

-   Build a solid understanding of AI, machine learning, deep learning,
    and modern AI systems.
-   Implement algorithms rather than only study their theory.
-   Learn how to prepare data, train models, evaluate results, and
    interpret model behaviour.
-   Develop practical projects that demonstrate engineering and
    problem-solving ability.
-   Build toward LLM applications, AI agents, and production-oriented AI
    engineering.

## Repository structure

The repository will grow as I complete each lesson. A suggested
structure is:

``` text
ai-ml-learning/
├── README.md
├── stage-01-ai-and-ml-foundations/
│   ├── lesson-01-ai-introduction/
│   ├── lesson-02-history-of-ai/
│   ├── lesson-03-traditional-programming-vs-ai/
│   ├── lesson-04-ai-vs-ml-vs-dl/
│   ├── lesson-05-supervised-learning/
│   └── lesson-06-unsupervised-learning/
│       └── customer-segmentation-ml/
│           ├── README.md
│           ├── TECHNICAL_REPORT.md
│           ├── main.py
│           ├── requirements.txt
│           ├── data/
│           ├── plots/
│           └── src/
├── stage-02-deep-learning/
├── stage-03-neural-networks/
├── stage-04-generative-ai-and-llms/
├── stage-05-agentic-ai/
└── projects/
```

The exact folder names may evolve as the curriculum develops. Each
lesson folder should contain a concise lesson README or notes, while
substantial practical work should live in its own project directory.

## Current progress

### Stage 1 --- AI and Machine Learning Foundations

  -----------------------------------------------------------------------
  Lesson                  Topic                   Status
  ----------------------- ----------------------- -----------------------
  1.1                     What is Artificial      Completed
                          Intelligence?           

  1.2                     History and evolution   Completed
                          of AI                   

  1.3                     Traditional programming Completed
                          vs. AI                  

  1.4                     AI vs. Machine Learning Completed
                          vs. Deep Learning       

  1.5                     Supervised Learning     Completed

  1.6                     Unsupervised Learning   Completed --- practical
                                                  project implemented

  1.7 onward              Upcoming lessons        In progress / planned
  -----------------------------------------------------------------------

Progress labels will be updated as each lesson and practical exercise is
completed.

## Completed practical: Customer Segmentation Using Unsupervised Learning

**Location:**
`stage-01-ai-and-ml-foundations/lesson-06-unsupervised-learning/customer-segmentation-ml/`

This project uses a synthetic dataset of 600 customer records to explore
customer groupings and unusual observations without predefined target
labels.

Implemented techniques:

-   **K-Means:** grouped customers into three clusters. The observed
    silhouette score was `0.6661`.
-   **Cluster-count evaluation:** compared K values from 2 to 7 using
    inertia and silhouette score. K=3 achieved the highest silhouette
    score among the tested values.
-   **DBSCAN:** discovered two clusters with the selected parameters
    (`eps=0.8`, `min_samples=5`) and identified zero noise points.
-   **Isolation Forest:** flagged 30 records as anomalous using a
    contamination setting of `0.05`.
-   **PCA:** reduced the three standardised input features to two
    principal components, retaining `97.25%` of the variance.

The dataset contains monthly spending, transactions per month, and
account age. It is synthetically generated with three balanced
underlying profiles, so the results are educational demonstrations
rather than evidence about a real customer population.

See the project's
[`TECHNICAL_REPORT.md`](stage-01-ai-and-ml-foundations/lesson-06-unsupervised-learning/customer-segmentation-ml/TECHNICAL_REPORT.md)
for methodology, results, interpretation, limitations, and future
improvements.

## Tools and technologies

Tools used so far include:

-   Python
-   NumPy
-   Pandas
-   Scikit-learn
-   Matplotlib
-   Git and GitHub

Additional technologies will be added as they are used in later lessons
and projects.

## How to run a practical project

Each project should document its own setup instructions. For the current
customer segmentation project, navigate to its directory, create and
activate a virtual environment, install dependencies, and run the entry
point:

``` bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

On Windows, activate the environment with:

``` powershell
venv\Scripts\activate
```

Generated datasets and plots are written to the project directories by
the scripts. Check each project's README for exact output locations.

## Documentation and reproducibility

For each substantial practical project, I aim to include:

1.  A project README explaining the problem, setup, and how to run it.
2.  A technical report explaining the methodology, implementation,
    results, and limitations.
3.  Source code organised into focused modules.
4.  Relevant plots and evaluation outputs.
5.  A dependency file such as `requirements.txt`.

Datasets, model files, credentials, virtual environments, and other
generated artefacts should be committed only when appropriate. Large
datasets and secrets should not be pushed to GitHub. A `.gitignore` file
should exclude items such as `venv/`, `__pycache__/`, `.env`, and
temporary files.

## Learning approach

This repository is a record of practical learning. The goal is not
simply to collect code snippets, but to understand:

-   why a method is appropriate for a problem;
-   how the implementation works;
-   how to evaluate and interpret its output;
-   where the approach can fail;
-   how the solution could be improved for real-world use.

I will continue updating this repository as I progress through the AI/ML
curriculum and build more advanced projects.
