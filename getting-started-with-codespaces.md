# Using Codespaces to work with the "Practical Deep Learning for Coders" course


To get started, create a codespace for this repository by clicking this 👇 

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=master&repo=485606685)

A codespace will open in a web-based version of Visual Studio Code.

**Note**: Dev containers is an open spec which is supported by [GitHub Codespaces](https://github.com/codespaces) and [other supporting tools](https://containers.dev/supporting).

## Opening a notebook

The [dev container](.devcontainer/devcontainer.json) is fully configured with software and [machine learning libraries](.devcontainer/requirements.txt) needed for this course.

In the VS Code editor, open any notebook file and start executing the notebook's cells.

## Opening your codespace in JupyterLab

You can open your codespace in JupyterLab from the "Your codespaces" page at [github.com/codespaces](https://github.com/codespaces), or by using [GitHub CLI](https://docs.github.com/en/codespaces/developing-in-codespaces/opening-an-existing-codespace?tool=cli#opening-an-existing-codespace) with `gh codespace jupyter`. For more information, see "[Opening an existing codespace](https://docs.github.com/en/codespaces/developing-in-codespaces/opening-an-existing-codespace)".

## GPU-powered Codespaces

GPU-powered Codespaces are now available in limited beta. Having access to a GPU from within a codespace allows developers to run complex Machine Learning models much more quickly. 

To request access to the GPU machine types, or any additional machine type, [please complete the sign up form](https://github.surveymonkey.com/r/Y75GX9T).

Once, GPU is enabled and configured for your codespace, uncomment [this section](.devcontainer/devcontainer.json#L9-L13) which installs NVIDIA CUDA.

**Note**: Notebooks [09-small-models-road-to-the-top-part-2](09-small-models-road-to-the-top-part-2.ipynb) and [10-scaling-up-road-to-the-top-part-3](10-scaling-up-road-to-the-top-part-3.ipynb) requires a powerful machine to ensure that the kernel does not crash. Hence, some notebook cells for these two notebooks might not execute without a GPU-powered codespace.
