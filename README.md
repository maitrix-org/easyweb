<a name="readme-top"></a>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<div align="center">
  <img src="./Fast-web Icon.png" alt="FastAgent Logo" width="200">
  <h1 align="center">FastAgent, Open Source LLM Web Agent</h1>
  <!-- Change based on updated links or names in the future -->
  <!-- <a href="https://discord.gg/NdQD6eJzch"><img src="https://img.shields.io/badge/Discord-Join-blue?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a> -->
</div>
<!-- <hr> -->
Introducing FastAgent, an open-source LLM text-based all-purpose web agent. Choose an agent, choose a backend LLM, and let the agent browse the web to find the answers.\
This project was forked from <a href="https://github.com/AI-App/OpenDevin.OpenDevin">OpenDevin</a>, but has been modified since then.

<!--Update if repository changes name or location-->
<!--TODO: change the video link-->
![Demo Video](https://raw.githubusercontent.com/mingkaid/web-agent-application/demo_videos/video.mov)

## Getting Started

Firstly, to run anything, you need to first install Docker. You must be using Linux, Mac OS, or WSL on Windows.

Then, **clone** this repo:

```
git clone https://github.com/mingkaid/web-agent-application.git
```

Then, use poetry to **install** the necessary dependencies:

```
poetry install
```

Start the **backend**:

```
make start-backend
```

Start the **frontend**:

```
python frontend.py
```

Once all of this is launched, navigate to:

```
localhost:7860
```

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## Join The Community

We welcome you to join our [Discord](https://discord.gg/NdQD6eJzch) server! Feel free to contribute the following:

**Code Contributions:** Collaborate on building new agents, enhancing core features, improving the frontend and other interfaces, or creating sandboxing solutions.\
**Research and Evaluation:** Advance our understanding of LLMs in software engineering, assist in model evaluation, or propose enhancements.\
**Feedback and Testing:** Test FastAgent, identify bugs, recommend features, and share insights on usability.

## Acknowledgments
We would like to thank <a href="https://github.com/AI-App/OpenDevin.OpenDevin">OpenDevin</a> for the base code for this project.
<!--TODO: Anything else to add?-->

## Cite

<!--TODO: Should edit this if github changes-->
```
@misc{opendevin2024,
  author       = {{FastAgent Team}},
  title        = {{FastAgent: an Open-Source LLM Web Agent}},
  year         = {2025},
  version      = {v1.0},
  howpublished = {\url{https://github.com/mingkaid/web-agent-application}},
  note         = {Accessed: ENTER THE DATE YOU ACCESSED THE PROJECT}
}
```
