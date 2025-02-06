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
  <img src="./easy-web-icon.png" alt="Easy Web Logo" width="200">
  <h1 align="center">EasyWeb: Web Agents at Your Fingertips</h1>
  <!-- Change based on updated links or names in the future -->
  <a href="https://easyweb.maitrix.org"><img src="https://img.shields.io/badge/Demo-Up-green?logo=gradio&logoColor=white&style=for-the-badge" alt="Try out our public demo"></a>
  <a href="https://discord.gg/b5NEhRbvJg"><img src="https://img.shields.io/badge/Discord-Join-blue?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://x.com/MaitrixOrg"><img src="https://img.shields.io/badge/Maitrix.org-Follow-black?logo=x&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
</div>
<!-- <hr> -->

<br>

EasyWeb is an open platform for building and serving AI agents that interact with web browsers.

**Using a web agent has never been easier:** Just open EasyWeb's interface, enter your command, and watch the agent take care of your browser-related tasks, whether it be travel planning, online shopping, news gathering, or anything you can think of.

**Deploy once, use everywhere:** EasyWeb comes with a full package for deploying web agents as a service. Built on [OpenHands](https://github.com/All-Hands-AI/OpenHands), EasyWeb introduces a parallelized architecture capable of fulfilling multiple user requests simultaneously, and supports toggling your favorite agent and LLM available as APIs.

<!--Update if repository changes name or location-->
<!--TODO: change the video link-->

## Demos

**Note:** The demos below were recorded using **ReasonerAgent with World Model Planning (Full)** and **GPT-4o** between January 2 and 21, 2025. Specific websites and more detailed instructions were included in the prompts to help guide the agent's behavior.

**Task:** Plan a travel from Pittsburgh to ICML 2025.

https://github.com/user-attachments/assets/6fd43d9c-2945-4a4c-ae33-5148df91aa66

<details>
<summary><b>Expand to see more demos</b></summary>

**Task:** Find a round-trip ticket from Chicago to Dubai next month, traveling in economy class, with non-stop flights only, departing after 8 AM, and returning within two weeks.

https://github.com/user-attachments/assets/0a50df50-52eb-423f-a270-7d25ab783fee

**Task:** I want to buy a black mattress. Can you look at Amazon, eBay, and Mattress Firm and give me one good option from each?

https://github.com/user-attachments/assets/3fd31131-3262-492b-a86d-a36d209417ee

**Task:** I'd like to learn how local news outlets covered Trump's inauguration. Please find one article from each of the following websites: *Times of San Diego*, *The Tennessee Tribune*, and *MinnPost*, and summarize the details to me.

https://github.com/user-attachments/assets/ce822017-4603-48d4-b400-07fe32db6803

</details>

## News
- [2025/02] We released **ReasonerAgent: A Fully Open Source, Ready-to-Run Agent That Uses a Web Browser to Answer Your Queries**. Check out the blog [post](https://reasoner-agent.maitrix.org/)

## Getting Started

### 1. Requirements


* Linux, Mac OS, or [WSL on Windows](https://learn.microsoft.com/en-us/windows/wsl/install)
* [Docker](https://docs.docker.com/engine/install/) (For those on MacOS, make sure to allow the default Docker socket to be used from advanced settings!)
* [Python](https://www.python.org/downloads/) = 3.11
* [NodeJS](https://nodejs.org/en/download/package-manager) >= 18.17.1
* [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) = 1.8.4

Make sure you have all these dependencies installed before moving on to `make build`.

<details>
<summary><b>Expand for setup without sudo access</b></summary>

If you want to develop without system admin/sudo access to upgrade/install `Python` and/or `NodeJs`, you can use `conda` or `mamba` to manage the packages for you:

```bash
# Download and install Mamba (a faster version of conda)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# Install Python 3.11, nodejs, and poetry
mamba install python=3.11
mamba install conda-forge::nodejs
mamba install conda-forge::poetry=1.8.4
```

</details>



### 2. Build and Setup The Environment

Begin by building the project, which includes setting up the environment and installing dependencies.

```bash
make build
```

### 3. Run the Application

Once the setup is complete, launching EasyWeb is as simple as running a single command. This command starts both the backend and frontend servers seamlessly.

```bash
make run
```

### 4. Individual Server Startup and Scaling Service

<details>
<summary><b>Expand to see details for scaling service</b></summary>

- **Start the Backend Server:** If you prefer, you can start the backend server independently to focus on backend-related tasks or configurations.
    ```bash
    make start-backend
    ```
- **Start the Frontend Server:** Similarly, you can start the frontend server on its own to work on frontend-related components or interface enhancements.
    ```bash
    make start-frontend
    ```
- **Start Multiple Backend Servers** If you prefer, you can also start multiple backend servers together with ports $5000$ and onwards for running multiple requests (one request per backend), given that you have sufficient memory on the machine.
    ```bash
    make start-backends NUM_BACKENDS={number_of_your_choice} START_PORT=5000
    ```
    Once you started multiple backend port, please start the frontend using:
    ```bash
    poetry run python frontend.py --num-backends {num_backend_opened}
    ```
    Then you can duplicate the frontend link you just opened to start running parallel requests.

    We aim to support a more scalable approach to multiple backends going forward.

</details>

<br>

Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.

## Join The Community

We welcome you to join our [Discord](https://discord.gg/b5NEhRbvJg) server! Feel free to contribute the following:

**Code Contributions:** Collaborate on building new agents, enabling new browser / UI environments, enhancing core features, improving the frontend and other interfaces, or creating sandboxing solutions.

**Research and Evaluation:** Advance our understanding of LLMs in automation, assist in model evaluation, or propose enhancements.

**Feedback and Testing:** Test EasyWeb, identify bugs, recommend features, and share insights on usability.

## Acknowledgments
We would like to thank [OpenHands](https://github.com/All-Hands-AI/OpenHands) for the base code for this project.
<!--TODO: Anything else to add?-->

## Cite

<!--TODO: Should edit this if github changes-->
```
@software{easyweb2025,
  author = {Maitrix Team},
  title = {EasyWeb: Open Platform for Building and Serving Web-Browsing Agents},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/maitrix-org/easyweb}
}
```
